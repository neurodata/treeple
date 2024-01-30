import numpy as np
from numbers import Integral, Real
from joblib import Parallel, delayed
from sklearn.base import _fit_context
from warnings import warn

from scipy.sparse import issparse

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.multiclass import (
    type_of_target,
)
from sklearn.utils.validation import (
    _check_sample_weight,
)
from .._lib.sklearn.tree._tree import DOUBLE, DTYPE
from .._lib.sklearn.ensemble._forest import (
    _parallel_build_trees,
)
from sktree import HonestForestClassifier



def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    XXX: Note this is copied from sklearn. We override the ability
    to sample a higher number of bootstrap samples to enable sampling
    closer to 80% unique training data points for in-bag computation.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0.0, 1.0]`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, Integral):
        # if max_samples > n_samples:
        #     msg = "`max_samples` must be <= n_samples={} but got value {}"
        #     raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, Real):
        return round(n_samples * max_samples)
        # return max(round(n_samples * max_samples), 1)



class PosteriorForest(HonestForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        stratify=False,
        honest_bootstrap=False,
        permute_per_tree=False,
    ):
        super().__init__(
            n_estimators,
            criterion,
            splitter,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_features,
            max_leaf_nodes,
            min_impurity_decrease,
            bootstrap,
            oob_score,
            n_jobs,
            random_state,
            verbose,
            warm_start,
            class_weight,
            ccp_alpha,
            max_samples,
            honest_prior,
            honest_fraction,
            tree_estimator,
            stratify,
            honest_bootstrap,
        )
        self.permute_per_tree = permute_per_tree

    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, classes=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        MAX_INT = np.iinfo(np.int32).max
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            force_all_finite=False,
        )
        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.
        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = estimator._compute_missing_values_in_feature_mask(
            X, estimator_name=self.__class__.__name__
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y, classes=classes)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if self.max_bins is not None:
            # `_openmp_effective_n_threads` is used to take cgroups CPU quotes
            # into account when determine the maximum number of threads to use.
            n_threads = _openmp_effective_n_threads()

            # Bin the data
            # For ease of use of the API, the user-facing GBDT classes accept the
            # parameter max_bins, which doesn't take into account the bin for
            # missing values (which is always allocated). However, since max_bins
            # isn't the true maximal number of bins, all other private classes
            # (binmapper, histbuilder...) accept n_bins instead, which is the
            # actual total number of bins. Everywhere in the code, the
            # convention is that n_bins == max_bins + 1
            n_bins = self.max_bins + 1  # + 1 for missing values
            self._bin_mapper = _BinMapper(
                n_bins=n_bins,
                # is_categorical=self.is_categorical_,
                known_categories=None,
                random_state=random_state,
                n_threads=n_threads,
            )

            # XXX: in order for this to work with the underlying tree submodule's Cython
            # code, we need to convert this into the original data's DTYPE because
            # the Cython code assumes that `DTYPE` is used.
            # The proper implementation will be a lot more complicated and should be
            # tackled once scikit-learn has finalized their inclusion of missing data
            # and categorical support for decision trees
            X = self._bin_data(X, is_training_data=True)  # .astype(DTYPE)
        else:
            self._bin_mapper = None

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not " "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            if self.permute_per_tree:
                # TODO: refactor to make this a more robust implementation
                permutation_arr_per_tree = [
                    random_state.choice(self._n_samples, size=self._n_samples, replace=False)
                    for _ in range(self.n_estimators)
                ]
                if sample_weight is None:
                    sample_weight = np.ones((self._n_samples,))

                trees = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    prefer="threads",
                )(
                    delayed(_parallel_build_trees)(
                        t,
                        self.bootstrap,
                        X,
                        y[perm_idx],
                        sample_weight[perm_idx],
                        i,
                        len(trees),
                        verbose=self.verbose,
                        class_weight=self.class_weight,
                        n_samples_bootstrap=n_samples_bootstrap,
                        missing_values_in_feature_mask=missing_values_in_feature_mask,
                        classes=classes,
                    )
                    for i, (t, perm_idx) in enumerate(
                        zip(
                            trees,
                            permutation_arr_per_tree,
                        )
                    )
                )
            else:
                trees = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    prefer="threads",
                )(
                    delayed(_parallel_build_trees)(
                        t,
                        self.bootstrap,
                        X,
                        y,
                        sample_weight,
                        i,
                        len(trees),
                        verbose=self.verbose,
                        class_weight=self.class_weight,
                        n_samples_bootstrap=n_samples_bootstrap,
                        missing_values_in_feature_mask=missing_values_in_feature_mask,
                        classes=classes,
                    )
                    for i, t in enumerate(trees)
                )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score and (n_more_estimators > 0 or not hasattr(self, "oob_score_")):
            y_type = type_of_target(y)
            if y_type == "unknown" or (
                self._estimator_type == "classifier" and y_type == "multiclass-multioutput"
            ):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )

            if callable(self.oob_score):
                self._set_oob_score_and_attributes(X, y, scoring_function=self.oob_score)
            else:
                self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
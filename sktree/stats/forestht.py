import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.model_selection import train_test_split

from sktree import HonestForestClassifier
from sktree._lib.sklearn.ensemble._forest import ForestClassifier
from sktree._lib.sklearn.tree import DecisionTreeClassifier

from .utils import METRIC_FUNCTIONS, _compute_null_distribution_coleman, _pvalue


def tree_posterior(
    tree: DecisionTreeClassifier,
    X: ArrayLike,
    y: ArrayLike,
    covariate_index: ArrayLike = None,
    test_size: float = 0.2,
    seed: int = None,
) -> ArrayLike:
    """Compute the posterior from each tree on the "OOB" samples.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        The tree to compute the posterior from.
    X : ArrayLike of shape (n_samples, n_features)
        The data matrix.
    y : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    covariate_index : ArrayLike of shape (n_covariates,), optional
        The indices of the covariates to permute, by default None, which
        does not permute any columns.
    test_size : float, optional
        The size of the OOB set of samples, by default 0.2.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    posterior : ArrayLike of shape (n_samples, n_outputs)
        The predicted posterior probabilities for each OOB sample from the tree.
        For any in-bag samples, the posterior is NaN.
    """
    rng = np.random.default_rng(seed)

    indices = np.arange(X.shape[0])

    if covariate_index is not None:
        # perform permutation of covariates
        index_arr = rng.choice(indices, size=X.shape[0], replace=False, shuffle=False)
        perm_X_cov = X[index_arr, covariate_index]
        X[:, covariate_index] = perm_X_cov

    # XXX: we can replace this using Forest's generator for the in-bag/oob sample indices when
    # https://github.com/scikit-learn/scikit-learn/pull/26736 is merged
    X_train, X_test, y_train, _, _, indices_test = train_test_split(
        X, y, indices, test_size=test_size
    )

    # individual tree permutation of y labels
    tree.fit(X_train, y_train)
    y_pred = tree.predict_proba(X_test)[:, 1]

    # Fill test set posteriors & set rest NaN
    posterior = np.full(y.shape, np.nan)
    posterior[indices_test] = y_pred

    return posterior


class ForestHT(MetaEstimatorMixin):
    """Forest hypothesis testing.

    For example, this allows Mutual information for gigantic hypothesis testing (MIGHT)
    via ``metric="mi"``.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a `joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base tree estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.

    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    tree_estimator : object, default=None
        Type of decision tree classifier to use. By default `None`, which
        defaults to :class:`sklearn.tree.DecisionTreeClassifier`.

    alpha : float, optional
        Rejection threshold, by default 0.05.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_final_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_final_ : ArrayLike of shape (n_samples_final,)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        alpha=0.05,
    ):
        self.estimator = HonestForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            tree_estimator=tree_estimator,
        )
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.honest_prior = honest_prior
        self.honest_fraction = honest_fraction
        self.tree_estimator = tree_estimator

    def _statistic(
        self,
        estimator: ForestClassifier,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="auc",
        test_size=0.2,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func = METRIC_FUNCTIONS[metric]

        # first run a dummy fit on just two samples to initialize the
        # internal data structure of the forest
        estimator.fit(X[:2], y[:2])

        # Fit each tree and ompute posteriors with train test splits
        posterior = Parallel(n_jobs=self.n_jobs)(
            delayed(tree_posterior)(tree, X, y, covariate_index, test_size)
            for tree in (estimator.estimators_)
        )

        # Average all posteriors
        posterior_final = np.nanmean(posterior, axis=0)
        samples = np.argwhere(~np.isnan(posterior_final).any(axis=1))[0]
        y_true_final = y[samples, :]
        posterior_final = posterior_final[samples, :]
        stat = metric_func(y_true=y_true_final, y_pred=posterior_final, **metric_kwargs)

        if covariate_index is None:
            # Ignore all NaN values (samples not tested) -> (n_samples_final, n_outputs)
            # arrays of y and predicted posterior
            self.samples_ = samples
            self.y_true_final_ = y_true_final
            self.posterior_final_ = posterior_final
            self.stat_ = stat
        else:
            if not np.array_equal(samples, self.samples_):
                raise ValueError(
                    "The samples used in the final test are not the same as the "
                    "samples used in the initial test on the non-permuted samples."
                )

        return stat

    def statistic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="auc",
        test_size=0.2,
        **metric_kwargs,
    ):
        if covariate_index is None:
            estimator = self.estimator
        else:
            self.permuted_estimator_ = clone(self.estimator)
            estimator = self.permuted_estimator_

        return self._statistic(
            estimator,
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            test_size=test_size,
            **metric_kwargs,
        )

    def test(
        self,
        X,
        y,
        covariate_index: ArrayLike,
        metric: str = "auc",
        test_size: float = 0.2,
        n_repeats: int = 1000,
        **metric_kwargs,
    ):
        # first compute the test statistic on the un-permuted data
        observe_stat = self.statistic(X, y, metric=metric, test_size=test_size, **metric_kwargs)

        if self.method == "permutation":
            # compute the null distribution by computing a second forest `n_repeats` times using
            # permutations of the covariate
            null_dist = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    [
                        delayed(self.statistic)(
                            self, X, y, covariate_index, metric, test_size, **metric_kwargs
                        )
                        for _ in range(n_repeats)
                    ]
                )
            )
        elif self.method == "coleman":
            # first compute the test statistic on the un-permuted data
            permute_stat = self.statistic(
                X,
                y,
                covariate_index=covariate_index,
                metric=metric,
                test_size=test_size,
                **metric_kwargs,
            )

            # XXX: make sure train/test split before everything; rn there is prolly data leakage
            metric_star, metric_star_pi = _compute_null_distribution_coleman(
                X,
                y,
                self.estimator,
                self.permuted_estimator_,
                metric=metric,
                n_repeats=n_repeats,
                seed=self.random_state,
            )
            # metric^\pi - metric
            observe_stat = permute_stat - observe_stat

            # metric^\pi_j - metric_j
            null_dist = metric_star_pi - metric_star

        pval = _pvalue(observe_stat=observe_stat, permuted_stat=null_dist, correction=True)
        self.null_dist_ = null_dist
        return observe_stat, pval

# Authors: Ronan Perry, Sambit Panda
# Adopted from: https://github.com/neurodata/honest-forests

import numpy as np
from sklearn_fork.base import MetaEstimatorMixin
from sklearn_fork.tree import DecisionTreeClassifier
from sklearn_fork.tree._classes import BaseDecisionTree
from sklearn_fork.utils.multiclass import check_classification_targets
from sklearn_fork.utils.validation import check_is_fitted


class HonestTreeClassifier(MetaEstimatorMixin, BaseDecisionTree):
    """
    A deecision tree classifier with honest predictions.

    Parameters
    ----------
    estimator : object, default=None
        Instatiated tree of type BaseDecisionTree.
        If None, then DecisionTreeClassifier with default parameters will
        be used.

    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the leaves. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", returns numpy.nan.
        If "uniform", the prior tree posterior is 1/(number of
        classes). If "empirical", the prior tree posterior is the relative
        class frequency in the voting subsample.

    Attributes
    ----------
    estimator_ : object
        The child estimator template used to create the collection
        of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    empirical_prior_ : float
        Proportion of each class in the training labels y

    structure_indices_ : numpy.ndarray, shape=(n_structure,)
        Indices of training samples used to learn the structure

    honest_indices_ : numpy.ndarray, shape=(n_honest,)
        Indices of training samples used to learn leaf estimates

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
            and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
            Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
            https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    .. [5] S. Athey, J. Tibshirani, and S. Wager. "Generalized
            Random Forests", Annals of Statistics, 2019.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from honest_forests import HonestTreeClassifier
    >>> clf = HonestTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([0.93333333, 0.93333333, 1.        , 1.        , 0.93333333,
           0.8       , 0.8       , 0.93333333, 1.        , 1.        ])
    """

    def __init__(
        self,
        estimator=None,
        honest_fraction=0.5,
        honest_prior="empirical",
    ):
        if not estimator:
            self.estimator_ = DecisionTreeClassifier()
        self.estimator_ = estimator
        self.honest_fraction = honest_fraction
        self.honest_prior = honest_prior

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build an honest tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : HonestTreeClassifier
            Fitted estimator.
        """
        # Account for bootstrapping too
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],), dtype=np.float64)

        nonzero_indices = np.where(sample_weight > 0)[0]

        self.structure_indices_ = np.random.choice(
            nonzero_indices,
            int((1 - self.honest_fraction) * len(nonzero_indices)),
            replace=False,
        )
        self.honest_indices_ = np.setdiff1d(nonzero_indices, self.structure_indices_)

        sample_weight[self.honest_indices_] = 0

        # Learn structure on subsample
        self.estimator_.fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )

        if self.n_outputs_ > 1:
            raise NotImplementedError(
                "Multi-target honest trees not yet \
                implemented"
            )

        # update the number of classes, unsplit
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        check_classification_targets(y)
        y = np.copy(y).astype(int)
        # Normally called by super
        X = self.estimator_._validate_X_predict(X, True)
        # Fit leaves using other subsample
        honest_leaves = self.tree_.apply(X[self.honest_indices_])

        self.tree_.value[:, :, :] = 0
        for leaf_id, yval in zip(honest_leaves, y[self.honest_indices_, 0]):
            self.tree_.value[leaf_id][0, yval] += 1

        # preserve from underlying tree
        # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/tree/_classes.py#L202
        self._tree_classes_ = self.classes_
        self._tree_n_classes_ = self.n_classes_
        self.classes_ = []
        self.n_classes_ = []
        self.empirical_prior_ = []

        y_encoded = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
            self.empirical_prior_.append(
                np.bincount(y_encoded[:, k], minlength=classes_k.shape[0]) / y.shape[0]
            )
        y = y_encoded

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            self.empirical_prior_ = self.empirical_prior_[0]
            y = y[:, 0]

        return self

    def _inherit_estimator_attributes(self):
        """Initialize necessary attributes from the provided estimator"""
        self.classes_ = self.estimator_.classes_
        self.feature_importances_ = self.estimator_.feature_importances_
        self.max_features_ = self.estimator_.max_features_
        self.n_classes_ = self.estimator_.n_classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        self.feature_names_in_ = self.estimator_.feature_names_in_
        self.n_outputs_ = self.estimator_.n_outputs_
        self.tree_ = self.estimator_.tree_

    def _empty_leaf_correction(self, proba, normalizer):
        """Leaves with empty posteriors are assigned values"""
        zero_mask = proba.sum(axis=1) == 0.0
        if self.honest_prior == "empirical":
            proba[zero_mask] = self.empirical_prior_
        elif self.honest_prior == "uniform":
            proba[zero_mask] = 1 / self.n_classes_
        elif self.honest_prior == "ignore":
            proba[zero_mask] = np.nan
        else:
            raise ValueError(f"honest_prior {self.honest_prior} not a valid input.")

        return proba

    def _impute_missing_classes(self, proba):
        """Due to splitting, provide proba outputs for some classes"""
        new_proba = np.zeros((proba.shape[0], self.n_classes_))
        for i, old_class in enumerate(self._tree_classes_):
            j = np.where(self.classes_ == old_class)[0][0]
            new_proba[:, j] = proba[:, i]

        return new_proba

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self.estimator_._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, : self._tree_n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            if self._tree_n_classes_ != self.n_classes_:
                proba = self._impute_missing_classes(proba)
            proba = self._empty_leaf_correction(proba, normalizer)

            return proba

        else:
            raise NotImplementedError(
                "Multi-target honest trees not yet \
                implemented"
            )

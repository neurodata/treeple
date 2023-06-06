# Authors: Ronan Perry
# Adopted from: https://github.com/neurodata/honest-forests

import threading

import numpy as np
from joblib import Parallel, delayed
from sklearn_fork.ensemble._base import _partition_estimators
from sklearn_fork.ensemble._forest import ForestClassifier
from sklearn_fork.tree import DecisionTreeClassifier
from sklearn_fork.utils.validation import check_is_fitted

from ..tree import HonestTreeClassifier


class HonestForestClassifier(ForestClassifier):
    """
    A forest classifier with honest leaf estimates.

    Parameters
    ----------
    estimator : object, default=None
        Instatiated tree of type BaseDecisionTree.
        If None, then DecisionTreeClassifier with default parameters will
        be used.

    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.

    Attributes
    ----------
    estimator_ : HonestTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of HonestTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    honest_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed on each sample, including only the trees
        for which it was in the honest subsample. It is possible that a sample
        is never in the honest subset in which case `honest_decision_function_`
        might contain NaN. This attribute exists when `honest_score_` is True.

    structure_indices_ : list of lists, shape=(n_estimators, n_structure)
        Indices of training samples used to learn the structure

    honest_indices_ : list of lists, shape=(n_estimators, n_honest)
        Indices of training samples used to learn leaf estimates

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    Honesty is a feature of trees that enables unbiased estimates of confidence
    intervals. The default implementation here is using double sampling to
    implement honesty. The amount of samples used for learning split nodes vs
    leaf nodes is controlled by the ``honest_fraction`` parameter. This forest
    classifier is a "meta-estimator" because any tree model can be used in the
    classification process, while enabling honesty separates the data used for
    split and leaf nodes.

    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] S. Athey, J. Tibshirani, and S. Wager. "Generalized
            Random Forests", Annals of Statistics, 2019.

    Examples
    --------
    >>> from honest_forests.estimators import HonestForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = HonestForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    HonestForestClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """

    def __init__(
        self,
        estimator=None,
        honest_fraction=0.5,
        honest_prior="empirical",
    ):
        if not estimator:
            estimator = DecisionTreeClassifier()
        self.estimator_ = HonestTreeClassifier(estimator, honest_fraction, honest_prior)
        self.honest_fraction = honest_fraction
        self.honest_prior = honest_prior

    def fit(self, X, y, sample_weight=None):
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

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super().fit(X, y, sample_weight)
        classes_k, y_encoded = np.unique(y, return_inverse=True)
        self.empirical_prior_ = np.bincount(y_encoded, minlength=classes_k.shape[0]) / len(y)

        # Compute honest decision function
        self.honest_decision_function_ = self._predict_proba(
            X, indices=self.honest_indices_, impute_missing=np.nan
        )
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return self._predict_proba(X)

    def _predict_proba(self, X, indices=None, impute_missing=None):
        """predict_proba helper class"""
        X = self._validate_X_predict(X)
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        posteriors = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        lock = threading.Lock()

        if indices is None:
            indices = [None] * self.n_estimators
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(tree, X, posteriors, lock, idx)
            for tree, idx in zip(self.estimators_, indices)
        )

        # Normalize to unit length, due to prior weighting
        zero_mask = posteriors.sum(1) == 0
        posteriors[~zero_mask] /= posteriors[~zero_mask].sum(1, keepdims=True)
        if impute_missing is None:
            posteriors[zero_mask] = self.empirical_prior_
        else:
            posteriors[zero_mask] = impute_missing

        return posteriors

    @property
    def structure_indices_(self):
        check_is_fitted(self)
        return [tree.structure_indices_ for tree in self.estimators_]

    @property
    def honest_indices_(self):
        check_is_fitted(self)
        return [tree.honest_indices_ for tree in self.estimators_]


def _accumulate_prediction(tree, X, out, lock, indices=None):
    """
    See https://github.com/scikit-learn/scikit-learn/blob/
    95119c13af77c76e150b753485c662b7c52a41a2/sklearn/ensemble/_forest.py#L460
    This is a utility function for joblib's Parallel.
    It cannot be placed in ForestClassifier or ForestRegressor due to joblib's
    compatibility issue with pickle.
    """

    if indices is None:
        indices = np.arange(X.shape[0])
    proba = tree.tree_.predict(X[indices])
    proba = proba[:, : tree._tree_n_classes_]
    normalizer = proba.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    proba /= normalizer

    if tree._tree_n_classes_ != tree.n_classes_:
        proba = tree._impute_missing_classes(proba)
    proba = tree._empty_leaf_correction(proba, normalizer)

    with lock:
        out[indices] += proba

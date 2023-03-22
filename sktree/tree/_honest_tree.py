"""Module for tree-based estimators"""
# Authors: Ronan Perry
# Adopted from: https://github.com/rflperry/ProgLearn/blob/UF/
# License: MIT
# and https://github.com/scikit-learn/scikit-learn/
# License: BSD 3 clause

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets


class HonestTreeClassifier(DecisionTreeClassifier):
    """
    A deecision tree classifier with honest predictions.

    Parameters
    ----------
    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the leaves. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
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

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", returns numpy.nan.
        If "uniform", the prior tree posterior is 1/(number of
        classes). If "empirical", the prior tree posterior is the relative
        class frequency in the voting subsample.

    Attributes
    ----------
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

    n_features_ : int
        The number of features when ``fit`` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in 1.0 and will be removed in
           1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

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
        honest_fraction=0.5,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        honest_prior="empirical",
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
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
        if check_input:
            X, y = check_X_y(X, y)

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
        super().fit(
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
        # self.n_samples_, self.n_features_in_ = X.shape
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        check_classification_targets(y)
        y = np.copy(y).astype(int)
        # Normally called by super
        X = self._validate_X_predict(X, True)
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
        X = self._validate_X_predict(X, check_input)
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
            # all_proba = []

            # for k in range(self.n_outputs_):
            #     proba_k = proba[:, k, : self._tree_n_classes_[k]]
            #     normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            #     normalizer[normalizer == 0.0] = 1.0
            #     proba_k /= normalizer
            #     proba = self._impute_missing_classes(proba)
            #     proba_k = self._empty_leaf_correction(proba_k, normalizer)
            #     all_proba.append(proba_k)

            # return all_proba

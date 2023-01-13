"""
Manifold forest of trees-based ensemble methods.

Those methods include various random forest methods that operate on manifolds.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

from sklearn.base import TransformerMixin
from sklearn.ensemble._forest import BaseForest

from .tree import UnsupervisedDecisionTree


class UnsupervisedRandomForest(TransformerMixin, BaseForest):
    """Unsupervised random forest.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees to fit, by default 100.
        
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
        Note: This parameter is tree-specific.

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
          `max(1, int(max_features * n_features_in_))` features are considered at each
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
    bootstrap : bool, optional
        Whether to bootstrap, by default False.
    oob_score : bool, optional
        Whether to compute OOB score, by default False. This computes
        the OOB "score metric" on the not-used samples per tree.
    n_jobs : int, optional
        Number of CPUs to use in `joblib` parallelization for constructing trees,
        by default None.
    random_state : int, optional
        Random seed, by default None.
    verbose : int, optional
        Verbosity, by default 0.
    warm_start : bool, optional
        Whether to continue constructing trees from previous instant, by default False.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion='twomeans',
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
        max_samples=None,
    ) -> None:
        super().__init__(
            estimator=UnsupervisedDecisionTree(),  # type: ignore
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease


    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The 2D data matrix with columns as features and rows as samples.
        y : ArrayLike, optional
            Not used. Passed in for API consistency.
        sample_weight : ArrayLike of shape (n_samples), optional
            The samples weight, by default None.

        Returns
        -------
        output : ArrayLike of shape (n_samples,)
            The transformed output from passing `X` through the forest.
        """
        super().fit(X, y, sample_weight)
        # apply to the leaves
        output = self.apply(X)
        return output

    def fit(self, X, y=None, sample_weight=None):
        """Fit unsupervised forest.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The 2D data matrix with columns as features and rows as samples.
        y : ArrayLike, optional
            Not used. Passed in for API consistency.
        sample_weight : ArrayLike of shape (n_samples), optional
            The samples weight, by default None.

        Returns
        -------
        self : UnsupervisedRandomForest
            The fitted forest.
        """
        # Parameters are validated in fit_transform
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

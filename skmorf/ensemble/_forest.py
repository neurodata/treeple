"""
Forest of trees-based ensemble methods.

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble._forest import ForestClassifier

from ..tree import ManifoldObliqueDecisionTreeClassifier


class ManifoldObliqueRandomForestClassifier(ForestClassifier):
    """Manifold oblique random forest classifier (MORF).

    Parameters
    ----------
    base_estimator : [type]
        [description]
    n_estimators : int, optional
        [description], by default 100
    max_depth : [type], optional
        [description], by default None
    min_samples_split : int, optional
        [description], by default 2
    min_samples_leaf : int, optional
        [description], by default 1
    max_features : str, optional
        [description], by default "auto"
    bootstrap : bool, optional
        [description], by default True
    oob_score : bool, optional
        [description], by default False
    n_jobs : [type], optional
        [description], by default None
    random_state : [type], optional
        [description], by default None
    verbose : int, optional
        [description], by default 0
    warm_start : bool, optional
        [description], by default False
    class_weight : [type], optional
        [description], by default None
    max_samples : [type], optional
        [description], by default None
    """

    def __init__(
        self,
        base_estimator,
        data_shape,
        patch_min_shape,
        patch_max_shape,
        kernel_func,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        feature_combinations=1.5,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        **estimator_params,
    ):
        super().__init__(
            base_estimator=ManifoldObliqueDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "feature_combinations",
                "min_weight_fraction_leaf",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.feature_combinations = feature_combinations

        # unused by MORF rn
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

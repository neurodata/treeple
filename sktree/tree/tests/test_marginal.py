from typing import Any, Dict

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises

from sktree import (
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    PatchObliqueRandomForestClassifier,
    PatchObliqueRandomForestRegressor,
    UnsupervisedObliqueRandomForest,
    UnsupervisedRandomForest,
)
from sktree._lib.sklearn.ensemble._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from sktree._lib.sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sktree.tree._marginalize import apply_marginal

CLF_TREES = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "ExtraTreeClassifier": ExtraTreeClassifier,
}

REG_TREES = {
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "ExtraTreeRegressor": ExtraTreeRegressor,
}


FOREST_CLASSIFIERS = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

FOREST_TRANSFORMERS = {
    "RandomTreesEmbedding": RandomTreesEmbedding,
}

FOREST_REGRESSORS = {
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}

FOREST_CLUSTERING = {
    "UnsupervisedRandomForest": UnsupervisedRandomForest,
}

OBLIQUE_FORESTS = {
    "ObliqueRandomForestClassifier": ObliqueRandomForestClassifier,
    "ObliqueRandomForestRegressor": ObliqueRandomForestRegressor,
    "PatchObliqueRandomForestClassifier": PatchObliqueRandomForestClassifier,
    "PatchObliqueRandomForestRegressor": PatchObliqueRandomForestRegressor,
    "UnsupervisedObliqueRandomForest": UnsupervisedObliqueRandomForest,
}

FOREST_ESTIMATORS: Dict[str, Any] = dict()
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)
FOREST_ESTIMATORS.update(FOREST_REGRESSORS)
FOREST_ESTIMATORS.update(FOREST_TRANSFORMERS)
FOREST_ESTIMATORS.update(FOREST_CLUSTERING)

FOREST_CLASSIFIERS_REGRESSORS: Dict[str, Any] = FOREST_CLASSIFIERS.copy()
FOREST_CLASSIFIERS_REGRESSORS.update(FOREST_REGRESSORS)


ALL_TREES: dict = dict()
ALL_TREES.update(CLF_TREES)
ALL_TREES.update(REG_TREES)


def assert_array_not_equal(x, y):
    return assert_raises(AssertionError, assert_array_equal, x, y)


X_small = np.array(
    [
        [0, 0, 4, 0, 0, 0, 1, -14, 0, -4, 0, 0, 0, 0],
        [0, 0, 5, 3, 0, -4, 0, 0, 1, -5, 0.2, 0, 4, 1],
        [-1, -1, 0, 0, -4.5, 0, 0, 2.1, 1, 0, 0, -4.5, 0, 1],
        [-1, -1, 0, -1.2, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 1],
        [-1, -1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
        [-1, -2, 0, 4, -3, 10, 4, 0, -3.2, 0, 4, 3, -4, 1],
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -1, 0],
        [2, 8, 5, 1, 0.5, -4, 10, 0, 1, -5, 3, 0, 2, 0],
        [2, 0, 1, 1, 1, -1, 1, 0, 0, -2, 3, 0, 1, 0],
        [2, 0, 1, 2, 3, -1, 10, 2, 0, -1, 1, 2, 2, 0],
        [1, 1, 0, 2, 2, -1, 1, 2, 0, -5, 1, 2, 3, 0],
        [3, 1, 0, 3, 0, -4, 10, 0, 1, -5, 3, 0, 3, 1],
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 1.5, 1, -1, -1],
        [2.11, 8, -6, -0.5, 0, 10, 0, 0, -3.2, 6, 0.5, 0, -1, -1],
        [2, 0, 5, 1, 0.5, -2, 10, 0, 1, -5, 3, 1, 0, -1],
        [2, 0, 1, 1, 1, -2, 1, 0, 0, -2, 0, 0, 0, 1],
        [2, 1, 1, 1, 2, -1, 10, 2, 0, -1, 0, 2, 1, 1],
        [1, 1, 0, 0, 1, -3, 1, 2, 0, -5, 1, 2, 1, 1],
        [3, 1, 0, 1, 0, -4, 1, 0, 1, -2, 0, 0, 1, 0],
    ]
)

y_small = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("est_name", FOREST_ESTIMATORS)
def test_apply_marginal(est_name):
    # Create sample data
    S = np.array([1, 0, 5])  # Example marginalization indices
    n_samples = len(X_small)

    # Test with RandomForestClassifier (forest input)
    est_forest = FOREST_ESTIMATORS[est_name](n_estimators=5, random_state=0)
    est_forest.fit(X_small, y_small)  # Example fitting
    X_leaves_forest = apply_marginal(est_forest, X_small, S)
    assert X_leaves_forest.shape == (
        n_samples,
        est_forest.n_estimators,
    )  # Check the shape of the output
    assert_array_not_equal(X_leaves_forest, est_forest.apply(X_small))  # Check the output

    # without marginalization, the tree should be exactly traversed the same.
    assert_array_equal(apply_marginal(est_forest, X_small, []), est_forest.apply(X_small))


@pytest.mark.parametrize("est_name", OBLIQUE_FORESTS)
def test_apply_marginal_error(est_name):
    S = np.array([1, 0, 5])  # Example marginalization indices
    est_forest = OBLIQUE_FORESTS[est_name](n_estimators=5, random_state=0)
    est_forest.fit(X_small, y_small)  # Example fitting

    with pytest.raises(RuntimeError, match="only supports axis-aligned trees"):
        apply_marginal(est_forest, X_small, S)

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import datasets

from treeple.tree import (
    ExtraObliqueDecisionTreeClassifier,
    ExtraObliqueDecisionTreeRegressor,
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeRegressor,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)

ALL_TREES = [
    ExtraObliqueDecisionTreeRegressor,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeRegressor,
    ExtraObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
]

rng = np.random.RandomState(1)

# load digits dataset and randomly permute it
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]


def test_splitters():
    """Test that splitters are picklable."""
    X, y = digits.data, digits.target
    X = X.astype(np.float32)
    sample_weight = np.ones(len(y), dtype=np.float64).squeeze()
    y = y.reshape(-1, 1).astype(np.float64)
    # missing_values_in_feature_mask = None

    print(X.shape, y.shape)
    from treeple._lib.sklearn.tree._criterion import Gini
    from treeple.tree.manifold._morf_splitter import BestPatchSplitterTester

    criterion = Gini(1, np.array((0, 1), dtype=np.intp))
    max_features = 6
    min_samples_leaf = 1
    min_weight_leaf = 0.0
    monotonic_cst = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    random_state = np.random.RandomState(100)
    boundary = None
    feature_weight = None
    min_patch_dims = np.array((1, 1), dtype=np.intp)
    max_patch_dims = np.array((3, 1), dtype=np.intp)
    dim_contiguous = np.array((True, True))
    data_dims = np.array((8, 8), dtype=np.intp)

    feature_combinations = 1.5

    splitter = BestPatchSplitterTester(
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
        feature_combinations,
        min_patch_dims,
        max_patch_dims,
        dim_contiguous,
        data_dims,
        boundary,
        feature_weight,
    )
    splitter.init_test(
        X,
        y,
        sample_weight,
    )
    assert_array_equal(splitter.X_reshaped.reshape(-1, 64), X)

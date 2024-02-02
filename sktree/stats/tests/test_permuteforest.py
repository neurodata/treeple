import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets

from sktree.stats import PermutationHonestForestClassifier

# load the iris dataset (n_samples, 4)
# and randomly permute it
iris = datasets.load_iris()
seed = 12345
rng = np.random.default_rng(seed)

# remove third class
iris_X = iris.data[iris.target != 2]
iris_y = iris.target[iris.target != 2]

p = rng.permutation(iris_X.shape[0])
iris_X = iris_X[p]
iris_y = iris_y[p]


def test_permutationforest_errors():
    """Test permutation forest errors when training."""
    n_samples = 10
    est = PermutationHonestForestClassifier(n_estimators=10, random_state=0)

    # covariate index must be an iterable
    with pytest.raises(RuntimeError, match="covariate_index must be an iterable"):
        est.fit(iris_X[:n_samples], iris_y[:n_samples], covariate_index=0)

    # covariate index must be an iterable of ints
    with pytest.raises(RuntimeError, match="Not all covariate_index"):
        est.fit(iris_X[:n_samples], iris_y[:n_samples], covariate_index=[0, 1.0])

    # covariate index must not have numbers greater than
    with pytest.raises(ValueError, match="The length of the covariate index"):
        est.fit(
            iris_X[:n_samples],
            iris_y[:n_samples],
            covariate_index=np.arange(iris_X.shape[1] + 1, dtype=np.intp),
        )


@pytest.mark.parametrize("permute_per_tree", [True, False])
def test_inbag_samples_different_across_forest(permute_per_tree):
    """Test that inbag samples are different across trees."""
    n_estimators = 10
    est = PermutationHonestForestClassifier(
        n_estimators=n_estimators, random_state=0, permute_per_tree=permute_per_tree
    )

    X = iris_X
    y = iris_y
    est.fit(X, y)

    # covariate index when None is all the features
    assert_array_equal(est.covariate_index_, np.arange(X.shape[1], dtype=np.intp))

    # inbag samples should be different across trees when permute_per_tree=True
    permutation_samples_ = est.permutation_indices_
    permutation_samples_ground = permutation_samples_[0]
    assert not all(
        np.array_equal(permutation_samples_ground, permutation_samples_[idx])
        for idx in range(1, n_estimators)
    )

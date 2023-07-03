import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.ensemble import (
    ObliqueRandomForestClassifier,
    PatchObliqueRandomForestClassifier,
    UnsupervisedObliqueRandomForest,
    UnsupervisedRandomForest,
)
from sktree.neighbors import NearestNeighborsMetaEstimator

FORESTS = [
    ObliqueRandomForestClassifier,
    PatchObliqueRandomForestClassifier,
    UnsupervisedRandomForest,
    UnsupervisedObliqueRandomForest,
]


@pytest.mark.parametrize("forest", FORESTS)
def test_similarity_matrix(forest):
    n_samples = 200
    n_classes = 2
    n_features = 5

    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=12345
    )

    clf = forest(random_state=12345)
    clf.fit(X, y)
    sim_mat = clf.compute_similarity_matrix(X)

    assert sim_mat.shape == (n_samples, n_samples)
    assert np.allclose(sim_mat, sim_mat.T)
    assert np.all((sim_mat.diagonal() == 1))


@pytest.fixture
def sample_data():
    # Generate sample data for testing
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(random_state=0),
        DecisionTreeRegressor(random_state=0),
        ExtraTreeClassifier(random_state=0),
        ExtraTreeRegressor(random_state=0),
        RandomForestClassifier(random_state=0, n_estimators=10),
        ExtraTreesClassifier(random_state=0, n_estimators=10),
    ],
)
def test_nearest_neighbors_meta_estimator(sample_data, estimator):
    X, y = sample_data
    estimator.fit(X, y)

    meta_estimator = NearestNeighborsMetaEstimator(estimator)

    # Fit the meta-estimator
    meta_estimator.fit(X, y)

    # Test the fitted estimator attribute
    assert hasattr(meta_estimator, "estimator_")

    # Test the nearest neighbors estimator
    assert isinstance(meta_estimator.neigh_est_, NearestNeighbors)

    # Test the kneighbors method
    neigh_dist, neigh_ind = meta_estimator.kneighbors()
    assert neigh_dist.shape == (X.shape[0], meta_estimator.n_neighbors)
    assert neigh_ind.shape == (X.shape[0], meta_estimator.n_neighbors)

    # Test the radius_neighbors method
    neigh_dist, neigh_ind = meta_estimator.radius_neighbors(radius=0.5)
    assert neigh_dist.shape == (X.shape[0],)
    assert neigh_ind.shape == (X.shape[0],)


@parametrize_with_checks(
    [
        NearestNeighborsMetaEstimator(DecisionTreeClassifier(random_state=0)),
    ]
)
def test_sklearn_compatible_transformer(estimator, check):
    check(estimator)

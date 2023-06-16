import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

from sktree.tree._neighbors import NearestNeighborsMetaEstimator


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

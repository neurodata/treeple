import numpy as np
import pytest
from sklearn.datasets import make_blobs

from sktree.ensemble import (
    ObliqueRandomForestClassifier,
    PatchObliqueRandomForestClassifier,
    UnsupervisedObliqueRandomForest,
    UnsupervisedRandomForest,
)

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

    assert np.allclose(sim_mat, sim_mat.T)
    assert np.all((sim_mat.diagonal() == 1))

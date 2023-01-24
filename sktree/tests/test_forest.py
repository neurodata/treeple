import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree import UnsupervisedRandomForest


@parametrize_with_checks([UnsupervisedRandomForest()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_urf():
    n_samples = 100
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=2**4)

    clf = UnsupervisedRandomForest()
    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)
    assert score > 0.9

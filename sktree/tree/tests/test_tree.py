import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import UnsupervisedDecisionTree


@parametrize_with_checks([UnsupervisedDecisionTree(random_state=12)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in [
        # Cannot apply agglomerative clustering on < 2 samples
        "check_methods_subset_invariance",
        # clustering accuracy is poor when using TwoMeans on 1 single tree
        "check_clustering",
        # sample weights do not necessarily imply a sample is not used in clustering
        "check_sample_weights_invariance",
        # sample order is not preserved in predict
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()
    check(estimator)


def test_unsupervisedtree():
    n_samples = 10
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1234)

    clf = UnsupervisedDecisionTree(random_state=1234)
    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    print(sim_mat)

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score > 0.05

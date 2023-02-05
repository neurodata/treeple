import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree import UnsupervisedObliqueRandomForest, UnsupervisedRandomForest


@parametrize_with_checks(
    [
        UnsupervisedRandomForest(random_state=12345, n_estimators=50),
        UnsupervisedObliqueRandomForest(random_state=12345, n_estimators=50),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in [
        # Cannot apply agglomerative clustering on < 2 samples
        "check_methods_subset_invariance",
        # # sample weights do not necessarily imply a sample is not used in clustering
        "check_sample_weights_invariance",
        # # sample order is not preserved in predict
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize(
    "CLF_NAME, ESTIMATOR",
    [
        ("UnsupervisedRandomForest", UnsupervisedRandomForest),
        ("UnsupervisedObliqueRandomForest", UnsupervisedObliqueRandomForest),
    ],
)
def test_urf(CLF_NAME, ESTIMATOR):
    n_samples = 100
    n_classes = 2

    #
    if CLF_NAME == "UnsupervisedRandomForest":
        n_features = 5
        n_estimators = 50
        expected_score = 0.4
    else:
        n_features = 20
        n_estimators = 20
        expected_score = 0.9
    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=12345
    )

    clf = ESTIMATOR(n_estimators=n_estimators, random_state=12345)
    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # XXX: This should be > 0.9 according to the UReRF. Hoewver, that could be because they used
    # the oblique projections by default
    assert score > expected_score

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn import datasets

from sktree import UnsupervisedRandomForest

CLUSTER_CRITERIONS = ("twomeans", "fastbic")

FOREST_CLUSTERS = {
    "UnsupervisedRandomForest": UnsupervisedRandomForest,
}

# load iris dataset
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

@parametrize_with_checks(
    [
        UnsupervisedRandomForest(random_state=12345, n_estimators=50),
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


def check_simulation_criterion(name, criterion):
    n_samples = 100
    n_classes = 2
    ForestCluster = FOREST_CLUSTERS[name]
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=2**4)

    clf = ForestCluster(criterion=criterion, random_state=12345)

    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # XXX: This should be > 0.9 according to the UReRF. However, that could be because they used
    # the oblique projections by default

    assert score > 0.6


def check_iris_criterion(name, criterion):
    # Check consistency on dataset iris.
    ForestCluster = FOREST_CLUSTERS[name]
    n_classes = 3
    clf = ForestCluster(criterion=criterion, random_state=12345)
    clf.fit(iris.data, iris.target)
    sim_mat = clf.affinity_matrix_

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(iris.target, predict_labels)

    # Two-means and fastBIC criterions perform similarly here
    assert score > 0.2, "Failed with criterion %s and score = %f" % (criterion, score)


@pytest.mark.parametrize("name", FOREST_CLUSTERS)
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_clusters(name, criterion):
    check_simulation_criterion(name, criterion)
    check_iris_criterion(name, criterion)


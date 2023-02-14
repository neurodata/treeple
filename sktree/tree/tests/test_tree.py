import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks
from itertools import product, chain
from sklearn import datasets

from sktree.tree import UnsupervisedDecisionTree

CLUSTER_CRITERIONS = ("twomeans", "fastbic")

TREE_CLUSTERS = {
    "UnsupervisedDecisionTree": UnsupervisedDecisionTree,
}

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

@parametrize_with_checks(
    [
        UnsupervisedDecisionTree(random_state=12)
    ]
)
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


def check_simulation(name, Tree, criterion):
    n_samples = 10
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=12345)

    clf = Tree(criterion=criterion, random_state=12345)

    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score >= 0.0


def check_iris(name, Tree, criterion):
    # Check consistency on dataset iris.
    n_classes = 3
    clf = Tree(criterion=criterion, random_state=12345)
    clf.fit(iris.data, iris.target)
    sim_mat = clf.affinity_matrix_

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(iris.target, predict_labels)

    # Two-means and fastBIC criterions doesn't perform well
    assert score > -0.01, "Failed with {0}, criterion = {1} and score = {2}".format(
    name, criterion, score)


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_trees(name, Tree, criterion):
    check_simulation(name, Tree, criterion)
    check_iris(name, Tree, criterion)
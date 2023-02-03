import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import (
    ObliqueDecisionTreeClassifier,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@parametrize_with_checks(
    [
        UnsupervisedDecisionTree(random_state=12),
        UnsupervisedObliqueDecisionTree(random_state=12),
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


@pytest.mark.parametrize("ESTIMATOR", [UnsupervisedDecisionTree, UnsupervisedObliqueDecisionTree])
def test_unsupervisedtree(ESTIMATOR):
    n_samples = 10
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=6, random_state=1234)

    clf = ESTIMATOR(random_state=1234)
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


def test_oblique_tree_sampling():
    """Test Oblique Decision Trees.

    Oblique trees can sample more candidate splits then
    a normal axis-aligned tree.
    """
    X, y = iris.data, iris.target
    n_samples, n_features = X.shape

    # add additional noise dimensions
    rng = np.random.RandomState(0)
    X_noise = rng.random((n_samples, n_features))
    X = np.concatenate((X, X_noise), axis=1)

    # oblique decision trees can sample significantly more
    # diverse sets of splits and will do better if allowed
    # to sample more
    tree_ri = DecisionTreeClassifier(random_state=0, max_features=n_features)
    tree_rc = ObliqueDecisionTreeClassifier(random_state=0, max_features=n_features * 2)
    ri_cv_scores = cross_val_score(tree_ri, X, y, scoring="accuracy", cv=10, error_score="raise")
    rc_cv_scores = cross_val_score(tree_rc, X, y, scoring="accuracy", cv=10, error_score="raise")
    assert rc_cv_scores.mean() > ri_cv_scores.mean()
    assert rc_cv_scores.std() < ri_cv_scores.std()
    assert rc_cv_scores.mean() > 0.91

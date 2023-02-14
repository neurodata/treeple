import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks
from itertools import product, chain
from sklearn import datasets

from sktree.tree import (
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)

CLUSTER_CRITERIONS = ("twomeans", "fastbic")

TREE_CLUSTERS = {
    "UnsupervisedDecisionTree": UnsupervisedDecisionTree,
    "UnsupervisedObliqueDecisionTree": UnsupervisedObliqueDecisionTree,
}

# load iris dataset
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# load digits dataset and randomly permute it
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]


@parametrize_with_checks(
    [
        ObliqueDecisionTreeClassifier(random_state=12),
        PatchObliqueDecisionTreeClassifier(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # TODO: investigate why this is the case, but not for oblique decision trees
    if isinstance(estimator, PatchObliqueDecisionTreeClassifier) and check.func.__name__ in [
        "check_methods_subset_invariance",
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()

    # TODO: remove when we implement Regressor classes
    if check.func.__name__ in ["check_requires_y_none"]:
        pytest.skip()
    check(estimator)


@parametrize_with_checks(
    [
        UnsupervisedDecisionTree(random_state=12),
        UnsupervisedObliqueDecisionTree(random_state=12),
    ]
)
def test_sklearn_compatible_transformer(estimator, check):
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
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=6, random_state=1234)

    est = Tree(criterion, random_state=1234)
    est.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score > 0.05, "Failed with {0}, criterion = {1} and score = {2}".format(
    name, criterion, score)
    assert score >= 0.0


def check_iris(name, Tree, criterion):
    # Check consistency on dataset iris.
    n_classes = 3
    est = Tree(criterion=criterion, random_state=12345)
    est.fit(iris.data, iris.target)
    sim_mat = est.affinity_matrix_

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


def test_oblique_tree_feature_combinations():
    """Test the hyperparameter ``feature_combinations`` behaves properly."""
    X, y = iris.data, iris.target
    _, n_features = X.shape

    X = X[:5, :]
    y = y[:5, ...]

    with pytest.raises(
        RuntimeError, match=f"Feature combinations {n_features + 1} should not be greater"
    ):
        clf = ObliqueDecisionTreeClassifier(random_state=0, feature_combinations=n_features + 1)
        clf.fit(X, y)

    # default option should make it 1.5 if n_features > 1.5
    clf = ObliqueDecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.feature_combinations_ == 1.5

    # setting the feature combinations explicitly is fine as long as it is < n_features
    clf = ObliqueDecisionTreeClassifier(random_state=0, feature_combinations=3)
    clf.fit(X, y)
    assert clf.feature_combinations_ == 3

    # edge-case of only a single feature should set feature_combinations properly
    X = X[:, 0:1]
    clf = ObliqueDecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.feature_combinations_ == 1


def test_patch_tree_errors():
    """Test errors that are specifically raised by manifold trees."""
    X, y = digits.data, digits.target

    # passed in data should match expected data shape
    with pytest.raises(RuntimeError, match="The passed in data height"):
        clf = PatchObliqueDecisionTreeClassifier(
            data_height=8,
            data_width=9,
        )
        clf.fit(X, y)

    # minimum patch height/width should be always less than or equal to
    # the maximum patch height/width
    with pytest.raises(RuntimeError, match="The minimum patch height"):
        clf = PatchObliqueDecisionTreeClassifier(
            min_patch_height=2,
            data_height=8,
            data_width=8,
        )
        clf.fit(X, y)
    with pytest.raises(RuntimeError, match="The minimum patch width"):
        clf = PatchObliqueDecisionTreeClassifier(
            min_patch_width=2,
            data_height=8,
            data_width=8,
        )
        clf.fit(X, y)

    # the maximum patch height/width should not exceed the data height/width
    with pytest.raises(RuntimeError, match="The maximum patch width"):
        clf = PatchObliqueDecisionTreeClassifier(
            max_patch_width=9,
            data_height=8,
            data_width=8,
        )
        clf.fit(X, y)

    with pytest.raises(RuntimeError, match="The maximum patch height"):
        clf = PatchObliqueDecisionTreeClassifier(
            max_patch_height=9,
            data_height=8,
            data_width=8,
        )
        clf.fit(X, y)


def test_patch_tree_overfits():
    """Test of performance of patch tree on image-like data."""
    X, y = digits.data, digits.target

    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_height=2,
        min_patch_width=2,
        max_patch_height=6,
        max_patch_width=6,
        data_height=8,
        data_width=8,
        random_state=1,
    )

    # the single tree without depth limitations should have almost 0 bias
    clf.fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.99


def test_patch_tree_compared():
    """Test patch tree against other tree models."""
    X, y = digits.data, digits.target
    _, n_features = X.shape

    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_height=1,
        min_patch_width=1,
        max_patch_height=8,
        max_patch_width=8,
        data_height=8,
        data_width=8,
        random_state=1,
        max_features=n_features,
    )

    clf.fit(X, y)
    print(clf.get_depth())

    # a well-parametrized patch tree should be relatively accurate
    patch_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    assert patch_tree_score > 0.6

    # similar to oblique trees, we can sample more and we should improve
    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_height=1,
        min_patch_width=1,
        max_patch_height=8,
        max_patch_width=8,
        data_height=8,
        data_width=8,
        random_state=1,
        max_features=n_features * 2,
    )
    new_patch_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    assert new_patch_tree_score > patch_tree_score

    clf.fit(X, y)
    print(clf.get_depth())

    # TODO: there is a performance difference that is not in favor of patch trees, so
    # either we can improve the implementation, hyperparameters, or choose an alternative dataset
    clf = ObliqueDecisionTreeClassifier(
        random_state=1,
        max_features=n_features,
    )
    oblique_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    clf.fit(X, y)
    print(clf.get_depth())

    clf = DecisionTreeClassifier(max_features=n_features, random_state=1)
    axis_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    clf.fit(X, y)
    print(clf.get_depth())

    print(axis_tree_score)
    print(oblique_tree_score)
    print(patch_tree_score)
    assert np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=2)) > 0.7

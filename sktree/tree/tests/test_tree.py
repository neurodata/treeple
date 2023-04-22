import joblib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from sklearn import datasets
from sklearn.base import is_classifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn_fork.tree import DecisionTreeClassifier

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

X_small = np.array(
    [
        [0, 0, 4, 0, 0, 0, 1, -14, 0, -4, 0, 0, 0, 0],
        [0, 0, 5, 3, 0, -4, 0, 0, 1, -5, 0.2, 0, 4, 1],
        [-1, -1, 0, 0, -4.5, 0, 0, 2.1, 1, 0, 0, -4.5, 0, 1],
        [-1, -1, 0, -1.2, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 1],
        [-1, -1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
        [-1, -2, 0, 4, -3, 10, 4, 0, -3.2, 0, 4, 3, -4, 1],
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -1, 0],
        [2, 8, 5, 1, 0.5, -4, 10, 0, 1, -5, 3, 0, 2, 0],
        [2, 0, 1, 1, 1, -1, 1, 0, 0, -2, 3, 0, 1, 0],
        [2, 0, 1, 2, 3, -1, 10, 2, 0, -1, 1, 2, 2, 0],
        [1, 1, 0, 2, 2, -1, 1, 2, 0, -5, 1, 2, 3, 0],
        [3, 1, 0, 3, 0, -4, 10, 0, 1, -5, 3, 0, 3, 1],
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 1.5, 1, -1, -1],
        [2.11, 8, -6, -0.5, 0, 10, 0, 0, -3.2, 6, 0.5, 0, -1, -1],
        [2, 0, 5, 1, 0.5, -2, 10, 0, 1, -5, 3, 1, 0, -1],
        [2, 0, 1, 1, 1, -2, 1, 0, 0, -2, 0, 0, 0, 1],
        [2, 1, 1, 1, 2, -1, 10, 2, 0, -1, 0, 2, 1, 1],
        [1, 1, 0, 0, 1, -3, 1, 2, 0, -5, 1, 2, 1, 1],
        [3, 1, 0, 1, 0, -4, 1, 0, 1, -2, 0, 0, 1, 0],
    ]
)

y_small = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
y_small_reg = [
    1.0,
    2.1,
    1.2,
    0.05,
    10,
    2.4,
    3.1,
    1.01,
    0.01,
    2.98,
    3.1,
    1.1,
    0.0,
    1.2,
    2,
    11,
    0,
    0,
    4.5,
    0.201,
    1.06,
    0.9,
    0,
]

# also load the iris dataset
# and randomly permute it
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


def assert_tree_equal(d, s, message):
    assert s.node_count == d.node_count, "{0}: inequal number of node ({1} != {2})".format(
        message, s.node_count, d.node_count
    )

    assert_array_equal(d.children_right, s.children_right, message + ": inequal children_right")
    assert_array_equal(d.children_left, s.children_left, message + ": inequal children_left")

    external = d.children_right == TREE_LEAF
    internal = np.logical_not(external)

    assert_array_equal(d.feature[internal], s.feature[internal], message + ": inequal features")
    assert_array_equal(
        d.threshold[internal], s.threshold[internal], message + ": inequal threshold"
    )
    assert_array_equal(
        d.n_node_samples.sum(),
        s.n_node_samples.sum(),
        message + ": inequal sum(n_node_samples)",
    )
    assert_array_equal(d.n_node_samples, s.n_node_samples, message + ": inequal n_node_samples")

    assert_almost_equal(d.impurity, s.impurity, err_msg=message + ": inequal impurity")

    assert_array_almost_equal(
        d.value[external], s.value[external], err_msg=message + ": inequal value"
    )


@parametrize_with_checks(
    [
        ObliqueDecisionTreeClassifier(random_state=12),
        PatchObliqueDecisionTreeClassifier(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
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


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_simulation(name, Tree, criterion):
    """Test axis-aligned Gaussian blobs."""
    n_samples = 100
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=6, random_state=1234)

    est = Tree(criterion=criterion, random_state=1234)
    est.fit(X)
    sim_mat = est.affinity_matrix_

    # there is quite a bit of variance in the performance at the tree level
    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.02
        else:
            expected_score = 0.3
    elif criterion == "fastbic":
        if "oblique" in name.lower():
            expected_score = 0.01
        else:
            expected_score = 0.4

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score >= expected_score, "Blobs failed with {0}, criterion = {1} and score = {2}".format(
        name, criterion, score
    )


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_rotated_blobs(name, Tree, criterion):
    """Test rotated axis-aligned Gaussian blobs, which should make oblique trees perform better."""
    n_samples = 100
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=6, random_state=1234)

    # apply rotation matrix to X

    est = Tree(criterion=criterion, random_state=1234)
    est.fit(X)
    sim_mat = est.affinity_matrix_

    # there is quite a bit of variance in the performance at the tree level
    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.02
        else:
            expected_score = 0.3
    elif criterion == "fastbic":
        if "oblique" in name.lower():
            expected_score = 0.01
        else:
            expected_score = 0.4

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score >= expected_score, "Blobs failed with {0}, criterion = {1} and score = {2}".format(
        name, criterion, score
    )


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_iris(name, Tree, criterion):
    # Check consistency on dataset iris.
    n_classes = len(np.unique(iris.target))
    est = Tree(criterion=criterion, random_state=12345)
    est.fit(iris.data, iris.target)
    sim_mat = est.affinity_matrix_

    # there is quite a bit of variance in the performance at the tree level
    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.2
        else:
            expected_score = 0.01
    elif criterion == "fastbic":
        if "oblique" in name.lower():
            expected_score = 0.001
        else:
            expected_score = 0.2

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(iris.target, predict_labels)

    # Two-means and fastBIC criterions doesn't perform well
    assert score > expected_score, "Iris failed with {0}, criterion = {1} and score = {2}".format(
        name, criterion, score
    )


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
    with pytest.raises(RuntimeError, match="Data dimensions"):
        clf = PatchObliqueDecisionTreeClassifier(
            data_dims=(8, 9),
        )
        clf.fit(X, y)

    # minimum patch height/width should be always less than or equal to
    # the maximum patch height/width
    with pytest.raises(RuntimeError, match="The minimum patch"):
        clf = PatchObliqueDecisionTreeClassifier(
            min_patch_dims=(2, 1),
            max_patch_dims=(1, 1),
            data_dims=(8, 8),
        )
        clf.fit(X, y)

    # the maximum patch height/width should not exceed the data height/width
    with pytest.raises(RuntimeError, match="The maximum patch width"):
        clf = PatchObliqueDecisionTreeClassifier(
            max_patch_dims=(9, 1),
            data_dims=(8, 8),
        )
        clf.fit(X, y)


def test_patch_tree_overfits():
    """Test of performance of patch tree on image-like data."""
    X, y = digits.data, digits.target

    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_dims=(2, 2),
        max_patch_dims=(6, 6),
        data_dims=(8, 8),
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
        min_patch_dims=(1, 1),
        max_patch_dims=(8, 8),
        data_dims=(8, 8),
        random_state=1,
        max_features=n_features,
    )

    clf.fit(X, y)

    # a well-parametrized patch tree should be relatively accurate
    patch_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    assert patch_tree_score > 0.6

    # similar to oblique trees, we can sample more and we should improve
    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_dims=(1, 1),
        max_patch_dims=(8, 8),
        data_dims=(8, 8),
        random_state=1,
        max_features=n_features * 2,
    )
    new_patch_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    assert new_patch_tree_score > patch_tree_score

    clf = ObliqueDecisionTreeClassifier(
        random_state=1,
        max_features=n_features,
    )
    assert np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=2)) < new_patch_tree_score

    clf = DecisionTreeClassifier(max_features=n_features, random_state=1)
    assert np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=2)) < new_patch_tree_score


@pytest.mark.parametrize(
    "TREE",
    [ObliqueDecisionTreeClassifier, UnsupervisedDecisionTree, UnsupervisedObliqueDecisionTree],
)
def test_tree_deserialization_from_read_only_buffer(tmpdir, TREE):
    """Check that Trees can be deserialized with read only buffers.

    Non-regression test for gh-25584.
    """
    pickle_path = str(tmpdir.join("clf.joblib"))
    clf = TREE(random_state=0)

    if is_classifier(TREE):
        clf.fit(X_small, y_small)
    else:
        clf.fit(X_small)

    joblib.dump(clf, pickle_path)
    loaded_clf = joblib.load(pickle_path, mmap_mode="r")

    assert_tree_equal(
        loaded_clf.tree_,
        clf.tree_,
        "The trees of the original and loaded classifiers are not equal.",
    )


def test_patch_oblique_tree_feature_weights():
    """Test patch oblique tree when feature weights are passed in."""
    X, y = digits.data, digits.target

    with pytest.raises(ValueError, match="feature_weight has shape"):
        clf = PatchObliqueDecisionTreeClassifier(
            min_patch_dims=(2, 2),
            max_patch_dims=(6, 6),
            data_dims=(8, 8),
            random_state=1,
            feature_weight=[1, 1],
        )
        clf.fit(X, y)


def test_patch_tree_higher_dims():
    """Test patch oblique tree when patch and data dimensions are higher."""
    pass

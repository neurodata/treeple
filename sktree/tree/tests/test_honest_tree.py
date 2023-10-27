import numpy as np
import pytest
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.tree import (
    HonestTreeClassifier,
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
)

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
        PatchObliqueDecisionTreeClassifier(),
    ],
)
def test_iris(criterion, max_features, estimator):
    # Check consistency on dataset iris.
    clf = HonestTreeClassifier(
        criterion=criterion, random_state=0, max_features=max_features, tree_estimator=estimator
    )
    clf.fit(iris.data, iris.target)
    score = accuracy_score(clf.predict(iris.data), iris.target)
    assert score > 0.5 and score < 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HonestTree", criterion, score
    )

    score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
    assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HonestTree", criterion, score
    )

    print(clf.honest_indices_)
    assert len(clf.honest_indices_) < len(iris.target)
    assert len(clf.structure_indices_) < len(iris.target)


def test_toy_accuracy():
    clf = HonestTreeClassifier()
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize(
    "honest_prior, val",
    [
        ("uniform", 0.5),
        ("empirical", 0.75),
        ("ignore", np.nan),
    ],
)
def test_impute_posteriors(honest_prior, val):
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestTreeClassifier(honest_fraction=0.02, random_state=0, honest_prior=honest_prior)
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    if np.isnan(val):
        assert len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50, f"Failed with {honest_prior}"
    else:
        assert len(np.where(y_proba[:, 0] == val)[0]) > 50, f"Failed with {honest_prior}"


def test_increasing_leaves():
    np.random.seed(0)
    X = np.random.normal(0, 1, (200, 2))
    y = [0] * 100 + [1] * 100

    n_leaves = []
    for hf in [0.9, 0.7, 0.4, 0.2]:
        clf = HonestTreeClassifier(honest_fraction=hf, random_state=0)
        clf = clf.fit(X, y)
        n_leaves.append(clf.get_n_leaves())

    assert np.all(np.diff(n_leaves) > 0)


def test_impute_classes():
    np.random.seed(0)
    X = np.random.normal(0, 1, (101, 2))
    y = [2] * 50 + [1] * 50 + [0]
    clf = HonestTreeClassifier(honest_fraction=0.02, random_state=0)
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)

    assert y_proba.shape[1] == 3


@parametrize_with_checks([HonestTreeClassifier(random_state=1)])
def test_sklearn_compatible_estimator(estimator, check):
    # 1. check_class_weight_classifiers is not supported since it requires sample weight
    # XXX: can include this "generalization" in the future if it's useful
    #  zero sample weight is not "really supported" in honest subsample trees since sample weight
    #  for fitting the tree's splits
    if check.func.__name__ in ["check_class_weight_classifiers", "check_classifier_multioutput"]:
        pytest.skip()
    check(estimator)


def test_with_sklearn_trees():
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10

    # with pytest.raises(RuntimeError, match="Instead of using sklearn.tree"):
    clf = HonestTreeClassifier(tree_estimator=skDecisionTreeClassifier())
    clf.fit(X, y)


def k_sample_transform(inputs, test_type="normal"):
    """
    Computes a `k`-sample transform of the inputs.

    For :math:`k` groups, this creates two matrices, the first vertically stacks the
    inputs.
    In order to use this function, the inputs must have the same number of dimensions
    :math:`p` and can have varying number of samples :math:`n`. The second output is a
    label
    matrix the one-hoc encodes the groups. The outputs are thus ``(N, p)`` and
    ``(N, k)`` where `N` is the total number of samples. In the case where the test
    a random forest based tests, it creates a ``(N, 1)`` where the entries are
    varlues from 1 to :math:`k` based on the number of samples.

    Parameters
    ----------
    inputs : list of ndarray
        A list of the inputs. All inputs must be ``(n, p)`` where `n` is the number
        of samples and `p` is the number of dimensions. `n` can vary between samples,
        but `p` must be the same among all the samples.
    test_type : {"normal", "rf"}, default: "normal"
        Whether to one-hoc encode the inputs ("normal") or use a one-dimensional
        categorical encoding ("rf").

    Returns
    -------
    u : ndarray
        The matrix of concatenated inputs of shape ``(N, p)``.
    v : ndarray
        The label matrix of shape ``(N, k)`` ("normal") or ``(N, 1)`` ("rf").
    """
    n_inputs = len(inputs)
    u = np.vstack(inputs)
    if np.var(u) == 0:
        raise ValueError("Test cannot be run, the inputs have 0 variance")

    if test_type == "rf":
        v = np.vstack([np.repeat(i, inputs[i].shape[0]).reshape(-1, 1) for i in range(n_inputs)])
    elif test_type == "normal":
        if n_inputs == 2:
            n1 = inputs[0].shape[0]
            n2 = inputs[1].shape[0]
            v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
        else:
            vs = []
            for i in range(n_inputs):
                n = inputs[i].shape[0]
                encode = np.zeros(shape=(n, n_inputs))
                encode[:, i] = np.ones(shape=n)
                vs.append(encode)
            v = np.concatenate(vs)
    else:
        raise ValueError("test_type must be normal or rf")

    return u, v


@pytest.mark.skip()
def test_sklearn_tree_regression():
    """Test against regression in power-curves discussed in:"""

    def quadratic(n, p, noise=False, seed=None):
        rng = np.random.default_rng(seed)

        x = rng.standard_normal(size=(n, p))
        coeffs = np.array([np.exp(-0.0325 * (i + 24)) for i in range(p)])
        eps = rng.standard_normal(size=(n, p))

        x_coeffs = x * coeffs
        y = x_coeffs**2 + noise * eps

        n1 = x.shape[0]
        n2 = y.shape[0]
        v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
        x = np.vstack((x, y))
        return x, v

    # generate the high-dimensional quadratic data
    X, y = quadratic(1024, 4096, noise=False, seed=0)
    print(X.shape, y.shape)
    print(np.sum(y) / len(y))
    assert False
    clf = HonestTreeClassifier(tree_estimator=skDecisionTreeClassifier(), random_state=0)
    honestsk_scores = cross_val_score(clf, X, y, cv=5)
    print(honestsk_scores)

    clf = HonestTreeClassifier(tree_estimator=DecisionTreeClassifier(), random_state=0)
    honest_scores = cross_val_score(clf, X, y, cv=5)
    print(honest_scores)

    clf = HonestTreeClassifier(random_state=0)
    honest_scores = cross_val_score(clf, X, y, cv=5)
    print(honest_scores)

    skest = skDecisionTreeClassifier(random_state=0)
    sk_scores = cross_val_score(skest, X, y, cv=5)

    est = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(est, X, y, cv=5)

    print(sk_scores, scores)
    print(np.mean(sk_scores), np.mean(scores))
    assert np.mean(sk_scores) == np.mean(scores)
    assert False

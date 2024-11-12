import numpy as np
import pytest
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from treeple.experimental import StreamDecisionForest

CLF_CRITERIONS = ("gini", "entropy")

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_toy_accuracy():
    clf = StreamDecisionForest(n_estimators=10)
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


def test_first_fit():
    clf = StreamDecisionForest(n_estimators=10)
    with pytest.raises(
        ValueError, match="classes must be passed on the first call to partial_fit."
    ):
        clf.partial_fit(iris.data, iris.target)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
def test_iris(criterion, max_features):
    # Check consistency on dataset iris.
    clf = StreamDecisionForest(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=10,
    )

    clf.partial_fit(iris.data, iris.target, classes=np.unique(iris.target))
    score = accuracy_score(clf.predict(iris.data), iris.target)

    assert score > 0.5 and score <= 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "SDF", criterion, score
    )

    score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
    assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "SDF", criterion, score
    )

    clf.partial_fit(iris.data, iris.target)
    score = accuracy_score(clf.predict(iris.data), iris.target)

    assert (
        score > 0.5 and score <= 1.0
    ), "Failed partial_fit with {0}, criterion = {1} and score = {2}".format(
        "SDF", criterion, score
    )

    score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
    assert score == 1.0, "Failed partial_fit with {0}, criterion = {1} and score = {2}".format(
        "SDF", criterion, score
    )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
def test_iris_multi(criterion, max_features):
    # Check consistency on dataset iris.
    clf = StreamDecisionForest(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=10,
    )

    second_y = np.concatenate([(np.ones(50) * 3), (np.ones(50) * 4), (np.ones(50) * 5)])

    X = iris.data
    y = np.stack((iris.target, second_y[perm])).T

    clf.fit(X, y)
    score = r2_score(clf.predict(X), y)
    assert score > 0.9 and score <= 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "SDF", criterion, score
    )


def test_max_samples():
    max_samples_list = [8, 0.5, None]
    depths = []
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    for ms in max_samples_list:
        uf = StreamDecisionForest(n_estimators=2, random_state=0, max_samples=ms, bootstrap=True)
        uf = uf.fit(X, y)
        depths.append(uf.estimators_[0].get_depth())

    assert all(np.diff(depths) > 0)


@parametrize_with_checks([StreamDecisionForest(n_estimators=10, random_state=0)])
def test_sklearn_compatible_estimator(estimator, check):
    # 1. check_class_weight_classifiers is not supported since it requires sample weight
    # XXX: can include this "generalization" in the future if it's useful
    if check.func.__name__ in [
        "check_class_weight_classifiers",
        "check_sample_weight_equivalence",
    ]:
        pytest.skip()
    check(estimator)

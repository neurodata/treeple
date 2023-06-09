import time
from typing import Any, Dict

import numpy as np
import pytest
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

from sktree.ensemble import HonestForestClassifier
from sktree.tree import (
    HonestTreeClassifier,
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
)

FOREST_CLASSIFIERS = {
    "HonestForestClassifier": HonestForestClassifier,
}

FOREST_ESTIMATORS: Dict[str, Any] = dict()
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)

CLF_CRITERIONS = ("gini", "entropy")

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_toy_accuracy():
    clf = HonestForestClassifier(n_estimators=10)
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


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
    clf = HonestForestClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=10,
        estimator=HonestTreeClassifier(estimator),
    )
    clf.fit(iris.data, iris.target)
    score = accuracy_score(clf.predict(iris.data), iris.target)
    assert score > 0.5 and score < 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HForest", criterion, score
    )

    score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
    assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HForest", criterion, score
    )


def test_impute_classes():
    np.random.seed(0)
    X = np.random.normal(0, 1, (101, 2))
    y = [0] * 50 + [1] * 50 + [2]
    clf = HonestForestClassifier(honest_fraction=0.02, random_state=0)
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)

    assert y_proba.shape[1] == 3


def test_parallel_trees():
    uf = HonestForestClassifier(n_estimators=100, n_jobs=1, max_features=1.0, honest_fraction=0.5)
    uf_parallel = HonestForestClassifier(
        n_estimators=100, n_jobs=10, max_features=1.0, honest_fraction=0.5
    )
    X = np.random.normal(0, 1, (1000, 100))
    y = [0, 1] * (len(X) // 2)

    time_start = time.time()
    uf.fit(X, y)
    time_diff = time.time() - time_start

    time_start = time.time()
    uf_parallel.fit(X, y)
    time_parallel_diff = time.time() - time_start
    assert time_parallel_diff / time_diff < 0.9


def test_max_samples():
    max_samples_list = [8, 0.5, None]
    depths = []
    X = np.random.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    for ms in max_samples_list:
        uf = HonestForestClassifier(n_estimators=2, max_samples=ms, bootstrap=True)
        uf = uf.fit(X, y)
        depths.append(uf.estimators_[0].get_depth())

    assert all(np.diff(depths) > 0)


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
    clf = HonestForestClassifier(
        honest_fraction=0.02, random_state=0, honest_prior=honest_prior, n_estimators=2
    )
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    if np.isnan(val):
        assert (
            len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50
        ), f"Failed with {honest_prior}, prior {clf.estimators_[0].empirical_prior_}"
    else:
        assert (
            len(np.where(y_proba[:, 0] == val)[0]) > 50
        ), f"Failed with {honest_prior}, prior {clf.estimators_[0].empirical_prior_}"


@pytest.mark.parametrize(
    "honest_fraction, val",
    [
        (0.8, 0.5),
        (0.02, np.nan),
    ],
)
def test_honest_decision_function(honest_fraction, val):
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestForestClassifier(honest_fraction=honest_fraction, random_state=0, n_estimators=2)
    clf = clf.fit(X, y)

    y_proba = clf.honest_decision_function_
    if np.isnan(val):
        assert len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50, f"Failed with {honest_fraction}"
    else:
        assert len(np.where(y_proba[:, 1] < val)[0]) > 50, f"Failed with {honest_fraction}"


@pytest.mark.parametrize("estimator", FOREST_ESTIMATORS)
def test_sklearn_compatible_estimator(estimator):
    # TODO: remove when we implement Regressor classes
    if FOREST_ESTIMATORS[estimator].__name__ in FOREST_CLASSIFIERS:
        pytest.skip()
    check_estimator(estimator)

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.ensemble import HonestForestClassifier
from sktree.tree import ObliqueDecisionTreeClassifier, PatchObliqueDecisionTreeClassifier

CLF_CRITERIONS = ("gini", "entropy")

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Larger classification sample used for testing feature importances
X_large, y_large = datasets.make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=0,
)


def test_toy_accuracy():
    clf = HonestForestClassifier(n_estimators=10)
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore", "error"])
@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
        PatchObliqueDecisionTreeClassifier(),
    ],
)
def test_iris(criterion, max_features, honest_prior, estimator):
    # Check consistency on dataset iris.
    clf = HonestForestClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=10,
        honest_prior=honest_prior,
        tree_estimator=estimator,
    )
    if honest_prior == "error":
        with pytest.raises(ValueError, match="honest_prior error not a valid input."):
            clf.fit(iris.data, iris.target)
    else:
        clf.fit(iris.data, iris.target)
        score = accuracy_score(clf.predict(iris.data), iris.target)

        assert (
            score > 0.5 and score < 1.0
        ), "Failed with {0}, criterion = {1} and score = {2}".format("HForest", criterion, score)

        score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
        assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
            "HForest", criterion, score
        )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore", "error"])
@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
        PatchObliqueDecisionTreeClassifier(),
    ],
)
def test_iris_multi(criterion, max_features, honest_prior, estimator):
    n_estimators = 10

    # Check consistency on dataset iris.
    clf = HonestForestClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=n_estimators,
        honest_prior=honest_prior,
        tree_estimator=estimator,
    )

    second_y = np.concatenate([(np.ones(50) * 3), (np.ones(50) * 4), (np.ones(50) * 5)])

    X = iris.data
    y = np.stack((iris.target, second_y[perm])).T
    if honest_prior == "error":
        with pytest.raises(ValueError, match="honest_prior error not a valid input."):
            clf.fit(X, y)
    else:
        clf.fit(X, y)
        score = r2_score(clf.predict(X), y)
        if honest_prior == "ignore":
            assert (
                score > 0.4 and score < 1.0
            ), "Failed with {0}, criterion = {1} and score = {2}".format(
                "HForest", criterion, score
            )
        else:
            assert (
                score > 0.9 and score < 1.0
            ), "Failed with {0}, criterion = {1} and score = {2}".format(
                "HForest", criterion, score
            )


def test_max_samples():
    max_samples_list = [8, 0.5, None]
    depths = []
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    for ms in max_samples_list:
        uf = HonestForestClassifier(n_estimators=2, random_state=0, max_samples=ms, bootstrap=True)
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
    X = rng.normal(0, 1, (100, 2))
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
    X = rng.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestForestClassifier(honest_fraction=honest_fraction, random_state=0, n_estimators=2)
    clf = clf.fit(X, y)

    y_proba = clf.honest_decision_function_
    if np.isnan(val):
        assert len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50, f"Failed with {honest_fraction}"
    else:
        assert len(np.where(y_proba[:, 1] < val)[0]) > 50, f"Failed with {honest_fraction}"


@parametrize_with_checks(
    [HonestForestClassifier(n_estimators=10, honest_fraction=0.5, random_state=0)]
)
def test_sklearn_compatible_estimator(estimator, check):
    # 1. check_class_weight_classifiers is not supported since it requires sample weight
    # XXX: can include this "generalization" in the future if it's useful
    #  zero sample weight is not "really supported" in honest subsample trees since sample weight
    #  for fitting the tree's splits
    if check.func.__name__ in [
        "check_class_weight_classifiers",
        # TODO: this is an error. Somehow a segfault is raised when fit is called first and
        # then partial_fit
        "check_fit_score_takes_y",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize("criterion", ["gini", "log_loss"])
def test_importances(dtype, criterion):
    """Ported from sklearn unit-test.

    Used to ensure that honest forest feature importances are consistent with sklearn's.
    """
    tolerance = 0.01

    # cast as dtype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    ForestEstimator = HonestForestClassifier

    est = ForestEstimator(n_estimators=10, criterion=criterion, random_state=0)
    est.fit(X, y)

    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert np.all(importances[:3] > 0.1)

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance

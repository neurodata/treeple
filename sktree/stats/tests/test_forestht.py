import numpy as np
import pytest
from flaky import flaky
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn import datasets

from sktree import HonestForestClassifier
from sktree.stats import (
    PermutationHonestForestClassifier,
    build_coleman_forest,
    build_permutation_forest,
)
from sktree.tree import MultiViewDecisionTreeClassifier

# load the iris dataset (n_samples, 4)
# and randomly permute it
iris = datasets.load_iris()
seed = 12345
rng = np.random.default_rng(seed)

# remove third class
iris_X = iris.data[iris.target != 2]
iris_y = iris.target[iris.target != 2]

p = rng.permutation(iris_X.shape[0])
iris_X = iris_X[p]
iris_y = iris_y[p]


@pytest.mark.parametrize("seed", [None, 0])
def test_small_dataset_independent(seed):
    n_samples = 64
    n_features = 20
    n_estimators = 100

    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    result = build_coleman_forest(
        clf, perm_clf, X, y, covariate_index=[1, 2], metric="mi", return_posteriors=False
    )

    assert ~np.isnan(result.pvalue)
    assert ~np.isnan(result.observe_test_stat)
    assert result.pvalue > 0.05, f"{result.pvalue}"

    result = build_coleman_forest(clf, perm_clf, X, y, metric="mi", return_posteriors=False)
    assert_almost_equal(result.observe_test_stat, 0.0, decimal=1)
    assert result.pvalue > 0.05, f"{result.pvalue}"


@flaky(max_runs=3)
@pytest.mark.parametrize("seed", [None, 0])
def test_small_dataset_dependent(seed):
    n_samples = 100
    n_features = 5
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n_samples, n_features))
    X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = X + 3
    X = np.vstack([X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    n_estimators = 50
    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    result = build_coleman_forest(
        clf, perm_clf, X, y, covariate_index=[1, 2], metric="mi", return_posteriors=False
    )
    assert ~np.isnan(result.pvalue)
    assert ~np.isnan(result.observe_test_stat)
    assert result.pvalue <= 0.05

    result = build_coleman_forest(clf, perm_clf, X, y, metric="mi", return_posteriors=False)
    assert result.pvalue <= 0.05


def test_comight_repeated_feature_sets():
    """Test COMIGHT when there are repeated feature sets."""
    n_samples = 50
    n_features = 500
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n_samples, 10))
    X2 = X + 3
    X = np.hstack((X, rng.standard_normal(size=(n_samples, n_features - 10))))
    X2 = np.hstack((X2, rng.standard_normal(size=(n_samples, n_features - 10))))
    X = np.vstack([X, X2])
    y = np.vstack([np.zeros((n_samples, 1)), np.ones((n_samples, 1))])  # Binary classification

    X = np.hstack((X, X))
    feature_set_ends = [n_features, n_features * 2]

    n_estimators = 50
    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
        max_features=0.3,
        tree_estimator=MultiViewDecisionTreeClassifier(
            feature_set_ends=feature_set_ends,
            apply_max_features_per_feature_set=True,
        ),
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
        max_features=0.3,
        tree_estimator=MultiViewDecisionTreeClassifier(
            feature_set_ends=feature_set_ends,
            apply_max_features_per_feature_set=True,
        ),
    )

    # first test MIGHT rejects the null, since there is information
    result = build_coleman_forest(clf, perm_clf, X, y, metric="mi", return_posteriors=False)
    assert result.pvalue < 0.05

    # second test CoMIGHT fails to reject the null, since the information
    # is entirely contained in the first feature set
    result = build_coleman_forest(
        clf,
        perm_clf,
        X,
        y,
        covariate_index=np.arange(n_features),
        metric="mi",
        return_posteriors=False,
    )
    assert result.pvalue > 0.05, f"{result.pvalue}"


def test_build_coleman_forest():
    """Simple test for building a Coleman forest.

    Test the function under alternative and null hypothesis for a very simple dataset.
    """
    n_estimators = 100
    n_samples = 30
    n_features = 5
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 3
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_coleman_forest(clf, clf, X, y)

    forest_result, orig_forest_proba, perm_forest_proba, clf_fitted, perm_clf_fitted = (
        build_coleman_forest(clf, perm_clf, X, y, metric="s@98", n_repeats=1000, seed=seed)
    )
    assert clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert perm_clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert_array_equal(perm_clf_fitted.permutation_indices_.shape, (n_samples, 1))

    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat > 0.1, f"{forest_result.observe_stat}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _, clf_fitted, perm_clf_fitted = build_coleman_forest(
        clf, perm_clf, X, y, metric="s@98"
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat < 0.05, f"{forest_result.observe_stat}"


def test_build_coleman_forest_multiview():
    """Simple test for building a Coleman forest.

    Test the function under alternative and null hypothesis for a very simple dataset.
    """
    n_estimators = 100
    n_samples = 30
    n_features = 5
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 3
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
        max_features=[1, 1],
        tree_estimator=MultiViewDecisionTreeClassifier(),
        feature_set_ends=[2, 5],
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
        max_features=[1, 1],
        tree_estimator=MultiViewDecisionTreeClassifier(),
        feature_set_ends=[2, 5],
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_coleman_forest(clf, clf, X, y)

    forest_result, orig_forest_proba, perm_forest_proba, clf_fitted, perm_clf_fitted = (
        build_coleman_forest(clf, perm_clf, X, y, metric="s@98", n_repeats=1000, seed=seed)
    )
    assert clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert perm_clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert_array_equal(perm_clf_fitted.permutation_indices_.shape, (n_samples, 1))

    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat > 0.1, f"{forest_result.observe_stat}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _, clf_fitted, perm_clf_fitted = build_coleman_forest(
        clf, perm_clf, X, y, metric="s@98"
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"


def test_build_permutation_forest():
    """Simple test for building a permutation forest."""
    n_estimators = 30
    n_samples = 100
    n_features = 3
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 10
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, honest_fraction=0.5, bootstrap=True
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, honest_fraction=0.5, bootstrap=True
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_permutation_forest(clf, clf, X, y, seed=seed)

    forest_result, orig_forest_proba, perm_forest_proba = build_permutation_forest(
        clf, perm_clf, X, y, metric="s@98", n_repeats=20, seed=seed
    )
    assert forest_result.observe_test_stat > 0.1, f"{forest_result.observe_stat}"
    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _ = build_permutation_forest(
        clf, perm_clf, X, y, metric="s@98", n_repeats=10, seed=seed
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_test_stat < 0.05, f"{forest_result.observe_test_stat}"

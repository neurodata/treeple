import importlib
import os

import numpy as np
import pytest
from flaky import flaky
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn import datasets

import treeple.stats as stats
import treeple.stats.utils as utils
from treeple import HonestForestClassifier, RandomForestClassifier
from treeple.stats import PermutationHonestForestClassifier, build_coleman_forest, build_oob_forest
from treeple.tree import MultiViewDecisionTreeClassifier

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


@pytest.mark.parametrize("seed", [10, 0])
def test_small_dataset_independent(seed):
    # XXX: unit test interestingly does not work for MI, possibly due to bias
    bootstrap = True
    n_samples = 100
    n_features = 500
    n_estimators = 100

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_samples, n_features))
    y = np.vstack(
        (np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1)))
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=bootstrap,
        max_samples=1.6,
        max_features=0.3,
        stratify=True,
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=bootstrap,
        max_samples=1.6,
        max_features=0.3,
        stratify=True,
    )
    result = build_coleman_forest(
        clf, perm_clf, X, y, n_repeats=1000, metric="s@98", return_posteriors=False, seed=seed
    )
    print(result.observe_stat, result.permuted_stat, result.pvalue, result.observe_test_stat)
    assert result.pvalue > 0.05, f"{result.pvalue}"
    assert_almost_equal(np.abs(result.observe_test_stat), 0.0, decimal=1)

    # now permute only some of the features
    feature_set_ends = [3, n_features]
    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=bootstrap,
        max_features=0.3,
        max_samples=1.6,
        stratify=True,
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
        bootstrap=bootstrap,
        max_samples=1.6,
        max_features=0.3,
        stratify=True,
        tree_estimator=MultiViewDecisionTreeClassifier(
            feature_set_ends=feature_set_ends,
            apply_max_features_per_feature_set=True,
        ),
    )
    result = build_coleman_forest(
        clf,
        perm_clf,
        X,
        y,
        covariate_index=[1, 2, 3, 4, 5],
        n_repeats=1000,
        metric="s@98",
        return_posteriors=False,
        seed=seed,
    )
    assert ~np.isnan(result.pvalue)
    assert ~np.isnan(result.observe_test_stat)
    assert result.pvalue > 0.05, f"{result.pvalue}"


@flaky(max_runs=3)
@pytest.mark.parametrize("seed", [10, 0])
def test_small_dataset_dependent(seed):
    n_samples = 100
    n_features = 5
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = X + 3
    X = np.vstack([X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    n_estimators = 100
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
        clf,
        perm_clf,
        X,
        y,
        covariate_index=[1, 2],
        n_repeats=1000,
        metric="mi",
        return_posteriors=False,
        seed=seed,
    )
    assert ~np.isnan(result.pvalue)
    assert ~np.isnan(result.observe_test_stat)
    assert result.pvalue <= 0.05

    result = build_coleman_forest(
        clf, perm_clf, X, y, metric="mi", return_posteriors=False, seed=seed
    )
    assert result.pvalue <= 0.05


@pytest.mark.parametrize("seed", [10, 0])
def test_comight_repeated_feature_sets(seed):
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

    X = np.hstack((X, X + rng.standard_normal(size=(n_samples * 2, n_features)) * 1e-5))
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
        stratify=True,
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
        stratify=True,
        tree_estimator=MultiViewDecisionTreeClassifier(
            feature_set_ends=feature_set_ends,
            apply_max_features_per_feature_set=True,
        ),
    )

    # first test MIGHT rejects the null, since there is information
    result = build_coleman_forest(
        clf, perm_clf, X, y, n_repeats=1000, metric="mi", return_posteriors=False
    )
    assert result.pvalue < 0.05

    # second test CoMIGHT fails to reject the null, since the information
    # is entirely contained in the first feature set
    result = build_coleman_forest(
        clf,
        perm_clf,
        X,
        y,
        n_repeats=1000,
        covariate_index=np.arange(n_features),
        metric="mi",
        return_posteriors=False,
    )
    assert result.pvalue > 0.05, f"{result.pvalue}"


@pytest.mark.parametrize("use_bottleneck", [True, False])
def test_build_coleman_forest(use_bottleneck: bool):
    """Simple test for building a Coleman forest.

    Test the function under alternative and null hypothesis for a very simple dataset.
    """
    if use_bottleneck and utils.DISABLE_BN_ENV_VAR in os.environ:
        del os.environ[utils.DISABLE_BN_ENV_VAR]
        importlib.reload(utils)
        importlib.reload(stats)
    else:
        os.environ[utils.DISABLE_BN_ENV_VAR] = "1"
        importlib.reload(utils)
        importlib.reload(stats)

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
        stats.build_coleman_forest(clf, clf, X, y)

    (
        forest_result,
        orig_forest_proba,
        perm_forest_proba,
        clf_fitted,
        perm_clf_fitted,
    ) = stats.build_coleman_forest(clf, perm_clf, X, y, metric="s@98", n_repeats=1000, seed=seed)
    assert clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert perm_clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert_array_equal(perm_clf_fitted.permutation_indices_.shape, (n_samples, 1))

    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat > 0.1, f"{forest_result.observe_stat}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _, clf_fitted, perm_clf_fitted = stats.build_coleman_forest(
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

    (
        forest_result,
        orig_forest_proba,
        perm_forest_proba,
        clf_fitted,
        perm_clf_fitted,
    ) = build_coleman_forest(clf, perm_clf, X, y, metric="s@98", n_repeats=1000, seed=seed)
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


def test_build_oob_honest_forest():
    bootstrap = True
    max_samples = 1.6

    n_estimators = 100
    est = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        bootstrap=bootstrap,
        max_samples=max_samples,
        honest_fraction=0.5,
        stratify=True,
    )
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = np.array([0, 1] * 50)
    samples = np.arange(len(y))

    est, proba = build_oob_forest(est, X, y)

    structure_samples = est.structure_indices_
    leaf_samples = est.honest_indices_
    oob_samples = est.oob_samples_
    for tree_idx in range(est.n_estimators):
        assert len(structure_samples[tree_idx]) + len(leaf_samples[tree_idx]) + len(
            oob_samples[tree_idx]
        ) == len(
            samples
        ), f"{tree_idx} {len(structure_samples[tree_idx])} {len(leaf_samples[tree_idx])} {len(samples)}"


def test_build_oob_random_forest():
    """Test building oob random forest."""
    bootstrap = True
    max_samples = 1.0

    n_estimators = 100
    est = RandomForestClassifier(
        n_estimators=n_estimators, random_state=0, bootstrap=bootstrap, max_samples=max_samples
    )
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = np.array([0, 1] * 50)
    samples = np.arange(len(y))

    est, proba = build_oob_forest(est, X, y)

    structure_samples = est.estimators_samples_
    all_samples = np.arange(X.shape[0])
    oob_samples_list = [
        np.setdiff1d(all_samples, structure_samples[i]) for i in range(len(structure_samples))
    ]
    for tree_idx in range(est.n_estimators):
        assert len(np.unique(structure_samples[tree_idx])) + len(oob_samples_list[tree_idx]) == len(
            samples
        ), f"{tree_idx} {len(structure_samples[tree_idx])} + {len(oob_samples_list[tree_idx])} != {len(samples)}"


def test_build_coleman_warning_with_multiview_without_covariate_index():
    """Test warning is raised in build_coleman_forest with multiview without covariate_index."""

    est = HonestForestClassifier(
        n_estimators=100,
        random_state=0,
        bootstrap=True,
        max_samples=1.0,
        honest_fraction=0.5,
        stratify=True,
        tree_estimator=MultiViewDecisionTreeClassifier(),
    )
    perm_est = PermutationHonestForestClassifier(
        n_estimators=100,
        random_state=0,
        bootstrap=True,
        max_samples=1.0,
        honest_fraction=0.5,
        stratify=True,
        tree_estimator=MultiViewDecisionTreeClassifier(),
    )
    X = rng.normal(0, 1, (100, 2))
    y = np.array([0, 1] * 50)
    with pytest.warns(UserWarning, match="Covariate index is not defined"):
        build_coleman_forest(est, perm_est, X, y, metric="s@98", n_repeats=1000, seed=0)

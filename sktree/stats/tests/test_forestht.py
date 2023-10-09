import pickle
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from flaky import flaky
from joblib import Parallel, delayed
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn import datasets

from sktree import HonestForestClassifier, RandomForestClassifier, RandomForestRegressor
from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.stats import FeatureImportanceForestClassifier, FeatureImportanceForestRegressor
from sktree.stats.utils import _non_nan_samples
from sktree.tree import ObliqueDecisionTreeClassifier

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


@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
def test_featureimportance_forest_permute_pertree(sample_dataset_per_tree):
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=10,
            random_state=seed,
        ),
        permute_per_tree=True,
        test_size=0.7,
        random_state=seed,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    n_samples = 50
    est.statistic(iris_X[:n_samples], iris_y[:n_samples], metric="mse")

    assert (
        len(est.train_test_samples_[0][1]) == n_samples * est.test_size
    ), f"{len(est.train_test_samples_[0][1])} {n_samples * est.test_size}"
    assert len(est.train_test_samples_[0][0]) == est._n_samples_ - n_samples * est.test_size

    est.test(iris_X[:n_samples], iris_y[:n_samples], [0, 1], n_repeats=10, metric="mse")
    assert (
        len(est.train_test_samples_[0][1]) == n_samples * est.test_size
    ), f"{len(est.train_test_samples_[0][1])} {n_samples * est.test_size}"
    assert len(est.train_test_samples_[0][0]) == est._n_samples_ - n_samples * est.test_size

    with pytest.raises(RuntimeError, match="Metric must be"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], metric="mi")

    # covariate index must be an iterable
    with pytest.raises(RuntimeError, match="covariate_index must be an iterable"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], 0, metric="mi")

    # covariate index must be an iterable of ints
    with pytest.raises(RuntimeError, match="Not all covariate_index"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], [0, 1.0], metric="mi")


def test_featureimportance_forest_errors():
    permute_per_tree = False
    sample_dataset_per_tree = True
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=10,
        ),
        test_size=0.5,
        permute_per_tree=permute_per_tree,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    with pytest.raises(RuntimeError, match="The estimator must be fitted"):
        est.train_test_samples_

    with pytest.raises(RuntimeError, match="There are less than 5 testing samples"):
        est.statistic(iris_X[:5], iris_y[:5])

    est = FeatureImportanceForestClassifier(estimator=RandomForestRegressor, test_size=0.5)
    with pytest.raises(RuntimeError, match="Estimator must be"):
        est.statistic(iris_X[:20], iris_y[:20])

    est = FeatureImportanceForestRegressor(estimator=RandomForestClassifier, test_size=0.5)
    with pytest.raises(RuntimeError, match="Estimator must be"):
        est.statistic(iris_X[:20], iris_y[:20])


@flaky(max_runs=2)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform"])
@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
    ],
)
@pytest.mark.parametrize(
    "permute_per_tree",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
def test_iris_pauc_statistic(
    criterion, honest_prior, estimator, permute_per_tree, sample_dataset_per_tree
):
    limit = 0.1
    max_features = "sqrt"
    n_repeats = 200
    n_estimators = 100
    test_size = 0.2

    # Check consistency on dataset iris.
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_features=max_features,
            tree_estimator=estimator,
            honest_prior=honest_prior,
            random_state=0,
            n_jobs=1,
        ),
        test_size=test_size,
        sample_dataset_per_tree=sample_dataset_per_tree,
        permute_per_tree=permute_per_tree,
    )
    # now add completely uninformative feature
    X = np.hstack((iris_X, rng.standard_normal(size=(iris_X.shape[0], 4))))

    # test for unimportant feature set
    clf.reset()
    if sample_dataset_per_tree and not permute_per_tree:
        # test in another test
        return

    stat, pvalue = clf.test(
        X,
        iris_y,
        np.arange(iris_X.shape[0], X.shape[1], dtype=int).tolist(),
        n_repeats=n_repeats,
        metric="auc",
    )
    print(pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"

    # test for important features that are permuted
    stat, pvalue = clf.test(X, iris_y, [0, 1, 2, 3], n_repeats=n_repeats, metric="auc")
    print(pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # one must call `reset()` to make sure the test is run on a "new" feature set properly
    with pytest.raises(RuntimeError, match="X must have 8 features"):
        clf.statistic(iris_X, iris_y, metric="auc", max_fpr=limit)

    clf.reset()
    score = clf.statistic(iris_X, iris_y, metric="auc", max_fpr=limit)
    assert score >= 0.8, "Failed with pAUC: {0} for max fpr: {1}".format(score, limit)

    assert isinstance(clf.estimator_, HonestForestClassifier)


@pytest.mark.parametrize(
    "forest_hyppo",
    [
        FeatureImportanceForestRegressor(
            estimator=RandomForestRegressor(
                n_estimators=10,
            ),
            random_state=seed,
        ),
        FeatureImportanceForestClassifier(
            estimator=RandomForestClassifier(
                n_estimators=10,
            ),
            random_state=seed,
            permute_per_tree=False,
            sample_dataset_per_tree=False,
        ),
    ],
)
def test_forestht_check_inputs(forest_hyppo):
    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    # Test case 1: Valid input
    forest_hyppo.statistic(X, y)

    # Test case 2: Invalid input with different number of samples
    X_invalid = np.random.rand(n_samples + 1, X.shape[1])
    y_invalid = rng.integers(0, 2, size=n_samples + 1)
    with pytest.raises(RuntimeError, match="X must have"):
        forest_hyppo.statistic(X_invalid, y_invalid)

    # Test case 3: Invalid input with different number of features
    X_invalid = np.random.rand(X.shape[0], n_features + 1)
    with pytest.raises(RuntimeError, match="X must have"):
        forest_hyppo.statistic(X_invalid, y)

    # Test case 4: Invalid input with incorrect y type target
    y_invalid = np.random.rand(X.shape[0])
    with pytest.raises(RuntimeError, match="y must have type"):
        forest_hyppo.statistic(X, y_invalid)


@pytest.mark.parametrize("backend", ["loky", "threading"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallelization(backend, n_jobs):
    """Test parallelization of training forests."""
    n_samples = 20
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    def run_forest(covariate_index=None):
        clf = FeatureImportanceForestClassifier(
            estimator=HonestForestClassifier(
                n_estimators=10, random_state=seed, n_jobs=n_jobs, honest_fraction=0.2
            ),
            test_size=0.5,
        )
        pvalue = clf.test(X, y, covariate_index=[covariate_index], metric="mi")
        return pvalue

    out = Parallel(n_jobs=-1, backend=backend)(
        delayed(run_forest)(covariate_index) for covariate_index in range(n_features)
    )
    assert len(out) == n_features


def test_pickle(tmpdir):
    """Test that pickling works and preserves fitted attributes."""
    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification
    n_repeats = 1000

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
    )
    stat, pvalue = clf.test(X, y, covariate_index=[1], metric="mi", n_repeats=n_repeats)

    with open(Path(tmpdir) / "clf.pkl", "wb") as fpath:
        pickle.dump(clf, fpath)

    with open(Path(tmpdir) / "clf.pkl", "rb") as fpath:
        clf_pickle = pickle.load(fpath)

    # recompute pvalue manually and compare
    pickle_pvalue = (
        1.0 + (clf_pickle.null_dist_ <= (clf_pickle.permute_stat_ - clf_pickle.observe_stat_)).sum()
    ) / (1.0 + n_repeats)
    assert pvalue == pickle_pvalue
    assert clf_pickle.permute_stat_ - clf_pickle.observe_stat_ == stat

    attr_list = [
        "test_size",
        "observe_samples_",
        "y_true_final_",
        "observe_posteriors_",
        "observe_stat_",
        "_is_fitted",
        "permute_samples_",
        "permute_posteriors_",
        "permute_stat_",
        "n_samples_test_",
        "_n_samples_",
        "_metric",
        "train_test_samples_",
    ]
    for attr in attr_list:
        assert_array_equal(getattr(clf, attr), getattr(clf_pickle, attr))


@pytest.mark.parametrize("permute_per_tree", [True, False])
@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
def test_sample_size_consistency_of_estimator_indices_(permute_per_tree, sample_dataset_per_tree):
    """Test that the test-sample indices are what is expected."""
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
        permute_per_tree=permute_per_tree,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )

    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    _, posteriors, samples = clf.statistic(
        X, y, covariate_index=None, return_posteriors=True, metric="mi"
    )
    if sample_dataset_per_tree:
        assert_array_equal(
            sorted(np.unique(samples)),
            sorted(np.unique(np.concatenate([x[1] for x in clf.train_test_samples_]).flatten())),
        )
    else:
        assert_array_equal(samples, sorted(clf.train_test_samples_[0][1]))
    assert len(_non_nan_samples(posteriors)) == len(samples)


@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
@pytest.mark.parametrize("seed", [None, 0])
def test_permute_per_tree_samples_consistency_with_sklearnforest(seed, sample_dataset_per_tree):
    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
        permute_per_tree=True,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    other_clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
        permute_per_tree=False,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )

    clf.statistic(X, y, covariate_index=None, metric="mi")
    other_clf.statistic(X, y, covariate_index=None, metric="mi")

    # estimator indices should be preserved over multiple calls
    estimator_train_test_indices = deepcopy(clf.train_test_samples_)
    for idx in range(clf.n_estimators):
        assert_array_equal(clf.train_test_samples_[idx][0], estimator_train_test_indices[idx][0])
        assert_array_equal(clf.train_test_samples_[idx][1], estimator_train_test_indices[idx][1])

    # if the sample_dataset_per_tree, then the indices should be different across all
    if sample_dataset_per_tree:
        for (indices, other_indices) in combinations(clf.train_test_samples_, 2):
            assert not np.array_equal(indices[0], other_indices[0])
            assert not np.array_equal(indices[1], other_indices[1])
    else:
        for (indices, other_indices) in combinations(clf.train_test_samples_, 2):
            assert_array_equal(indices[0], other_indices[0])
            assert_array_equal(indices[1], other_indices[1])

    # estimator indices should be preserved over multiple calls
    estimator_train_test_indices = deepcopy(other_clf.train_test_samples_)
    for idx in range(clf.n_estimators):
        assert_array_equal(
            other_clf.train_test_samples_[idx][0], estimator_train_test_indices[idx][0]
        )
        assert_array_equal(
            other_clf.train_test_samples_[idx][1], estimator_train_test_indices[idx][1]
        )

        # when seed is passed, the indices should be deterministic
        if seed is not None:
            assert_array_equal(
                clf.train_test_samples_[idx][0], other_clf.train_test_samples_[idx][0]
            )
            assert_array_equal(
                clf.train_test_samples_[idx][1], other_clf.train_test_samples_[idx][1]
            )


@pytest.mark.parametrize("seed", [None, 0])
def test_small_dataset(seed):
    n_samples = 32
    n_features = 5

    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.5
        ),
        test_size=0.2,
        permute_per_tree=False,
        sample_dataset_per_tree=False,
    )
    stat, pvalue = clf.test(X, y, covariate_index=[1, 2], metric="mi")
    assert ~np.isnan(pvalue)
    assert ~np.isnan(stat)
    assert pvalue > 0.05

    stat, pvalue = clf.test(X, y, metric="mi")
    if seed is not None:
        assert stat == 0.0
    else:
        assert_almost_equal(stat, 0.0, decimal=1)
    assert pvalue > 0.05


# @pytest.mark.monitor_test
# def test_memory_usage():
#     n_samples = 1000
#     n_features = 5000
#     X = rng.uniform(size=(n_samples, n_features))
#     y = rng.integers(0, 2, size=n_samples)  # Binary classification

#     clf = FeatureImportanceForestClassifier(
#         estimator=HonestForestClassifier(
#             n_estimators=10, random_state=seed, n_jobs=-1, honest_fraction=0.5
#         ),
#         test_size=0.2,
#         permute_per_tree=False,
#         sample_dataset_per_tree=False,
#     )

#     stat, pvalue = clf.test(X, y, covariate_index=[1, 2], metric="mi")

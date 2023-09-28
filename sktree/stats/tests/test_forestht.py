import numpy as np
import pytest
from flaky import flaky
from scipy.special import expit
from sklearn import datasets

from sktree import HonestForestClassifier, RandomForestClassifier, RandomForestRegressor
from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.stats import (
    FeatureImportanceForestClassifier,
    FeatureImportanceForestRegressor,
    PermutationForestClassifier,
    PermutationForestRegressor,
)
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


def test_featureimportance_forest_permute_pertree():
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=10,
        ),
        permute_per_tree=True,
        sample_dataset_per_tree=False,
    )
    est.statistic(iris_X[:10], iris_y[:10])

    assert (
        len(est.train_test_samples_[0][1]) == 10 * est.test_size
    ), f"{len(est.train_test_samples_[0][1])} {10 * est.test_size}"
    assert len(est.train_test_samples_[0][0]) == est._n_samples_ - 10 * est.test_size

    est.test(iris_X[:10], iris_y[:10], [0, 1], n_repeats=10, metric="mse")
    assert (
        len(est.train_test_samples_[0][1]) == 10 * est.test_size
    ), f"{len(est.train_test_samples_[0][1])} {10 * est.test_size}"
    assert len(est.train_test_samples_[0][0]) == est._n_samples_ - 10 * est.test_size


def test_featureimportance_forest_errors():
    permute_per_tree = False
    sample_dataset_per_tree = True
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=10,
        ),
        permute_per_tree=permute_per_tree,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    with pytest.raises(ValueError, match="sample_dataset_per_tree"):
        est.statistic(iris_X[:10], iris_y[:10])

    est = FeatureImportanceForestClassifier(estimator=RandomForestRegressor)
    with pytest.raises(RuntimeError, match="Estimator must be"):
        est.statistic(iris_X[:10], iris_y[:10])

    est = FeatureImportanceForestRegressor(estimator=RandomForestClassifier)
    with pytest.raises(RuntimeError, match="Estimator must be"):
        est.statistic(iris_X[:10], iris_y[:10])


@flaky(max_runs=3)
@pytest.mark.slowtest
@pytest.mark.parametrize(
    "hypotester, model_kwargs, n_samples, n_repeats, test_size",
    [
        [
            PermutationForestRegressor,
            {
                "estimator": RandomForestRegressor(
                    max_features="sqrt",
                    random_state=seed,
                    n_estimators=75,
                    n_jobs=-1,
                ),
                "random_state": seed,
            },
            300,
            50,
            0.1,
        ],
        [
            FeatureImportanceForestRegressor,
            {
                "estimator": RandomForestRegressor(
                    max_features="sqrt",
                    random_state=seed,
                    n_estimators=125,
                    n_jobs=-1,
                ),
                "random_state": seed,
                "permute_per_tree": True,
                "sample_dataset_per_tree": True,
            },
            300,  # n_samples
            500,  # n_repeats
            0.1,  # test_size
        ],
    ],
)
def test_linear_model(hypotester, model_kwargs, n_samples, n_repeats, test_size):
    r"""Test hypothesis testing forests using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    Y = Beta * X_1 + Beta * I(X_6 = 2) + \epsilon
    """
    beta = 15.0
    sigma = 0.05
    metric = "mse"

    # sample covariates
    X_15 = rng.uniform(0, 1, size=(n_samples, 5))
    X_610 = np.zeros((n_samples, 5))
    for idx in range(5):
        X_610[:, idx] = np.argwhere(
            rng.multinomial(1, [1.0 / 3, 1.0 / 3, 1.0 / 3], size=(n_samples,))
        )[:, 1]
    X = np.concatenate((X_15, X_610), axis=1)
    assert X.shape == (n_samples, 10)

    # sample noise
    epsilon = rng.normal(size=n_samples, loc=0.0, scale=sigma)

    # compute final y of (n_samples,)
    y = beta * X[:, 0] + (beta * (X[:, 5] == 2.0)) + epsilon
    est = hypotester(test_size=test_size, **model_kwargs)

    # test for X_1
    stat, pvalue = est.test(X, y, [0], metric=metric, n_repeats=n_repeats)
    print("X1: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_6
    stat, pvalue = est.test(X, y, [5], metric=metric, n_repeats=n_repeats)
    print("X6: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for a few unimportant other X
    for covariate_index in [1, 6]:
        # test for X_2, X_7
        stat, pvalue = est.test(X, y, [covariate_index], metric=metric, n_repeats=n_repeats)
        print("X2/7: ", pvalue)
        assert pvalue > 0.05, f"pvalue: {pvalue}"


@flaky(max_runs=3)
@pytest.mark.slowtest
@pytest.mark.parametrize(
    "hypotester, model_kwargs, n_samples, n_repeats, test_size",
    [
        [
            PermutationForestClassifier,
            {
                "estimator": RandomForestClassifier(
                    max_features="sqrt",
                    random_state=seed,
                    n_estimators=50,
                    n_jobs=-1,
                ),
                "random_state": seed,
            },
            600,
            50,
            1.0 / 6,
        ],
        [
            FeatureImportanceForestClassifier,
            {
                "estimator": RandomForestClassifier(
                    max_features="sqrt",
                    random_state=seed,
                    n_estimators=100,
                    n_jobs=-1,
                ),
                "random_state": seed,
                "permute_per_tree": True,
                "sample_dataset_per_tree": True,
            },
            600,
            200,
            1.0 / 6,
        ],
    ],
)
def test_correlated_logit_model(hypotester, model_kwargs, n_samples, n_repeats, test_size):
    r"""Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    P(Y = 1 | X) = expit(beta * \\sum_{j=2}^5 X_j)
    """
    beta = 10.0
    metric = "mse"

    n = 100  # Number of time steps
    ar_coefficient = 0.015

    X = np.zeros((n_samples, n))
    for idx in range(n_samples):
        # sample covariates
        white_noise = rng.standard_normal(size=n)

        # Create an array to store the simulated AR(1) time series
        ar1_series = np.zeros(n)
        ar1_series[0] = white_noise[0]

        # Simulate the AR(1) process
        for t in range(1, n):
            ar1_series[t] = ar_coefficient * ar1_series[t - 1] + white_noise[t]

        X[idx, :] = ar1_series

    # now compute the output labels
    y_proba = expit(beta * X[:, 1:5].sum(axis=1))
    assert y_proba.shape == (n_samples,)
    y = rng.binomial(1, y_proba, size=n_samples)  # .reshape(-1, 1)

    est = hypotester(test_size=test_size, **model_kwargs)

    # test for X_2 important
    stat, pvalue = est.test(X.copy(), y.copy(), [1], n_repeats=n_repeats, metric=metric)
    print("X2: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_1 unimportant
    stat, pvalue = est.test(X.copy(), y.copy(), [0], n_repeats=n_repeats, metric=metric)
    print("X1: ", pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"

    # test for X_500 unimportant
    stat, pvalue = est.test(X.copy(), y.copy(), [n - 1], n_repeats=n_repeats, metric=metric)
    print("X500: ", pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"


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
            n_jobs=-1,
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
        pytest.skip()

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

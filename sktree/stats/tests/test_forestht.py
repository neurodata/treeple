import numpy as np
import pytest
from scipy.special import expit
from sklearn import datasets

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.stats import (
    FeatureImportanceForestClassifier,
    FeatureImportanceForestRegressor,
    PermutationForestClassifier,
    PermutationForestRegressor,
)
from sktree.tree import ObliqueDecisionTreeClassifier

# load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)

# remove third class
iris_X = iris.data[iris.target != 2]
iris_y = iris.target[iris.target != 2]

p = rng.permutation(iris_X.shape[0])
iris_X = iris_X[p]
iris_y = iris_y[p]


seed = 12345


def test_forestht_proper_attributes():
    """Forest HTs should have n_classes_ and n_outputs_ properly set.

    This requires the first dummy fit to always get all classes.
    """
    pass


@pytest.mark.slowtest
@pytest.mark.parametrize(
    "hypotester, model_kwargs, n_samples, n_repeats, test_size",
    [
        [
            PermutationForestRegressor,
            {
                "max_features": 1.0,
                "random_state": seed,
                "n_estimators": 50,
                "n_jobs": -1,
            },
            550,
            50,
            0.1,
        ],
        [
            FeatureImportanceForestRegressor,
            {
                "max_features": 1.0,
                "random_state": seed,
                "n_estimators": 125,
                "permute_per_tree": True,
                "n_jobs": -1,
            },
            600,
            200,
            0.1,
        ],
    ],
)
def test_linear_model(hypotester, model_kwargs, n_samples, n_repeats, test_size):
    r"""Test hypothesis testing forests using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    Y = Beta * X_1 + Beta * I(X_6 = 2) + \epsilon
    """
    beta = 10.0
    sigma = 0.5
    metric = "mse"

    rng = np.random.default_rng(seed)

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
    est = hypotester(**model_kwargs)

    # test for X_1
    stat, pvalue = est.test(X, y, [0], metric=metric, test_size=test_size, n_repeats=n_repeats)
    print("X1: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_6
    stat, pvalue = est.test(X, y, [5], metric=metric, test_size=test_size, n_repeats=n_repeats)
    print("X6: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for a few unimportant other X
    for covariate_index in [1, 6]:
        # test for X_2, X_7
        stat, pvalue = est.test(
            X, y, [covariate_index], metric=metric, test_size=test_size, n_repeats=n_repeats
        )
        print("X2/7: ", pvalue)
        assert pvalue > 0.05, f"pvalue: {pvalue}"


@pytest.mark.slowtest
@pytest.mark.parametrize(
    "hypotester, model_kwargs, n_samples, n_repeats, test_size",
    [
        [
            PermutationForestClassifier,
            {
                "max_features": "sqrt",
                "random_state": seed,
                "n_estimators": 75,
                "n_jobs": -1,
            },
            500,
            50,
            1.0 / 6,
        ],
        [
            FeatureImportanceForestClassifier,
            {
                "max_features": "sqrt",
                "random_state": seed,
                "n_estimators": 125,
                "permute_per_tree": True,
                "sample_dataset_per_tree": True,
                "n_jobs": -1,
            },
            500,
            100,
            1.0 / 6,
        ],
    ],
)
def test_correlated_logit_model(hypotester, model_kwargs, n_samples, n_repeats, test_size):
    r"""Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    P(Y = 1 | X) = expit(beta * \\sum_{j=2}^5 X_j)
    """
    beta = 5.0
    metric = "mse"

    n = 200  # Number of time steps
    ar_coefficient = 0.0015

    rng = np.random.default_rng(seed)

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

    est = hypotester(**model_kwargs)

    # test for X_2 important
    stat, pvalue = est.test(
        X.copy(), y.copy(), [1], test_size=test_size, n_repeats=n_repeats, metric=metric
    )
    print("X2: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_1 unimportant
    stat, pvalue = est.test(
        X.copy(), y.copy(), [0], test_size=test_size, n_repeats=n_repeats, metric=metric
    )
    print("X1: ", pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"

    # test for X_500 unimportant
    stat, pvalue = est.test(
        X.copy(), y.copy(), [n - 1], test_size=test_size, n_repeats=n_repeats, metric=metric
    )
    print("X500: ", pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, "sqrt"])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore"])
@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
    ],
)
@pytest.mark.parametrize("limit", [0.05, 0.1])
def test_iris_pauc(criterion, max_features, honest_prior, estimator, limit):
    # Check consistency on dataset iris.
    clf = FeatureImportanceForestClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=100,
        honest_prior=honest_prior,
        tree_estimator=estimator,
    )
    score = clf.statistic(iris_X, iris_y, metric="auc", max_fpr=limit)
    assert score >= 0.9, "Failed with pAUC: {0} for max fpr: {1}".format(score, limit)

    # now add completely uninformative feature
    X = np.hstack((iris_X, rng.standard_normal(size=(iris_X.shape[0], 1))))

    # test for unimportant feature
    test_size = 0.2
    clf.reset()
    stat, pvalue = clf.test(X, iris_y, [X.shape[1] - 1], test_size=test_size, metric="auc")
    print(pvalue)
    # assert pvalue > 0.05, f"pvalue: {pvalue}"

    stat, pvalue = clf.test(X, iris_y, [2, 3], test_size=test_size, metric="auc")
    print(pvalue)
    # assert pvalue < 0.05, f"pvalue: {pvalue}"
    assert False

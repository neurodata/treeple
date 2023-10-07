"""Tests with respect to down-sized simulations from Coleman et al. 2022.

Coleman, T., Peng, W., & Mentch, L. (2022). Scalable and efficient hypothesis
testing with random forests. The Journal of Machine Learning Research, 23(1),
7679-7713.
"""

import numpy as np
import pytest
from flaky import flaky
from scipy.special import expit

from sktree import RandomForestClassifier, RandomForestRegressor
from sktree.stats import (
    FeatureImportanceForestClassifier,
    FeatureImportanceForestRegressor,
    PermutationForestClassifier,
    PermutationForestRegressor,
)

seed = 12345
rng = np.random.default_rng(seed)


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
                    # random_state=seed,
                    n_estimators=125,
                    n_jobs=-1,
                ),
                "random_state": seed,
                "permute_per_tree": False,
                "sample_dataset_per_tree": False,
            },
            300,  # n_samples
            1000,  # n_repeats
            0.2,  # test_size
        ],
        # XXX: Currently does not work with permute and sample dataset per tree
        # [
        #     FeatureImportanceForestRegressor,
        #     {
        #         "estimator": RandomForestRegressor(
        #             max_features=1.0,
        #             # random_state=seed,
        #             n_estimators=150,
        #             n_jobs=-1,
        #         ),
        #         "permute_per_tree": True,
        #         "sample_dataset_per_tree": True,
        #     },
        #     300,  # n_samples
        #     1000,  # n_repeats
        #     0.2,  # test_size
        # ],
        [
            FeatureImportanceForestRegressor,
            {
                "estimator": RandomForestRegressor(
                    max_features="sqrt",
                    # random_state=seed,
                    n_estimators=125,
                    n_jobs=-1,
                ),
                # "random_state": seed,
                "permute_per_tree": True,
                "sample_dataset_per_tree": False,
            },
            300,  # n_samples
            1000,  # n_repeats
            0.2,  # test_size
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
            # XXX: Currently does not work with permute and sample dataset per tree
            FeatureImportanceForestClassifier,
            {
                "estimator": RandomForestClassifier(
                    max_features="sqrt",
                    # random_state=seed,
                    n_estimators=100,
                    n_jobs=-1,
                ),
                # "random_state": seed,
                "permute_per_tree": False,
                "sample_dataset_per_tree": False,
            },
            600,  # n_samples
            1000,  # n_repeats
            1.0 / 6,  # test_size
        ],
        # XXX: Currently does not work with permute and sample dataset per tree
        # [
        #     FeatureImportanceForestClassifier,
        #     {
        #         "estimator": RandomForestClassifier(
        #             max_features=1.0,
        #             # random_state=seed,
        #             n_estimators=150,
        #             n_jobs=-1,
        #         ),
        #         # "random_state": seed,
        #         "permute_per_tree": True,
        #         "sample_dataset_per_tree": True,
        #     },
        #     600,  # n_samples
        #     1000,  # n_repeats
        #     1.0 / 6,  # test_size
        # ],
        [
            FeatureImportanceForestClassifier,
            {
                "estimator": RandomForestClassifier(
                    max_features="sqrt",
                    # random_state=seed,
                    n_estimators=100,
                    n_jobs=-1,
                ),
                # "random_state": seed,
                "permute_per_tree": True,
                "sample_dataset_per_tree": False,
            },
            600,  # n_samples
            1000,  # n_repeats
            1.0 / 6,  # test_size
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

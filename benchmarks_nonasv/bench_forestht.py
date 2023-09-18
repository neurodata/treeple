"""The following functions reproduce the results from the paper, :footcite:`coleman2022scalable`.

Specifically, the simulations for model 1, 2, 3 and 4 are reproduced.

.. note:: This script will take a long time to run, since a power curve is generated.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit

from sktree.stats import PermutationForestClassifier, PermutationForestRegressor

seed = 12345


def linear_model_ancova(sigma_factor=2.0, seed=None):
    r"""Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    Y = Beta * X_1 + Beta * I(X_6 = 2) + \epsilon
    """
    beta = 10.0
    sigma = 10.0 / sigma_factor
    n_samples = 2200
    n_estimators = 125
    test_size = 0.1

    rng = np.random.default_rng(seed)

    # sample covariates
    X_15 = rng.uniform(0, 1, size=(n_samples, 5))
    X_610 = np.zeros((n_samples, 5))
    for idx in range(5):
        buff = np.argwhere(rng.multinomial(1, [1.0 / 3, 1.0 / 3, 1.0 / 3], size=n_samples))[:, 1]

        X_610[:, idx] = buff

    X = np.concatenate((X_15, X_610), axis=1)
    assert X_15.shape == (n_samples, 5)
    assert X_610.shape == (n_samples, 5)
    assert X.shape == (n_samples, 10)

    # sample noise
    epsilon = rng.normal(size=n_samples, loc=0.0, scale=sigma)

    # compute final y of (n_samples,)
    y = beta * X[:, 0] + (beta * (X[:, 5] - 2)) + epsilon

    # initialize hypothesis tester
    est = PermutationForestRegressor(
        max_features=1.0,
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=-1,
        # bootstrap=True,
        # max_samples=subsample_size
    )
    pvalue_dict = {}

    # test for X_1
    stat, pvalue = est.test(X.copy(), y.copy(), [0], n_repeats=100, test_size=test_size)
    print("X1: ", pvalue)
    pvalue_dict["X1"] = pvalue
    # assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_6
    stat, pvalue = est.test(X.copy(), y.copy(), [5], n_repeats=100, test_size=test_size)
    print("X6: ", pvalue)
    pvalue_dict["X6"] = pvalue
    # assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for a few unimportant other X
    for name, covariate_index in zip(["X2", "X7"], [1, 6]):
        # test for X_2, X_7
        stat, pvalue = est.test(
            X.copy(), y.copy(), [covariate_index], n_repeats=100, test_size=test_size
        )
        print("X2/7: ", pvalue)
        pvalue_dict[name] = pvalue
        # assert pvalue > 0.05, f"pvalue: {pvalue}"

    return pvalue_dict


def correlated_logit_model(beta=5.0, seed=None):
    n_samples = 600
    n_estimators = 125
    n_jobs = -1
    max_features = 1.0
    test_size = 1.0 / 6
    metric = "mse"
    n_repeats = 100

    n = 500  # Number of time steps
    ar_coefficient = 0.15

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

    est = PermutationForestClassifier(
        max_features=max_features, random_state=seed, n_estimators=n_estimators, n_jobs=n_jobs
    )

    # test for X_2 important
    stat, pvalue = est.test(X, y, [1], test_size=test_size, n_repeats=n_repeats, metric=metric)
    print("X2: ", pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_1 unimportant
    stat, pvalue = est.test(X, y, [0], test_size=test_size, n_repeats=n_repeats, metric=metric)
    print("X1: ", pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"

    # test for X_500 unimportant
    stat, pvalue = est.test(X, y, [n - 1], test_size=test_size, n_repeats=n_repeats, metric=metric)
    print("X500: ", pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"


def random_forest_model():
    pass


def evaluate_correlated_logit_model():
    pvalue_dict = defaultdict(list)
    rng = np.random.default_rng(seed)

    beta_space = np.hstack((np.linspace(0.01, 2.5, 8), np.linspace(5, 20, 7)))
    for beta in beta_space:
        for idx in range(5):
            new_seed = rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)

            elements_dict = correlated_logit_model(beta, new_seed)
            for key, value in elements_dict.items():
                pvalue_dict[key].append(value)
            pvalue_dict["sigma_factor"].append(beta)

    df = pd.DataFrame(pvalue_dict)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharey=True, sharex=True)
    axs = axs.flatten()

    for ax, name in zip(axs, ["X1", "X2", "X500"]):
        sns.lineplot(data=df, x="sigma_factor", y=name, ax=ax, marker="o")

        ax.axhline([0.05], ls="--", color="red", label="alpha")
        ax.set(title=name, ylabel="pvalue", xlabel="SNR (beta)")
        ax.legend()
    fig.suptitle("Correlated Logit model")
    fig.tight_layout()


def evaluate_linear_ancova_model():
    pvalue_dict = defaultdict(list)
    rng = np.random.default_rng(seed)

    j_space = np.linspace(0.005, 2.25, 9)

    for sigma_factor in j_space:
        for idx in range(5):
            new_seed = rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)

            elements_dict = linear_model_ancova(sigma_factor, new_seed)
            for key, value in elements_dict.items():
                pvalue_dict[key].append(value)
            pvalue_dict["sigma_factor"].append(sigma_factor)

    df = pd.DataFrame(pvalue_dict)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=True)
    axs = axs.flatten()

    for ax, name in zip(axs, ["X1", "X2", "X6", "X7"]):
        sns.lineplot(data=df, x="sigma_factor", y=name, ax=ax, marker="o")

        ax.axhline([0.05], ls="--", color="red", label="alpha")
        ax.set(title=name, ylabel="pvalue", xlabel="SNR (10 / x)")
        ax.legend()
    fig.suptitle("Linear ANCOVA model")
    fig.tight_layout()


if __name__ == "__main__":
    evaluate_linear_ancova_model()

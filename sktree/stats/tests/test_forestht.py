import numpy as np
import pytest
from scipy.special import expit
from sklearn import datasets

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.stats import MIGHT
from sktree.stats.forestht import ForestHT, HyppoForestRegressor
from sktree.tree import ObliqueDecisionTreeClassifier, PatchObliqueDecisionTreeClassifier

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


def test_iris():
    pass


def test_linear_model():
    r"""Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    Y = Beta * X_1 + Beta * I(X_6 = 2) + \epsilon
    """
    # j = np.linspace(0.005, 2.25, 9)[]
    beta = 10.0
    sigma = 0.05  # / j
    n_samples = 2500
    n_estimators = 125
    test_size = 0.1
    # subsample_size = 0.8

    rng = np.random.default_rng(seed)

    # sample covariates
    X_15 = rng.uniform(0, 1, size=(n_samples, 5))
    X_610 = np.zeros((n_samples, 5))
    for idx in range(5):
        X_610[:, idx] = np.argwhere(
            rng.multinomial(1, [1.0 / 3, 1.0 / 3, 1.0 / 3], size=(n_samples,))
        )[:, 1]
    X = np.concatenate((X_15, X_610), axis=1, dtype=np.float32)
    assert X.shape == (n_samples, 10)

    # sample noise
    epsilon = rng.normal(size=n_samples, loc=0.0, scale=sigma)

    # compute final y of (n_samples,)
    y = beta * X[:, 0] + (beta * (X[:, 5] == 2.0)) + epsilon
    est = HyppoForestRegressor(
        max_features=1.0,
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=-1,
        permute_per_tree=False,
        # bootstrap=True, max_samples=subsample_size
    )

    # test for X_1
    stat, pvalue = est.test(X, y, [0], test_size=test_size)
    print(pvalue)
    # assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for X_6
    stat, pvalue = est.test(X, y, [5], test_size=test_size)
    print(pvalue)
    # assert pvalue < 0.05, f"pvalue: {pvalue}"

    # test for a few unimportant other X
    for covariate_index in [1, 6]:
        # test for X_2, X_3, X_4
        stat, pvalue = est.test(X, y, [covariate_index], test_size=test_size)
        print(pvalue)
        # assert pvalue > 0.05, f"pvalue: {pvalue}"

    assert False


def test_correlated_logit_model():
    r"""Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    P(Y = 1 | X) = expit(beta * \\sum_{j=2}^5 X_j)
    """
    beta = 15.0
    n_samples = 600
    n_estimators = 125
    n_jobs = -1

    n = 100  # Number of time steps
    ar_coefficient = 0.015
    rng = np.random.default_rng(seed)
    test_size = 0.5

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

    est = ForestHT(max_features=n, random_state=seed, n_estimators=n_estimators, n_jobs=n_jobs)

    # test for X_2 important
    stat, pvalue = est.test(X, y, [1], test_size=test_size, metric="mse")
    print(pvalue)
    assert pvalue < 0.6, f"pvalue: {pvalue}"

    # test for X_1
    stat, pvalue = est.test(X, y, [0], metric="mse")
    print(pvalue)
    assert pvalue > 0.9, f"pvalue: {pvalue}"

    # test for X_500
    stat, pvalue = est.test(X, y, [n - 1], metric="mse")
    print(pvalue)
    assert pvalue > 0.9, f"pvalue: {pvalue}"


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
    clf = ForestHT(
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

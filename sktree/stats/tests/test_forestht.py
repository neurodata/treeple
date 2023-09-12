import numpy as np

from sktree.stats.forestht import ForestHT

seed = 12345


def test_iris():
    pass


def test_linear_model():
    """Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    Y = Beta * X_1 + Beta * I(X_6 = 2) + epsilon
    """
    j = np.linspace(0.005, 2.25, 9)[0]
    beta = 10
    sigma = 10 / j
    n_samples = 2000
    n_estimators = 125
    # subsample_size = np.power(n_samples, 0.6)

    rng = np.random.default_rng(seed)

    # sample covariates
    X_15 = rng.uniform(0, 1, size=(n_samples, 5))
    X_610 = rng.multinomial(1, [1.0 / 3, 1.0 / 3, 1.0 / 3], size=(n_samples, 5))
    X = np.concatenate((X_15, X_610), axis=1)

    # sample noise
    epsilon = rng.normal(size=n_samples, scale=sigma)

    # compute final y of (n_samples,)
    y = beta * X[:, 0] + beta * (X[:, 5] == 2) + epsilon

    est = ForestHT(random_state=seed, n_estimators=n_estimators)

    # test for X_1
    stat, pvalue = est.test(X, y, [0])
    assert pvalue < 0.05

    # test for X_6
    stat, pvalue = est.test(X, y, [5])
    assert pvalue < 0.05

    # test for a few unimportant other X
    for covariate_index in [1, 2, 3]:
        # test for X_2, X_3, X_4
        stat, pvalue = est.test(X, y, [covariate_index])
        assert pvalue > 0.05


def test_mars_model():
    """Test MIGHT using MSE from linear model simulation.

    See https://arxiv.org/pdf/1904.07830.pdf Figure 1.

    Y = Beta * X_1 + Beta * I(X_6 = 2) + epsilon
    """
    pass

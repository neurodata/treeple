import numpy as np


def quadratic(n_samples: int, n_features: int, noise=False, seed=None):
    """Simulate classification data from a quadratic model.

    Parameters
    ----------
    n_samples : int
        _description_
    n_features : int
        _description_
    noise : bool, optional
        _description_, by default False
    seed : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(size=(n_samples, n_features))
    coeffs = np.array([np.exp(-0.0325 * (i + 24)) for i in range(n_features)])
    eps = rng.standard_normal(size=(n_samples, n_features))

    x_coeffs = x * coeffs
    y = x_coeffs**2 + noise * eps

    n1 = x.shape[0]
    n2 = y.shape[0]
    v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
    x = np.vstack((x, y))
    return x, v

import numpy as np


def make_quadratic_classification(n_samples: int, n_features: int, noise=False, seed=None):
    """Simulate classification data from a quadratic model.

    This is a form of the simulation used in :footcite:`panda2018learning`.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_features : int
        The number of dimensions in the dataset.
    noise : bool, optional
        Whether or not to add noise, by default False.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    x : array-like, shape (2 * n_samples, n_features)
        Data array.
    v : array-like, shape (2 * n_samples,)
        Target array of 1's and 0's.

    References
    ----------
    .. footbibliography::
    """
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(size=(n_samples, n_features))
    coeffs = np.array([np.exp(-0.0325 * (i + 24)) for i in range(n_features)])
    eps = rng.standard_normal(size=(n_samples, n_features))

    x_coeffs = x * coeffs
    y = x_coeffs**2 + noise * eps

    # generate the classification labels
    n1 = x.shape[0]
    n2 = y.shape[0]
    v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
    x = np.vstack((x, y))
    return x, v

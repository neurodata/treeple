import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats


def simulate_helix(
    radius_a=0,
    radius_b=1,
    obs_noise_func=None,
    nature_noise_func=None,
    alpha=0.005,
    n_samples=1000,
    return_mi_lb=False,
    random_seed=None,
):
    """Simulate data from a helix.

    Parameters
    ----------
    radius_a : int, optional
        The value of the smallest radius, by default 0.0.
    radius_b : int, optional
        The value of the largest radius, by default 1.0
    obs_noise_func : Callable, optional
        By default None, which defaults to a Uniform distribution from
        (-0.005, 0.005). If passed in, then must be a callable that when
        called returns a random number denoting the noise.
    nature_noise_func : callable, optional
        By defauult None, which will add no noise. The nature noise func
        is just an independent noise term added to ``P`` before it is
        passed to the generation of the X, Y, and Z terms.
    alpha : float, optional
        The value of the noise, by default 0.005.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    return_mi_lb : bool, optional
        Whether to return the mutual information lower bound, by default False.
    random_seed : int, optional
        The random seed.

    Returns
    -------
    P : array-like of shape (n_samples,)
        The sampled P.
    X : array-like of shape (n_samples,)
        The X dimension.
    Y : array-like of shape (n_samples,)
        The X dimension.
    Z : array-like of shape (n_samples,)
        The X dimension.
    lb : float
        The mutual information lower bound.

    Notes
    -----
    Data is generated as follows: We first sample a radius that
    defines the helix, :math:`R \\approx Unif(radius_a, radius_b)`.
    Afterwards, we generate one sample as follows::

        P = 5\\pi + 3\\pi R
        X = (P + \\epsilon_1) cos(P + \\epsilon_1) / 8\\pi + N_1
        Y = (P + \\epsilon_2) sin(P + \\epsilon_2) / 8\\pi + N_2
        Z = (P + \\epsilon_3) / 8\\pi + N_3

    where :math:`N_1,N_2,N_3` are noise variables that are independently
    sampled for each sample point. And
    :math:`\\epsilon_1, \\epsilon_2, \\epsilon_3` are "nature noise" terms
    which are off by default. This process is repeated ``n_samples`` times.

    Note, that this forms the graphical model::

        R \\rightarrow P

        P \\rightarrow X
        P \\rightarrow Y
        P \\rightarrow Z

    such that P is a confounder among X, Y and Z. This implies that X, Y and Z
    are conditionally dependent on P, whereas
    """
    rng = np.random.default_rng(random_seed)

    Radii = np.zeros((n_samples,))
    P_arr = np.zeros((n_samples,))
    X_arr = np.zeros((n_samples,))
    Y_arr = np.zeros((n_samples,))
    Z_arr = np.zeros((n_samples,))

    if obs_noise_func is None:
        obs_noise_func = lambda: rng.uniform(-alpha, alpha)
    if nature_noise_func is None:
        nature_noise_func = lambda: 0.0

    for idx in range(n_samples):
        Radii[idx] = rng.uniform(radius_a, radius_b)
        P_arr[idx] = 5 * np.pi + 3 * np.pi * Radii[idx]
        X_arr[idx] = (P_arr[idx] + nature_noise_func()) * np.cos(
            P_arr[idx] + nature_noise_func()
        ) / (8 * np.pi) + obs_noise_func()
        Y_arr[idx] = (P_arr[idx] + nature_noise_func()) * np.sin(
            P_arr[idx] + nature_noise_func()
        ) / (8 * np.pi) + obs_noise_func()
        Z_arr[idx] = (P_arr[idx] + nature_noise_func()) / (8 * np.pi) + obs_noise_func()

    if return_mi_lb:
        lb = alpha / 2 - np.log(alpha)
        return P_arr, X_arr, Y_arr, Z_arr, lb

    return P_arr, X_arr, Y_arr, Z_arr


def simulate_sphere(
    radius=1, noise_func=None, alpha=0.005, n_samples=1000, return_mi_lb=False, random_seed=None
):
    """Simulate samples generated on a sphere.

    Parameters
    ----------
    radius : int, optional
        The radius of the sphere, by default 1.
    noise_func : callable, optional
        The noise function to call to add to samples, by default None,
        which defaults to sampling from the uniform distribution [-alpha, alpha].
    alpha : float, optional
        The value of the noise, by default 0.005.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    return_mi_lb : bool, optional
        Whether to return the mutual information lower bound, by default False.
    random_seed : int, optional
        Random seed, by default None.

    Returns
    -------
    latitude : float
        Latitude.
    longitude : float
        Longitude.
    Y1 : array-like of shape (n_samples,)
        The X coordinate.
    Y2 : array-like of shape (n_samples,)
        The Y coordinate.
    Y3 : array-like of shape (n_samples,)
        The Z coordinate.
    lb : float
        The mutual information lower bound.
    """
    rng = np.random.default_rng(random_seed)
    if noise_func is None:
        noise_func = lambda: rng.uniform(-alpha, alpha)

    latitude = np.zeros((n_samples,))
    longitude = np.zeros((n_samples,))
    Y1 = np.zeros((n_samples,))
    Y2 = np.zeros((n_samples,))
    Y3 = np.zeros((n_samples,))

    # sample latitude and longitude
    for idx in range(n_samples):
        # sample latitude and longitude
        latitude[idx] = rng.uniform(0, 2 * np.pi)
        longitude[idx] = np.arccos(1 - 2 * rng.uniform(0, 1))

        Y1[idx] = np.sin(longitude[idx]) * np.cos(latitude[idx]) * radius + noise_func()
        Y2[idx] = np.sin(longitude[idx]) * np.sin(latitude[idx]) * radius + noise_func()
        Y3[idx] = np.cos(longitude[idx]) * radius + noise_func()

    if return_mi_lb:
        lb = alpha / 2 - np.log(alpha)
        return latitude, longitude, Y1, Y2, Y3, lb

    return latitude, longitude, Y1, Y2, Y3


def simulate_multivariate_gaussian(mean=None, cov=None, d=2, n_samples=1000, seed=1234):
    """Multivariate gaussian simulation for testing entropy and MI estimators.

    Simulates samples from a "known" multivariate gaussian distribution
    and then passes those samples, along with the true analytical MI/CMI.

    Parameters
    ----------
    mean : array-like of shape (n_features,)
        The optional mean array. If None (default), a random standard normal vector is drawn.
    cov : array-like of shape (n_features, n_features)
        The covariance array. If None (default), a random standard normal 2D array is drawn.
        It is then converted to a PD matrix.
    d : int
        The dimensionality of the multivariate gaussian. By default 2.
    n_samples : int
        The number of samples to generate. By default 1000.
    seed : int
        The random seed to feed to :func:`numpy.random.default_rng`.

    Returns
    -------
    data : array-like of shape (n_samples, n_features)
        The generated data from the distribution.
    mean : array-like of shape (n_features,)
        The mean vector of the distribution.
    cov : array-like of shape (n_features, n_features)
        The covariance matrix of the distribution.
    """
    rng = np.random.default_rng(seed)

    if mean is None:
        mean = rng.normal(size=(d,))
    if cov is None:
        # generate random covariance matrix and enure it is symmetric and positive-definite
        cov = rng.normal(size=(d, d))
        cov = 0.5 * (cov.dot(cov.T))
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise RuntimeError("Passed in covariance matrix should be positive definite")
    if not scipy.linalg.issymmetric(cov):
        raise RuntimeError(f"Passed in covariance matrix {cov} should be symmetric")

    if len(mean) != d or len(cov) != d:
        raise RuntimeError(f"Dimensionality of mean and covariance matrix should match {d}")

    data = rng.multivariate_normal(mean=mean, cov=cov, size=(n_samples))

    return data, mean, cov

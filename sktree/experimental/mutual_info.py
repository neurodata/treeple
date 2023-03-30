from typing import Optional

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sktree.ensemble import UnsupervisedObliqueRandomForest, pairwise_forest_distance


def simulate_helix(
    radius_a=0,
    radius_b=1,
    obs_noise_func=None,
    nature_noise_func=None,
    n_samples=1000,
    random_seed=None,
):
    """Simulate data from a helix.

    Parameters
    ----------
    radius_a : int, optional
        The value of the smallest radius, by default 0.0.
    radius_b : int, optional
        The value of the largest radius, by default 1.0
    obs_noise_func : scipy.stats.distribution, optional
        By default None, which defaults to a Uniform distribution from
        (-0.005, 0.005). If passed in, then must be a callable that when
        called returns a random number denoting the noise.
    nature_noise_func : callable, optional
        By defauult None, which will add no noise. The nature noise func
        is just an independent noise term added to ``P`` before it is
        passed to the generation of the X, Y, and Z terms.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
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
        obs_noise_func = lambda: rng.uniform(-0.005, 0.005)
    if nature_noise_func is None:
        nature_noise_func = lambda: 0.0

    for idx in range(n_samples):
        Radii[idx] = rng.uniform(radius_a, radius_b)
        P_arr[idx] = 5 * np.pi + 3 * np.pi * Radii[idx]
        X_arr[idx] = (P_arr[idx] + nature_noise_func) * np.cos(P_arr[idx] + nature_noise_func) / (
            8 * np.pi
        ) + obs_noise_func()
        Y_arr[idx] = (P_arr[idx] + nature_noise_func) * np.sin(P_arr[idx] + nature_noise_func) / (
            8 * np.pi
        ) + obs_noise_func()
        Z_arr[idx] = (P_arr[idx] + nature_noise_func) / (8 * np.pi) + obs_noise_func()

    return P_arr, X_arr, Y_arr, Z_arr


def simulate_sphere(radius=1, noise_func=None, n_samples=1000, random_seed=None):
    """Simulate samples generated on a sphere.

    Parameters
    ----------
    radius : int, optional
        The radius of the sphere, by default 1.
    noise_func : callable, optional
        The noise function to call to add to samples, by default None,
        which defaults to sampling from the uniform distribution [-0.005, 0.005].
    n_samples : int, optional
        Number of samples to generate, by default 1000.
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
    """
    rng = np.random.default_rng(random_seed)
    if noise_func is None:
        noise_func = lambda: rng.uniform(-0.005, 0.005)

    latitude = np.zeros((n_samples,))
    longitude = np.zeros((n_samples,))
    Y1 = np.zeros((n_samples,))
    Y2 = np.zeros((n_samples,))
    Y3 = np.zeros((n_samples,))

    for idx in range(n_samples):
        # sample latitude and longitude
        latitude[idx] = rng.uniform(0, 1)
        longitude[idx] = rng.uniform(0, 1)

        Y1[idx] = np.cos(latitude[idx]) * np.cos(longitude[idx]) * radius + noise_func()
        Y2[idx] = np.cos(latitude[idx]) * np.sin(longitude[idx]) * radius + noise_func()
        Y3[idx] = np.sin(longitude[idx]) * radius + noise_func()

    return latitude, longitude, Y1, Y2, Y3


def simulate_multivariate_gaussian(mean=None, cov=None, d=2, n_samples=1000, seed=1234):
    """Multivariate gaussian simulation for testing entropy and MI estimators.

    Simulates samples from a "known" multivariate gaussian distribution
    and then passes those samples, along with the true analytical MI/CMI.

    Parameters
    ----------
    mean : array-like of shape (d,)
        The optional mean array. If None (default), a random standard normal vector is drawn.
    cov : array-like of shape (d,d)
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
    data : array-like of shape (n_samples, d)
        The generated data from the distribution.
    mean : array-like of shape (d,)
        The mean vector of the distribution.
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.
    """
    rng = np.random.default_rng(seed)

    if mean is None:
        mean = rng.normal(size=(d,))
    if cov is None:
        # generate random covariance matrix and enure it is symmetric and positive-definite
        cov = rng.normal(size=(d, d))
        cov = 0.5 * (cov + cov.T)
    else:
        if not np.all(np.linalg.eigvals(cov) > 0):
            raise RuntimeError("Passed in covariance matrix should be positive definite")
        if not scipy.linalg.issymmetric(cov):
            raise RuntimeError("Passed in covariance matrix should be symmetric")

    data = rng.multivariate_normal(mean=mean, cov=cov, size=(n_samples))

    return data, mean, cov


def entropy_gaussian(cov):
    """Compute entropy of a multivariate Gaussian.

    Computes the analytical solution due to :footcite:`Darbellay1999Entropy`.

    Parameters
    ----------
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.

    Returns
    -------
    true_mi : float
        The true analytical mutual information of the generated multivariate Gaussian distribution.

    Notes
    -----
    The analytical solution for the true mutual information, ``I(X; Y)`` is given by::

        I(X; Y) = H(X) + H(Y) - H(X, Y) = -\\frac{1}{2} log(det(C))

    References
    ----------
    .. footbibliography::
    """
    d = cov.shape[0]

    true_entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(cov))
    return true_entropy


def mi_gaussian(cov):
    """Compute mutual information of a multivariate Gaussian.

    Parameters
    ----------
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.

    Returns
    -------
    true_mi : float
        The true analytical entropy of the generated multivariate Gaussian distribution.

    Notes
    -----
    Analytical solution for entropy, :math:`H(X)` of a multivariate Gaussian is given by::

        H(X) = \\frac{d}{2} (1 + log(2\\pi)) + \\frac{1}{2} log(det(C))
    """
    # computes the true MI
    true_mi = -0.5 * np.log(np.linalg.det(cov))
    return true_mi


def cmi_gaussian(cov, x_index, y_index, z_index):
    """Computes the analytical CMI for a multivariate Gaussian distribution.

    Parameters
    ----------
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.
    x_index : list or int
        List of indices in ``cov`` that are for the X variable.
    y_index : list or int
        List of indices in ``cov`` that are for the Y variable.
    z_index : list or int
        List of indices in ``cov`` that are for the Z variable.

    Returns
    -------
    true_mi : float
        The true analytical mutual information of the generated multivariate Gaussian distribution.

    Notes
    -----
    Analytical solution for conditional mutual information, :math:`I(X;Y|Z)` of a
    multivariate Gaussian is given by::

        I(X;Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where we plug in the analytical solutions for entropy as shown in :func:`entropy_gaussian`.
    """
    x_index = np.atleast_1d(x_index)
    y_index = np.atleast_1d(y_index)
    z_index = np.atleast_1d(z_index)

    xz_index = np.concatenate((x_index, z_index)).squeeze()
    yz_index = np.concatenate((y_index, z_index)).squeeze()

    cov_xz = cov[xz_index, xz_index]
    cov_yz = cov[yz_index, yz_index]
    cov_z = cov[z_index, z_index]

    cmi = (
        entropy_gaussian(cov_xz)
        + entropy_gaussian(cov_yz)
        - entropy_gaussian(cov_z)
        - entropy_gaussian(cov)
    )
    return cmi


def entropy_weibull(alpha, k):
    """Analytical solution for entropy of Weibull distribution.

    https://en.wikipedia.org/wiki/Weibull_distribution
    """
    return np.euler_gamma * (1.0 - 1.0 / alpha) - np.log(alpha * np.power(k, 1.0 / alpha)) + 1


def mi_gamma(theta):
    """Analytical solution for"""
    return scipy.special.digamma(theta + 1) - np.log(theta)


def mi_from_entropy(hx, hy, hxy):
    """Analytic formula for MI given plug-in estimates of entropies."""
    return hx + hy - hxy


def cmi_from_entropy(hxz, hyz, hz, hxyz):
    """Analytic formula for CMI given plug-in estimates of entropies."""
    return hxz + hyz - hz - hxyz


def mutual_info_ksg(
    X,
    Y,
    Z=None,
    k: float = 0.2,
    metric="forest",
    algorithm="kd_tree",
    n_jobs: int = -1,
    transform: str = "rank",
    random_seed: int = None,
):
    """Compute the generalized (conditional) mutual information KSG estimate.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features_x)
        The Xcovariate space.
    Y : ArrayLike of shape (n_samples, n_features_y)
        The Y covariate space.
    Z : ArrayLike of shape (n_samples, n_features_z), optional
        The Z covariate space, by default None. If None, then the MI is computed.
        If Z is defined, then the CMI is computed.
    k : float, optional
        The number of neighbors to use in defining the radius, by default 0.2.
    metric : str
        Any distance metric accepted by :class:`sklearn.neighbors.NearestNeighbors`.
        If 'forest' (default), then uses an :class:`UnsupervisedObliqueRandomForest` to compute
        geodesic distances.
    algorithm : str, optional
        Method to use, by default 'knn'. Can be ('ball_tree', 'kd_tree', 'brute').
    n_jobs : int, optional
        Number of parallel jobs, by default -1.
    transform : one of {'rank', 'standardize', 'uniform'}
        Preprocessing, by default "rank".
    random_seed : int, optional
        Random seed, by default None.

    Returns
    -------
    val : float
        The estimated MI, or CMI value.

    Notes
    -----
    Given a dataset with ``n`` samples, the KSG estimator proceeds by:

    1. For fixed k, get the distance to the kth nearest-nbr in XYZ subspace, call it 'r'
    2. Get the number of NN in XZ subspace within radius 'r'
    3. Get the number of NN in YZ subspace within radius 'r'
    4. Get the number of NN in Z subspace within radius 'r'
    5. Apply analytic solution for KSG estimate

    For MI :footcite:`Kraskov_2004`, the analytical solution is::

        \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)

    For CMI :footcite:`Frenzel2007`m the analytical solution is::

        \\psi(k) - E[(\\psi(n_{xz}) + \\psi(n_{yz}) - \\psi(n_{z}))]

    where :math:`\\psi` is the DiGamma function, and each expectation term
    is estimated by taking the sample average.

    Note that the :math:`n_i` terms denote the number of neighbors within
    radius 'r' in the subspace of 'i', where 'i' could be for example the
    X, Y, XZ, etc. subspaces. This term does not include the sample itself.
    """
    rng = np.random.default_rng(random_seed)

    n_samples, n_features_x = X.shape
    _, n_features_y = Y.shape
    data = np.hstack((X, Y))

    if Z is not None:
        _, n_features_z = Z.shape
        data = np.hstack((data, Z))
    else:
        n_features_z = 0
    n_features = n_features_x + n_features_y + n_features_z

    # add minor noise to make sure there are no ties
    random_noise = rng.random((n_samples, n_features_x))
    data += 1e-5 * random_noise @ np.std(data, axis=0).reshape(n_features, 1)

    if transform == "standardize":
        # standardize with standard scaling
        data = data.astype(np.float64)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    elif transform == "uniform":
        data = _trafo2uniform(data)
    elif transform == "rank":
        # rank transform each column
        data = scipy.stats.rankdata(data, axis=0)

    if k < 1:
        knn_here = max(1, int(k * n_samples))
    else:
        knn_here = max(1, int(k))

    if Z is not None:
        val = _cmi_ksg(data, X, Y, Z, metric, algorithm, knn_here, n_jobs)
    else:
        val = _mi_ksg(data, X, Y, metric, algorithm, knn_here, n_jobs)
    return val


def _mi_ksg(data, X, Y, metric, algorithm, knn_here, n_jobs):
    """Compute KSG estimate of MI."""
    n_samples = X.shape[0]

    # estimate distance to the kth NN in XYZ subspace for each sample
    neigh = _compute_nn(data, algorithm=algorithm, metric=metric, k=knn_here, n_jobs=n_jobs)

    # get the radius we want to use per sample as the distance to the kth neighbor
    # in the joint distribution space
    dists, _ = neigh.kneighbors()
    radius_per_sample = dists[:, -1]

    # compute on the subspace of X
    num_nn_x = _compute_radius_nbrs(
        X,
        radius_per_sample,
        knn_here,
        algorithm=algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )

    # compute on the subspace of Y
    num_nn_y = _compute_radius_nbrs(
        Y,
        radius_per_sample,
        knn_here,
        algorithm=algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )

    # compute the final MI value
    # \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)
    hxy = scipy.special.digamma(knn_here)
    hx = scipy.special.digamma(num_nn_x)
    hy = scipy.special.digamma(num_nn_y)
    hn = scipy.special.digamma(n_samples)
    val = hxy - (hx + hy).mean() + hn
    return val


def _cmi_ksg(data, X, Y, Z, metric, algorithm, knn_here, n_jobs):
    """Compute KSG estimate of CMI."""
    # estimate distance to the kth NN in XYZ subspace for each sample
    neigh = _compute_nn(data, algorithm=algorithm, metric=metric, k=knn_here, n_jobs=n_jobs)

    # get the radius we want to use per sample as the distance to the kth neighbor
    # in the joint distribution space
    dists, _ = neigh.kneighbors()
    radius_per_sample = dists[:, -1]

    # compute on the subspace of XZ
    xz_data = np.hstack((X, Z))
    num_nn_xz = _compute_radius_nbrs(
        xz_data,
        radius_per_sample,
        knn_here,
        algorithm=algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )

    # compute on the subspace of YZ
    yz_data = np.hstack((Y, Z))
    num_nn_yz = _compute_radius_nbrs(
        yz_data,
        radius_per_sample,
        knn_here,
        algorithm=algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )

    # compute on the subspace of XZ
    num_nn_z = _compute_radius_nbrs(
        Z,
        radius_per_sample,
        knn_here,
        algorithm=algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )

    # compute the final CMI value
    hxyz = scipy.special.digamma(knn_here)
    hxz = scipy.special.digamma(num_nn_xz)
    hyz = scipy.special.digamma(num_nn_yz)
    hz = scipy.special.digamma(num_nn_z)
    val = hxyz - (hxz + hyz - hz).mean()
    return val


def _compute_radius_nbrs(
    data,
    radius_per_sample,
    k,
    algorithm: str = "kd_tree",
    metric="l2",
    n_jobs: Optional[int] = None,
):
    neigh = _compute_nn(data, algorithm=algorithm, metric=metric, k=k, n_jobs=n_jobs)

    n_samples = radius_per_sample.shape[0]

    num_nn_data = np.zeros((n_samples,))
    for idx in range(n_samples):
        num_nn = neigh.radius_neighbors(radius=radius_per_sample[idx])
        num_nn_data[idx] = num_nn
    return num_nn_data


def _compute_nn(
    X: ArrayLike, algorithm: str = "kd_tree", metric="l2", k: int = 1, n_jobs: Optional[int] = None
) -> ArrayLike:
    """Compute kNN in subspace.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        The covariate space.
    algorithm : str, optional
        Method to use, by default 'knn'. Can be ('ball_tree', 'kd_tree', 'brute').
    metric : str
        Any distance metric accepted by :class:`sklearn.neighbors.NearestNeighbors`.
        If 'forest', then uses an :class:`UnsupervisedObliqueRandomForest` to compute
        geodesic distances.
    k : int, optional
        The number of k-nearest neighbors to query, by default 1.
    n_jobs : int,
        The number of CPUs to use for joblib. By default, None.

    Returns
    -------
    neigh : instance of sklearn.neighbors.NearestNeighbor
        A fitted instance of the nearest-neighbor algorithm on ``X`` input.

    Notes
    -----
    Can query for the following, using the ``neigh.kneighbors(X)`` function, which would
    return:

    dists : ArrayLike of shape (n_samples, k)
        The distance array of every sample with its k-nearest neighbors. The columns
        are ordered from closest to furthest neighbors.
    indices : ArrayLike of shape (n_samples, k)
        The sample indices of the k-nearest-neighbors for each sample. These
        contain the row indices of ``X`` for each sample. The columns
        are ordered from closest to furthest neighbors.
    """
    if metric == "forest":
        est = UnsupervisedObliqueRandomForest()
        dists = pairwise_forest_distance(est, X, n_jobs=n_jobs)

        # we have a precomputed distance matrix, so we can use the NearestNeighbor
        # implementation of sklearn
        metric = "precomputed"
    else:
        dists = X

    # compute the nearest neighbors in the space using specified NN algorithm
    # then get the K nearest nbrs and their distances
    neigh = NearestNeighbors(n_neighbors=k, algorithm=algorithm, metric=metric, n_jobs=n_jobs).fit(
        dists
    )
    return neigh


def _trafo2uniform(X):
    """Transforms input array to uniform marginals.

    Assumes x.shape = (dim, T)

    Parameters
    ----------
    X : arraylike
        The input data with (n_samples,) rows and (n_features,) columns.

    Returns
    -------
    u : array-like
        array with uniform marginals.
    """

    def trafo(xi):
        xisorted = np.sort(xi)
        yi = np.linspace(1.0 / len(xi), 1, len(xi))
        return np.interp(xi, xisorted, yi)

    _, n_features = X.shape

    # apply a uniform transformation for each feature
    for idx in range(n_features):
        marginalized_feature = trafo(X[:, idx].to_numpy().squeeze())
        X[:, idx] = marginalized_feature
    return X

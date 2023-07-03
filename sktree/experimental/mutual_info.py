from typing import Optional

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sktree.ensemble import UnsupervisedObliqueRandomForest
from sktree.tree import compute_forest_similarity_matrix


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
    """Compute the analytical CMI for a multivariate Gaussian distribution.

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
        The X covariate space.
    Y : ArrayLike of shape (n_samples, n_features_y)
        The Y covariate space.
    Z : ArrayLike of shape (n_samples, n_features_z), optional
        The Z covariate space, by default None. If None, then the MI is computed.
        If Z is defined, then the CMI is computed.
    k : float, optional
        The number of neighbors to use in defining the radius, by default 0.2.
    metric : str
        Any distance metric accepted by :class:`sklearn.neighbors.NearestNeighbors`.
        If 'forest' (default), then uses an
        :class:`sktree.UnsupervisedObliqueRandomForest` to compute geodesic distances.
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

    For MI, the analytical solution is:

    .. math::

        \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)

    For CMI, the analytical solution is:

    .. math::

        \\psi(k) - E[(\\psi(n_{xz}) + \\psi(n_{yz}) - \\psi(n_{z}))]

    where :math:`\\psi` is the DiGamma function, and each expectation term
    is estimated by taking the sample average.

    Note that the :math:`n_i` terms denote the number of neighbors within
    radius 'r' in the subspace of 'i', where 'i' could be for example the
    X, Y, XZ, etc. subspaces. This term does not include the sample itself.
    """
    rng = np.random.default_rng(random_seed)

    data = np.hstack((X, Y))
    if Z is not None:
        data = np.hstack((data, Z))

    data = _preprocess_data(data, transform, rng)
    n_samples = data.shape[0]

    if k < 1:
        knn_here = max(1, int(k * n_samples))
    else:
        knn_here = max(1, int(k))

    if Z is not None:
        val = _cmi_ksg(data, X, Y, Z, metric, algorithm, knn_here, n_jobs)
    else:
        val = _mi_ksg(data, X, Y, metric, algorithm, knn_here, n_jobs)
    return val


def _preprocess_data(data, transform, rng):
    n_samples, n_features = data.shape

    # add minor noise to make sure there are no ties
    random_noise = rng.random((n_samples, n_features))
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
    return data


def _mi_ksg(data, X, Y, metric, algorithm, knn_here, n_jobs):
    """Compute KSG estimate of MI."""
    n_samples = X.shape[0]

    # estimate distance to the kth NN in XYZ subspace for each sample
    # - get the radius we want to use per sample as the distance to the kth neighbor
    #   in the joint distribution space
    neigh = _compute_nn(data, algorithm=algorithm, metric=metric, k=knn_here, n_jobs=n_jobs)
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
        nn = neigh.radius_neighbors(radius=radius_per_sample[idx], return_distance=False)
        num_nn_data[idx] = len(nn)
    return num_nn_data


def _compute_nn(
    X: ArrayLike, algorithm: str = "kd_tree", metric="l2", k: int = 1, n_jobs: Optional[int] = None
) -> NearestNeighbors:
    """Compute kNN in subspace.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        The covariate space.
    algorithm : str, optional
        Method to use, by default 'knn'. Can be ('ball_tree', 'kd_tree', 'brute').
    metric : str
        Any distance metric accepted by :class:`sklearn.neighbors.NearestNeighbors`.
        If 'forest', then uses an :class:`sktree.UnsupervisedObliqueRandomForest`
        to compute geodesic distances.
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
        dists = compute_forest_similarity_matrix(est, X)

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

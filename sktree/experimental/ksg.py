from typing import Optional

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def mutual_info_ksg(
    X,
    Y,
    Z=None,
    k: float = 0.2,
    nn_estimator=None,
    n_jobs: int = -1,
    transform: str = "rank",
    random_seed: int = None,
    verbose: bool=False
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
        If a number less than 1, then the number of neighbors is computed as
        ``k * n_samples``.
    nn_estimator : str, optional
        The nearest neighbor estimator to use, by default None. If None willl default
        to using :class:`sklearn.neighbors.NearestNeighbors` with default parameters.
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

    The hyperparamter ``k`` defines the number of points in the D-dimensional
    ball with a specified radius. The larger the k, the higher the bias in
    the estimate, but the lower the variance. The smaller the k, the lower
    the bias, but the higher the variance. The default value of 0.2 is set
    to allow scaling with the number of samples. 
    """
    rng = np.random.default_rng(random_seed)
    
    if nn_estimator is None:
        nn_estimator = NearestNeighbors(n_jobs=n_jobs)

    data = np.hstack((X, Y))
    x_idx = np.arange(X.shape[1])
    y_idx = np.arange(Y.shape[1]) + X.shape[1]
    z_idx = np.array([])
    if Z is not None:
        z_idx = np.arange(Z.shape[1]) + data.shape[1]
        data = np.hstack((data, Z))

    data = _preprocess_data(data, transform, rng)
    if verbose:
        print(f"Data shape: {data.shape}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        if Z is not None:
            print(f"Z shape: {Z.shape}")
        print('Preprocessing complete.')
    n_samples = data.shape[0]

    # get the number of neighbors to use in estimating the CMI
    if k < 1:
        knn_here = max(1, int(k * n_samples))
    else:
        knn_here = max(1, int(k))

    if verbose:
        print(f"Using {knn_here} neighbors to define D-dimensional volume.")

    if Z is not None:
        val = _cmi_ksg(data, x_idx, y_idx, z_idx, nn_estimator, knn_here)
    else:
        val = _mi_ksg(data, x_idx, y_idx, nn_estimator, knn_here)
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


def _mi_ksg(data, x_idx, y_idx, nn_estimator: BaseEstimator, knn_here: int, verbose: bool=False)-> float:
    """Compute KSG estimate of MI.

    Parameters
    ----------
    data : ArrayLike
        Stacked data of X and Y.
    x_idx : ArrayLike
        Indices for the X data stored as a 1D array.
    y_idx : ArrayLike
        Indices for the Y data stored as a 1D array.
    nn_estimator : BaseEstimator
        Nearest neighbor estimator.
    knn_here : int
        Number of nearest neighbors used in nn_estimator to estimate the volume
        of the joint distribution.

    Returns
    -------
    val : float
        Estimated MI value.
    """
    n_samples = data.shape[0]

    # estimate distance to the kth NN in XYZ subspace for each sample
    neigh = nn_estimator.fit(data, force_fit=True)
    dists, _ = neigh.kneighbors(n_neighbors=knn_here)

    # - get the radius we want to use per sample as the distance to the kth neighbor
    #   in the joint distribution space
    radius_per_sample = dists[:, -1]

    # compute on the subspace of X
    num_nn_x = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=x_idx
    )

    # compute on the subspace of Y
    num_nn_y = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=y_idx
    )

    # compute the final MI value
    # \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)
    hxy = scipy.special.digamma(knn_here)
    hx = scipy.special.digamma(num_nn_x)
    hy = scipy.special.digamma(num_nn_y)
    hn = scipy.special.digamma(n_samples)
    val = hxy - (hx + hy).mean() + hn
    return val


def _cmi_ksg(data: ArrayLike, x_idx, y_idx, z_idx, nn_estimator: BaseEstimator, knn_here: int) -> float:
    """Compute KSG estimate of CMI.

    Parameters
    ----------
    data : ArrayLike
        Stacked data of X, Y and Z.
    x_idx : ArrayLike
        Indices for the X data stored as a 1D array.
    y_idx : ArrayLike
        Indices for the Y data stored as a 1D array.
    z_idx : ArrayLike
        Indices for the Z data stored as a 1D array.
    nn_estimator : BaseEstimator
        Nearest neighbor estimator.
    knn_here : int
        Number of nearest neighbors used in nn_estimator to estimate the volume
        of the joint distribution.

    Returns
    -------
    val : float
        Estimated CMI value.
    """
    # estimate distance to the kth NN in XYZ subspace for each sample
    neigh = nn_estimator.fit(data, force_fit=True)

    # get the radius we want to use per sample as the distance to the kth neighbor
    # in the joint distribution space
    dists, _ = neigh.kneighbors(knn_here)
    radius_per_sample = dists[:, -1]

    # compute on the subspace of XZ
    xz_idx = np.concatenate((x_idx, z_idx))
    num_nn_xz = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=xz_idx,
    )

    # compute on the subspace of YZ
    yz_idx = np.concatenate((y_idx, z_idx))
    num_nn_yz = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=yz_idx,
    )

    # compute on the subspace of XZ
    num_nn_z = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=z_idx,
    )

    # compute the final CMI value
    hxyz = scipy.special.digamma(knn_here)
    hxz = scipy.special.digamma(num_nn_xz)
    hyz = scipy.special.digamma(num_nn_yz)
    hz = scipy.special.digamma(num_nn_z)
    val = hxyz - (hxz + hyz - hz).mean()
    return val


def _compute_radius_nbrs(
    data: ArrayLike,
    radius_per_sample: ArrayLike,
    nn_estimator: BaseEstimator,
    col_idx: Optional[ArrayLike] = None,
):
    n_samples = radius_per_sample.shape[0]

    # compute distances in the subspace defined by data
    nn_estimator.fit(data[:, col_idx], force_fit=True)

    num_nn_data = np.zeros((n_samples,))
    for idx in range(n_samples):
        nn = nn_estimator.radius_neighbors(radius=radius_per_sample[idx], return_distance=False)
        num_nn_data[idx] = len(nn)
    return num_nn_data


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

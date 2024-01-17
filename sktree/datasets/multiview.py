# Original source: https://github.com/mvlearn/mvlearn
# License: MIT

import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state


def make_gaussian_mixture(
    centers,
    covariances,
    n_samples=100,
    transform="linear",
    noise=None,
    noise_dims=None,
    class_probs=None,
    random_state=None,
    shuffle=False,
    return_latents=False,
    add_latent_noise=False,
):
    r"""Two-view Gaussian mixture model dataset generator.

    This creates a two-view dataset from a Gaussian mixture model and
    a (possibly nonlinear) transformation.

    Parameters
    ----------
    centers : 1D array-like or list of 1D array-likes
        The mean(s) of the Gaussian(s) from which the latent
        points are sampled. If is a list of 1D array-likes, each is the
        mean of a distinct Gaussian, sampled from with
        probability given by `class_probs`. Otherwise is the mean of a
        single Gaussian from which all are sampled.
    covariances : 2D array-like or list of 2D array-likes
        The covariance matrix(s) of the Gaussian(s), matched
        to the specified centers.
    n_samples : int
        The number of points in each view, divided across Gaussians per
        `class_probs`.
    transform : 'linear' | 'sin' | 'poly' | callable, (default 'linear')
        Transformation to perform on the latent variable. If a function,
        applies it to the latent. Otherwise uses an implemented function.
    noise : float or None (default=None)
        Variance of mean zero Gaussian noise added to the first view.
    noise_dims : int or None (default=None)
        Number of additional dimensions of standard normal noise to add.
    class_probs : array-like, default=None
        A list of probabilities specifying the probability of a latent
        point being sampled from each of the Gaussians. Must sum to 1. If
        None, then is taken to be uniform over the Gaussians.
    random_state : int, default=None
        If set, can be used to reproduce the data generated.
    shuffle : bool, default=False
        If ``True``, data is shuffled so the labels are not ordered.
    return_latents : bool (default False)
        If true, returns the non-noisy latent variables.
    add_latent_noise : bool (default False)
        If true, adds noise to the latent variables before applying the
        transformation.

    Returns
    -------
    Xs : list of np.ndarray, of shape (n_samples, n_features)
        The latent data and its noisy transformation.

    y : np.ndarray, shape (n_samples,)
        The integer labels for each sample's Gaussian membership.

    latents : np.ndarray, shape (n_samples, n_features)
        The non-noisy latent variables. Only returned if
        ``return_latents=True``.

    Notes
    -----
    For each class :math:`i` with prior probability :math:`p_i`,
    center and covariance matrix :math:`\mu_i` and :math:`\Sigma_i`, and
    :math:`n` total samples, the latent data is sampled such that:

    .. math::
        (X_1, y_1), \dots, (X_{np_i}, Y_{np_i}) \overset{i.i.d.}{\sim}
            \mathcal{N}(\mu_i, \Sigma_i)

    Two views of data are returned, the first being the latent samples and
    the second being a specified transformation of the latent samples.
    Additional noise may be added to the first view or added as noise
    dimensions to both views.

    Examples
    --------
    >>> from sktree.datasets.multiview import make_gaussian_mixture
    >>> import numpy as np
    >>> n_samples = 10
    >>> centers = [[0,1], [0,-1]]
    >>> covariances = [np.eye(2), np.eye(2)]
    >>> Xs, y = make_gaussian_mixture(n_samples, centers, covariances,
    ...                               shuffle=True, shuffle_random_state=42)
    >>> print(y)
    [1. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
    """
    centers = np.asarray(centers)
    covariances = np.asarray(covariances)

    if centers.ndim == 1:
        centers = centers[np.newaxis, :]
    if covariances.ndim == 2:
        covariances = covariances[np.newaxis, :]
    if not centers.ndim == 2:
        msg = "centers is of the incorrect shape"
        raise ValueError(msg)
    if not covariances.ndim == 3:
        msg = "covariance matrix is of the incorrect shape"
        raise ValueError(msg)
    if centers.shape[0] != covariances.shape[0]:
        msg = "The first dimensions of 2D centers and 3D covariances" + " must be equal"
        raise ValueError(msg)
    if class_probs is None:
        class_probs = np.ones(centers.shape[0])
        class_probs /= centers.shape[0]
    elif sum(class_probs) != 1.0:
        msg = "elements of `class_probs` must sum to 1"
        raise ValueError(msg)
    if len(centers) != len(class_probs) or len(covariances) != len(class_probs):
        msg = "centers, covariances, and class_probs must be of equal length"
        raise ValueError(msg)

    rng = np.random.default_rng(seed=random_state)
    latent = np.concatenate(
        [
            rng.multivariate_normal(
                centers[i],
                covariances[i],
                size=int(class_probs[i] * n_samples),
            )
            for i in range(len(class_probs))
        ]
    )
    y = np.concatenate(
        [i * np.ones(int(class_probs[i] * n_samples)) for i in range(len(class_probs))]
    )

    if add_latent_noise:
        latent += rng.standard_normal(size=latent.shape) * 0.1

    # shuffle latent samples and labels
    if shuffle:
        indices = np.arange(latent.shape[0]).squeeze()
        rng.shuffle(indices)
        latent = latent[indices, :]
        y = y[indices]

    if callable(transform):
        X = np.asarray([transform(x) for x in latent])
    elif not isinstance(transform, str):
        raise TypeError(
            "'transform' must be of type string or a callable function," + f"not {type(transform)}"
        )
    elif transform == "linear":
        X = _linear2view(latent, rng)
    elif transform == "poly":
        X = _poly2view(latent)
    elif transform == "sin":
        X = _sin2view(latent)
    else:
        raise ValueError(
            "Transform type must be one of {'linear', 'poly', 'sin'}"
            + f" or a callable function. Not {transform}"
        )

    if noise is not None:
        Xs = [latent + np.sqrt(noise) * rng.standard_normal(size=latent.shape), X]
    else:
        Xs = [latent, X]

    # if random_state is not None, make sure both views are independent
    # but reproducible
    if noise_dims is not None:
        Xs = [_add_noise(X, noise_dims, rng) for X in Xs]

    if return_latents:
        return Xs, y, latent
    else:
        return Xs, y


def _add_noise(X, n_noise, rng):
    """Appends dimensions of standard normal noise to X"""
    noise_vars = rng.standard_normal(size=(X.shape[0], n_noise))
    return np.hstack((X, noise_vars))


def _linear2view(X, rng):
    """Rotates the data, a linear transformation"""
    if X.shape[1] == 1:
        X = -X
    else:
        X = X @ ortho_group.rvs(X.shape[1], random_state=rng)
    return X


def _poly2view(X):
    """Applies a degree 2 polynomial transform to the data"""
    X = np.asarray([np.power(x, 2) for x in X])
    return X


def _sin2view(X):
    """Applies a sinusoidal transformation to the data"""
    X = np.asarray([np.sin(x) for x in X])
    return X


def _rand_orthog(n, K, random_state=None):
    """
    Samples a random orthonormal matrix.

    Parameters
    ----------
    n : int, positive
        Number of rows in the matrix

    K : int, positive
        Number of columns in the matrix

    random_state : None | int | instance of RandomState, optional
        Seed to set randomization for reproducible results

    Returns
    -------
    A: array-like, (n, K)
        A random, column orthonormal matrix.

    Notes
    -----
    See Section A.1.1 of :footcite:`perry2009crossvalidation`.

    References
    ----------
    .. footbibliography::
    """
    rng = check_random_state(random_state)

    Z = rng.normal(size=(n, K))
    Q, R = np.linalg.qr(Z)

    s = np.ones(K)
    neg_mask = rng.uniform(size=K) > 0.5
    s[neg_mask] = -1

    return Q * s


def make_joint_factor_model(
    n_views,
    n_features,
    n_samples=100,
    joint_rank=1,
    noise_std=1,
    m=1.5,
    random_state=None,
    return_decomp=False,
):
    """Joint factor model data generator.

    Samples from a low rank, joint factor model where there is one set of
    shared scores.

    Parameters
    ----------
    n_views : int
        Number of views to sample. This corresponds to ``B`` in the notes.

    n_features : int, or list of int
        Number of features in each view. A list specifies a different number
        of features for each view.

    n_samples : int
        Number of samples in each view

    joint_rank : int (default 1)
        Rank of the common signal across views.

    noise_std : float (default 1)
        Scale of noise distribution.

    m : float (default 1.5)
        Signal strength.

    random_state : int or RandomState instance, optional (default=None)
        Controls random orthonormal matrix sampling and random noise
        generation. Set for reproducible results.

    return_decomp : bool, default=False
        If ``True``, returns the ``view_loadings`` as well.

    Returns
    -------
    Xs : list of array-likes
        List of samples data matrices with the following attributes.

        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i).

    U: (n_samples, joint_rank)
        The true orthonormal joint scores matrix. Returned if
        ``return_decomp`` is True.

    view_loadings: list of numpy.ndarray
        The true view loadings matrices. Returned if
        ``return_decomp`` is True.

    Notes
    -----
    The data is generated as follows, where:

    - :math:`b` are the different views
    - :math:`U` is is a (n_samples, joint_rank) matrix of rotation matrices.
    - ``svals`` are the singular values sampled.
    - :math:`W_b` are (n_features_b, joint_rank) view loadings matrices, which are
        orthonormal matrices to linearly transform the data, while preserving inner
        products (i.e. a unitary transformation).

    For b = 1, .., B
        X_b = U @ diag(svals) @ W_b^T + noise_std * E_b

    where U and each W_b are orthonormal matrices. The singular values are
    linearly increasing following :footcite:`choi2017selectingpca` section 2.2.3.

    References
    ----------
    .. footbibliography::
    """
    rng = check_random_state(random_state)
    if isinstance(n_features, int):
        n_features = [n_features] * n_views

    # generate W_b orthonormal matrices
    view_loadings = [_rand_orthog(d, joint_rank, random_state=rng) for d in n_features]

    # sample monotonically increasing singular values
    # the signal increases linearly and ``m`` determines the strength of the signal
    svals = np.arange(1, 1 + joint_rank).astype(float)
    svals *= m * noise_std * (n_samples * max(n_features)) ** (1 / 4)

    # rotation operators that are generated via standard random normal
    U = rng.standard_normal(size=(n_samples, joint_rank))
    U = np.linalg.qr(U)[0]

    # random noise for each view
    Es = [noise_std * rng.standard_normal(size=(n_samples, d)) for d in n_features]
    Xs = [(U * svals) @ view_loadings[b].T + Es[b] for b in range(n_views)]

    if return_decomp:
        return Xs, U, view_loadings
    else:
        return Xs

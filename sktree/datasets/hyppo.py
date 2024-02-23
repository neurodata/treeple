import numpy as np
from scipy.integrate import nquad
from scipy.stats import entropy, multivariate_normal


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


# Dictionary of simulations from Marron and Wand 1992
# keys: names of each simulation corresponding to the class MarronWandSims
# values: probabilities associated with the mixture of Gaussians
MARRON_WAND_SIMS = {
    "gaussian": [1],
    "skewed_unimodal": [1 / 5, 1 / 5, 3 / 5],
    "strongly_skewed": [1 / 8] * 8,
    "kurtotic_unimodal": [2 / 3, 1 / 3],
    "outlier": [1 / 10, 9 / 10],
    "bimodal": [1 / 2] * 2,
    "separated_bimodal": [1 / 2] * 2,
    "skewed_bimodal": [3 / 4, 1 / 4],
    "trimodal": [9 / 20, 9 / 20, 1 / 10],
    "claw": [1 / 2, *[1 / 10] * 5],
    "double_claw": [49 / 100, 49 / 100, *[1 / 350] * 7],
    "asymmetric_claw": [1 / 2, *[2 ** (1 - i) / 31 for i in range(-2, 3)]],
    "asymmetric_double_claw": [*[46 / 100] * 2, *[1 / 300] * 3, *[7 / 300] * 3],
    "smooth_comb": [2 ** (5 - i) / 63 for i in range(6)],
    "discrete_comb": [*[2 / 7] * 3, *[1 / 21] * 3],
}


def make_marron_wand_classification(
    n_samples,
    n_dim=4096,
    n_informative=256,
    simulation: str = "gaussian",
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    seed=None,
):
    """Generate Marron-Wand binary classification dataset.

    The simulation is similar to that of :func:`sktree.datasets.make_trunk_classification`
    where the first class is generated from a multivariate-Gaussians with mean vector of
    0's. The second class is generated from a mixture of Gaussians with mean vectors
    specified by the Marron-Wand simulations, but as the dimensionality increases, the second
    class distribution approaches the first class distribution by a factor of :math:`1 / sqrt(d)`.

    Full details for the Marron-Wand simulations can be found in :footcite:`marron1992exact`.

    Instead of the identity covariance matrix, one can implement a banded covariance matrix
    that follows :footcite:`Bickel_2008`.

    Parameters
    ----------
    n_samples : int
        Number of sample to generate.
    n_dim : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 4096.
    n_informative : int, optional
        The informative dimensions. All others for ``n_dim - n_informative``
        are Gaussian noise. Default is 256.
    simulation : str, optional
        Which simulation to run. Must be one of the
        following Marron-Wand simulations: 'gaussian', 'skewed_unimodal', 'strongly_skewed',
        'kurtotic_unimodal', 'outlier', 'bimodal', 'separated_bimodal', 'skewed_bimodal',
        'trimodal', 'claw', 'double_claw', 'asymmetric_claw', 'asymmetric_double_claw',
        'smooth_comb', 'discrete_comb'.
        When calling the Marron-Wand simulations, only the covariance parameters are considered
        (`rho` and `band_type`). Means are taken from :footcite:`marron1992exact`.
        By default 'gaussian'.
    rho : float, optional
        The covariance value of the bands. By default 0 indicating, an identity matrix is used.
    band_type : str
        The band type to use. For details, see Example 1 and 2 in :footcite:`Bickel_2008`.
        Either 'ma', or 'ar'.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal distributions.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n_samples,), dtype=np.intp
        Labels of the dataset.
    G : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        The mixture of Gaussians for the Marron-Wand simulations.
        Returned if ``return_params`` is True.
    w : np.ndarray of shape (n_dim,), dtype=np.float64
        The weight vector for the Marron-Wand simulations.
        Returned if ``return_params`` is True.

    Notes
    -----
    **Marron-Wand Simulations**: The Marron-Wand simulations generate two classes of data with the
    setup specified in the paper.

    Covariance: The covariance matrix among different dimensions is controlled by the ``rho`` parameter
    and the ``band_type`` parameter. The ``band_type`` parameter controls the type of band to use, while
    the ``rho`` parameter controls the specific scaling factor for the covariance matrix while going
    from one dimension to the next.

    For each dimension in the first distribution, there is a mean of :math:`1 / d`, where
    ``d`` is the dimensionality. The covariance is the identity matrix.

    The second distribution has a mean vector that is the negative of the first.
    As ``d`` increases, the two distributions become closer and closer.
    Full details for the trunk simulation can be found in :footcite:`trunk1982`.

    References
    ----------
    .. footbibliography::
    """
    if n_dim < n_informative:
        raise ValueError(
            f"Number of informative dimensions {n_informative} must be less than number "
            f"of dimensions, {n_dim}"
        )
    if simulation not in MARRON_WAND_SIMS.keys():
        raise ValueError(
            f"Simulation must be: trunk, trunk_overlap, trunk_mix, {MARRON_WAND_SIMS.keys()}"
        )

    rng = np.random.default_rng(seed=seed)

    if rho != 0:
        if band_type == "ma":
            cov = _moving_avg_cov(n_informative, rho)
        elif band_type == "ar":
            cov = _autoregressive_cov(n_informative, rho)
        else:
            raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
    else:
        cov = np.identity(n_informative)

    # speed up computations for large multivariate normal matrix with SVD approximation
    if n_informative > 1000:
        mvg_sampling_method = "cholesky"
    else:
        mvg_sampling_method = "svd"

    mixture_idx = rng.choice(
        len(MARRON_WAND_SIMS[simulation]),  # type: ignore
        size=n_samples // 2,
        replace=True,
        p=MARRON_WAND_SIMS[simulation],
    )
    # the parameters used for each Gaussian in the mixture for each Marron Wand simulation
    norm_params = MarronWandSims(n_dim=n_informative, cov=cov)(simulation)
    G = np.fromiter(
        (
            rng.multivariate_normal(*(norm_params[i]), size=1, method=mvg_sampling_method)
            for i in mixture_idx
        ),
        dtype=np.dtype((float, n_informative)),
    )

    # as the dimensionality of the simulations increasing, we are adding more and
    # more noise to the data using the w parameter
    w_vec = np.array([1.0 / np.sqrt(i) for i in range(1, n_informative + 1)])
    X = np.vstack(
        (
            rng.multivariate_normal(
                np.zeros(n_informative), cov, n_samples // 2, method=mvg_sampling_method
            ),
            (1 - w_vec)
            * rng.multivariate_normal(
                np.zeros(n_informative), cov, n_samples // 2, method=mvg_sampling_method
            )
            + w_vec * G.reshape(n_samples // 2, n_informative),
        )
    )
    if n_dim > n_informative:
        X = np.hstack((X, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim - n_informative))))

    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params:
        returns = [X, y]
        returns += [*list(zip(*norm_params)), G, w_vec]
        return returns
    return X, y


def make_trunk_mixture_classification(
    n_samples,
    n_dim=4096,
    n_informative=256,
    mu_0: int = 0,
    mu_1: int = 1,
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    mix: float = 0.5,
    seed=None,
):
    """Generate trunk mixture binary classification dataset.

    The first class is generated from a multivariate-Gaussians with mean vector of
    0's. The second class is generated from a mixture of Gaussians with mean vectors
    specified by ``mu_0`` and ``mu_1``. The mixture is specified by the ``mix`` parameter,
    which is the probability of the first Gaussian in the mixture.

    For each dimension in the first distribution, there is a mean of :math:`1 / d`, where
    ``d`` is the dimensionality. The covariance is the identity matrix.
    The second distribution has a mean vector that is the negative of the first.
    As ``d`` increases, the two distributions become closer and closer.

    Full details for the trunk simulation can be found in :footcite:`trunk1982`.

    Instead of the identity covariance matrix, one can implement a banded covariance matrix
    that follows :footcite:`Bickel_2008`.

    Parameters
    ----------
    n_samples : int
        Number of sample to generate.
    n_dim : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 4096.
    n_informative : int, optional
        The informative dimensions. All others for ``n_dim - n_informative``
        are Gaussian noise. Default is 256.
    mu_0 : int, optional
        The mean of the first distribution. By default -1. The mean of the distribution will decrease
        by a factor of ``sqrt(i)`` for each dimension ``i``.
    mu_1 : int, optional
        The mean of the second distribution. By default 1. The mean of the distribution will decrease
        by a factor of ``sqrt(i)`` for each dimension ``i``.
    rho : float, optional
        The covariance value of the bands. By default 0 indicating, an identity matrix is used.
    band_type : str
        The band type to use. For details, see Example 1 and 2 in :footcite:`Bickel_2008`.
        Either 'ma', or 'ar'.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal distributions.
    mix : int, optional
        The probabilities associated with the mixture of Gaussians in the ``trunk-mix`` simulation.
        By default 0.5.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n_samples,), dtype=np.intp
        Labels of the dataset.
    means : list of ArrayLike of shape (n_dim,), dtype=np.float64
        The mean vector for each class starting with class 0.
        Returned if ``return_params`` is True.
    covs : list of ArrayLike of shape (n_dim, n_dim), dtype=np.float64
        The covariance for each class. Returned if ``return_params`` is True.
    X_mixture : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        The mixture of Gaussians.
        Returned if ``return_params`` is True.

    Notes
    -----
    **Trunk**: The trunk simulation decreases the signal-to-noise ratio as the dimensionality
    increases. This is implemented by decreasing the mean of the distribution by a factor of
    ``sqrt(i)`` for each dimension ``i``. Thus for instance if the means of distribution one
    and two are 1 and -1 respectively, the means for the first dimension will be 1 and -1,
    for the second dimension will be 1/sqrt(2) and -1/sqrt(2), and so on.

    **Trunk Mix**: The trunk mix simulation generates two classes of data with the same covariance
    matrix. The first class (label 0) is generated from a multivariate-Gaussians with mean vector of
    zeros and the second class is generated from a mixture of Gaussians with mean vectors
    specified by ``mu_0`` and ``mu_1``. The mixture is specified by the ``mix`` parameter, which
    is the probability of the first Gaussian in the mixture.

    Covariance: The covariance matrix among different dimensions is controlled by the ``rho`` parameter
    and the ``band_type`` parameter. The ``band_type`` parameter controls the type of band to use, while
    the ``rho`` parameter controls the specific scaling factor for the covariance matrix while going
    from one dimension to the next.

    References
    ----------
    .. footbibliography::
    """
    if n_dim < n_informative:
        raise ValueError(
            f"Number of informative dimensions {n_informative} must be less than number "
            f"of dimensions, {n_dim}"
        )
    if mix < 0 or mix > 1:  # type: ignore
        raise ValueError("Mix must be between 0 and 1.")

    rng = np.random.default_rng(seed=seed)

    mu_1_vec = np.array([mu_1 / np.sqrt(i) for i in range(1, n_informative + 1)])
    mu_0_vec = np.array([mu_0 / np.sqrt(i) for i in range(1, n_informative + 1)])

    if rho != 0:
        if band_type == "ma":
            cov = _moving_avg_cov(n_informative, rho)
        elif band_type == "ar":
            cov = _autoregressive_cov(n_informative, rho)
        else:
            raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
    else:
        cov = np.identity(n_informative)

    # speed up computations for large multivariate normal matrix with SVD approximation
    if n_informative > 1000:
        method = "cholesky"
    else:
        method = "svd"

    mixture_idx = rng.choice(2, n_samples // 2, replace=True, shuffle=True, p=[mix, 1 - mix])  # type: ignore

    # When variance is 1, trunk-mix does not look bimodal at low dimensions.
    # It is set it to (2/3)**2 since that is consistent with Marron and Wand bimodal
    norm_params = [[mu_0_vec, cov * (2 / 3) ** 2], [mu_1_vec, cov * (2 / 3) ** 2]]
    X_mixture = np.fromiter(
        (rng.multivariate_normal(*(norm_params[i]), size=1, method=method) for i in mixture_idx),
        dtype=np.dtype((float, n_informative)),
    )

    X = np.vstack(
        (
            rng.multivariate_normal(
                np.zeros(n_informative), cov * (2 / 3) ** 2, n_samples // 2, method=method
            ),
            X_mixture.reshape(n_samples // 2, n_informative),
        )
    )

    if n_dim > n_informative:
        X = np.hstack((X, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim - n_informative))))

    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params:
        returns = [X, y]
        returns += [*list(zip(*norm_params)), X_mixture]
        return returns
    return X, y


def make_trunk_classification(
    n_samples,
    n_dim=4096,
    n_informative=256,
    mu_0: int = 0,
    mu_1: int = 1,
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    seed=None,
):
    """Generate trunk binary classification dataset.

    For each dimension in the first distribution, there is a mean of :math:`1 / d`, where
    ``d`` is the dimensionality. The covariance is the identity matrix.
    The second distribution has a mean vector that is the negative of the first.
    As ``d`` increases, the two distributions become closer and closer.

    Full details for the trunk simulation can be found in :footcite:`trunk1982`.

    Instead of the identity covariance matrix, one can implement a banded covariance matrix
    that follows :footcite:`Bickel_2008`.

    Parameters
    ----------
    n_samples : int
        Number of sample to generate.
    n_dim : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 4096.
    n_informative : int, optional
        The informative dimensions. All others for ``n_dim - n_informative``
        are Gaussian noise. Default is 256.
    mu_0 : int, optional
        The mean of the first distribution. By default -1. The mean of the distribution will decrease
        by a factor of ``sqrt(i)`` for each dimension ``i``.
    mu_1 : int, optional
        The mean of the second distribution. By default 1. The mean of the distribution will decrease
        by a factor of ``sqrt(i)`` for each dimension ``i``.
    rho : float, optional
        The covariance value of the bands. By default 0 indicating, an identity matrix is used.
    band_type : str
        The band type to use. For details, see Example 1 and 2 in :footcite:`Bickel_2008`.
        Either 'ma', or 'ar'.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal distributions.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n_samples,), dtype=np.intp
        Labels of the dataset.
    means : list of ArrayLike of shape (n_dim,), dtype=np.float64
        The mean vector for each class starting with class 0.
        Returned if ``return_params`` is True.
    covs : list of ArrayLike of shape (n_dim, n_dim), dtype=np.float64
        The covariance for each class. Returned if ``return_params`` is True.

    Notes
    -----
    **Trunk**: The trunk simulation decreases the signal-to-noise ratio as the dimensionality
    increases. This is implemented by decreasing the mean of the distribution by a factor of
    ``sqrt(i)`` for each dimension ``i``. Thus for instance if the means of distribution one
    and two are 1 and -1 respectively, the means for the first dimension will be 1 and -1,
    for the second dimension will be 1/sqrt(2) and -1/sqrt(2), and so on.

    **Trunk Overlap**: The trunk overlap simulation generates two classes of data with the same
    covariance matrix and mean vector of zeros.

    Covariance: The covariance matrix among different dimensions is controlled by the ``rho`` parameter
    and the ``band_type`` parameter. The ``band_type`` parameter controls the type of band to use, while
    the ``rho`` parameter controls the specific scaling factor for the covariance matrix while going
    from one dimension to the next.

    References
    ----------
    .. footbibliography::
    """
    if n_dim < n_informative:
        raise ValueError(
            f"Number of informative dimensions {n_informative} must be less than number "
            f"of dimensions, {n_dim}"
        )
    rng = np.random.default_rng(seed=seed)

    mu_1_vec = np.array([mu_1 / np.sqrt(i) for i in range(1, n_informative + 1)])
    mu_0_vec = np.array([mu_0 / np.sqrt(i) for i in range(1, n_informative + 1)])

    if rho != 0:
        if band_type == "ma":
            cov = _moving_avg_cov(n_informative, rho)
        elif band_type == "ar":
            cov = _autoregressive_cov(n_informative, rho)
        else:
            raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
    else:
        cov = np.identity(n_informative)

    # speed up computations for large multivariate normal matrix with SVD approximation
    if n_informative > 1000:
        method = "cholesky"
    else:
        method = "svd"

    X = np.vstack(
        (
            rng.multivariate_normal(mu_1_vec, cov, n_samples // 2, method=method),
            rng.multivariate_normal(mu_0_vec, cov, n_samples // 2, method=method),
        )
    )

    if n_dim > n_informative:
        X = np.hstack((X, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim - n_informative))))

    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params:
        returns = [X, y]
        returns += [[mu_0_vec, mu_1_vec], [cov, cov]]
        return returns
    return X, y


class MarronWandSims:
    def __init__(self, n_dim=1, cov=1):
        self.n_dim = n_dim
        self.cov = cov

    def __call__(self, simulation):
        sims = self._my_method_generator()
        if simulation in sims.keys():
            return sims[simulation]()
        else:
            raise ValueError(f"simulation is not one of these: {sims.keys()}")

    def _my_method_generator(self):
        return {
            method: getattr(self, method) for method in dir(self) if not method.startswith("__")
        }

    def gaussian(self):
        return [[np.zeros(self.n_dim), self.cov]]

    def skewed_unimodal(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            [np.full(self.n_dim, 1 / 2), self.cov * (2 / 3) ** 2],
            [np.full(self.n_dim, 13 / 12), self.cov * (5 / 9) ** 2],
        ]

    def strongly_skewed(self):
        return [
            [np.full(self.n_dim, 3 * ((2 / 3) ** l_mix - 1)), self.cov * (2 / 3) ** (2 * l_mix)]
            for l_mix in range(8)
        ]

    def kurtotic_unimodal(self):
        return [[np.zeros(self.n_dim), self.cov], [np.zeros(self.n_dim), self.cov * (1 / 10) ** 2]]

    def outlier(self):
        return [[np.zeros(self.n_dim), self.cov], [np.zeros(self.n_dim), self.cov * (1 / 10) ** 2]]

    def bimodal(self):
        return [
            [-np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
            [np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
        ]

    def separated_bimodal(self):
        return [
            [-np.full(self.n_dim, 3 / 2), self.cov * (1 / 2) ** 2],
            [np.full(self.n_dim, 3 / 2), self.cov * (1 / 2) ** 2],
        ]

    def skewed_bimodal(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            [np.full(self.n_dim, 3 / 2), self.cov * (1 / 3) ** 2],
        ]

    def trimodal(self):
        return [
            [np.full(self.n_dim, -6 / 5), self.cov * (3 / 5) ** 2],
            [np.full(self.n_dim, 6 / 5), self.cov * (3 / 5) ** 2],
            [np.zeros(self.n_dim), self.cov * (1 / 4) ** 2],
        ]

    def claw(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            *[
                [np.full(self.n_dim, (l_mix / 2) - 1), self.cov * (1 / 10) ** 2]
                for l_mix in range(5)
            ],
        ]

    def double_claw(self):
        return [
            [-np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
            [np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
            *[
                [np.full(self.n_dim, (l_mix - 3) / 2), self.cov * (1 / 100) ** 2]
                for l_mix in range(7)
            ],
        ]

    def asymmetric_claw(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            *[
                [np.full(self.n_dim, l_mix + 1 / 2), self.cov * (1 / ((2**l_mix) * 10)) ** 2]
                for l_mix in range(-2, 3)
            ],
        ]

    def asymmetric_double_claw(self):
        return [
            *[[np.full(self.n_dim, 2 * l_mix - 1), self.cov * (2 / 3) ** 2] for l_mix in range(2)],
            *[
                [-np.full(self.n_dim, l_mix / 2), self.cov * (1 / 100) ** 2]
                for l_mix in range(1, 4)
            ],
            *[[np.full(self.n_dim, l_mix / 2), self.cov * (7 / 100) ** 2] for l_mix in range(1, 4)],
        ]

    def smooth_comb(self):
        return [
            [
                np.full(self.n_dim, (65 - 96 * ((1 / 2) ** l_mix)) / 21),
                self.cov * (32 / 63) ** 2 / (2 ** (2 * l_mix)),
            ]
            for l_mix in range(6)
        ]

    def discrete_comb(self):
        return [
            *[
                [np.full(self.n_dim, (12 * l_mix - 15) / 7), self.cov * (2 / 7) ** 2]
                for l_mix in range(3)
            ],
            *[
                [np.full(self.n_dim, (2 * l_mix) / 7), self.cov * (1 / 21) ** 2]
                for l_mix in range(8, 11)
            ],
        ]


def _moving_avg_cov(n_dim, rho):
    # Create a meshgrid of indices
    i, j = np.meshgrid(np.arange(1, n_dim + 1), np.arange(1, n_dim + 1), indexing="ij")

    # Calculate the covariance matrix using the corrected formula
    cov_matrix = rho ** np.abs(i - j)

    # Apply the banding condition
    cov_matrix[abs(i - j) > 1] = 0
    return cov_matrix


def _autoregressive_cov(n_dim, rho):
    # Create a meshgrid of indices
    i, j = np.meshgrid(np.arange(1, n_dim + 1), np.arange(1, n_dim + 1), indexing="ij")

    # Calculate the covariance matrix using the corrected formula
    cov_matrix = rho ** np.abs(i - j)

    return cov_matrix


def approximate_clf_mutual_information(
    means, covs, class_probs=[0.5, 0.5], base=np.exp(1), seed=None
):
    """Approximate MI for multivariate Gaussian for a classification setting.

    Parameters
    ----------
    means : list of ArrayLike of shape (n_dim,)
        A list of means to sample from for each class.
    covs : list of ArrayLike of shape (n_dim, n_dim)
        A list of covariances to sample from for each class.
    class_probs : list, optional
        List of class probabilities, by default [0.5, 0.5] for
        balanced binary classification.
    base : float, optional
        The bit base to use, by default np.exp(1) for natural logarithm.
    seed : int, optional
        Random seed for the multivariate normal, by default None.

    Returns
    -------
    I_XY : float
        Estimated mutual information.
    H_X : float
        Estimated entropy of X, the mixture of multivariate Gaussians.
    H_XY : float
        The conditional entropy of X given Y.
    int_err : float
        The integration error for ``H_X``.
    """
    # this implicitly assumes that the signal of interest is between -10 and 10
    scale = 10
    n_dims = [cov.shape[1] for cov in covs]
    lims = [[-scale, scale]] * max(n_dims)

    # Compute entropy and X and Y.
    def func(*args):
        x = np.array(args)
        p = 0
        for k in range(len(means)):
            p += class_probs[k] * multivariate_normal.pdf(x, means[k], covs[k])
        return -p * np.log(p) / np.log(base)

    # numerically integrate H(X)
    # opts = dict(limit=1000)
    H_X, int_err = nquad(func, lims)

    # Compute MI.
    H_XY = 0
    for k in range(len(means)):
        H_XY += (
            class_probs[k]
            * (n_dims[k] * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + n_dims[k])
            / (2 * np.log(base))
        )
    I_XY = H_X - H_XY
    return I_XY, H_X, H_XY, int_err


def approximate_clf_mutual_information_with_monte_carlo(
    means, covs, n_samples=100_000, class_probs=[0.5, 0.5], base=np.exp(1), seed=None
):
    """Approximate MI for multivariate Gaussian for a classification setting.

    Parameters
    ----------
    means : list of ArrayLike of shape (n_dim,)
        A list of means to sample from for each class.
    covs : list of ArrayLike of shape (n_dim, n_dim)
        A list of covariances to sample from for each class.
    n_samples : int
        The total number of simulation samples
    class_probs : list, optional
        List of class probabilities, by default [0.5, 0.5] for
        balanced binary classification.
    base : float, optional
        The bit base to use, by default np.exp(1) for natural logarithm.
    seed : int, optional
        Random seed for the multivariate normal, by default None.

    Returns
    -------
    I_XY : float
        Estimated mutual information.
    H_Y : float
        Estimated entropy of Y, the class labels.
    H_Y_on_X : float
        The conditional entropy of Y given X.
    """
    rng = np.random.default_rng(seed=seed)
    P_Y = class_probs

    # Generate samples
    pdf_class = []
    X = []
    for i in range(len(means)):
        pdf_class.append(multivariate_normal(means[i], covs[i], allow_singular=True))
        X.append(
            rng.multivariate_normal(means[i], covs[i], size=int(n_samples * P_Y[i])).reshape(-1, 1)
        )

    X = np.vstack(X)

    # Calculate P(X) by law of total probability
    P_X_l = []
    P_X_on_Y = []
    for i in range(len(means)):
        P_X_on_Y.append(pdf_class[i].pdf(X))
        P_X_l.append(P_X_on_Y[-1] * P_Y[i])
    P_X = sum(P_X_l)

    # Calculate P(Y|X) by Bayes' theorem
    P_Y_on_X = []
    for i in range(len(means)):
        P_Y_on_X.append((P_X_on_Y[i] * P_Y[i] / P_X).reshape(-1, 1))

    P_Y_on_X = np.hstack(P_Y_on_X)
    P_Y_on_X = P_Y_on_X[~np.isnan(P_Y_on_X)].reshape(-1, 2)

    # Calculate the entropy of Y by class counts
    H_Y = entropy(P_Y, base=base)

    # Calculate the conditional entropy of Y on X
    H_Y_on_X = np.mean(entropy(P_Y_on_X, base=base, axis=1))

    MI = H_Y - H_Y_on_X
    return MI, H_Y, H_Y_on_X


def _compute_mi_bounds(means, covs, class_probs):
    # compute bounds using https://arxiv.org/pdf/2101.11670.pdf
    prior_y = np.array(class_probs)
    H_Y = entropy(prior_y, base=np.exp(1))

    # number of mixtures
    N = len(class_probs)

    cov = covs[0]

    H_YX_lb = 0.0
    H_YX_ub = 0.0
    for idx in range(N):
        num = 0.0
        denom = 0.0
        mean_i = means[idx]

        for jdx in range(N):
            mean_j = means[jdx]
            c_alpha, kl_div = _compute_c_alpha_and_kl(mean_i, mean_j, cov, alpha=0.5)
            num += class_probs[jdx] * np.exp(-c_alpha)

        for kdx in range(N):
            mean_k = means[kdx]
            c_alpha, kl_div = _compute_c_alpha_and_kl(mean_i, mean_k, cov, alpha=0.5)
            denom += class_probs[jdx] * np.exp(-kl_div)
        H_YX_lb += class_probs[idx] * np.log(num / denom)
        H_YX_ub += class_probs[idx] * np.log(denom / num)
    I_lb = H_Y - H_YX_lb
    I_ub = H_Y - H_YX_ub
    return I_lb, I_ub


def _compute_c_alpha_and_kl(mean_i, mean_j, cov, alpha=0.5):
    mean_ij = mean_i - mean_j
    lambda_ij = mean_ij.T @ np.linalg.inv(cov) @ mean_ij
    return alpha * (1.0 - alpha) * lambda_ij / 2.0, lambda_ij / 2

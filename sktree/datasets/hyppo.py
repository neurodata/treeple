import numpy as np
from scipy.stats import multivariate_normal, entropy, norm
from scipy.integrate import nquad


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


def make_trunk_classification(
    n_samples,
    n_dim=10,
    m_factor: int = -1,
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    simulation: str = "trunk",
    seed=None,
):
    """Generate trunk dataset.

    For each dimension in the first distribution, there is a mean of :math:`1 / d`, where
    ``d`` is the dimensionality. The covariance is the identity matrix.
    The second distribution has a mean vector that is the negative of the first.
    As ``d`` increases, the two distributions become closer and closer.

    See full details in :footcite:`trunk1982`.

    Instead of the identity covariance matrix, one can implement a banded covariance matrix
    that follows :footcite:`Bickel_2008`.

    Parameters
    ----------
    n_samples : int
        Number of sample to generate.
    n_dim : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 10.
    m_factor : int, optional
        The multiplicative factor to apply to the mean-vector of the first
        distribution to obtain the mean-vector of the second distribution.
        By default -1.
    rho : float, optional
        The covariance value of the bands. By default 0 indicating, an identity matrix is used.
    band_type : str
        The band type to use. For details, see Example 1 and 2 in :footcite:`Bickel_2008`.
        Either 'ma', or 'ar'.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal distributions.
    simulation : str, optional
        Which simulation to run. Must be: 'trunk', 'trunk_overlap', 'trunk_mix', or one of the
        following Marron-Wand simulations: 'gaussian', 'skewed_unimodal', 'strongly_skewed',
        'kurtotic_unimodal', 'outlier', 'bimodal', 'separated_bimodal', 'skewed_bimodal',
        'trimodal', 'claw', 'double_claw', 'asymmetric_claw', 'asymmetric_double_claw',
        'smooth_comb', 'discrete_comb'.
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

    References
    ----------
    .. footbibliography::
    """
    rng = np.random.default_rng(seed=seed)

    mu_1 = m_factor * np.array([1 / np.sqrt(i) for i in range(1, n_dim + 1)])
    mu_0 = -mu_1
    w = mu_1

    if rho != 0:
        if band_type == "ma":
            cov = _moving_avg_cov(n_dim, rho)
        elif band_type == "ar":
            cov = _autoregressive_cov(n_dim, rho)
        else:
            raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
    else:
        cov = np.identity(n_dim)

    if n_dim > 1000:
        method = "cholesky"
    else:
        method = "svd"

    MARRON_WAND_SIMS = {
        "gaussian": [1, [1]],
        "skewed_unimodal": [3, [1 / 5, 1 / 5, 3 / 5]],
        "strongly_skewed": [8, None],
        "kurtotic_unimodal": [2, [2 / 3, 1 / 3]],
        "outlier": [2, [1 / 10, 9 / 10]],
        "bimodal": [2, None],
        "separated_bimodal": [2, None],
        "skewed_bimodal": [2, [3 / 4, 1 / 4]],
        "trimodal": [3, [9 / 20, 9 / 20, 1 / 10]],
        "claw": [6, [1 / 2, *[1 / 10] * 5]],
        "double_claw": [9, [49 / 100, 49 / 100, *[1 / 350] * 7]],
        "asymmetric_claw": [6, [1 / 2, *[2 ** (1 - i) / 31 for i in range(-2, 3)]]],
        "asymmetric_double_claw": [8, [*[46 / 100] * 2, *[1 / 300] * 3, *[7 / 300] * 3]],
        "smooth_comb": [6, [2 ** (5 - i) / 63 for i in range(6)]],
        "discrete_comb": [6, [*[2 / 7] * 3, *[1 / 21] * 3]],
    }

    if simulation == "trunk":
        X = np.vstack(
            (
                rng.multivariate_normal(mu_0, cov, n_samples // 2, method=method),
                rng.multivariate_normal(mu_1, cov, n_samples // 2, method=method),
            )
        )
    elif simulation == "trunk_overlap":
        X = np.vstack(
            (
                rng.multivariate_normal(np.zeros(n_dim), cov, n_samples // 2, method=method),
                rng.multivariate_normal(np.zeros(n_dim), cov, n_samples // 2, method=method),
            )
        )
    elif simulation == "trunk_mix":
        mixture_idx = rng.choice(2, n_samples // 2, replace=True, shuffle=True, p=None)
        X_mixture = np.zeros((n_samples // 2, len(mu_1)))
        for idx in range(n_samples // 2):
            if mixture_idx[idx] == 1:
                X_sample = rng.multivariate_normal(mu_1, cov * (2 / 3) ** 2, 1, method=method)
            else:
                X_sample = rng.multivariate_normal(mu_0, cov * (2 / 3) ** 2, 1, method=method)
            X_mixture[idx, :] = X_sample

        X = np.vstack(
            (
                rng.multivariate_normal(np.zeros(n_dim), cov, n_samples // 2, method=method),
                X_mixture,
            )
        )
    elif simulation in MARRON_WAND_SIMS.keys():
        mixture_idx = rng.choice(
            MARRON_WAND_SIMS[simulation][0],
            n_samples // 2,
            replace=True,
            shuffle=True,
            p=MARRON_WAND_SIMS[simulation][1],
        )
        G = np.zeros((n_samples // 2, len(w)))
        for idx in range(n_samples // 2):
            G[idx, :] = MarronWandSims(
                n_dim=n_dim,
                cov=cov,
                method=method,
                mix_id=mixture_idx[idx],
                rng=rng,
                max_mix_id=MARRON_WAND_SIMS[simulation][0],
            )(simulation)

        X = np.vstack(
            (
                rng.multivariate_normal(np.zeros(n_dim), cov, n_samples // 2, method=method),
                (1 - w)
                * rng.multivariate_normal(np.zeros(n_dim), cov, n_samples // 2, method=method)
                + w * G,
            )
        )
    else:
        raise ValueError(
            f"Simulation must be trunk, trunk_overlap, trunk_mix, {MARRON_WAND_SIMS.keys()}"
        )

    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params:
        return X, y, [mu_0, mu_1], [cov, cov]
    return X, y


class MarronWandSims:
    def __init__(self, n_dim=0, cov=1, method="svd", rng=None, mix_id=None, max_mix_id=1):
        self.n_dim = n_dim
        self.cov = cov
        self.method = method
        self.rng = rng
        self.mix_id = mix_id
        self.max_mix_id = max_mix_id

    def __call__(self, simulation):
        sims = self._my_method_generator()
        if simulation in sims.keys():
            if self.mix_id in range(self.max_mix_id):
                return sims[simulation]()
            else:
                return ValueError(
                    f"Invalid value for mix_id, one of these: {range(self.max_mix_id)}"
                )
        else:
            raise ValueError(f"simulation is not one of these: {sims.keys()}")

    def _my_method_generator(self):
        return {
            method: getattr(self, method) for method in dir(self) if not method.startswith("__")
        }

    def gaussian(self):
        return self.rng.multivariate_normal(np.zeros(self.n_dim), self.cov, 1, method=self.method)

    def skewed_unimodal(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * 1 / 2, self.cov * (2 / 3) ** 2, 1, method=self.method
            )
        elif self.mix_id == 2:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * 13 / 12, self.cov * (5 / 9) ** 2, 1, method=self.method
            )

    def strongly_skewed(self):
        return self.rng.multivariate_normal(
            np.ones(self.n_dim) * 3 * ((2 / 3) ** self.mix_id - 1),
            self.cov * (2 / 3) ** (2 * self.mix_id),
            1,
            method=self.method,
        )

    def kurtotic_unimodal(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov * (1 / 10) ** 2, 1, method=self.method
            )

    def outlier(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov * (1 / 10) ** 2, 1, method=self.method
            )

    def bimodal(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                -np.ones(self.n_dim), self.cov * (2 / 3) ** 2, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim), self.cov * (2 / 3) ** 2, 1, method=self.method
            )

    def separated_bimodal(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                -np.ones(self.n_dim) * (3 / 2), self.cov * (1 / 2) ** 2, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (3 / 2), self.cov * (1 / 2) ** 2, 1, method=self.method
            )

    def skewed_bimodal(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (3 / 2), self.cov * (1 / 3) ** 2, 1, method=self.method
            )

    def trimodal(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (-6 / 5), self.cov * (3 / 5) ** 2, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (6 / 5), self.cov * (3 / 5) ** 2, 1, method=self.method
            )
        elif self.mix_id == 2:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov * (1 / 4) ** 2, 1, method=self.method
            )

    def claw(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov, 1, method=self.method
            )
        elif self.mix_id in range(1, 7):
            i = self.mix_id - 1
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (i / 2 - 1), self.cov * (1 / 10) ** 2, 1, method=self.method
            )

    def double_claw(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                -np.ones(self.n_dim), self.cov * (2 / 3) ** 2, 1, method=self.method
            )
        elif self.mix_id == 1:
            return self.rng.multivariate_normal(
                np.ones(self.n_dim), self.cov * (2 / 3) ** 2, 1, method=self.method
            )
        elif self.mix_id in range(2, 9):
            i = self.mix_id - 2
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (float(i) - 3) / 2,
                self.cov * (1 / 100) ** 2,
                1,
                method=self.method,
            )

    def asymmetric_claw(self):
        if self.mix_id == 0:
            return self.rng.multivariate_normal(
                np.zeros(self.n_dim), self.cov, 1, method=self.method
            )
        elif self.mix_id in range(1, 6):
            i = self.mix_id - 3
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (i + 1 / 2),
                self.cov * (2 ** -float(i) / 10) ** 2,
                1,
                method=self.method,
            )

    def asymmetric_double_claw(self):
        if self.mix_id in range(2):
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (2 * self.mix_id - 1),
                self.cov * (2 / 3) ** 2,
                1,
                method=self.method,
            )
        elif self.mix_id in range(2, 5):
            i = self.mix_id - 1
            return self.rng.multivariate_normal(
                -np.ones(self.n_dim) * (i / 2), self.cov * (1 / 100) ** 2, 1, method=self.method
            )
        elif self.mix_id in range(5, 8):
            i = self.mix_id - 4
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (i / 2), self.cov * (7 / 100) ** 2, 1, method=self.method
            )

    def smooth_comb(self):
        return self.rng.multivariate_normal(
            np.ones(self.n_dim) * (65 - 96 * (1 / 2) ** self.mix_id) / 21,
            self.cov * (32 / 63) ** 2 / (2 ** (2 * self.mix_id)),
            1,
            method=self.method,
        )

    def discrete_comb(self):
        if self.mix_id in range(3):
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (12 * self.mix_id - 15) / 7,
                self.cov * (2 / 7) ** 2,
                1,
                method=self.method,
            )
        if self.mix_id in range(3, 6):
            i = self.mix_id + 5
            return self.rng.multivariate_normal(
                np.ones(self.n_dim) * (2 * i) / 7, self.cov * (1 / 21) ** 2, 1, method=self.method
            )


class TrueSimulationDensities:
    def __init__(self, min_max=[-3, 3], n_samples=200):
        self.min_max = min_max
        self.n_samples = n_samples

    def __call__(self, simulation, n_dim=10):
        self.xs = np.linspace(self.min_max[0], self.min_max[1], self.n_samples)
        self.mean = 1 / np.sqrt(n_dim)
        self.weight = self.mean
        sims = self._my_method_generator()
        if simulation in sims.keys():
            return (self.xs, *sims[simulation]())
        else:
            raise ValueError(f"simulation is not one of these: {sims.keys()}")

    def _my_method_generator(self):
        return {
            method: getattr(self, method) for method in dir(self) if not method.startswith("__")
        }

    def trunk(self):
        pop1 = norm.pdf(self.xs, loc=self.mean, scale=1)
        pop2 = norm.pdf(self.xs, loc=-self.mean, scale=1)
        return pop1, pop2

    def trunk_overlap(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)
        pop2 = norm.pdf(self.xs, loc=0, scale=1)
        return pop1, pop2

    def trunk_mix(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=(2 / 3) ** 2)

        means = [-self.mean, self.mean]
        stds = [(2 / 3) ** 2, (2 / 3) ** 2]
        weights = [1 / 2] * 2

        pop2 = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            pop2 += norm.pdf(self.xs, loc=l, scale=s) * w
        return pop1, pop2

    def gaussian(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)
        G = self.weight * norm.pdf(self.xs, loc=0, scale=1)
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def skewed_unimodal(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [0, 1 / 2, 13 / 12]
        stds = [1, (2 / 3) ** 2, (5 / 9) ** 2]
        weights = [1 / 5, 1 / 5, 3 / 5]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def strongly_skewed(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [3 * (((2 / 3) ** l) - 1) for l in range(8)]
        stds = [(2 / 3) ** (2 * l) for l in range(8)]
        weights = [1 / 8] * 8

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def kurtotic_unimodal(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [0, 0]
        stds = [1, (1 / 10) ** 2]
        weights = [2 / 3, 1 / 3]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def outlier(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [0, 0]
        stds = [1, (1 / 10) ** 2]
        weights = [1 / 10, 9 / 10]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def bimodal(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [-1, 1]
        stds = [(2 / 3) ** 2, (2 / 3) ** 2]
        weights = [1 / 2] * 2

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def separated_bimodal(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [-3 / 2, 3 / 2]
        stds = [(2 / 3) ** 2, (2 / 3) ** 2]
        weights = [1 / 2] * 2

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def skewed_bimodal(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [0, 3 / 2]
        stds = [1, (1 / 3) ** 2]
        weights = [3 / 4, 1 / 4]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def trimodal(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [-6 / 5, 6 / 5, 0]
        stds = [(3 / 5) ** 2, (3 / 5) ** 2, (1 / 4) ** 2]
        weights = [9 / 20, 9 / 20, 1 / 10]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def claw(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [0, *[l / 2 - 1 for l in range(5)]]
        stds = [1, *[1 / 10] * 5]
        weights = [1 / 2, *[1 / 10] * 5]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def double_claw(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [-1, 1, *[(l - 3) / 2 for l in range(7)]]
        stds = [(2 / 3) ** 2, (2 / 3) ** 2, *[(1 / 100) ** 2] * 7]
        weights = [49 / 100, 49 / 100, *[1 / 350] * 7]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def asymmetric_claw(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [0, *[l + 1 / 2 for l in range(7)]]
        stds = [1, *[(2 ** -float(l) / 10) ** 2 for l in range(-2, 3)]]
        weights = [1 / 2, *[2 ** (1 - float(l)) / 31 for l in range(-2, 3)]]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def asymmetric_double_claw(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [
            *[2 * l - 1 for l in range(2)],
            *[-l / 2 for l in range(1, 4)],
            *[l / 2 for l in range(1, 4)],
        ]
        stds = [*[(2 / 3) ** 2] * 2, *[(1 / 100) ** 2] * 3, *[(7 / 100) ** 2] * 3]
        weights = [*[46 / 100] * 2, *[1 / 300] * 3, *[7 / 300] * 3]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def smooth_comb(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [(65 - 96 * ((1 / 2) ** l)) / 21 for l in range(6)]
        stds = [((32 / 63) ** 2) / (2 ** (2 * l)) for l in range(6)]
        weights = [(2 ** (5 - l)) / 63 for l in range(6)]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2

    def discrete_comb(self):
        pop1 = norm.pdf(self.xs, loc=0, scale=1)

        means = [*[(12 * l - 15) / 7 for l in range(3)], *[2 * l / 7 for l in range(8, 11)]]
        stds = [*[(2 / 7) ** 2] * 3, *[(1 / 21) ** 2] * 3]
        weights = [*[2 / 7] * 3, *[1 / 21] * 3]

        G = np.zeros_like(self.xs)
        for l, s, w in zip(means, stds, weights):
            G += norm.pdf(self.xs, loc=l, scale=s) * w
        pop2 = (1 - self.weight) * norm.pdf(self.xs, loc=0, scale=1) + self.weight * G
        return pop1, pop2


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
    lims = [[-scale, scale]] * n_dims

    # Compute entropy and X and Y.
    def func(*args):
        x = np.array(args)
        p = 0
        for k in range(len(means)):
            p += class_probs[k] * multivariate_normal(seed=seed).pdf(x, means[k], covs[k])
        return -p * np.log(p) / np.log(base)

    # numerically integrate H(X)
    opts = dict(limit=1000)
    H_X, int_err = nquad(func, lims, opts=opts)

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

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from treeple.datasets import (
    approximate_clf_mutual_information,
    approximate_clf_mutual_information_with_monte_carlo,
    make_marron_wand_classification,
    make_quadratic_classification,
    make_trunk_classification,
    make_trunk_mixture_classification,
)
from treeple.datasets.hyppo import MARRON_WAND_SIMS


def test_make_quadratic_classification_v():
    n_samples = 100
    n_features = 5
    x, v = make_quadratic_classification(n_samples, n_features)
    assert all(val in [0, 1] for val in v)
    assert x.shape == (n_samples * 2, n_features)
    assert len(x) == len(v)


@pytest.mark.parametrize(
    "trunk_gen", [make_trunk_classification, make_trunk_mixture_classification]
)
def test_make_trunk_classification_custom_parameters(trunk_gen):
    # Test with custom parameters
    X, y = trunk_gen(
        n_samples=50,
        n_dim=5,
        n_informative=2,
        mu_0=0,
        rho=0.5,
        band_type="ma",
        return_params=False,
    )
    assert X.shape == (50, 5)
    assert y.shape == (50,)


@pytest.mark.parametrize(
    "trunk_gen", [make_trunk_classification, make_trunk_mixture_classification]
)
def test_make_trunk_classification_autoregressive_cov(trunk_gen):
    # Test with default parameters
    n_dim = 10
    n_informative = 10
    rho = 0.5
    data = trunk_gen(
        n_samples=100,
        n_dim=n_dim,
        n_informative=n_informative,
        rho=rho,
        band_type="ar",
        return_params=True,
    )
    cov_list = data[3]
    if trunk_gen == make_trunk_classification:
        assert len(data) == 4
        assert len(data[2]) == 2
        assert len(data[3]) == 2
        assert_array_equal(cov_list[0][0, :], [rho**idx for idx in range(n_dim)])
    elif trunk_gen == make_trunk_mixture_classification:
        assert len(data) == 5
        assert_array_equal(cov_list[0][0, :], [rho**idx for idx in range(n_dim)])
    assert_array_equal(cov_list[0], cov_list[1])
    assert cov_list[0].shape == (n_dim, n_dim)


def test_make_trunk_classification_return_params():
    # Test with return_params=True and uneven number of samples
    n_informative = 5
    X, y, means, covs = make_trunk_classification(
        n_samples=75, n_dim=10, n_informative=n_informative, return_params=True
    )
    assert X.shape == (74, 10), X.shape
    assert y.shape == (74,)
    assert len(means) == 2
    assert len(covs) == 2


@pytest.mark.parametrize(
    "trunk_gen", [make_trunk_classification, make_trunk_mixture_classification]
)
def test_make_trunk_generator_errors(trunk_gen):
    # Test with an invalid band type
    with pytest.raises(ValueError, match=r"Band type .* must be one of"):
        trunk_gen(n_samples=50, rho=0.5, band_type="invalid_band_type")

    # Test with an invalid band type
    with pytest.warns(RuntimeWarning, match="Number of informative dimensions"):
        trunk_gen(n_samples=50, n_dim=10, n_informative=11, rho=0.5)


def test_make_trunk_mixture_errors():
    # Test with an invalid band type
    with pytest.raises(ValueError, match="Mix must be between 0 and 1."):
        make_trunk_mixture_classification(n_samples=50, rho=0.5, mix=2)


def test_make_marron_wand_errors():
    # Test with an invalid band type
    with pytest.raises(ValueError, match="Simulation must be"):
        make_marron_wand_classification(n_samples=50, rho=0.5, simulation=None)


@pytest.mark.parametrize("simulation", [*MARRON_WAND_SIMS.keys()])
def test_make_marron_wand_simulations(simulation):
    # Test with default parameters
    n_samples = 100
    n_dim = 10
    n_informative = 10
    X, y = make_marron_wand_classification(
        n_samples=n_samples,
        n_dim=n_dim,
        n_informative=n_informative,
        simulation=simulation,
    )
    assert X.shape == (n_samples, n_dim)
    assert y.shape == (n_samples,)


def test_approximate_clf_mutual_information_numerically_close():
    mean1 = np.array([1, 1])
    cov1 = np.array([[1, 0.5], [0.5, 1]])

    mean2 = np.array([-1, -1])
    cov2 = np.array([[1, -0.5], [-0.5, 1]])

    means = [mean1, mean2]
    covs = [cov1, cov2]

    result_approximate = approximate_clf_mutual_information(means, covs)
    result_monte_carlo = approximate_clf_mutual_information_with_monte_carlo(means, covs)

    assert np.isclose(
        result_approximate[0], result_monte_carlo[0], atol=5e-2
    ), f"{result_approximate[0]}, {result_monte_carlo[0]}"


@pytest.mark.parametrize(
    "method",
    [make_trunk_classification, make_trunk_mixture_classification],
)
def test_consistent_fixed_seed_trunk_and_mix(method):
    seed = 0
    n_samples = 10
    n_dim = 5
    n_informative = 3
    X1, _ = method(n_samples=n_samples, n_dim=n_dim, n_informative=n_informative, seed=seed)
    X2, _ = method(n_samples=n_samples + 2, n_dim=n_dim + 1, n_informative=n_informative, seed=seed)

    # informative is the same
    assert_array_equal(X1[: n_samples // 2, :n_informative], X2[: n_samples // 2, :n_informative])
    assert_array_equal(
        X1[n_samples // 2 :, :n_informative], X2[n_samples // 2 + 1 : -1, :n_informative]
    )

    # noise is the same
    assert_array_equal(X1[:, n_informative:], X2[:n_samples, n_informative:n_dim])


@pytest.mark.parametrize(
    "method",
    [make_marron_wand_classification],
)
@pytest.mark.parametrize(
    "simulation",
    [*MARRON_WAND_SIMS.keys()],
)
def test_consistent_fixed_seed_marron_wand(method, simulation):
    seed = 0
    n_samples = 10
    n_dim = 5
    n_informative = 3
    X1, _ = method(
        n_samples=n_samples,
        n_dim=n_dim,
        n_informative=n_informative,
        seed=seed,
        simulation=simulation,
    )
    X2, _ = method(
        n_samples=n_samples + 2,
        n_dim=n_dim + 1,
        n_informative=n_informative,
        seed=seed,
        simulation=simulation,
    )

    # informative is the same
    assert_array_equal(X1[: n_samples // 2, :n_informative], X2[: n_samples // 2, :n_informative])
    assert_array_equal(
        X1[n_samples // 2 :, :n_informative], X2[n_samples // 2 + 1 : -1, :n_informative]
    )

    # noise is the same
    assert_array_equal(X1[:, n_informative:], X2[:n_samples, n_informative:n_dim])

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sktree.datasets import (
    make_quadratic_classification,
    make_trunk_classification,
    approximate_clf_mutual_information,
    approximate_clf_mutual_information_with_monte_carlo,
)


def test_make_quadratic_classification_v():
    n_samples = 100
    n_features = 5
    x, v = make_quadratic_classification(n_samples, n_features)
    assert all(val in [0, 1] for val in v)
    assert x.shape == (n_samples * 2, n_features)
    assert len(x) == len(v)


def test_make_trunk_classification_default():
    # Test with default parameters
    X, y = make_trunk_classification(n_samples=100)
    assert X.shape == (100, 10)
    assert y.shape == (100,)


def test_make_trunk_classification_custom_parameters():
    # Test with custom parameters
    X, y = make_trunk_classification(
        n_samples=50,
        n_dim=5,
        n_informative=2,
        m_factor=2,
        rho=0.5,
        band_type="ma",
        return_params=False,
    )
    assert X.shape == (50, 5)
    assert y.shape == (50,)


def test_make_trunk_classification_autoregressive_cov():
    # Test with default parameters
    n_dim = 10
    rho = 0.5
    _, _, _, cov_list = make_trunk_classification(
        n_samples=100, n_dim=n_dim, rho=rho, band_type="ar", return_params=True
    )
    assert_array_equal(cov_list[0], cov_list[1])
    assert cov_list[0].shape == (n_dim, n_dim)
    assert_array_equal(cov_list[0][0, :], [rho**idx for idx in range(n_dim)])


def test_make_trunk_classification_mixture():
    # Test with default parameters
    X, y, _, _ = make_trunk_classification(n_samples=100, mix=0.5, return_params=True)
    assert X.shape == (100, 10)
    assert y.shape == (100,)


def test_make_trunk_classification_return_params():
    # Test with return_params=True and uneven number of samples
    X, y, means, covs = make_trunk_classification(n_samples=75, n_dim=10, return_params=True)
    assert X.shape == (74, 10), X.shape
    assert y.shape == (74,)
    assert len(means) == 2
    assert len(covs) == 2


def test_make_trunk_classification_invalid_band_type():
    # Test with an invalid band type
    with pytest.raises(ValueError, match=r"Band type .* must be one of"):
        make_trunk_classification(n_samples=50, rho=0.5, band_type="invalid_band_type")


def test_make_trunk_classification_invalid_mix():
    # Test with an invalid band type
    with pytest.raises(ValueError, match="Mix must be between 0 and 1."):
        make_trunk_classification(n_samples=50, rho=0.5, mix=2)


def test_make_trunk_classification_invalid_n_informative():
    # Test with an invalid band type
    with pytest.raises(ValueError, match="Number of informative dimensions"):
        make_trunk_classification(n_samples=50, n_dim=10, n_informative=11, rho=0.5, mix=2)


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

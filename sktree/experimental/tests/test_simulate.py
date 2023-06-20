from sktree.experimental.simulate import (
    simulate_helix,
    simulate_multivariate_gaussian,
    simulate_sphere,
)


# Test simulate_helix function
def test_simulate_helix():
    P, X, Y, Z = simulate_helix(n_samples=1000)
    assert len(P) == 1000
    assert len(X) == 1000
    assert len(Y) == 1000
    assert len(Z) == 1000

    # Add more specific tests if necessary


# Test simulate_sphere function
def test_simulate_sphere():
    latitude, longitude, Y1, Y2, Y3 = simulate_sphere(n_samples=1000)
    assert len(latitude) == 1000
    assert len(longitude) == 1000
    assert len(Y1) == 1000
    assert len(Y2) == 1000
    assert len(Y3) == 1000

    # Add more specific tests if necessary


# Test simulate_multivariate_gaussian function
def test_simulate_multivariate_gaussian():
    data, mean, cov = simulate_multivariate_gaussian(d=2, n_samples=1000)
    assert data.shape == (1000, 2)
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)

    # Add more specific tests if necessary

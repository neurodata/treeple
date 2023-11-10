import numpy as np
import pytest

from sktree.datasets.multiview import make_gaussian_mixture, make_joint_factor_model


def test_make_gaussian_mixture_errors():
    with pytest.raises(ValueError, match="centers is of the incorrect shape"):
        # Test centers dimension error
        make_gaussian_mixture(np.zeros((2, 2, 2)), np.eye(2))

    with pytest.raises(ValueError, match="covariance matrix is of the incorrect shape"):
        # Test covariances dimension error
        make_gaussian_mixture([0, 1], np.zeros((2, 2, 2, 2)), random_state=0)

    with pytest.raises(ValueError, match="The first dimensions of 2D centers and 3D covariances"):
        # Test centers and covariances shape mismatch
        make_gaussian_mixture(np.zeros((3, 2)), np.zeros((2, 2, 2)), random_state=0)

    with pytest.raises(ValueError, match="elements of `class_probs` must sum to 1"):
        # Test class_probs not summing to 1
        make_gaussian_mixture([[0, 1], [2, 3]], [np.eye(2), np.eye(2)], class_probs=[0.5, 0.6])

    with pytest.raises(
        ValueError, match="centers, covariances, and class_probs must be of equal length"
    ):
        make_gaussian_mixture([[0, 1], [2, 3]], [np.eye(2), np.eye(2)], class_probs=[1.0])

    # Test invalid transform type
    with pytest.raises(ValueError, match="Transform type must be one of {'linear', 'poly', 'sin'}"):
        make_gaussian_mixture([[0, 1], [2, 3]], [np.eye(2), np.eye(2)], transform="invalid")

    # Test invalid transform type (callable)
    with pytest.raises(
        TypeError, match="'transform' must be of type string or a callable function"
    ):
        make_gaussian_mixture([[0, 1], [2, 3]], [np.eye(2), np.eye(2)], transform=123)


def test_make_gaussian_mixture():
    # Test basic functionality
    Xs, y = make_gaussian_mixture([[0, 1], [2, 3]], [np.eye(2), np.eye(2)], random_state=0)
    assert len(Xs) == 2
    assert Xs[0].shape == (100, 2)
    assert Xs[1].shape == (100, 2)
    assert y.shape == (100,)

    # Test with noise
    Xs, y = make_gaussian_mixture(
        [[0, 1], [2, 3]],
        [np.eye(2), np.eye(2)],
        noise=0.1,
        noise_dims=2,
        random_state=0,
    )
    assert len(Xs) == 2
    assert Xs[0].shape == (100, 4)  # 2 original dimensions + 2 noise dimensions
    assert Xs[1].shape == (100, 4)  # 2 original dimensions + 2 noise dimensions
    assert y.shape == (100,)


def custom_transform(x):
    return x + 2


@pytest.mark.parametrize("transform", ["linear", "poly", "sin", custom_transform])
def test_make_gaussian_mixture_with_transform(transform):
    # Test with any transformation
    Xs, y = make_gaussian_mixture(
        [[0, 1], [2, 3]],
        [np.eye(2), np.eye(2)],
        transform=transform,
        random_state=0,
    )
    assert len(Xs) == 2
    assert Xs[0].shape == (100, 2)
    assert Xs[1].shape == (100, 2)
    assert y.shape[0] == 100
    old_sum = np.sum(y)

    # Test with any transformation
    Xs, y = make_gaussian_mixture(
        [[0, 1], [2, 3]], [np.eye(2), np.eye(2)], transform=transform, random_state=0, shuffle=True
    )
    assert len(Xs) == 2
    assert Xs[0].shape == (100, 2)
    assert Xs[1].shape == (100, 2)
    assert y.shape[0] == 100
    assert np.sum(y) == old_sum


def test_make_joint_factor_model():
    Xs = make_joint_factor_model(1, 3, n_samples=100, random_state=0)
    assert len(Xs) == 1
    assert Xs[0].shape == (100, 3)

    Xs, U, view_loadings = make_joint_factor_model(
        2, 3, n_samples=100, random_state=0, return_decomp=True
    )
    assert len(Xs) == 2
    assert Xs[0].shape == (100, 3)
    assert Xs[1].shape == (100, 3)
    assert U.shape == (100, 1)
    assert len(view_loadings) == 2
    assert view_loadings[0].shape == (3, 1)
    assert view_loadings[1].shape == (3, 1)

    Xs, U, view_loadings = make_joint_factor_model(
        3,
        [2, 4, 3],
        n_samples=100,
        random_state=0,
        return_decomp=True,
    )
    assert len(Xs) == 3, f"Expected 3 views, got {len(Xs)}"
    assert Xs[0].shape == (100, 2)
    assert Xs[1].shape == (100, 4)
    assert Xs[2].shape == (100, 3)
    assert U.shape == (100, 1)
    assert len(view_loadings) == 3
    assert view_loadings[0].shape == (2, 1)
    assert view_loadings[1].shape == (4, 1)
    assert view_loadings[2].shape == (3, 1)

import numpy as np
import pytest

from treeple.tree._kernel import sample_gaussian_kernels


def gaussian_2d(x, y, mu, sigma):
    """Generate the 2D Gaussian value at point (x, y).

    Used to generate the expected Gaussian kernel for testing.
    """
    return np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma**2))


@pytest.mark.parametrize(
    "n_kernels, min_size, max_size, mu, sigma",
    [
        (1, 3, 3, 0.0, 1.0),  # Single fixed kernel
        (2, 3, 5, 0.5, 1.1),  # Two fixed-size kernels with varying sigma
        (3, 3, 5, (0.0, 0.5), (0.1, 1.0)),  # Varying sizes, fixed sigma
    ],
)
def test_sample_gaussian_kernels(n_kernels, min_size, max_size, mu, sigma):
    kernel_matrix, kernel_params = sample_gaussian_kernels(n_kernels, min_size, max_size, mu, sigma)

    # Ensure kernel_matrix and kernel_sizes have correct lengths
    assert kernel_matrix.shape[0] == n_kernels
    assert kernel_params["size"].shape[0] == n_kernels

    for i in range(n_kernels):
        size = kernel_params["size"][i]
        mu_sample = kernel_params["mu"][i]
        sigma_sample = kernel_params["sigma"][i]

        # Extract and reshape the kernel
        start_idx = kernel_matrix.indptr[i]
        end_idx = kernel_matrix.indptr[i + 1]
        kernel_vector = kernel_matrix.data[start_idx:end_idx]
        kernel = kernel_vector.reshape(size, size)

        # Generate the expected Gaussian kernel
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        expected_kernel = gaussian_2d(X, Y, mu_sample, sigma_sample)

        # Normalize the expected kernel for comparison
        expected_kernel /= expected_kernel.sum()
        kernel /= kernel.sum()

        # Check that the kernel matches the expected Gaussian distribution
        np.testing.assert_allclose(kernel, expected_kernel, rtol=1e-2, atol=1e-2)

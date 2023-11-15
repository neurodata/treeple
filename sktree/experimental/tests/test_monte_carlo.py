import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

from sktree.experimental import conditional_resample


def test_conditional_resample_with_default_params():
    # Create a simple example dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()
    # Test conditional resampling with default parameters
    resampled_arrays = conditional_resample(
        conditional_array, X, y, nn_estimator=NearestNeighbors()
    )

    # Check if the number of samples in resampled_arrays is the same as the input arrays
    assert len(resampled_arrays) == 2
    assert len(resampled_arrays[0]) == len(X)
    assert len(resampled_arrays[1]) == len(y)


def test_conditional_resample_without_replacement():
    # Create a simple example dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()

    # Test conditional resampling without replacement
    resampled_arrays = conditional_resample(
        conditional_array, X, y, nn_estimator=NearestNeighbors(), replace=False
    )

    # Check if the number of samples in resampled_arrays is the same as the input arrays
    assert len(resampled_arrays) == 2
    assert len(resampled_arrays[0]) == len(X)
    assert len(resampled_arrays[1]) == len(y)

    # Check if the samples are unique (no replacement)
    assert len(np.unique(resampled_arrays[1])) == len(
        np.unique(y)
    ), f"{len(np.unique(resampled_arrays[1]))} != {len(y)}"


def test_conditional_resample_with_sparse_matrix():
    # Create a simple example dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_sparse = csr_matrix(X)  # Convert X to a sparse matrix

    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()

    # Test conditional resampling with a sparse matrix
    resampled_arrays = conditional_resample(
        conditional_array, X_sparse, y, nn_estimator=NearestNeighbors()
    )

    # Check if the number of samples in resampled_arrays is the same as the input arrays
    assert len(resampled_arrays) == 2
    assert resampled_arrays[0].shape[0] == len(X)
    assert len(resampled_arrays[1]) == len(y)


def test_conditional_resample_with_stratify():
    # Create a simple example dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()

    # Define a custom stratify function
    def custom_stratify(y, category):
        # Create an array where each entry is True if it belongs to the specified category,
        # False otherwise
        stratify_array = y == category
        return stratify_array

    category_to_stratify = 1  # Change this to the category you want to stratify

    # Get the distribution of the specified category before resampling
    category_distribution_before = np.sum(y == category_to_stratify)

    # Test conditional resampling with the custom stratify function
    stratify = custom_stratify(y, category_to_stratify)
    resampled_arrays = conditional_resample(
        conditional_array, X, y, nn_estimator=NearestNeighbors(), stratify=stratify, random_state=0
    )

    # Get the distribution of the specified category after resampling
    category_distribution_after = np.sum(resampled_arrays[1] == category_to_stratify)

    # Check if the distribution of the specified category is preserved
    assert category_distribution_before == category_distribution_after, (
        f"Expected {category_distribution_before} samples, got "
        f"{category_distribution_after} samples"
    )


def test_conditional_resample_with_replace_nbrs():
    # Create a simple example dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()

    # Test conditional resampling with replace_nbrs=False
    resampled_arrays = conditional_resample(
        conditional_array, X, y, nn_estimator=NearestNeighbors(), replace_nbrs=False
    )

    # Check if the number of samples in resampled_arrays is the same as the input arrays
    assert len(resampled_arrays) == 2, f"Expected 2 arrays, got {len(resampled_arrays)} arrays"
    assert len(resampled_arrays[0]) == len(X)
    assert len(resampled_arrays[1]) == len(y)


def test_conditional_resample_errors():
    # 01: Test with invalid number of samples
    # Create a simple example dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Test conditional resampling with an invalid stratify array (should raise an error)
    with pytest.raises(ValueError, match="Cannot sample"):
        conditional_resample(X, y, nn_estimator=NearestNeighbors(), replace=False, n_samples=1000)

    # 02: Test inconsistent_length
    # Create an additional array with a different number of samples
    additional_array = np.random.rand(80, 5)

    # Test conditional resampling with inconsistent length of input arrays (should raise an error)
    with pytest.raises(ValueError):
        conditional_resample(X, y, additional_array, nn_estimator=NearestNeighbors())

    # 03: Test with invalid sample size when replace=False
    # Test conditional resampling with n_samples larger than the input arrays
    # (should raise an error)
    with pytest.raises(ValueError):
        conditional_resample(X, y, nn_estimator=NearestNeighbors(), n_samples=200, replace=False)


def test_conditional_resample():
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Convert X to sparse matrix
    X_sparse = csr_matrix(X)

    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()

    # Perform conditional resampling
    resampled_X = conditional_resample(conditional_array, X, replace=False, replace_nbrs=False)
    resampled_X_sparse = conditional_resample(
        conditional_array, X_sparse, replace=False, replace_nbrs=False
    )

    # Check that the resampled arrays have the correct shape
    assert resampled_X.shape == X.shape
    assert resampled_X_sparse.shape == X_sparse.shape

    # Check that the resampled arrays have the correct number of unique samples
    assert len(np.unique(resampled_X, axis=0)) == X.shape[0]
    assert len(np.unique(resampled_X_sparse.toarray(), axis=0)) == X_sparse.shape[0]

    # Check that the conditional distribution is preserved
    for i in range(X.shape[1]):
        unique_values, counts = np.unique(resampled_X[:, i], return_counts=True)
        original_values, original_counts = np.unique(X[:, i], return_counts=True)

        assert np.all(unique_values == original_values)
        assert np.all(counts == original_counts)

        unique_values_sparse, counts_sparse = np.unique(
            resampled_X_sparse[:, i].toarray(), return_counts=True
        )
        original_values_sparse, original_counts_sparse = np.unique(
            X_sparse[:, i].toarray(), return_counts=True
        )

        assert np.all(unique_values_sparse == original_values_sparse)
        assert np.all(counts_sparse == original_counts_sparse)

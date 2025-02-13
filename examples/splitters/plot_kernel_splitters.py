"""
====================
Random Kernel Splits
====================

This example shows how to build a manifold oblique decision tree classifier using
a custom set of user-defined kernel/filter library, such as the Gaussian, or Gabor
kernels.

The example demonstrates superior performance on a 2D dataset with structured images
as samples. The dataset is the downsampled MNIST dataset, where each sample is a
28x28 image. The dataset is downsampled to 14x14, and then flattened to a 196
dimensional vector. The dataset is then split into a training and testing set.

See :ref:`sphx_glr_auto_examples_plot_projection_matrices` for more information on
projection matrices and the way they can be sampled.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from treeple.tree.manifold._kernel_splitter import Kernel2D

# %%
# Create a synthetic image
image_height, image_width = 50, 50
image = np.random.rand(image_height, image_width).astype(np.float32)

# Generate a Gaussian kernel (example)
kernel_size = 7
x = np.linspace(-2, 2, kernel_size)
y = np.linspace(-2, 2, kernel_size)
x, y = np.meshgrid(x, y)
kernel = np.exp(-(x**2 + y**2))
kernel = kernel / kernel.sum()  # Normalize the kernel

# Vectorize and create a sparse CSR matrix
kernel_vector = kernel.flatten().astype(np.float32)
kernel_indices = np.arange(kernel_vector.size)
kernel_indptr = np.array([0, kernel_vector.size])
kernel_csr = csr_matrix(
    (kernel_vector, kernel_indices, kernel_indptr), shape=(1, kernel_vector.size)
)

# %%
# Initialize the Kernel2D class
kernel_sizes = np.array([kernel_size], dtype=np.intp)
random_state = np.random.RandomState(42)
print(kernel_csr.dtype, kernel_sizes.dtype, np.intp)
kernel_2d = Kernel2D(kernel_csr, kernel_sizes, random_state)

# Apply the kernel to the image
result_value = kernel_2d.apply_kernel_py(image, 0)

# %%
# Plot the original image, kernel, and result
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(image, cmap="gray")
axs[0].set_title("Original Image")

axs[1].imshow(kernel, cmap="viridis")
axs[1].set_title("Gaussian Kernel")

# Highlight the region where the kernel was applied
start_x, start_y = random_state.randint(0, image_width - kernel_size + 1), random_state.randint(
    0, image_height - kernel_size + 1
)
image_with_kernel = image.copy()
image_with_kernel[start_y : start_y + kernel_size, start_x : start_x + kernel_size] *= kernel

axs[2].imshow(image_with_kernel, cmap="gray")
axs[2].set_title(f"Result: {result_value:.4f}")

plt.tight_layout()
plt.show()

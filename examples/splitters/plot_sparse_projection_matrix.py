"""
======================================================
Plot the sparse projection matrices of an oblique tree
======================================================

This example shows how projection matrices are generated for an oblique tree,
specifically the :class:`sktree.tree.ObliqueDecisionTreeClassifier`.

The projection matrix here samples a subset of features from the input ``X``
controlled by the parameter ``feature_combinations``. The projection matrix
is sparse when ``feature_combinations`` is small, meaning that it is mostly zero.
The non-zero elements of the projection matrix are the features that are sampled
from the input ``X`` and linearly combined to form candidate split dimensions.

For details on how to use the hyperparameters related to the patches, see
:class:`sktree.tree.ObliqueDecisionTreeClassifier`.
"""

# import matplotlib.pyplot as plt

# import modules
# .. note:: We use a private Cython module here to demonstrate what the patches
#           look like. This is not part of the public API. The Cython module used
#           is just a Python wrapper for the underlying Cython code and is not the
#           same as the Cython splitter used in the actual implementation.
#           To use the actual splitter, one should use the public API for the
#           relevant tree/forests class.
import numpy as np

from sktree._lib.sklearn.tree._criterion import Gini
from sktree.tree._oblique_splitter import BestObliqueSplitterTester

# %%
# Initialize patch splitter
# -------------------------
# The patch splitter is used to generate patches for the projection matrices.
# We will initialize the patch with some dummy values for the sake of this
# example.

criterion = Gini(1, np.array((0, 1)))
max_features = 6
min_samples_leaf = 1
min_weight_leaf = 0.0
random_state = np.random.RandomState(100)

feature_combinations = 1
monotonic_cst = None
missing_value_feature_mask = None

# initialize some dummy data
X = np.repeat(np.arange(25).astype(np.float32), 5).reshape(5, -1)
y = np.array([0, 0, 0, 1, 1]).reshape(-1, 1).astype(np.float64)
sample_weight = np.ones(5)

print("The shape of our dataset is: ", X.shape, y.shape, sample_weight.shape)

# %%
# Generate 1D patches
# -------------------


splitter = BestObliqueSplitterTester(
    criterion,
    max_features,
    min_samples_leaf,
    min_weight_leaf,
    random_state,
    monotonic_cst,
    feature_combinations,
)
splitter.init_test(X, y, sample_weight, missing_value_feature_mask)

# sample the projection matrix that consists of 1D patches
proj_mat = splitter.sample_projection_matrix()
print(proj_mat.shape)

# Visualize 1D patches
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True, squeeze=True)
# axs = axs.flatten()
# for idx, ax in enumerate(axs):
#     ax.imshow(proj_mat[idx, :].reshape(data_dims), cmap="viridis")
#     ax.set(
#         xlim=(-1, data_dims[1]),
#         ylim=(-1, data_dims[0]),
#         title=f"Patch {idx}",
#     )

# fig.suptitle("1D Patch Visualization")
# plt.show()

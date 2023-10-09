"""
===========================================================================
Demonstrate and visualize a multi-view projection matrix of an oblique tree
===========================================================================

This example shows how multi-view projection matrices are generated for an oblique tree,
specifically the :class:`sktree.tree.ObliqueDecisionTreeClassifier`.

Multi-view projection matrices operate under the assumption that the input ``X`` array
consists of multiple feature-sets that are groups of features important for predicting
``y``. 

For details on how to use the hyperparameters related to the multi-view, see
:class:`sktree.tree.ObliqueDecisionTreeClassifier`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

from sktree._lib.sklearn.tree._criterion import Gini
from sktree.tree._oblique_splitter import MultiViewSplitterTester


criterion = Gini(1, np.array((0, 1)))
max_features = 6
min_samples_leaf = 1
min_weight_leaf = 0.0
random_state = np.random.RandomState(100)

feature_set_ends = np.array([3, 5, 9], dtype=np.int8)
n_feature_sets = len(feature_set_ends)
uniform_sampling = True

feature_combinations = 1
monotonic_cst = None
missing_value_feature_mask = None

# initialize some dummy data
X = np.repeat(np.arange(feature_set_ends[-1]).astype(np.float32), 5).reshape(5, -1)
y = np.array([0, 0, 0, 1, 1]).reshape(-1, 1).astype(np.float64)
sample_weight = np.ones(5)

print("The shape of our dataset is: ", X.shape, y.shape, sample_weight.shape)

# %%
# Initialize the multi-view splitter
# ----------------------------------
# The multi-view splitter is a Cython class that is initialized internally
# in scikit-tree. However, we expose a Python tester object to demonstrate
# how the splitter works in practice.

splitter = MultiViewSplitterTester(
    criterion,
    max_features,
    min_samples_leaf,
    min_weight_leaf,
    random_state,
    monotonic_cst,
    feature_combinations,
    feature_set_ends,
    n_feature_sets,
    uniform_sampling
)
splitter.init_test(X, y, sample_weight, missing_value_feature_mask)

# %%
# Sample the projection matrix
# ----------------------------
# The projection matrix is sampled by the splitter. The projection
# matrix is a (max_features, n_features) matrix that linearly combines random
# features from ``X`` to define candidate split dimensions. The multi-view
# splitter's projection matrix though samples from multiple feature sets,
# which are aligned contiguously over the columns of ``X``.

projection_matrix = splitter.sample_projection_matrix_py()

cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c'][:n_feature_sets])

# Create a heatmap to visualize the indices
fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(projection_matrix, cmap=cmap, aspect='auto')
ax.set(
    title="Sampled Projection Matrix",
    xlabel="Feature Index",
    ylabel="Projection Vector Index"
)
ax.set_xticks(np.arange(feature_set_ends[-1]))
ax.set_yticks(np.arange(max_features))
ax.set_yticklabels(np.arange(max_features, dtype=int) + 1)
ax.set_xticklabels(np.arange(feature_set_ends[-1], dtype=int) + 1)

# Create a mappable object
sm = ScalarMappable(cmap=cmap)
sm.set_array([])  # You can set an empty array or values here

# Create a color bar with labels for each feature set
colorbar = fig.colorbar(sm, ax=ax, ticks=np.arange(n_feature_sets) + 0.5, format="%d")
colorbar.set_label("Feature Set")

plt.show()
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

from sktree._lib.sklearn.tree._criterion import Gini
from sktree.tree._oblique_splitter import MultiViewSplitterTester


criterion = Gini(1, np.array((0, 1)))
max_features = 6
min_samples_leaf = 1
min_weight_leaf = 0.0
random_state = np.random.RandomState(100)

feature_set_ends = [3, 5, 9]
n_feature_sets = 3

boundary = None
feature_weight = None
missing_value_feature_mask = None

# initialize some dummy data
X = np.repeat(np.arange(25).astype(np.float32), 5).reshape(5, -1)
y = np.array([0, 0, 0, 1, 1]).reshape(-1, 1).astype(np.float64)
sample_weight = np.ones(5)

print("The shape of our dataset is: ", X.shape, y.shape, sample_weight.shape)

# %%
# Generate multiple views
# -------------------
# Now that we have th patch splitter initialized, we can generate some patches
# and visualize how they appear on the data. We will make the patch 1D, which
# samples multiple rows contiguously. This is a 1D patch of size 3.

splitter = MultiViewSplitterTester(
    criterion,
    max_features,
    min_samples_leaf,
    min_weight_leaf,
    random_state,
    missing_value_feature_mask,
)
splitter.init_test(X, y, sample_weight, None)


# Sample the projection matrix
projection_matrix = multi_view_splitter.sample_projection_matrix_py()

# Create a heatmap to visualize the indices
plt.figure(figsize=(10, 6))
plt.imshow(projection_matrix, cmap='binary', aspect='auto')
plt.title("Sampled Projection Matrix")
plt.xlabel("Feature Index")
plt.ylabel("Projection Vector Index")
plt.colorbar(label="Weight (1/-1)")
plt.show()
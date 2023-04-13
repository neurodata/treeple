import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from sktree.tree._morf_splitter import BestPatchSplitter

# %% Initialize patch

splitter = BestPatchSplitter()

# %% Generate 1D patches
arr_1d = np.arange(20)
start_1d = (5,)
patch_size_1d = (3,)
patch_1d = sample_patch(start_1d, patch_size_1d, arr_1d)

# Visualize 1D patch
plt.plot(arr_1d)
plt.plot(patch_1d, arr_1d[patch_1d], "o")
plt.title("1D Patch Visualization")
plt.show()

# %% Generate 2D patches
arr_2d = np.arange(25).reshape(5, 5)
start_2d = (1, 1)
patch_size_2d = (3, 3)
patch_2d = sample_patch(start_2d, patch_size_2d, arr_2d)

# Visualize 2D patch
plt.imshow(arr_2d)
plt.scatter(*zip(*np.argwhere(arr_2d)), color="w", marker=".")
plt.scatter(*zip(*np.argwhere(arr_2d)[patch_2d]), color="r")
plt.title("2D Patch Visualization")
plt.show()

# %% Generate 3D patches
arr_3d = np.arange(27).reshape(3, 3, 3)
start_3d = (1, 1, 1)
patch_size_3d = (3, 3, 3)
patch_3d = sample_patch(start_3d, patch_size_3d, arr_3d)

# Visualize 3D patch
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(*zip(*np.argwhere(arr_3d)), color="w", marker=".")
ax.scatter(*zip(*np.argwhere(arr_3d)[patch_3d]), color="r")
ax.set_title("3D Patch Visualization")
plt.show()

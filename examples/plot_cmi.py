"""
========================================================
Forest-based (Conditional) Mutual Information Estimators
========================================================

Mutual information (MI) and conditional mutual information (CMI) are
fundamentally important nonparametric measures of dependence and conditional
dependence. MI and CMI are useful in the analysis conditional independences (CI),
which are commonly used to determine trends and relationships in data. Therefore,
good, robust, efficient estimates of MI and CMI are very desirable.

However, estimating MI and CMI in general is very difficult, especially when there
is high-dimensionality in the data. In this example, we will show how a
forest-based approach estimate to MI and CMI is more robust than the traditional
KSG-estimator for MI and CMI (based on kNN statistics).

We will perform two simulation illustrating the robustness of using tree-based models
to estimate pairwise distances. For each simulation, we will generate data from
the data generating model and then augment the dataset with artificial number of
noise dimensions. 

For MI, this means in one, or both of the variables, ``X``, or ``Y``, the dimensionality
will be very high. For CMI, we will also additionally augment the conditioning variable
``Z``. In the simulations with exact analytical solutions to the MI and CMI, we will compare
with the ground-truth. In the simulations from a manifold, we will compare our estimates
to a lower-bound of the MI and CMI.

See :ref:`decision tree <tree>` for more information on the estimators.
"""

import numpy as np
import scipy
import scipy.spatial

from sktree.ensemble import pairwise_forest_distance
from sktree.experimental.mutual_info import (
    simulate_multivariate_gaussian,
    cmi_gaussian,
    mi_gaussian,
)

import matplotlib.pyplot as plt
import seaborn as sns


n_jobs = -1
x, y = np.mgrid[0:5, 2:8]
X = np.c_[x.ravel(), y.ravel()]

print(X)
k = 2
tree_xyz = scipy.spatial.cKDTree(X)
dists, indices = tree_xyz.query(X, k=[k + 1], p=np.inf, eps=0.0, workers=n_jobs)


# %%
# Generate Dataset Showing Robustness to High-Dimensionality
# ----------------------------------------------------------
# We will perform two simulation illustrating the robustness of using tree-based models
# to estimate pairwise distances. The first is a simulation of uniformly generated data
# with augmented noise dimensions.


# %%
# Augment dataset with noise terms


# %%
# Compare KSG estimator with Forest-based KSG estimator
# -----------------------------------------------------
# Since this is a simulation from known parametric distributions,
# we have the luxury of knowing the true analytical solution of the
# MI and CMI values. We will compare the KSG estimate

# %%
# Generate Dataset Showing Robustness to Manifold Structure
# ---------------------------------------------------------
# that follows a generate data from the manifold of a helix.


# %%
# Augment dataset with noise terms


# %%
# Compare KSG estimator with Forest-based KSG estimator
# -----------------------------------------------------
#

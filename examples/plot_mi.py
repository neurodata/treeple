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

from sktree.experimental.mutual_info import (
    simulate_multivariate_gaussian,
    simulate_helix,
    simulate_sphere,
    cmi_gaussian,
    mi_gaussian,
    mutual_info_ksg,
)

import matplotlib.pyplot as plt
import seaborn as sns

seed = 12345
n_samples = 5000
d = 10
d_noise = 90
n_jobs = -1
k = 2
rng = np.random.default_rng(seed)

# %%
# Generate Dataset Showing Robustness to High-Dimensionality
# ----------------------------------------------------------
# We will perform two simulation illustrating the robustness of using tree-based models
# to estimate pairwise distances. The first is a simulation of uniformly generated data
# with augmented noise dimensions.
data, mean, cov = simulate_multivariate_gaussian(d=d, n_samples=n_samples, seed=seed)

# compute the analytical solution for the MI
true_mi = mi_gaussian(cov)

# compute the analytical solution for the CMI
# true_cmi = cmi_gaussian(cov, x_index=x_index, y_index=y_index, z_index=z_index)

# %%
# Augment dataset with noise terms, so that data is (n_samples, dims + dims_noise)

data_noise = rng.standard_normal(size=(n_samples, d_noise))
data = np.concatenate((data, data_noise), axis=1)

# %%
# Compare KSG estimator with Forest-based KSG estimator
# -----------------------------------------------------
# Since this is a simulation from known parametric distributions,
# we have the luxury of knowing the true analytical solution of the
# MI and CMI values. We will compare the KSG estimate with that of the
# forest-KSG estimate.
ksg_mi = mutual_info_ksg(
    data[:, 0], data[:, 1:], metric="l2", algorithm="kd_tree", k=k, n_jobs=n_jobs, random_seed=seed
)

forestksg_mi = mutual_info_ksg(
    data[:, 0],
    data[:, 1:],
    metric="forest",
    algorithm="kd_tree",
    k=k,
    n_jobs=n_jobs,
    random_seed=seed,
)

ksg_error = ksg_mi - true_mi
forestksg_error = forestksg_mi - true_mi

print(f"Ground truth MI, KSG estimate and Forest-KSG estimate: {true_mi}, {ksg_mi}, {forestksg_mi}")
print(f"KSG MI estimate error: {ksg_error}")
print(f"Forest-KSG MI estimate error: {forestksg_error}")

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

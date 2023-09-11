"""
===============================
ExtendedIsolationForest example
===============================

An example using :class:`~sktree.ensemble.ExtendedIsolationForest` for anomaly
detection, which compares to the :class:`~sklearn.ensemble.IsolationForest`.

In the present example we demo two ways to visualize the decision boundary of an
Isolation Forest trained on a toy dataset.

"""

# %%
# Data generation
# ---------------
#
# We generate two clusters (each one containing `n_samples`) by randomly
# sampling the standard normal distribution as returned by
# :func:`numpy.random.randn`. One of them is spherical and the other one is
# slightly deformed.
#
# For consistency with the :class:`~sklearn.ensemble.IsolationForest` notation,
# the inliers (i.e. the gaussian clusters) are assigned a ground truth label `1`
# whereas the outliers (created with :func:`numpy.random.uniform`) are assigned
# the label `-1`.

import numpy as np
from sklearn.model_selection import train_test_split

n_samples, n_outliers = 120, 40
rng = np.random.RandomState(0)
covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate([np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
# We can visualize the resulting clusters:

import matplotlib.pyplot as plt

scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
handles, labels = scatter.legend_elements()
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.title("Gaussian inliers with \nuniformly distributed outliers")
plt.show()

# %%
# Training of the model
# ---------------------

from sklearn.ensemble import IsolationForest

from sktree.ensemble import ExtendedIsolationForest

extended_clf = ExtendedIsolationForest(max_samples=100, random_state=0)
extended_clf.fit(X_train)

clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)

# %%
# Plot discrete decision boundary
# -------------------------------
#
# We use the class :class:`~sklearn.inspection.DecisionBoundaryDisplay` to
# visualize a discrete decision boundary. The background color represents
# whether a sample in that given area is predicted to be an outlier
# or not. The scatter plot displays the true labels.

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

for model in [clf, extended_clf]:
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        alpha=0.5,
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
    plt.axis("square")
    plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
    plt.show()


# %%
# Plot path length decision boundary
# ----------------------------------
#
# By setting the `response_method="decision_function"`, the background of the
# :class:`~sklearn.inspection.DecisionBoundaryDisplay` represents the measure of
# normality of an observation. Such score is given by the path length averaged
# over a forest of random trees, which itself is given by the depth of the leaf
# (or equivalently the number of splits) required to isolate a given sample.
#
# When a forest of random trees collectively produce short path lengths for
# isolating some particular samples, they are highly likely to be anomalies and
# the measure of normality is close to `0`. Similarly, large paths correspond to
# values close to `1` and are more likely to be inliers.

for model in [clf, extended_clf]:
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="decision_function",
        alpha=0.5,
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
    plt.axis("square")
    plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
    plt.colorbar(disp.ax_.collections[1])
    plt.show()

# %%
# Generate Data
# Produce 2-D dataset with a sinusoidal shape and Gaussian noise added on top.

N = 1000
x = np.random.rand(N) * 8 * np.pi
y = np.sin(x) + np.random.randn(N) / 4.0
X = np.array([x, y]).T

fig = plt.figure(figsize=(6, 6))
fig.add_subplot(111)
plt.plot(X[:, 0], X[:, 1], "o", color=[0.5, 0.5, 0.5])

plt.xlim([-5, 30])
plt.ylim([-3.0, 3.0])

plt.show()

# %%

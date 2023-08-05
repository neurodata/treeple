"""
====================================================================
Plot the decision surfaces of ensembles of trees on the iris dataset
====================================================================

Plot the decision surfaces of forests of randomized trees trained on pairs of
features of the iris dataset.

This plot compares the decision surfaces learned by a decision tree classifier
(first column), by a oblique decision tree classifier (second column), by an extra-
oblique decision tree classifier (third column).

In the first row, the classifiers are built using the sepal width and
the sepal length features only, on the second row using the petal length and
sepal length only, and on the third row using the petal width and the
petal length only.

"""
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sktree.tree import ExtraObliqueDecisionTreeClassifier, ObliqueDecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
max_depth = 10
random_state = 12345

models = [
    DecisionTreeClassifier(max_depth=max_depth),
    ObliqueDecisionTreeClassifier(max_depth=max_depth),
    ExtraObliqueDecisionTreeClassifier(max_depth=max_depth),
]

cmap = plt.cm.Spectral
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.25  # step widths for coarse classifier guesses
# figure size for plotting
figure_size = (30, 30)
pairs = [[0, 1], [0, 2], [2, 3]]
N = len(pairs) * len(models)
plot_idx = 1

n_rows = 3
fig, ax = plt.subplots(n_rows, math.ceil(N / n_rows))
fig.set_size_inches(6 * N, 6)

# Load data
iris = load_iris()

for pair in pairs:
    for model in models:
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target
        # starting time
        t0 = datetime.now()

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(random_state)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][: -len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print(
            model_details + " with features",
            pair,
            "has a score of",
            round(scores, 5),
            "took",
            (datetime.now() - t0).total_seconds(),
            "seconds",
        )

        plt.subplot(3, 3, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if (
            isinstance(model, DecisionTreeClassifier)
            or isinstance(model, ObliqueDecisionTreeClassifier)
            or isinstance(model, ExtraObliqueDecisionTreeClassifier)
        ):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)

        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser),
        )
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(
            xx_coarser.shape
        )
        cs_points = plt.scatter(
            xx_coarser,
            yy_coarser,
            s=15,
            c=Z_points_coarser,
            cmap=cmap,
            edgecolors="none",
        )

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=ListedColormap(["r", "y", "b"]),
            edgecolor="k",
            s=20,
        )
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()

# Discussion
# ----------
# This section demonstrates the decision boundaries of the classification task with
# ObliqueDecisionTree and ExtraObliqueDecisionTree. While the decision boundaries are different
# for each tree, the performance are similar. However, the ExtraObliqueDecisionTree classifier
# runs faster with similar or better performance in some cases.

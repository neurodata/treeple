"""
===========================================
Comparison of Decision Tree and Honest Tree
===========================================

This example compares the :class:`treeple.tree.HonestTreeClassifier` from the
``treeple`` library with the :class:`sklearn.tree.DecisionTreeClassifier`
from scikit-learn on the Iris dataset.

Both classifiers are fitted on the same dataset and their decision trees
are plotted side by side.
"""

import matplotlib.pyplot as plt
from sklearn import config_context
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

from treeple.tree import HonestTreeClassifier

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize classifiers
honest_clf = HonestTreeClassifier(honest_method="prune", max_features="sqrt", random_state=0)
honest_noprune_clf = HonestTreeClassifier(
    honest_method="apply", max_features="sqrt", random_state=0
)
sklearn_clf = DecisionTreeClassifier(max_features="sqrt", random_state=0)

# Fit classifiers
honest_noprune_clf.fit(X, y)
honest_clf.fit(X, y)
sklearn_clf.fit(X, y)

# Plotting the trees
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# .. note:: We skip parameter validation because internally the `plot_tree`
#           function checks if the estimator is a DecisionTreeClassifier
#           instance from scikit-learn, but the ``HonestTreeClassifier`` is
#           a subclass of a forked version of the DecisionTreeClassifier.

# Plot HonestTreeClassifier tree
with config_context(skip_parameter_validation=True):
    plot_tree(honest_clf, filled=True, ax=axes[0])
axes[0].set_title("HonestTreeClassifier")

# Plot HonestTreeClassifier tree
with config_context(skip_parameter_validation=True):
    plot_tree(honest_noprune_clf, filled=False, ax=axes[1])
axes[1].set_title("HonestTreeClassifier (No pruning)")

# Plot scikit-learn DecisionTreeClassifier tree
plot_tree(sklearn_clf, filled=True, ax=axes[2])
axes[2].set_title("DecisionTreeClassifier")

plt.show()

# %%
# Discussion
# ----------
# The HonestTreeClassifier is a variant of the DecisionTreeClassifier that
# provides honest inference. The honest inference is achieved by splitting the
# dataset into two parts: the training set and the validation set. The training
# set is used to build the tree, while the validation set is used to fit the
# leaf nodes for posterior prediction. This results in calibrated posteriors
# (see :ref:`sphx_glr_auto_examples_calibration_plot_overlapping_gaussians`).
#
# Compared to the ``honest_prior='apply'`` method, the ``honest_prior='prune'``
# method builds a tree that will not contain empty leaves, and also leverages
# the validation set to check split conditions. Thus we see that the pruned
# honest tree is significantly smaller than the regular decision tree.

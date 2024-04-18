"""
================================
1-2: Calculating p-value (MIGHT)
================================
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

from sktree.datasets import make_trunk_classification
from sktree.ensemble import HonestForestClassifier
from sktree.stats import build_hyppo_oob_forest

# %%
# Independence Testing
# --------------------
#
# Given samples from ``X`` and ``Y``, the independent hypothesis and its
# alternative are stated as:
#
# .. math:: H_0 : F_{XY} = F_X F_Y
#
# .. math:: H_A : F_{XY} \neq F_X F_Y
#
# P-value is the probability of observing a statistic more extreme than the null.
# By computing the p-value using ``treeple``, we can test if :math:`H_0`
# would be rejected, which confirms that X and Y are not independent. The p-value is
# generated by comparing the observed statistic difference with permuted
# differences, using mutual information as an example.

# %%
# MI
# --
#
# Mutual Information (*MI*) measures the mutual dependence between *X* and
# *Y*. It can be calculated by the difference between the class entropy
# (``H(Y)``) and the conditional entropy (``H(Y | X)``):
#
# .. math:: I(X; Y) = H(Y) - H(Y\mid X)
#
# With a binary class simulation as an example, this tutorial will show
# how to use ``treeple`` to use the statistic and the p-value.

# %%
# Create a simulation with two gaussians
# --------------------------------------


# create a binary class simulation with two gaussians
# 500 samples for each class, class zero is standard
# gaussian, and class one has a mean at one
X, y = make_trunk_classification(
    n_samples=1000,
    n_dim=1,
    mu_0=0,
    mu_1=1,
    n_informative=1,
    seed=1,
)


# scatter plot the samples
plt.hist(X[:500], bins=15, alpha=0.6, color="blue", label="negative")
plt.hist(X[500:], bins=15, alpha=0.6, color="red", label="positive")
plt.legend()
plt.show()

# %%
# Generate observed posteriors
# ----------------------------
#
# The observed posteriors represent the original distribution of ``X`` and ``Y``.


# initialize the forest with 100 trees
est = HonestForestClassifier(
    n_estimators=100,
    max_samples=1.6,
    max_features=0.3,
    bootstrap=True,
    stratify=True,
    random_state=1,
)

# fit the model and obtain the tree posteriors
_, observe_proba_tree = build_hyppo_oob_forest(est, X, y)

# generate forest posteriors for the two classes
observe_proba = np.nanmean(observe_proba_tree, axis=0)


# scatter plot the posterior probabilities for class one
plt.hist(observe_proba[:500][:, 1], bins=30, alpha=0.6, color="blue", label="negative")
plt.hist(observe_proba[500:][:, 1], bins=30, alpha=0.6, color="red", label="positive")
plt.legend()
plt.show()

# %%
# Generate null posteriors
# ------------------------


# shuffle the labels
X_null = np.copy(X)
y_null = np.copy(y)
np.random.shuffle(y_null)

# initialize another forest with 100 trees
est_null = HonestForestClassifier(
    n_estimators=100,
    max_samples=1.6,
    max_features=0.3,
    bootstrap=True,
    stratify=True,
    random_state=1,
)

# fit the model and obtain the tree posteriors
_, null_proba_tree = build_hyppo_oob_forest(est, X_null, y_null)

# generate forest posteriors for the two classes
null_proba = np.nanmean(null_proba_tree, axis=0)


# scatter plot the posterior probabilities for class one
plt.hist(null_proba[:500][:, 1], bins=30, alpha=0.6, color="blue", label="negative")
plt.hist(null_proba[500:][:, 1], bins=30, alpha=0.6, color="red", label="positive")
plt.legend()
plt.show()

# %%
# Find the observed statistic difference
# --------------------------------------


def Calculate_MI(y_true, y_pred_proba):
    # calculate the conditional entropy
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))

    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    # calculate the entropy of labels
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


mi = Calculate_MI(y, observe_proba)
mi_null = Calculate_MI(y, null_proba)

observed_diff = mi - mi_null
print("Observed statistic difference =", round(observed_diff, 2))

# %%
# Permute the trees
# -----------------
#
# In this case, permuting the tree posteriors is equivalent to permuting
# the trees in the two forests.


PERMUTE = 10000
mix_diff = []

# Collect all the tree posteriors
proba = np.vstack((observe_proba_tree, null_proba_tree))
for i in range(PERMUTE):

    # permute the posteriors
    np.random.shuffle(proba)

    # calculate the statistic for
    # the two mixed forest posteriors
    mi_mix_one = Calculate_MI(y, np.nanmean(proba[:100], axis=0))
    mi_mix_two = Calculate_MI(y, np.nanmean(proba[100:], axis=0))
    mix_diff.append(mi_mix_one - mi_mix_two)

# %%
# Calculate the p-value
# ---------------------


pvalue = (1 + (mix_diff >= observed_diff).sum()) / (1 + PERMUTE)

print("p-value is:", round(pvalue, 2))
if pvalue < 0.05:
    print("The null hypothesis is rejected.")
else:
    print("The null hypothesis is not rejected.")

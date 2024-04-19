"""
======================================================
2-2: Calculating p-value with multiview data (CoMIGHT)
======================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

from sktree.datasets import make_trunk_classification
from sktree.ensemble import HonestForestClassifier
from sktree.stats import build_hyppo_oob_forest
from sktree.tree import MultiViewDecisionTreeClassifier

sns.set(color_codes=True, style="white", context="talk", font_scale=1.5)
PALETTE = sns.color_palette("Set1")
sns.set_palette(PALETTE[1:5] + PALETTE[6:], n_colors=9)
sns.set_style("white", {"axes.edgecolor": "#dddddd"})
# %%
# Independence Testing
# --------------------
#
# Given samples from ``X``, ``Z``, and ``Y``, the independent hypothesis
# and its alternative are stated as:
#
# .. math:: H_0 : F_{X,Y|Z} = F_{X|Z} F_{Y|Z}
#
# .. math:: H_A : F_{X,Y|Z} \neq F_{X|Z} F_{Y|Z}
#
# By computing the p-value using ``treeple``, we can test if :math:`H_0`
# would be rejected, which confirms that ``X|Z`` and ``Y|Z`` are not independent.
# The p-value is generated by comparing the observed
# statistic difference with permuted differences, using conditional mutual
# information as a test statistic in this example.

# %%
# CMI
# ---
#
# Conditional mutual information (*CMI*) measures the dependence of *Y* on
# *X* conditioned on *Z*. It can be calculated by the difference between
# the joint MI (``I([X, Z]; Y)``) and the MI on Z (``I(Y; Z)``):
#
# .. math:: I(X; Y | Z) = I([X, Z]; Y) - I(Y; Z)
#
# With a multiview binary class simulation as an example, this tutorial
# will show how to use ``treeple`` to calculate the statistic and test the
# hypothesis with multiview data.

# %%
# Create a simulation with two gaussians
# --------------------------------------


# create a binary class simulation with two gaussians
# 500 samples for each class, class zero is standard
# gaussian, and class one has a mean at one for Z
Z, y = make_trunk_classification(
    n_samples=1000,
    n_dim=1,
    mu_0=0,
    mu_1=1,
    n_informative=1,
    seed=1,
)


X, y = make_trunk_classification(
    n_samples=1000,
    n_dim=1,
    mu_0=0,
    mu_1=2,
    n_informative=1,
    seed=2,
)

Z_X = np.hstack((Z, X))


Z_X_y = np.hstack((Z_X, y.reshape(-1, 1)))
Z_X_y = pd.DataFrame(Z_X_y, columns=["Z", "X", "y"])
Z_X_y = Z_X_y.replace({"y": 0.0}, "Class Zero")
Z_X_y = Z_X_y.replace({"y": 1.0}, "Class One")

fig, ax = plt.subplots(figsize=(5, 5))
ax.tick_params(labelsize=15)
sns.scatterplot(data=Z_X_y, x="Z", y="X", hue="y", palette=PALETTE[:2], alpha=0.2)
sns.kdeplot(data=Z_X_y, x="Z", y="X", hue="y", palette=PALETTE[:2], alpha=0.6)
ax.set_ylabel("X", fontsize=15)
ax.set_xlabel("Z", fontsize=15)
plt.legend(frameon=False, fontsize=15)

# %%
# Generate observed posteriors with X and Z
# -----------------------------------------


# initialize the forest with 100 trees
est = HonestForestClassifier(
    n_estimators=100,
    max_samples=1.6,
    max_features=0.3,
    bootstrap=True,
    stratify=True,
    tree_estimator=MultiViewDecisionTreeClassifier(),
    random_state=1,
)

# fit the model and obtain the tree posteriors
_, observe_proba_tree = build_hyppo_oob_forest(est, Z_X, y)

# generate forest posteriors for the two classes
observe_proba = np.nanmean(observe_proba_tree, axis=0)


fig, ax = plt.subplots(figsize=(5, 5))
ax.tick_params(labelsize=15)

# histogram plot the posterior probabilities for class one
ax.hist(observe_proba[:500][:, 1], bins=50, alpha=0.6, color=PALETTE[1], label="negative")
ax.hist(observe_proba[500:][:, 1], bins=50, alpha=0.3, color=PALETTE[0], label="positive")
ax.set_ylabel("# of Samples", fontsize=15)
ax.set_xlabel("Class One Posterior", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

#
# Generate observed posteriors with Z only
# ----------------------------------------


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
_, single_proba_tree = build_hyppo_oob_forest(est, Z, y)

# generate forest posteriors for the two classes
single_proba = np.nanmean(single_proba_tree, axis=0)


fig, ax = plt.subplots(figsize=(5, 5))
ax.tick_params(labelsize=15)

# histogram plot the posterior probabilities for class one
ax.hist(single_proba[:500][:, 1], bins=50, alpha=0.6, color=PALETTE[1], label="negative")
ax.hist(single_proba[500:][:, 1], bins=50, alpha=0.3, color=PALETTE[0], label="positive")
ax.set_ylabel("# of Samples", fontsize=15)
ax.set_xlabel("Class One Posterior", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

#
# Generate null posteriors with Z and permuted X
# ----------------------------------------------


# shuffle the labels
X_null = np.copy(X)
np.random.shuffle(X_null)

# initialize another forest with 100 trees
est_null = HonestForestClassifier(
    n_estimators=100,
    max_samples=1.6,
    max_features=0.3,
    bootstrap=True,
    stratify=True,
    tree_estimator=MultiViewDecisionTreeClassifier(),
    random_state=1,
)

# fit the model and obtain the tree posteriors
_, null_proba_tree = build_hyppo_oob_forest(est, np.hstack((Z, X_null)), y)

# generate forest posteriors for the two classes
null_proba = np.nanmean(null_proba_tree, axis=0)


fig, ax = plt.subplots(figsize=(5, 5))
ax.tick_params(labelsize=15)

# histogram plot the posterior probabilities for class one
ax.hist(null_proba[:500][:, 1], bins=50, alpha=0.6, color=PALETTE[1], label="negative")
ax.hist(null_proba[500:][:, 1], bins=50, alpha=0.3, color=PALETTE[0], label="positive")
ax.set_ylabel("# of Samples", fontsize=15)
ax.set_xlabel("Class One Posterior", fontsize=15)
plt.legend(frameon=False, fontsize=15)
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


joint_mi = Calculate_MI(y, observe_proba)
mi = Calculate_MI(y, single_proba)
cmi = joint_mi - mi

joint_mi_null = Calculate_MI(y, null_proba)
cmi_null = joint_mi_null - mi

observed_diff = cmi - cmi_null
print("Observed conditional mutual information difference =", round(observed_diff, 2))

# %%
# Permute the trees
# -----------------


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

    # same difference of joint MI and CMI
    mix_diff.append(mi_mix_one - mi_mix_two)


# %%
# Calculate the p-value
# ---------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.tick_params(labelsize=15)

# histogram plot the statistic differences
ax.hist(mix_diff, bins=50, alpha=0.6, color=PALETTE[1], label="null")
ax.axvline(x=observed_diff, color=PALETTE[0], linestyle="--", label="observed")
ax.set_xlabel("Conditional Mutual Information Diff", fontsize=15)
ax.set_ylabel("# of Samples", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

pvalue = (1 + (mix_diff >= observed_diff).sum()) / (1 + PERMUTE)

print("p-value is:", round(pvalue, 2))
if pvalue < 0.05:
    print("The null hypothesis is rejected.")
else:
    print("The null hypothesis is not rejected.")
# sphinx_gallery_thumbnail_number = -1

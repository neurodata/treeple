"""
====================================
Calculating S@98 with multiview data
====================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from treeple.datasets import make_trunk_classification
from treeple.ensemble import HonestForestClassifier
from treeple.stats import build_oob_forest
from treeple.tree import MultiViewDecisionTreeClassifier

sns.set(color_codes=True, style="white", context="talk", font_scale=1.5)
PALETTE = sns.color_palette("Set1")
sns.set_palette(PALETTE[1:5] + PALETTE[6:], n_colors=9)
sns.set_style("white", {"axes.edgecolor": "#dddddd"})
# %%
# S@98 with multiview data
# ------------------------
#
# Sensitivity at 98% specificity (*S@98*) measures, namely, the true
# positive rate (*TPR*) when the false positive rate (*FPR*) is at 98%.
#
# .. math:: S@r = \mathbb{P}[\eta(X) > T_r \mid Y=1]
#
# With a multiview binary class simulation as an example, this tutorial
# will show how to use ``treeple`` to calculate the statistic with
# multiview data. For data with a single feature set, you can check out
# the simpler S@98 tutorial.

# %%
# Create a two-dimensional simulation with gaussians
# --------------------------------------------------


# create a binary class simulation with two gaussians
# 500 samples for each class, class zero is standard
# gaussian, and class one has a mean at one
Z, y = make_trunk_classification(
    n_samples=1000,
    n_dim=1,
    mu_0=0,
    mu_1=1,
    n_informative=1,
    seed=1,
)

# class one has a mean at two for X
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

fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
ax.tick_params(labelsize=15)
sns.scatterplot(data=Z_X_y, x="Z", y="X", hue="y", palette=PALETTE[:2][::-1], alpha=0.2)
sns.kdeplot(data=Z_X_y, x="Z", y="X", hue="y", palette=PALETTE[:2][::-1], alpha=0.6)
ax.set_ylabel("Variable Two", fontsize=15)
ax.set_xlabel("Variable One", fontsize=15)
plt.legend(frameon=False, fontsize=15)


# %%
# Fit the model with X and Z
# --------------------------


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
_, observe_proba = build_oob_forest(est, Z_X, y)

# generate forest posteriors for the two classes
observe_proba = np.nanmean(observe_proba, axis=0)


fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
ax.tick_params(labelsize=15)

# histogram plot the posterior probabilities for class one
ax.hist(observe_proba[:500][:, 1], bins=50, alpha=0.6, color=PALETTE[1], label="negative")
ax.hist(observe_proba[500:][:, 1], bins=50, alpha=0.3, color=PALETTE[0], label="positive")
ax.set_ylabel("# of Samples", fontsize=15)
ax.set_xlabel("Class One Posterior", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

# %%
# Calculate the statistic
# -----------------------


def Calculate_SA(y_true, y_pred_proba, max_fpr=0.02) -> float:
    """Calculate the sensitivity at a specific specificity"""
    # check the shape of true labels
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # find the positive class and calculate fpr and tpr
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    sa98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    ax.tick_params(labelsize=15)
    ax.set_xlim([-0.005, 1.005])
    ax.set_ylim([-0.005, 1.005])
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)

    ax.plot(fpr, tpr, label="ROC curve", color=PALETTE[1])

    spec = int((1 - max_fpr) * 100)
    ax.axvline(
        x=max_fpr,
        color=PALETTE[0],
        ymin=0,
        ymax=sa98,
        label="S@" + str(spec) + " = " + str(round(sa98, 2)),
        linestyle="--",
    )
    ax.axhline(y=sa98, xmin=0, xmax=max_fpr, color="r", linestyle="--")
    ax.legend(frameon=False, fontsize=15)

    return sa98


sa98 = Calculate_SA(y, observe_proba, max_fpr=0.02)
print("S@98 =", round(sa98, 2))
# sphinx_gallery_thumbnail_number = -1

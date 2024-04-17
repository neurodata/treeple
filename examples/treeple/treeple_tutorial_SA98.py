"""
=====================================
Treeple tutorial for calculating S@98
=====================================
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, roc_curve

from sktree.datasets import make_trunk_classification
from sktree.ensemble import HonestForestClassifier
from sktree.stats import build_hyppo_oob_forest

# %%
# S@98
# ----
#
# Sensitivity at 98% specificity (*S@98*) measures, namely, the true
# positive rate (*TPR*) when the false positive rate (*FPR*) is at 98%.
#
# With a binary class simulation as an example, this tutorial will show
# how to use ``treeple`` to calculate the statistic.

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
# Fit the model
# -------------


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
_, observe_proba = build_hyppo_oob_forest(est, X, y)

# generate forest posteriors for the two classes
observe_proba = np.nanmean(observe_proba, axis=0)


# scatter plot the posterior probabilities for class one
plt.hist(observe_proba[:500][:, 1], bins=30, alpha=0.6, color="blue", label="negative")
plt.hist(observe_proba[500:][:, 1], bins=30, alpha=0.6, color="red", label="positive")
plt.legend()
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
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label="ROC Curve")

    spec = int((1 - max_fpr) * 100)
    plt.axvline(
        x=max_fpr,
        color="r",
        ymin=0,
        ymax=sa98,
        label="S@" + str(spec) + " = " + str(round(sa98, 2)),
        linestyle="--",
    )
    plt.axhline(y=sa98, xmin=0, xmax=max_fpr, color="r", linestyle="--")
    plt.legend()

    return sa98


sa98 = Calculate_SA(y, observe_proba, max_fpr=0.02)
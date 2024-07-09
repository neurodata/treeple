"""
=======================================
Estimating true posteriors & statistics
=======================================
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy, multivariate_normal
from sklearn.metrics import roc_auc_score, roc_curve

from treeple.datasets import make_trunk_mixture_classification

sns.set(color_codes=True, style="white", context="talk", font_scale=1.5)
PALETTE = sns.color_palette("Set1")
sns.set_palette(PALETTE[1:5] + PALETTE[6:], n_colors=9)
sns.set_style("white", {"axes.edgecolor": "#dddddd"})
warnings.filterwarnings("ignore")


# %%
# True posterior estimation
# -------------------------
#
# As we know the true priors of each class, we can generate a sufficient
# amount of samples to estimate the true posteriors and corresponding
# statistics like *MI*, *pAUC*, and *S@98*.

# %%
# Generate gaussian mixture simulations
# -------------------------------------


# set the simulation parameters and generate samples
X, y = make_trunk_mixture_classification(
    n_samples=10000,
    n_dim=1,
    n_informative=1,
    mu_0=0,
    mu_1=5,
    mix=0.75,
    seed=1,
)


fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
ax.tick_params(labelsize=15)

# histogram plot the samples
ax.hist(X[:5000], bins=50, alpha=0.6, color=PALETTE[1], label="negative")
ax.hist(X[5000:], bins=50, alpha=0.3, color=PALETTE[0], label="positive")
ax.set_xlabel("Variable One", fontsize=15)
ax.set_ylabel("Likelihood", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

# %%
# Calculate X priors with true pdfs
# ---------------------------------
#
# .. math:: f_{X}(x) = f_{X  \mid Y = 0}(x)\mathbb{P}(Y = 0) + f_{X  \mid Y = 1}(x)\mathbb{P}(Y = 1)


# calculate pdf for class zero
pdf_class0 = multivariate_normal.pdf(X, mean=0)

# calculate pdf for each component of class one
pdf_class1_0 = multivariate_normal.pdf(X, mean=0)
pdf_class1_1 = multivariate_normal.pdf(X, mean=5)

# combine the class one pdfs
pdf_class1 = 0.75 * pdf_class1_0 + 0.25 * pdf_class1_1


# Y prior is 0.5 for balanced class
p_x = pdf_class0 * 0.5 + pdf_class1 * 0.5

# %%
# Calculate true posteriors
# -------------------------
#
# .. math:: \mathbb{P}(Y  \mid X = x) = \frac{f_{X  \mid Y}(x)\mathbb{P}(Y )}{f_{X}(x)}


pos_class0 = pdf_class0 * 0.5 / p_x
pos_class1 = pdf_class1 * 0.5 / p_x

pos = np.hstack((pos_class0.reshape(-1, 1), pos_class1.reshape(-1, 1)))

# %%
# Generate true statistic estimates: S@98
# ---------------------------------------


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


sa98 = Calculate_SA(y, pos, max_fpr=0.02)
print("S@98 =", round(sa98, 2))

# %%
# Generate true statistic estimates: MI
# -------------------------------------


def Calculate_MI(y_true, y_pred_proba):
    # calculate the conditional entropy
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))

    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    # calculate the entropy of labels
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


mi = Calculate_MI(y, pos)
print("MI =", round(mi, 2))


# %%
# Generate true statistic estimates: pAUC@90
# ------------------------------------------


def Calculate_pAUC(y_true, y_pred_proba, max_fpr=0.1) -> float:
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

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    ax.tick_params(labelsize=15)
    ax.set_xlim([-0.005, 1.005])
    ax.set_ylim([-0.005, 1.005])
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)

    ax.plot(fpr, tpr, label="ROC curve", color=PALETTE[1])
    # Calculate pAUC at the specific threshold
    pAUC = roc_auc_score(y_true, y_pred_proba[:, 1], max_fpr=max_fpr)

    pos = np.where(fpr == max_fpr)[0][-1]
    ax.fill_between(
        fpr[:pos],
        tpr[:pos],
        color=PALETTE[0],
        alpha=0.6,
        label="pAUC@90 = " + str(round(pAUC, 2)),
        linestyle="--",
    )
    ax.legend(frameon=False, fontsize=15)
    return pAUC


pAUC = Calculate_pAUC(y, pos, max_fpr=0.1)
print("pAUC@90 =", round(pAUC, 2))

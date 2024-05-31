"""
===========================
Calculating p-value (MIGHT)
===========================
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sktree.datasets import make_trunk_classification
from sktree.ensemble import HonestForestClassifier
from sktree.stats import build_coleman_forest

sns.set(color_codes=True, style="white", context="talk", font_scale=1.5)
PALETTE = sns.color_palette("Set1")
sns.set_palette(PALETTE[1:5] + PALETTE[6:], n_colors=9)
sns.set_style("white", {"axes.edgecolor": "#dddddd"})

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
# Under the null hypothesis :math:`H_0`, the conditional entropy ``H(Y | X)``
# is equal to the class entropy ``H(Y)``, so the *MI* becomes zero. Thus, if
# the *MI* is significantly larger than zero, we can reject the null hypothesis
# :math:`H_0`.
#
# With a binary class simulation as an example, this tutorial will show
# how to use ``treeple`` to calculate the statistic and test the
# hypothesis with data.

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

fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
ax.tick_params(labelsize=15)

# histogram plot the samples
ax.hist(X[:500], bins=50, alpha=0.6, color=PALETTE[1], label="negative")
ax.hist(X[500:], bins=50, alpha=0.3, color=PALETTE[0], label="positive")
ax.set_xlabel("Variable One", fontsize=15)
ax.set_ylabel("Likelihood", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

# %%
# Fit the models and calculate the p-value
# ----------------------------------------
#
# Construct two forests, one trained with original data,
# and the other trained with permuted data. The test randomly
# permutes the two forests for a specified number of times.
#
# Each pair of forest outputs a set of mutual information statistics,
# and the statistic differences are used to calculate the p-vale.


# initialize the forest with 100 trees
est = HonestForestClassifier(
    n_estimators=100,
    max_samples=1.6,
    max_features=0.3,
    bootstrap=True,
    stratify=True,
    random_state=1,
)

# initialize another forest with 100 trees
est_null = HonestForestClassifier(
    n_estimators=100,
    max_samples=1.6,
    max_features=0.3,
    bootstrap=True,
    stratify=True,
    random_state=1,
)

PERMUTE = 10000

# conduct the hypothesis test with mutual information
observed_diff, _, _, pvalue, mix_diff = build_coleman_forest(
    est, est_null, X, y, metric="mi", n_repeats=PERMUTE, return_posteriors=False, seed=1
)

fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
ax.tick_params(labelsize=15)

# histogram plot the statistic differences
ax.hist(mix_diff, bins=50, alpha=0.6, color=PALETTE[1], label="null")
ax.axvline(x=observed_diff, color=PALETTE[0], linestyle="--", label="observed")
ax.set_xlabel("Mutual Information Diff", fontsize=15)
ax.set_ylabel("# of Samples", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.show()

print("p-value is:", round(pvalue, 2))
if pvalue < 0.05:
    print("The null hypothesis is rejected.")
else:
    print("The null hypothesis is not rejected.")
# sphinx_gallery_thumbnail_number = -1

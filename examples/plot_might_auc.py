"""
===================================================================================
Compute partial AUC using Mutual Information for Genuine Hypothesis Testing (MIGHT)
===================================================================================

An example using :class:`~sktree.stats.FeatureImportanceForestClassifier` for nonparametric
multivariate hypothesis test, on simulated datasets. Here, we present a simulation
of how MIGHT is used to evaluate how a "feature set is important for predicting the target".

We simulate a dataset with 1000 features, 500 samples, and a binary class target
variable. Within each feature set, there is 500 features associated with one feature
set, and another 500 features associated with another feature set. One could think of
these for example as different datasets collected on the same patient in a biomedical setting.
The first feature set (X) is strongly correlated with the target, and the second
feature set (W) is weakly correlated with the target (y).

We then use MIGHT to calculate the partial AUC of these sets.
"""

import numpy as np
from scipy.special import expit

from sktree import HonestForestClassifier
from sktree.stats import FeatureImportanceForestClassifier
from sktree.tree import DecisionTreeClassifier

seed = 12345
rng = np.random.default_rng(seed)

# %%
# Simulate data
# -------------
# We simulate the two feature sets, and the target variable. We then combine them
# into a single dataset to perform hypothesis testing.

n_samples = 1000
n_features_set = 500
mean = 1.0
sigma = 2.0
beta = 5.0

unimportant_mean = 0.0
unimportant_sigma = 4.5

# first sample the informative features, and then the uniformative features
X_important = rng.normal(loc=mean, scale=sigma, size=(n_samples, 10))
X_important = np.hstack(
    [
        X_important,
        rng.normal(
            loc=unimportant_mean, scale=unimportant_sigma, size=(n_samples, n_features_set - 10)
        ),
    ]
)

X_unimportant = rng.normal(
    loc=unimportant_mean, scale=unimportant_sigma, size=(n_samples, n_features_set)
)

# simulate the binary target variable
y = rng.binomial(n=1, p=expit(beta * X_important[:, :10].sum(axis=1)), size=n_samples)

# %%
# Use partial AUC as test statistic
# ---------------------------------
# You can specify the maximum specificity by modifying ``max_fpr`` in ``statistic``.

n_estimators = 125
max_features = "sqrt"
metric = "auc"
test_size = 0.2
n_jobs = -1
honest_fraction = 0.7
max_fpr = 0.1

est = FeatureImportanceForestClassifier(
    estimator=HonestForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        tree_estimator=DecisionTreeClassifier(),
        random_state=seed,
        honest_fraction=honest_fraction,
        n_jobs=n_jobs,
    ),
    random_state=seed,
    test_size=test_size,
    permute_per_tree=True,
    sample_dataset_per_tree=True,
)

# we test for the first feature set, which is important and thus should return a higher AUC
stat, posterior_arr, samples = est.statistic(
    X_important,
    y,
    metric=metric,
    return_posteriors=True,
)

print(f"ASH-90 / Partial AUC: {stat}")
print(f"Shape of Observed Samples: {samples.shape}")
print(f"Shape of Tree Posteriors for the positive class: {posterior_arr.shape}")

# %%
# Repeat for the second feature set
# ---------------------------------
# This feature set has a smaller statistic, which is expected due to its weak correlation.

stat, posterior_arr, samples = est.statistic(
    X_unimportant,
    y,
    metric=metric,
    return_posteriors=True,
)

print(f"ASH-90 / Partial AUC: {stat}")
print(f"Shape of Observed Samples: {samples.shape}")
print(f"Shape of Tree Posteriors for the positive class: {posterior_arr.shape}")

# %%
# All posteriors are saved within the model
# -----------------------------------------
# Extract the results from the model variables anytime. You can save the model with ``pickle``.
#
# ASH-90 / Partial AUC: ``est.observe_stat_``
# Observed Samples: ``est.observe_samples_``
# Tree Posteriors for the positive class: ``est.observe_posteriors_`` (n_trees, n_samples_test, 1)
# True Labels: ``est.y_true_final_``

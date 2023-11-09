"""
==============================================================================
Mutual Information for Genuine Hypothesis Testing (MIGHT) with Imbalanced Data
==============================================================================

Here, we demonstrate how to do hypothesis testing on highly imbalanced data
in terms of their feature-set dimensionalities.
using mutual information as a test statistic. We use the framework of
:footcite:`coleman2022scalable` to estimate pvalues efficiently.

Here, we simulate two feature sets, one of which is important for the target,
but significantly smaller in dimensionality than the other feature set, which
is unimportant for the target. We then use the MIGHT framework to test for
the importance of each feature set. Instead of leveraging a normal honest random
forest to estimate the posteriors, here we leverage a multi-view honest random
forest, with knowledge of the multi-view structure of the ``X`` data.

For other examples of hypothesis testing, see the following:

- :ref:`sphx_glr_auto_examples_hypothesis_testing_plot_MI_genuine_hypothesis_testing_forest.py`
- :ref:`sphx_glr_auto_examples_hypothesis_testing_plot_might_auc.py`

For more information on the multi-view decision-tree, see
:ref:`sphx_glr_auto_examples_multiview_plot_multiview_dtc.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from sktree import HonestForestClassifier
from sktree.stats import FeatureImportanceForestClassifier
from sktree.tree import DecisionTreeClassifier, MultiViewDecisionTreeClassifier

# %%
# Simulate data
# -------------
# We simulate the two feature sets, and the target variable. We then combine them
# into a single dataset to perform hypothesis testing.
#
# Our data will follow the following graphical model:
#
# $(X_1 \rightarrow Y; X_2)$
#
# where $X_1$ is our signal feature set, $X_2$ is our noise feature set, and $Y$ is our target.
# $X_1$ will be low-dimensional, but $X_2$ is high-dimensional noise.

seed = 12345
rng = np.random.default_rng(seed)


def make_multiview_classification(
    n_samples=100, n_features_1=10, n_features_2=1000, cluster_std=2.0, seed=None
):
    rng = np.random.default_rng(seed=seed)

    # Create a high-dimensional multiview dataset with a low-dimensional informative
    # subspace in one view of the dataset.
    X0_first, y0 = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        n_features=n_features_1,
        random_state=rng.integers(1, 10000),
        centers=1,
        center_box=(-2, 2.0),
    )

    X1_first, y1 = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        n_features=n_features_1,
        random_state=rng.integers(1, 10000),
        centers=1,
        center_box=(-2.0, 2.0),
    )

    y1[:] = 1

    # add the second view for y=0 and y=1, which is completely noise
    X0 = np.concatenate([X0_first, rng.standard_normal(size=(n_samples, n_features_2))], axis=1)
    X1 = np.concatenate([X1_first, rng.standard_normal(size=(n_samples, n_features_2))], axis=1)

    # combine the views and targets
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1)).T

    # add noise to the data
    X = X + rng.standard_normal(size=X.shape)

    return X, y


n_samples = 100
n_features_1 = 10
n_features_2 = 10_000
n_features_views = [n_features_1, n_features_1 + n_features_2]

X, y = make_multiview_classification(
    n_samples=n_samples,
    n_features_1=n_features_1,
    n_features_2=n_features_2,
    cluster_std=5.0,
    seed=seed,
)

print(X.shape, y.shape, n_features_views)
# %%
# Perform hypothesis testing using Mutual Information
# ---------------------------------------------------
# Here, we use :class:`~sktree.stats.FeatureImportanceForestClassifier` to perform the hypothesis
# test. The test statistic is computed by comparing the metric (i.e. mutual information) estimated
# between two forests. One forest is trained on the original dataset, and one forest is trained
# on a permuted dataset, where the rows of the ``covariate_index`` columns are shuffled randomly.
#
# The null distribution is then estimated in an efficient manner using the framework of
# :footcite:`coleman2022scalable`. The sample evaluations of each forest (i.e. the posteriors)
# are sampled randomly ``n_repeats`` times to generate a null distribution. The pvalue is then
# computed as the proportion of samples in the null distribution that are less than the
# observed test statistic.

n_estimators = 100
max_features = "sqrt"
test_size = 0.2
n_repeats = 1000
n_jobs = -1

est = FeatureImportanceForestClassifier(
    estimator=HonestForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        tree_estimator=MultiViewDecisionTreeClassifier(
            feature_set_ends=n_features_views,
            apply_max_features_per_feature_set=True,
        ),
        random_state=seed,
        honest_fraction=0.5,
        n_jobs=n_jobs,
    ),
    random_state=seed,
    test_size=test_size,
)

mv_results = dict()

# we test for the overall MI of X vs y
stat, pvalue = est.test(X, y, metric="mi", n_repeats=n_repeats)
mv_results["feature_stat"] = stat
mv_results["feature_pvalue"] = pvalue
print(f"Estimated MI difference: {stat} with Pvalue: {pvalue}")

# we test for the first feature set, which is important and thus should return a pvalue < 0.05
stat, pvalue = est.test(
    X, y, covariate_index=np.arange(n_features_1, dtype=int), metric="mi", n_repeats=n_repeats
)
mv_results["important_feature_stat"] = stat
mv_results["important_feature_pvalue"] = pvalue
print(f"Estimated MI difference: {stat} with Pvalue: {pvalue}")

# we test for the second feature set, which is unimportant and thus should return a pvalue > 0.05
stat, pvalue = est.test(
    X,
    y,
    covariate_index=np.arange(n_features_1, n_features_2, dtype=int),
    metric="mi",
    n_repeats=n_repeats,
)
mv_results["unimportant_feature_stat"] = stat
mv_results["unimportant_feature_pvalue"] = pvalue
print(f"Estimated MI difference: {stat} with Pvalue: {pvalue}")

# %%
# Let's investigate what happens when we do not use a multi-view decision tree.
# All other parameters are kept the same.

# to ensure max-features is the same across the two models
max_features = int(np.sqrt(n_features_1) + np.sqrt(n_features_2))

est = FeatureImportanceForestClassifier(
    estimator=HonestForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        tree_estimator=DecisionTreeClassifier(),
        random_state=seed,
        honest_fraction=0.5,
        n_jobs=n_jobs,
    ),
    random_state=seed,
    test_size=test_size,
)

rf_results = dict()

# we test for the overall MI of X vs y
stat, pvalue = est.test(X, y, metric="mi", n_repeats=n_repeats)
rf_results["feature_stat"] = stat
rf_results["feature_pvalue"] = pvalue
print("\n\nAnalyzing regular decision tree models:")
print(f"Estimated MI difference using regular decision-trees: {stat} with Pvalue: {pvalue}")

# we test for the first feature set, which is important and thus should return a pvalue < 0.05
stat, pvalue = est.test(
    X, y, covariate_index=np.arange(n_features_1, dtype=int), metric="mi", n_repeats=n_repeats
)
rf_results["important_feature_stat"] = stat
rf_results["important_feature_pvalue"] = pvalue
print(f"Estimated MI difference using regular decision-trees: {stat} with Pvalue: {pvalue}")

# we test for the second feature set, which is unimportant and thus should return a pvalue > 0.05
stat, pvalue = est.test(
    X,
    y,
    covariate_index=np.arange(n_features_1, n_features_2, dtype=int),
    metric="mi",
    n_repeats=n_repeats,
)
rf_results["unimportant_feature_stat"] = stat
rf_results["unimportant_feature_pvalue"] = pvalue
print(f"Estimated MI difference using regular decision-trees: {stat} with Pvalue: {pvalue}")

fig, ax = plt.subplots(figsize=(5, 3))

# plot pvalues
ax.bar(0, rf_results["important_feature_pvalue"], label="Permuting $X_1$ (RF)")
ax.bar(1, rf_results["unimportant_feature_pvalue"], label="Permuting $X_2$ (RF)")
ax.bar(2, mv_results["important_feature_pvalue"], label="Permuting $X_1$ (MV)")
ax.bar(3, mv_results["unimportant_feature_pvalue"], label="Permuting $X_2$ (MV)")
ax.bar(4, mv_results["feature_pvalue"], label="Overall Feature Set (MV)")
ax.bar(5, rf_results["feature_pvalue"], label="Overall Feature Set (RF)")
ax.axhline(0.05, color="k", linestyle="--", label="alpha=0.05")
ax.set(ylabel="Log10(PValue)", xlim=[-0.5, 5.5], yscale="log")
ax.legend()

fig.tight_layout()
plt.show()

# %%
# Discussion
# ----------
# We see that the multi-view decision tree is able to detect the important feature set,
# while the regular decision tree is not. This is because the regular decision tree
# is not aware of the multi-view structure of the data, and thus is challenged
# by the imbalanced dimensionality of the feature sets. I.e. it rarely splits on
# the first low-dimensional feature set, and thus is unable to detect its importance.
#
# Note both approaches still fail to reject the null hypothesis (for alpha of 0.05)
# when testing the unimportant feature set. The difference in the two approaches
# show the statistical power of the multi-view decision tree is higher than the
# regular decision tree in this simulation.

# %%
# References
# ----------
# .. footbibliography::

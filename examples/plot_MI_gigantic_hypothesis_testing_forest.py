"""
===========================================================
Mutual Information for Gigantic Hypothesis Testing (MIGHT)
===========================================================

An example using :class:`~sktree.stats.FeatureImportanceForestClassifier` for nonparametric
multivariate hypothesis test, on simulated datasets. Here, we present a simulation
of how MIGHT is used to test the hypothesis that a "feature set is important for
predicting the target". This is a generalization of the framework presented in
:footcite:`coleman2022scalable`.

We simulate a dataset with 1000 features, 500 samples, and a binary class target
variable. Within each feature set, there is 500 features associated with one feature
set, and another 500 features associated with another feature set. One could think of
these for example as different datasets collected on the same patient in a biomedical setting.
The first feature set (X) is strongly correlated with the target, and the second
feature set (W) is weakly correlated with the target (y). Here, we are testing the
null hypothesis:

- ``H0: I(X; y) - I(X, W; y) = 0``
- ``HA: I(X; y) - I(X, W; y) < 0`` indicating that there is more mutual information with
    respect to ``y``

where ``I`` is mutual information. For example, this could be true in the following settings,
where X is our informative feature set and W is our uninformative feature set.

- ``W    X -> y``: here ``W`` is completely disconnected from X and y.
- ``W -> X -> y``: here ``W`` is d-separated from y given X.
- ``W <- X -> y``: here ``W`` is d-separated from y given X.

We then use MIGHT to test the hypothesis that the first feature set is important for
predicting the target, and the second feature set is not important for predicting the
target. We use :class:`~sktree.stats.FeatureImportanceForestClassifier`.
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
X = np.hstack([X_important, X_unimportant])

# simulate the binary target variable
y = rng.binomial(n=1, p=expit(beta * X_important[:, :10].sum(axis=1)), size=n_samples)

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

n_estimators = 200
max_features = "sqrt"
test_size = 0.2
n_repeats = 1000
n_jobs = -1

est = FeatureImportanceForestClassifier(
    estimator=HonestForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        tree_estimator=DecisionTreeClassifier(),
        random_state=seed,
        honest_fraction=0.7,
        n_jobs=n_jobs,
    ),
    random_state=seed,
    test_size=test_size,
    permute_per_tree=True,
    sample_dataset_per_tree=False,
)

print(
    f"Permutation per tree: {est.permute_per_tree} and sampling dataset per tree: "
    f"{est.sample_dataset_per_tree}"
)
# we test for the first feature set, which is important and thus should return a pvalue < 0.05
stat, pvalue = est.test(
    X, y, covariate_index=np.arange(n_features_set, dtype=int), metric="mi", n_repeats=n_repeats
)
print(f"Estimated MI difference: {stat} with Pvalue: {pvalue}")

# we test for the second feature set, which is unimportant and thus should return a pvalue > 0.05
stat, pvalue = est.test(
    X,
    y,
    covariate_index=np.arange(n_features_set, dtype=int) + n_features_set,
    metric="mi",
    n_repeats=n_repeats,
)
print(f"Estimated MI difference: {stat} with Pvalue: {pvalue}")

# %%
# References
# ----------
# .. footbibliography::

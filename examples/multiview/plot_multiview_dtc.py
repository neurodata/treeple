"""
============================================================
Analyze a multi-view dataset with a multi-view random forest
============================================================

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
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sktree import MultiViewRandomForestClassifier, RandomForestClassifier

seed = 12345
rng = np.random.default_rng(seed)


def make_multiview_classification(n_samples=100, n_features_1=5, n_features_2=1000, cluster_std=2.0, seed=None):
    rng = np.random.default_rng(seed=seed)

    # Create a high-dimensional multiview dataset with a low-dimensional informative
    # subspace in one view of the dataset.
    X0_first, y0 = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        n_features=n_features_1,
        random_state=rng.integers(1, 10000),
        centers=1,
    )

    X1_first, y1 = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        n_features=n_features_1,
        random_state=rng.integers(1, 10000),
        centers=1,
    )
    y1[:] = 1
    X0 = np.concatenate([X0_first, rng.standard_normal(size=(n_samples, n_features_2))], axis=1)
    X1 = np.concatenate([X1_first, rng.standard_normal(size=(n_samples, n_features_2))], axis=1)
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1)).T

    return X, y

# %%
# Simulate data
# -------------
# We simulate a 2-view dataset with both views containing informative low-dimensional features.
# The first view has five dimensions, while the second view will vary from five to a thousand
# dimensions. The sample-size will be kept fixed, so we can compare the performance of
# regular Random forests with Multi-view Random Forests.

n_samples = 1000
n_features_views = np.linspace(5, 1000, 2).astype(int)

datasets = []
for idx, n_features in enumerate(n_features_views):
    X, y = make_multiview_classification(n_samples=n_samples, n_features_1=5, n_features_2=n_features, cluster_std=2.0, seed=seed + idx)
    datasets.append((X, y))

# %%
# Fit Random Forest and Multi-view Random Forest
# ----------------------------------------------
# Here, we fit both forests over all the datasets.

n_estimators = 50
n_jobs = -1

scores = defaultdict(list)

for ((X, y), n_features) in zip(datasets, n_features_views):
    feature_set_ends = np.array([5, n_features + 5])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )

    mvrf = MultiViewRandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        feature_set_ends=n_features_views,
    )

    # obtain the cross-validation score
    rf_score = cross_val_score(rf, X, y, cv=2).mean()
    mvrf_score = cross_val_score(mvrf, X, y, cv=2).mean()

    scores['rf'].append(rf_score)
    scores['mvrf'].append(mvrf_score)

# %%
# Visualize scores and compare performance
# ----------------------------------------
# Now, we can compare the performance from the cross-validation experiment.

scores['n_features'] = n_features_views
df = pd.DataFrame(scores)

# melt the dataframe, to make it easier to plot
df = pd.melt(df, id_vars=['n_features'], var_name='model', value_name='score')

fig, ax = plt.subplots()
sns.lineplot(data=df, x='n_features', y='score', hue='model', label='CV Score', ax=ax)
ax.set_ylabel('CV Score')
ax.set_xlabel('Number of features in second view')
ax.set_title('Random Forest vs Multi-view Random Forest')
plt.show()

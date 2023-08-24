"""
================================================================================
Plot extra oblique forest and oblique random forest predictions on cc18 datasets
================================================================================

A performance comparison between extra oblique forest and standard oblique random
forest using four datasets from OpenML benchmarking suites.

Extra oblique forest uses extra oblique trees as base model which differ from classic
decision trees in the way they are built. When looking for the best split to
separate the samples of a node into two groups, random splits are drawn for each of
the `max_features` randomly selected features and the best split among those is chosen.
When `max_features` is set 1, this amounts to building a totally random
decision tree.

The datasets used in this example are from the OpenML benchmarking suite are:

[Phishing Website](https://www.openml.org/search?type=data&sort=runs&id=4534),
[WDBC](https://www.openml.org/search?type=data&sort=runs&id=1510),
[Lsvt](https://www.openml.org/search?type=data&sort=runs&id=1484),
[har]((https://www.openml.org/search?type=data&sort=runs&id=1478), and
[cnae-9](https://www.openml.org/search?type=data&sort=runs&id==1468).
 All datasets are subsampled due to computational constraints. Note that `cnae-9` is
 an high dimensional dataset with very sparse 856 features, mostly consisting of zeros.

dataset| samples | features | datatype
-------|---------|----------|---------
Phishing Website | 8844 | 30 | nominal
WDBC | 455 | 30 | numeric
Lsvt | 100 | 310 | numeric
har | 100 | 561 | numeric
cnae-9 | 100 | 856 | numeric

References
----------
.. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
        Machine Learning, 63(1), 3-42, 2006.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import RepeatedKFold, cross_validate

from sktree import ExtraObliqueRandomForestClassifier, ObliqueRandomForestClassifier

# Parameters
random_state = 12345
phishing_website = 4534
wdbc = 1510
lsvt = 1484
har = 1478
cnae_9 = 1468

data_ids = [phishing_website, wdbc, lsvt, har, cnae_9]
df = pd.DataFrame()


def load_cc18(data_id):
    df = fetch_openml(data_id=data_id, as_frame=True, parser="pandas")

    # extract the dataset name
    d_name = df.details["name"]

    # Subsampling large datasets
    if data_id in [1468, 1478]:
        n = 100
    else:
        n = int(df.frame.shape[0] * 0.8)

    df = df.frame.sample(n, random_state=random_state)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y, d_name


def get_scores(X, y, d_name, n_cv=5, n_repeats=1, **kwargs):
    clfs = [ExtraObliqueRandomForestClassifier(**kwargs), ObliqueRandomForestClassifier(**kwargs)]
    dim = X.shape
    tmp = []

    for i, clf in enumerate(clfs):
        t0 = datetime.now()
        cv = RepeatedKFold(n_splits=n_cv, n_repeats=n_repeats, random_state=kwargs["random_state"])
        test_score = cross_validate(estimator=clf, X=X, y=y, cv=cv, scoring="accuracy")
        time_taken = datetime.now() - t0
        # convert the time taken to seconds
        time_taken = time_taken.total_seconds()

        tmp.append(
            [
                d_name,
                dim,
                ["EORF", "ORF"][i],
                test_score["test_score"],
                test_score["test_score"].mean(),
                time_taken,
            ]
        )

    df = pd.DataFrame(tmp, columns=["dataset", "dimension", "model", "score", "mean", "time_taken"])
    df = df.explode("score")
    df["score"] = df["score"].astype(float)
    df.reset_index(inplace=True, drop=True)

    return df


params = {
    "max_features": None,
    "n_estimators": 50,
    "max_depth": None,
    "random_state": random_state,
    "n_cv": 10,
    "n_repeats": 1,
}

for data_id in data_ids:
    X, y, d_name = load_cc18(data_id=data_id)
    tmp = get_scores(X=X, y=y, d_name=d_name, **params)
    df = pd.concat([df, tmp])

# Show the time taken to train each model
print(df.groupby(["dataset", "dimension", "model"])[["time_taken"]].mean())

# Draw a comparison plot
d_names = df.dataset.unique()
N = d_names.shape[0]

fig, ax = plt.subplots(1, N)
fig.set_size_inches(6 * N, 6)

for i, name in enumerate(d_names):
    sns.stripplot(
        data=df.query(f'dataset == "{name}"'),
        x="model",
        y="score",
        ax=ax[i],
        dodge=True,
    )
    sns.boxplot(
        data=df.query(f'dataset == "{name}"'),
        x="model",
        y="score",
        ax=ax[i],
        color="white",
    )
    ax[i].set_title(name)
    if i != 0:
        ax[i].set_ylabel("")
    ax[i].set_xlabel("")
# show the figure
plt.show()


# Discussion
# ----------
# Extra Oblique Tree demonstrates performance similar to that of regular Oblique Tree on average
# with some increase in variance.
# However, Extra Oblique Tree runs substantially faster than Oblique Tree on some datasets due to
# the random_splits process which omits the computationally expensive search for the best split.

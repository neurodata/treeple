"""
================================================================================
Plot extra oblique forest and oblique random forest predictions on cc18 datasets
================================================================================

A performance comparison between extra oblique forest and standard oblique random
forest using three datasets from OpenML benchmarking suites.

Extra oblique forest uses extra oblique trees as base model which differ from classic
decision trees in the way they are built. When looking for the best split to
separate the samples of a node into two groups, random splits are drawn for each
of the `max_features` randomly selected features and the best split among those is
chosen. When `max_features` is set 1, this amounts to building a totally random
decision tree.

Two of these datasets, namely
[WDBC](https://www.openml.org/search?type=data&sort=runs&id=1510)
and [Phishing Website](https://www.openml.org/search?type=data&sort=runs&id=4534)
datasets consist of 31 features where the former dataset is entirely numeric
and the latter dataset is entirely norminal. The third dataset, dubbed
[cnae-9](https://www.openml.org/search?type=data&status=active&id=1468), is a
numeric dataset that has notably large feature space of 857 features. As you
will notice, of these three datasets, the oblique forest outperforms axis-aligned
random forest on cnae-9 utilizing sparse random projection mechanism. All datasets
are subsampled due to computational constraints.

References
----------
.. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
        Machine Learning, 63(1), 3-42, 2006.
"""

import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from sktree import ExtraObliqueRandomForestClassifier, ObliqueRandomForestClassifier
from sktree.tree import ExtraObliqueDecisionTreeClassifier, ObliqueDecisionTreeClassifier

random_state = 12345
# data_ids = [4534, 1510, 1468]  # openml dataset id
data_ids = [4534, 1510, 53, 1468]
df = pd.DataFrame()


def load_cc18(data_id):
    df = fetch_openml(data_id=data_id, as_frame=True, parser="pandas")

    # extract the dataset name
    d_name = df.details["name"]

    # Subsampling large datasets
    if data_id == 1468:
        n = 100
    else:
        n = int(df.frame.shape[0] * 0.8)

    df = df.frame.sample(n, random_state=random_state)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y, d_name


def get_scores(X, y, d_name, n_cv=5, n_repeats=1, **kwargs):
    clfs = [ExtraObliqueRandomForestClassifier(**kwargs), ObliqueRandomForestClassifier(**kwargs)]

    tmp = []

    for i, clf in enumerate(clfs):
        t0 = datetime.now()
        cv = RepeatedKFold(n_splits=n_cv, n_repeats=n_repeats, random_state=kwargs["random_state"])
        test_score = cross_validate(estimator=clf, X=X, y=y, cv=cv, scoring="accuracy")
        time_taken = datetime.now() - t0
        # convert the time taken to seconds
        time_taken = time_taken.total_seconds()
        print(f"Dataset [{d_name}] - [{clf.__class__.__name__}] - {time_taken}")

        tmp.append(
            [
                d_name,
                ["EORF", "ORF"][i],
                test_score["test_score"],
                test_score["test_score"].mean(),
                time_taken,
            ]
        )

    df = pd.DataFrame(tmp, columns=["dataset", "model", "score", "mean", "time_taken"])
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
    print(f"Loading [{d_name}] dataset..")
    tmp = get_scores(X=X, y=y, d_name=d_name, **params)
    df = pd.concat([df, tmp])

# Show the time taken to train each model
print(df.groupby(["dataset", "model"])[["time_taken"]].mean())

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
# ----------``
# Extra Oblique Tree runs faster compared to the standard Oblique Tree,
# while the performance is comparable or better in some cases that are
# tested.

"""
====================================================================
Plot the decision surfaces of ensembles of trees on the iris dataset
====================================================================

Plot the decision surfaces of forests of randomized trees trained on pairs of
features of the iris dataset.

This plot compares the decision surfaces learned by a decision tree classifier
(first column), by a oblique decision tree classifier (second column), by an extra-
oblique decision tree classifier (third column).

In the first row, the classifiers are built using the sepal width and
the sepal length features only, on the second row using the petal length and
sepal length only, and on the third row using the petal width and the
petal length only.

"""

# Plot the decision boundaries of the ObliqueDecisionTree and ExtraObliqueDecisionTree
# on iris dataset

# Parameters
n_classes = 3
n_estimators = 30
max_depth = 10
cmap = plt.cm.Spectral
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.25  # step widths for coarse classifier guesses
# figure size for plotting
figure_size = (30, 30)
pairs = [[0, 1], [0, 2], [2, 3]]

# Load data
iris = load_iris()

plot_idx = 1

models = [
    DecisionTreeClassifier(max_depth=max_depth),
    ObliqueDecisionTreeClassifier(max_depth=max_depth),
    ExtraObliqueDecisionTreeClassifier(max_depth=max_depth),
]

# figure size for plotting
figure_size = (30, 30)
pairs = [[0, 1], [0, 2], [2, 3]]
N = len(pairs) * len(models)

plot_idx = 1

n_rows = 3
fig, ax = plt.subplots(n_rows, math.ceil(N / n_rows))
fig.set_size_inches(6 * N, 6)

for pair in pairs:
    for model in models:
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target
        # starting time
        t0 = datetime.now()

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(random_state)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][: -len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print(
            model_details + " with features",
            pair,
            "has a score of",
            round(scores, 5),
            "took",
            (datetime.now() - t0).total_seconds(),
            "seconds",
        )

        plt.subplot(3, 3, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if (
            isinstance(model, DecisionTreeClassifier)
            or isinstance(model, ObliqueDecisionTreeClassifier)
            or isinstance(model, ExtraObliqueDecisionTreeClassifier)
        ):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)

        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser),
        )
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(
            xx_coarser.shape
        )
        cs_points = plt.scatter(
            xx_coarser,
            yy_coarser,
            s=15,
            c=Z_points_coarser,
            cmap=cmap,
            edgecolors="none",
        )

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=ListedColormap(["r", "y", "b"]),
            edgecolor="k",
            s=20,
        )
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()

# Discussion
# ----------
# This section demonstrates the decision boundaries of the classification task with
# ObliqueDecisionTree and ExtraObliqueDecisionTree. While the decision boundaries are different
# for each tree, the performance are similar. However, the ExtraObliqueDecisionTree classifier
# runs faster with similar or better performance in some cases.

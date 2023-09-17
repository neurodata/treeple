import numpy as np
import pytest
from sklearn import datasets

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.stats import MIGHT
from sktree.tree import ObliqueDecisionTreeClassifier

# load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)

# remove third class
iris_X = iris.data[iris.target != 2]
iris_y = iris.target[iris.target != 2]

p = rng.permutation(iris_X.shape[0])
iris_X = iris_X[p]
iris_y = iris_y[p]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore", "error"])
@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
    ],
)
@pytest.mark.parametrize("limit", [0.05, 0.1])
def test_iris(criterion, max_features, honest_prior, estimator, limit):
    # Check consistency on dataset iris.
    clf = MIGHT(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=10,
        honest_prior=honest_prior,
        tree_estimator=estimator,
        limit=limit,
    )
    if honest_prior == "error":
        with pytest.raises(ValueError, match="honest_prior error not a valid input."):
            clf.statistic(iris_X, iris_y)
    else:
        score = clf.statistic(iris_X, iris_y, stat="AUC")
        assert score >= 0.9, "Failed with pAUC: {0} for max fpr: {1}".format(score, limit)

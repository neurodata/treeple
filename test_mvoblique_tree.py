import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import (
    DecisionTreeClassifier,
    MultiViewDecisionTreeClassifier,
    MultiViewObliqueDecisionTreeClassifier,
)

seed = 12345

rng = np.random.default_rng(seed=seed)

X = np.random.random((20, 10))
y = np.random.randint(0, 2, size=20)

# test with max_features as a float
clf = MultiViewDecisionTreeClassifier(
    random_state=seed,
    feature_set_ends=[6, 10],
    max_features=0.5,
)
clf.fit(X, y)

assert_array_equal(clf.max_features_per_set_, [3, 2])
assert clf.max_features_ == 5

# test with max_features as sqrt
# X = np.random.random((20, 13))
# clf = MultiViewDecisionTreeClassifier(
#     random_state=seed,
#     feature_set_ends=[9, 13],
#     max_features="sqrt",
# )
# clf.fit(X, y)
# assert_array_equal(clf.max_features_per_set_, [3, 2])
# assert clf.max_features_ == 5

# # test with max_features as 'sqrt' but not a perfect square
# X = np.random.random((20, 9))
# clf = MultiViewDecisionTreeClassifier(
#     random_state=seed,
#     feature_set_ends=[5, 9],
#     max_features="sqrt",
# )
# clf.fit(X, y)
# assert_array_equal(clf.max_features_per_set_, [3, 2])
# assert clf.max_features_ == 5
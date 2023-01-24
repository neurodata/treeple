import sys

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import UnsupervisedDecisionTree

# @parametrize_with_checks([UnsupervisedDecisionTree()])
# def test_sklearn_compatible_estimator(estimator, check):
#     if check.func.__name__ in [
#         # methods should never have only one sample
#         # 'check_fit2d_1sample',
#         # samples should be ordered wrt
#         # 'check_methods_sample_order_invariance',
#         # negative window dimension not allowed
#         # 'check_methods_subset_invariance',
#         "check_estimators_dtype",
#         # 'check_dtype_object',
#     ]:
#         pytest.skip()

#     print("here...")
#     print(check)

#     check(estimator)


def test_unsupervisedtree():
    n_samples = 100
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=2**4)

    clf = UnsupervisedDecisionTree(
        criterion="twomeans",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        random_state=None,
        min_impurity_decrease=0.0,
    )
    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)
    assert score > 0.9


if __name__ == "__main__":
    test_unsupervisedtree()

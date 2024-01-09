import pytest
import numpy as np
from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.datasets import make_quadratic_classification
from sktree.ensemble import HonestForestClassifier
from sklearn.exceptions import NotFittedError

from sktree.posterior import predict_proba_per_tree


@pytest.mark.parametrize("BaseTree", [None, DecisionTreeClassifier()])
def test_honest_forest_predict_proba_for_all_trees(BaseTree):
    n_samples = 200
    honest_fraction = 0.5
    X, y = make_quadratic_classification(n_samples // 2, 32, noise=True, seed=0)
    y = y.squeeze()

    est = HonestForestClassifier(
        n_estimators=10,
        tree_estimator=BaseTree,
        random_state=0,
        honest_fraction=honest_fraction,
    )
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        predict_proba_per_tree(est, X)

    est.fit(X, y)

    est.estimators_samples_
    y_pred_proba = predict_proba_per_tree(est, X, oob=False)
    assert y_pred_proba.shape == (10, n_samples, 2), y_pred_proba.shape
    assert np.isnan(y_pred_proba).sum() == 0

    # When using oob=True, we only predict for the samples that were not used for constructing the tree
    y_pred_proba = predict_proba_per_tree(est, X, oob=True)
    assert y_pred_proba.shape == (10, n_samples, 2), y_pred_proba.shape
    assert all(np.isnan(y_pred_proba).sum(axis=1)[:, -1] == n_samples * honest_fraction), np.isnan(
        y_pred_proba
    ).sum(axis=1)[:, -1]

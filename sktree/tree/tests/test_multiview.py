import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import DecisionTreeClassifier, MultiViewDecisionTreeClassifier

seed = 12345


@parametrize_with_checks(
    [
        MultiViewDecisionTreeClassifier(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("baseline_est", [MultiViewDecisionTreeClassifier, DecisionTreeClassifier])
def test_multiview_classification(baseline_est):
    """Test that explicit knowledge of multi-view structure improves classification accuracy.

    In very high-dimensional noise setting across two views, when the max_depth and max_features
    are constrained, the multi-view decision tree will still obtain perfect accuracy, while
    the single-view decision tree will not.
    """
    rng = np.random.default_rng(seed=seed)

    n_samples = 20
    n_features_1 = 5
    n_features_2 = 1000
    cluster_std = 2.0

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

    # Compare multiview decision tree vs single-view decision tree
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[n_features_1, X.shape[1]],
        max_features="sqrt",
        max_depth=5,
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) == 1.0
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=5).mean() == 1.0
    ), f"CV score: {cross_val_score(clf, X, y, cv=5).mean()}"

    clf = baseline_est(
        random_state=seed,
        max_depth=5,
        max_features="sqrt",
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) == 1.0
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=5).mean() <= 0.6
    ), f"CV score: {cross_val_score(clf, X, y, cv=5).mean()}"

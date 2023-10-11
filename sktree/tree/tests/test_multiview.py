import numpy as np
import pytest

from numpy.testing import assert_allclose
from sklearn.datasets import make_classification, make_blobs
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sktree.tree import MultiViewDecisionTreeClassifier, DecisionTreeClassifier

seed = 12345


@parametrize_with_checks(
    [
        MultiViewDecisionTreeClassifier(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_multiview_classification():
    """Test that explicit knowledge of multi-view structure improves classification accuracy."""
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
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) > 0.99
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=5).mean() > 0.82
    ), f"CV score: {cross_val_score(clf, X, y, cv=5).mean()}"

    clf = DecisionTreeClassifier(
        random_state=seed,
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) >= 0.5 and accuracy_score(y, clf.predict(X)) < 0.99
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=5).mean() >= 0.5 and cross_val_score(clf, X, y, cv=5).mean() < 0.82
    ), f"CV score: {cross_val_score(clf, X, y, cv=5).mean()}"
    assert False

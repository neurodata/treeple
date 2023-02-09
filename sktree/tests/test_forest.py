import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_random_state

from sktree import (
    ObliqueRandomForestClassifier,
    UnsupervisedObliqueRandomForest,
    UnsupervisedRandomForest,
)

# Larger classification sample used for testing feature importances
X_large, y_large = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=0,
)


def _sparse_parity(n, p=20, p_star=3, random_state=None):
    """Generate sparse parity dataset.

    Sparse parity is a multivariate generalization of the
    XOR problem.

    Parameters
    ----------
    n : int
        Number of sample to generate.
    p : int, optional
        The dimensionality of the dataset, by default 20
    p_star : int, optional
        The number of informative dimensions, by default 3.
    random_state : Random State, optional
        Random state, by default None.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Sparse parity dataset as a dense array.
    y : np.ndarray of shape (n,)
        Labels of the dataset
    """
    rng = np.random.RandomState(seed=random_state)
    X = rng.uniform(-1, 1, (n, p))
    y = np.zeros(n)

    for i in range(0, n):
        y[i] = sum(X[i, :p_star] > 0) % 2

    return X, y


def _orthant(n, p=6, random_state=None):
    """Generate orthant dataset.

    Parameters
    ----------
    n : int
        Number of sample to generate.
    p : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 6.
    rec : int, optional
        _description_, by default 1
    random_state : Random State, optional
        Random state, by default None.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Orthant dataset as a dense array.
    y : np.ndarray of shape (n,)
        Labels of the dataset
    """
    rng = np.random.RandomState(seed=random_state)
    orth_labels = np.asarray([2**i for i in range(0, p)][::-1])

    X = rng.uniform(-1, 1, (n, p))
    y = np.zeros(n)

    for i in range(0, n):
        idx = np.where(X[i, :] > 0)[0]
        y[i] = sum(orth_labels[idx])

    if len(np.unique(y)) < 2**p:
        raise RuntimeError("Increase sample size to get a label in each orthant.")

    return X, y


def _trunk(n, p=10, random_state=None):
    """Generate trunk dataset.

    Parameters
    ----------
    n : int
        Number of sample to generate.
    p : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 10.
    random_state : Random State, optional
        Random state, by default None.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n,)
        Labels of the dataset

    References
    ----------
    [1] Gerard V. Trunk. A problem of dimensionality: A
    simple example. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 1(3):306–307, 1979.
    """
    rng = np.random.RandomState(seed=random_state)

    mu_1 = np.array([1 / i for i in range(1, p + 1)])
    mu_0 = -1 * mu_1
    cov = np.identity(p)

    X = np.vstack(
        (
            rng.multivariate_normal(mu_0, cov, int(n / 2)),
            rng.multivariate_normal(mu_1, cov, int(n / 2)),
        )
    )
    y = np.concatenate((np.zeros(int(n / 2)), np.ones(int(n / 2))))
    return X, y


@parametrize_with_checks(
    [
        UnsupervisedRandomForest(random_state=12345, n_estimators=50),
        UnsupervisedObliqueRandomForest(random_state=12345, n_estimators=50),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in [
        # Cannot apply agglomerative clustering on < 2 samples
        "check_methods_subset_invariance",
        # # sample weights do not necessarily imply a sample is not used in clustering
        "check_sample_weights_invariance",
        # # sample order is not preserved in predict
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize(
    "CLF_NAME, ESTIMATOR",
    [
        ("UnsupervisedRandomForest", UnsupervisedRandomForest),
        ("UnsupervisedObliqueRandomForest", UnsupervisedObliqueRandomForest),
    ],
)
def test_urf(CLF_NAME, ESTIMATOR):
    n_samples = 100
    n_classes = 2

    #
    if CLF_NAME == "UnsupervisedRandomForest":
        n_features = 5
        n_estimators = 50
        expected_score = 0.4
    else:
        n_features = 20
        n_estimators = 20
        expected_score = 0.9
    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=12345
    )

    clf = ESTIMATOR(n_estimators=n_estimators, random_state=12345)
    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # XXX: This should be > 0.9 according to the UReRF. Hoewver, that could be because they used
    # the oblique projections by default
    assert score > expected_score


def test_oblique_forest_sparse_parity():
    # Sparse parity dataset
    n = 1000
    X, y = _sparse_parity(n, random_state=0)
    n_test = 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=0,
    )

    rc_clf = ObliqueRandomForestClassifier(max_features=None, random_state=0)
    rc_clf.fit(X_train, y_train)
    y_hat = rc_clf.predict(X_test)
    rc_accuracy = accuracy_score(y_test, y_hat)

    ri_clf = RandomForestClassifier(random_state=0)
    ri_clf.fit(X_train, y_train)
    y_hat = ri_clf.predict(X_test)
    ri_accuracy = accuracy_score(y_test, y_hat)

    assert ri_accuracy < rc_accuracy
    assert ri_accuracy > 0.45
    assert rc_accuracy > 0.5


def test_oblique_forest_orthant():
    """Test oblique forests on orthant problem.

    It is expected that axis-aligned and oblique-aligned
    forests will perform similarly.
    """
    n = 500
    X, y = _orthant(n, p=6, random_state=0)
    n_test = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=0,
    )

    rc_clf = ObliqueRandomForestClassifier(max_features=None, random_state=0)
    rc_clf.fit(X_train, y_train)
    y_hat = rc_clf.predict(X_test)
    rc_accuracy = accuracy_score(y_test, y_hat)

    ri_clf = RandomForestClassifier(max_features="sqrt", random_state=0)
    ri_clf.fit(X_train, y_train)
    y_hat = ri_clf.predict(X_test)
    ri_accuracy = accuracy_score(y_test, y_hat)

    assert rc_accuracy >= ri_accuracy
    assert ri_accuracy > 0.84
    assert rc_accuracy > 0.85


def test_oblique_forest_trunk():
    """Test oblique vs axis-aligned forests on Trunk."""
    n = 1000
    X, y = _trunk(n, p=100, random_state=0)
    n_test = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=0,
    )

    rc_clf = ObliqueRandomForestClassifier(max_features=X.shape[1], random_state=0)
    rc_clf.fit(X_train, y_train)
    y_hat = rc_clf.predict(X_test)
    rc_accuracy = accuracy_score(y_test, y_hat)

    ri_clf = RandomForestClassifier(max_features="sqrt", random_state=0)
    ri_clf.fit(X_train, y_train)
    y_hat = ri_clf.predict(X_test)
    ri_accuracy = accuracy_score(y_test, y_hat)

    assert rc_accuracy > ri_accuracy
    assert ri_accuracy > 0.83
    assert rc_accuracy > 0.86


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize(
    "criterion",
    ["gini", "log_loss"],
)
def test_check_importances(criterion, dtype):
    """Test checking feature importances for oblique trees."""
    tolerance = 0.01

    # cast as dype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    est = ObliqueRandomForestClassifier(n_estimators=10, criterion=criterion, random_state=0)
    est.fit(X, y)
    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert np.all(importances[:3] > 0.1)

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = ObliqueRandomForestClassifier(n_estimators=10, random_state=0, criterion=criterion)
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = ObliqueRandomForestClassifier(n_estimators=10, random_state=0, criterion=criterion)
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance

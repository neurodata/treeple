import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
)
import pickle
import copy
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import pytest

from oblique_forests.sporf import ObliqueForestClassifier, PythonObliqueForestClassifier


@pytest.mark.skip(reason='Python version can skip')
def test_sparse_parity_py():
    clf = PythonObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )

    train = np.load("data/sparse_parity_train_1000.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/sparse_parity_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    accuracy = np.sum(y_test == y_hat) / len(y_test)

    assert accuracy >= 0.8


def test_sparse_parity():
    clf = ObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )

    train = np.load("data/sparse_parity_train_1000.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/sparse_parity_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    accuracy = np.sum(y_test == y_hat) / len(y_test)

    assert accuracy >= 0.8


def test_orthant():
    clf = ObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )

    train = np.load("data/orthant_train_400.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/orthant_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    accuracy = np.sum(y_test == y_hat) / len(y_test)

    assert accuracy >= 0.95


def test_clone():
    clf = ObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )
    train = np.load("data/orthant_train_400.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]
    clf.fit(X_train, y_train)
    print('Did first fit...')
    new_clf = clone(clf)
    new_clf.fit(X_train, y_train)
    print('did second fit...')

    pickled = pickle.dumps(clf)

def test_orthant_crossval():
    clf = ObliqueForestClassifier(
        random_state=1,
        n_estimators=500,
        max_features=1.0,
        feature_combinations=2.0,
        n_jobs=-1,
    )
    print('Trying to make new cpy of ', clf)
    new_clf = copy.deepcopy(clf)
    print(new_clf)

    train = np.load("data/orthant_train_400.npy")
    X_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.load("data/orthant_test.npy")
    X_test = test[:, :-1]
    y_test = test[:, -1]
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # get cross-validation score
    score = cross_val_score(clf, X, y, cv=3, verbose=10)
    assert score.mean() >= 0.95


test_orthant_crossval()
# test_clone()
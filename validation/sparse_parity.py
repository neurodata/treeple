import sys
from pathlib import Path

import numpy as np
from oblique_forests.sporf import ObliqueForestClassifier as SPORF
from oblique_forests.sporf import PythonObliqueForestClassifier as PySPORF
from sklearn.ensemble import RandomForestClassifier as RF

# Change the rerf import as needed:
sys.path.append(
    "/Users/ChesterHuynh/OneDrive - Johns Hopkins/research/seeg localization/SPORF/Python/"
)
from rerf.rerfClassifier import rerfClassifier as RERF


def load_data(n):
    ftrain = "data/sparse_parity_train_" + str(n) + ".npy"
    ftest = "data/sparse_parity_test.npy"

    dftrain = np.load(ftrain)
    dftest = np.load(ftest)

    X_train = dftrain[:, :-1]
    y_train = dftrain[:, -1]

    X_test = dftest[:, :-1]
    y_test = dftest[:, -1]

    return X_train, y_train, X_test, y_test


def test_rf(n, reps, n_estimators, max_features):
    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        clf = RF(n_estimators=n_estimators, max_features=max_features)

        import yep

        yep.start(f"profiling/rf_fit_sparse_parity{n}.prof")
        clf.fit(X_train, y_train)
        yep.stop()

        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/rf_sparse_parity_preds_" + str(n) + ".npy", preds)
    return acc


def test_rerf(n, reps, n_estimators, feature_combinations, max_features):
    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        clf = RERF(
            n_estimators=n_estimators,
            projection_matrix="RerF",
            feature_combinations=feature_combinations,
            max_features=max_features,
        )

        import yep

        yep.start(f"profiling/rerf_fit_sparse_parity{n}.prof")
        clf.fit(X_train, y_train)
        yep.stop()
        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/rerf_sparse_parity_preds_" + str(n) + ".npy", preds)
    return acc


def test_pysporf(n, reps, n_estimators, feature_combinations, max_features):
    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        clf = PySPORF(
            n_estimators=n_estimators,
            feature_combinations=feature_combinations,
            max_features=max_features,
            n_jobs=1,
        )

        clf.fit(X_train, y_train)
        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/of_sparse_parity_preds_" + str(n) + ".npy", preds)
    return acc


def test_sporf(n, reps, n_estimators, feature_combinations, max_features):
    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        clf = SPORF(
            n_estimators=n_estimators,
            feature_combinations=feature_combinations,
            max_features=max_features,
            n_jobs=-1,
        )

        import yep

        yep.start(f"profiling/cysporf_fit_sparse_parity{n}.prof")
        clf.fit(X_train, y_train)
        yep.stop()
        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/sporf_sparse_parity_preds_" + str(n) + ".npy", preds)
    return acc


def main():
    # How many samples to train on
    ns = [1000, 5000, 10000]

    # How many repetitions
    reps = 3

    # Tree parameters
    n_estimators = 100
    feature_combinations = 2
    max_features = 1.0

    np.random.seed(0)
    for n in ns:
        # acc = test_rf(n, reps, n_estimators)
        # print(acc)

        acc = test_rerf(n, reps, n_estimators, feature_combinations, max_features)
        print(acc)

        acc = test_sporf(n, reps, n_estimators, feature_combinations, max_features)
        print(acc)

        # acc = test_pysporf(n, reps, n_estimators, feature_combinations, max_features_pysporf)
        # print(acc)


if __name__ == "__main__":
    main()

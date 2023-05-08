import sys

import numpy as np
from oblique_forests.ensemble import RandomForestClassifier as ObliqueRF
from oblique_forests.sporf import ObliqueForestClassifier as ofc

sys.path.append(
    "/Users/ChesterHuynh/OneDrive - Johns Hopkins/research/seeg localization/SPORF/Python"
)
from rerf.rerfClassifier import rerfClassifier as rfc


def load_data(n):
    ftrain = "data/orthant_train_" + str(n) + ".npy"
    ftest = "data/orthant_test.npy"

    dftrain = np.load(ftrain)
    dftest = np.load(ftest)

    X_train = dftrain[:, :-1]
    y_train = dftrain[:, -1]

    X_test = dftest[:, :-1]
    y_test = dftest[:, -1]

    return X_train, y_train, X_test, y_test


def test_rf(n, reps, n_estimators, max_features):
    from sklearn.ensemble import RandomForestClassifier

    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        # clf = rfc(n_estimators=n_estimators,
        #           projection_matrix="Base")
        # clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=8)
        clf = ObliqueRF(n_estimators=n_estimators, max_features=max_features, n_jobs=8)

        import yep

        yep.start(f"profiling/rf_fit_orthant{n}.prof")
        clf.fit(X_train, y_train)
        # print(list(map(lambda x: x.tree_.node_count, clf.estimators_)))
        yep.stop()

        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/rf_orthant_preds_" + str(n) + ".npy", preds)
    return acc


def test_rerf(n, reps, n_estimators, feature_combinations, max_features):
    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        clf = rfc(
            n_estimators=n_estimators,
            projection_matrix="RerF",
            feature_combinations=feature_combinations,
            max_features=max_features,
            n_jobs=8,
        )

        import yep

        yep.start(f"profiling/rerf_fit_orthant{n}.prof")
        clf.fit(X_train, y_train)
        yep.stop()

        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/rerf_orthant_preds_" + str(n) + ".npy", preds)
    return acc


def test_of(n, reps, n_estimators, feature_combinations, max_features):
    preds = np.zeros((reps, 10000))
    acc = np.zeros(reps)
    for i in range(reps):
        X_train, y_train, X_test, y_test = load_data(n)

        clf = ofc(
            n_estimators=n_estimators,
            feature_combinations=feature_combinations,
            max_features=max_features,
            n_jobs=8,
        )

        # Profile fitting
        import yep

        yep.start(f"profiling/cysporf_fit_orthant{n}.prof")
        clf.fit(X_train, y_train)
        # print(list(map(lambda x: x.tree_.node_count, clf.estimators_)))
        yep.stop()

        preds[i] = clf.predict(X_test)
        acc[i] = np.sum(preds[i] == y_test) / len(y_test)

    np.save("output/of_orthant_preds_" + str(n) + ".npy", preds)
    return acc


def main():
    n = 4000
    reps = 3
    n_estimators = 100
    feature_combinations = 2
    max_features = 1.0

    acc = test_rf(n, reps, n_estimators, max_features)
    acc = test_rerf(n, reps, n_estimators, feature_combinations, max_features)
    acc = test_of(n, reps, n_estimators, feature_combinations, max_features)
    print(acc)


if __name__ == "__main__":
    main()

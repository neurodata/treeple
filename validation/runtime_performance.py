import logging
import sys
import timeit
from collections import defaultdict
from pathlib import Path

import numpy as np
from oblique_forests.ensemble import RandomForestClassifier
from oblique_forests.sporf import ObliqueForestClassifier, PythonObliqueForestClassifier

sys.path.append(
    "/Users/ChesterHuynh/OneDrive - Johns Hopkins/research/seeg localization/SPORF/Python"
)
from rerf.rerfClassifier import rerfClassifier


def load_data(n, data_path, exp_name):
    """Function to load in data as a function of sample size."""
    ftrain = data_path / f"{exp_name}_train_{n}.npy"
    ftest = data_path / f"{exp_name}_test.npy"

    dftrain = np.load(ftrain)
    dftest = np.load(ftest)

    X_train = dftrain[:, :-1]
    y_train = dftrain[:, -1]

    X_test = dftest[:, :-1]
    y_test = dftest[:, -1]

    return X_train, y_train, X_test, y_test


def main():
    logging.basicConfig(
        filename=Path(__file__).parents[0] / "runtime_performance.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    data_path = Path(__file__).parents[0] / "data"
    exps = {"sparse_parity": [1000, 5000, 10000], "orthant": [400, 2000, 4000]}
    n_reps = 7  # Number of times to execute fit

    # Experiment parameters
    n_estimators = 100
    n_jobs = 8
    feature_combinations = 2
    max_features = 1.0

    np.random.seed(0)
    for exp_name, ns in exps.items():
        for n in ns:
            n_list = defaultdict(list)
            X_train, y_train, _, _ = load_data(n, data_path, exp_name)

            # ==================================
            # Sklearn RF
            # ==================================
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs
            )
            timings = timeit.repeat(
                "clf.fit(X_train, y_train)",
                repeat=n_reps,
                number=1,
                globals={"clf": clf, "X_train": X_train, "y_train": y_train},
            )
            n_list["SklearnRF"] = timings

            # ==================================
            # C++ RF
            # ==================================
            clf = rerfClassifier(
                n_estimators=n_estimators,
                projection_matrix="Base",
                max_features=max_features,
                n_jobs=n_jobs,
            )
            timings = timeit.repeat(
                "clf.fit(X_train, y_train)",
                repeat=n_reps,
                number=1,
                globals={"clf": clf, "X_train": X_train, "y_train": y_train},
            )
            n_list["ReRF-Base"] = timings

            # ==================================
            # C++ SPORF
            # ==================================
            clf = rerfClassifier(
                n_estimators=n_estimators,
                projection_matrix="RerF",
                max_features=max_features,
                n_jobs=n_jobs,
            )
            timings = timeit.repeat(
                "clf.fit(X_train, y_train)",
                repeat=n_reps,
                number=1,
                globals={"clf": clf, "X_train": X_train, "y_train": y_train},
            )
            n_list["ReRF-Sporf"] = timings

            # ==================================
            # Cython SPORF
            # ==================================
            clf = ObliqueForestClassifier(
                n_estimators=n_estimators,
                feature_combinations=feature_combinations,
                max_features=max_features,
                n_jobs=n_jobs,
            )
            timings = timeit.repeat(
                "clf.fit(X_train, y_train)",
                repeat=n_reps,
                number=1,
                globals={"clf": clf, "X_train": X_train, "y_train": y_train},
            )
            n_list["Cy-Sporf"] = timings

            logging.info(f"Results for {exp_name}{n}")
            for clf_name, timings in n_list.items():
                mean = np.mean(timings)
                std = np.std(timings)
                unit_mean = "s"
                unit_std = "s"
                if mean < 1.0:
                    unit_mean = "ms"
                    mean *= 1000
                if std < 1.0:
                    unit_std = "ms"
                    std *= 1000
                logging.info(
                    f"{clf_name}: {mean} {unit_mean} ± {std} {unit_std} per loop (mean ± std. dev. of {n_reps} runs, 1 loop each)"
                )


if __name__ == "__main__":
    main()

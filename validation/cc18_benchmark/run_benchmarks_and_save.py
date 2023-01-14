import argparse
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import openml
from joblib import Parallel, delayed
from oblique_forests.ensemble import RandomForestClassifier as ObliqueRF
from oblique_forests.sporf import ObliqueForestClassifier as ObliqueSPORF
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def _check_nested_equality(lst1, lst2):
    if isinstance(lst1, list) and isinstance(lst2, list):
        for l1, l2 in zip(lst1, lst2):
            if not _check_nested_equality(l1, l2):
                return False
    elif isinstance(lst1, np.ndarray) and isinstance(lst2, np.ndarray):
        return np.all(lst1 == lst2)
    else:
        return lst1 == lst2

    return True


def stratify_samplesizes(y, block_lengths):
    """
    Sort data and labels into blocks that preserve class balance

    Parameters
    ----------
    X: data matrix
    y : 1D class labels
    block_lengths : Block sizes to sort X,y into that preserve class balance
    """
    clss, counts = np.unique(y, return_counts=True)
    ratios = counts / sum(counts)
    class_idxs = [np.where(y == i)[0] for i in clss]

    sort_idxs = []

    prior_idxs = np.zeros(len(clss)).astype(int)
    for n in block_lengths:
        get_idxs = np.rint((n - len(clss)) * ratios).astype(int) + 1
        for idxs, prior_idx, next_idx in zip(class_idxs, prior_idxs, get_idxs):
            sort_idxs.append(idxs[prior_idx:next_idx])
        prior_idxs = get_idxs

    sort_idxs = np.hstack(sort_idxs)

    return sort_idxs


def train_test(X, y, task_name, task_id, nominal_indices, args, clfs, save_path):
    # Set up Cross validation

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=0)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    if args.vary_samples:
        sample_sizes = np.logspace(
            np.log10(n_classes * 2),
            np.log10(np.floor(len(y) * (args.cv - 1.1) / args.cv)),
            num=10,
            endpoint=True,
            dtype=int,
        )
    else:
        sample_sizes = [len(y)]

    # Check if existing experiments
    results_dict = {
        "task": task_name,
        "task_id": task_id,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "y": y,
        "test_indices": [],
        "n_estimators": args.n_estimators,
        "cv": args.cv,
        "nominal_features": len(nominal_indices),
        "sample_sizes": sample_sizes,
    }

    # Get numeric indices first
    numeric_indices = np.delete(np.arange(X.shape[1]), nominal_indices)

    # Numeric Preprocessing
    numeric_transformer = SimpleImputer(strategy="median")

    # Nominal preprocessing
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    transformers = []
    if len(numeric_indices) > 0:
        transformers += [("numeric", numeric_transformer, numeric_indices)]
    if len(nominal_indices) > 0:
        transformers += [("nominal", nominal_transformer, nominal_indices)]
    preprocessor = ColumnTransformer(transformers=transformers)

    _, n_features_fitted = preprocessor.fit_transform(X, y).shape
    results_dict["n_features_fitted"] = n_features_fitted
    print(
        f"Features={n_features}, nominal={len(nominal_indices)} (After transforming={n_features_fitted})"
    )

    # Store training indices (random state insures consistent across clfs)
    for train_index, test_index in skf.split(X, y):
        results_dict["test_indices"].append(test_index)

    for clf_name, clf in clfs:
        pipeline = Pipeline(steps=[("Preprocessor", preprocessor), ("Estimator", clf)])

        fold_probas = []
        oob_fold_probas = []
        if not f"{clf_name}_metadata" in results_dict.keys():
            results_dict[f"{clf_name}_metadata"] = {}
        results_dict[f"{clf_name}_metadata"]["train_times"] = []
        results_dict[f"{clf_name}_metadata"]["test_times"] = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if args.vary_samples:
                stratified_sort = stratify_samplesizes(y_train, sample_sizes)
                X_train = X_train[stratified_sort]
                y_train = y_train[stratified_sort]

            probas_vs_sample_sizes = []
            oob_probas_vs_sample_sizes = []

            for n_samples in sample_sizes:
                start_time = time.time()
                # Fix too few samples for internal CV of these methods
                if (
                    clf_name in ["IRF", "SigRF"]
                    and np.min(np.unique(y_train[:n_samples], return_counts=True)[1]) < 5
                ):
                    print(
                        f"{clf_name} requires more samples of minimum class. Skipping n={n_samples}"
                    )
                    y_proba = np.repeat(
                        np.bincount(y_train[:n_samples]).reshape(1, -1) / len(y_train[:n_samples]),
                        X_test.shape[0],
                        axis=0,
                    )
                    # y_proba_oob = y_proba
                    train_time = time.time() - start_time
                else:
                    pipeline = pipeline.fit(X_train[:n_samples], y_train[:n_samples])
                    train_time = time.time() - start_time
                    y_proba = pipeline.predict_proba(X_test)
                    # y_proba_oob = predict_proba_oob(pipeline['Estimator'], pipeline['Preprocessor'].transform(X_train[:n_samples]))

                test_time = time.time() - (train_time + start_time)

                probas_vs_sample_sizes.append(y_proba)
                # oob_probas_vs_sample_sizes.append(y_proba_oob)
                results_dict[f"{clf_name}_metadata"]["train_times"].append(train_time)
                results_dict[f"{clf_name}_metadata"]["test_times"].append(test_time)

            fold_probas.append(probas_vs_sample_sizes)
            # oob_fold_probas.append(oob_probas_vs_sample_sizes)

        results_dict[clf_name] = fold_probas
        # results_dict[clf_name + '_oob'] = oob_fold_probas
        print(
            f"{clf_name} Time: train_time={train_time:.3f}, test_time={test_time:.3f}, Cohen Kappa={cohen_kappa_score(y_test, y_proba.argmax(1)):.3f}"
        )

    # If existing data, load and append to. Else save
    if os.path.isfile(save_path) and args.mode == "OVERWRITE":
        logging.info(f"OVERWRITING {task_name} ({task_id})")
        with open(save_path, "rb") as f:
            prior_results = pickle.load(f)

        # Check these keys have the same values
        verify_keys = [
            "task",
            "task_id",
            "n_samples",
            "n_features",
            "n_classes",
            "y",
            "test_indices",
            "n_estimators",
            "cv",
            "nominal_features",
            "n_features_fitted",
            "sample_sizes",
        ]
        for key in verify_keys:
            assert _check_nested_equality(
                prior_results[key], results_dict[key]
            ), f"OVERWRITE {key} does not match saved value"

        # Replace/add data
        replace_keys = [name for name, _ in clfs]
        replace_keys += [f"{name}_metadata" for name in replace_keys]
        for key in replace_keys:
            prior_results[key] = results_dict[key]

        results_dict = prior_results

    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)


def run_cc18(args, clfs, data_dir):
    logging.basicConfig(
        filename="run_all.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )

    for key, val in vars(args).items():
        logging.info(f"{key}={val}")

    benchmark_suite = openml.study.get_suite("OpenML-CC18")  # obtain the benchmark suite

    folder = data_dir / f"sporf_benchmarks/results_cv{args.cv}_features={args.max_features}_{name}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    def _run_task_helper(task_id):
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        task_name = task.get_dataset().name

        save_path = f"{folder}/{task_name}_results_dict.pkl"
        if args.mode == "OVERWRITE":
            if not os.path.isfile(save_path):
                logging.info(f"OVERWRITE MODE: Skipping {task_name} (doesn't  exist)")
                return
        elif args.mode == "APPEND" and os.path.isfile(save_path):
            logging.info(f"APPEND MODE: Skipping {task_name} (already exists)")
            return

        print(f"{args.mode} {task_name} ({task_id})")
        logging.info(f"Running {task_name} ({task_id})")

        X, y = task.get_X_and_y()  # get the data

        nominal_indices = task.get_dataset().get_features_by_type("nominal", [task.target_name])
        try:
            train_test(X, y, task_name, task_id, nominal_indices, args, clfs, save_path)
        except Exception as e:
            print(
                f"Test {task_name} ({task_id}) Failed | X.shape={X.shape} | {len(nominal_indices)} nominal indices"
            )
            print(e)
            logging.error(
                f"Test {task_name} ({task_id}) Failed | X.shape={X.shape} | {len(nominal_indices)} nominal indices"
            )
            import traceback

            logging.error(e)
            traceback.sprint_exc()

    task_ids_to_run = []
    for task_id in benchmark_suite.tasks:
        if args.start_id is not None and task_id < args.start_id:
            print(f"Skipping task_id={task_id}")
            logging.info(f"Skipping task_id={task_id}")
            continue
        if args.stop_id is not None and task_id >= args.stop_id:
            print(f"Stopping at task_id={task_id}")
            logging.info(f"Stopping at task_id={task_id}")
            break
        task_ids_to_run.append(task_id)

    if args.parallel_tasks is not None and args.parallel_tasks > 1:
        Parallel(n_jobs=args.parallel_tasks, verbose=1)(
            delayed(_run_task_helper)(task_id) for task_id in tqdm(task_ids_to_run)
        )
    else:
        for task_id in tqdm(task_ids_to_run):  # iterate over all tasks
            _run_task_helper(task_id)


parser = argparse.ArgumentParser(description="Run CC18 dataset.")
parser.add_argument(
    "--mode", action="store", default="CREATE", choices=["OVERWRITE", "CREATE", "APPEND"]
)
parser.add_argument("--cv", action="store", type=int, default=10)
parser.add_argument("--n_estimators", action="store", type=int, default=500)
parser.add_argument("--n_jobs", action="store", type=int, default=6)
# parser.add_argument("--uf_kappa", action="store", type=float, default=None)
# parser.add_argument("--uf_construction_prop", action="store", type=float, default=0.63)
# parser.add_argument("--uf_max_samples", action="store", type=float, default=1.0)
parser.add_argument(
    "--max_features",
    action="store",
    default=None,
    help="Either an integer, float, or string in {'sqrt', 'log2'}. Default uses all features.",
)
# parser.add_argument("--uf_poisson", action="store_true", default=False)
parser.add_argument("--start_id", action="store", type=int, default=None)
parser.add_argument("--stop_id", action="store", type=int, default=None)
# parser.add_argument("--honest_prior", action="store", default="ignore", choices=["ignore", "uniform", "empirical"])
parser.add_argument("--parallel_tasks", action="store", default=1, type=int)
parser.add_argument("--vary_samples", action="store_true", default=False)


args = parser.parse_args()

max_features = args.max_features
try:
    max_features = int(max_features)
except:
    try:
        max_features = float(max_features)
    except:
        pass

clfs = [
    (
        "RF",
        RandomForestClassifier(
            n_estimators=args.n_estimators, max_features=max_features, n_jobs=args.n_jobs
        ),
    ),
    # (
    #     "Oblique-RF",
    #     ObliqueRF(
    #         n_estimators=args.n_estimators,
    #         max_features=max_features,
    #         n_jobs=args.n_jobs)
    # ),
    (
        "SPORF",
        ObliqueSPORF(
            n_estimators=args.n_estimators,
            max_features=max_features,
            feature_combinations=1.5,
            n_jobs=args.n_jobs,
        ),
    ),
]


data_dir = Path("/mnt/ssd3/ronan/")
data_dir = Path("/home/adam2392/Downloads/")
run_cc18(args, clfs, data_dir)
# python run_all.py --mode CREATE --n_jobs 45 --max_features sqrt --stop_id 29 --honest_prior ignore --uf_kappa 1.5
# python run_all.py --mode CREATE --n_jobs 45 --max_features 0.33 --honest_prior ignore
# python run_all.py --mode CREATE --n_jobs 25 --max_features 0.33 --stop_id 220 --honest_prior ignore --vary_samples --parallel_tasks 20 --cv 5

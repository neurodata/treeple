from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.utils.validation import check_X_y

from sktree._lib.sklearn.ensemble._forest import ForestClassifier
from sktree._lib.sklearn.tree import DecisionTreeClassifier


def _mutual_information(y_true, y_pred_proba):
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return max(H_Y - H_YX, 0)


METRIC_FUNCTIONS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "auc": roc_auc_score,
    "mi": _mutual_information,
    "balanced_accuracy": balanced_accuracy_score,
}
REGRESSOR_METRICS = ("mse", "mae")


def train_tree(
    tree: DecisionTreeClassifier,
    X: ArrayLike,
    y: ArrayLike,
    covariate_index: ArrayLike = None,
) -> ArrayLike:
    """Compute the posterior from each tree on the "OOB" samples.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        The tree to compute the posterior from.
    X : ArrayLike of shape (n_samples, n_features)
        The data matrix.
    y : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    covariate_index : ArrayLike of shape (n_covariates,), optional
        The indices of the covariates to permute, by default None, which
        does not permute any columns.
    """
    # seed the random number generator using each tree's random seed(?)
    rng = np.random.default_rng(tree.random_state)

    indices = np.arange(X.shape[0], dtype=int)

    if covariate_index is not None:
        # perform permutation of covariates
        index_arr = rng.choice(indices, size=(X.shape[0], 1), replace=False, shuffle=True)
        perm_X_cov = X[index_arr, covariate_index]
        X[:, covariate_index] = perm_X_cov

    # individual tree permutation of y labels
    tree.fit(X, y, check_input=False)


def _compute_null_distribution_perm(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    covariate_index: ArrayLike,
    est: ForestClassifier,
    metric: str = "mse",
    n_repeats: int = 1000,
    seed: int = None,
) -> ArrayLike:
    """Compute null distribution using permutation method.

    Parameters
    ----------
    X_test : ArrayLike of shape (n_samples, n_features)
        The data matrix.
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    covariate_index : ArrayLike of shape (n_covariates,)
        The indices of the covariates to permute.
    est : ForestClassifier
        The forest that will be used to recompute the test statistic on the permuted data.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.
    """
    rng = np.random.default_rng(seed)
    X_test, y_test = check_X_y(X_test, y_test, ensure_2d=True, multi_output=True)
    n_samples_test = len(y_test)
    n_samples_train = len(y_train)
    metric_func = METRIC_FUNCTIONS[metric]

    # pre-allocate memory for the index array
    train_index_arr = np.arange(n_samples_train, dtype=int).reshape(-1, 1)
    test_index_arr = np.arange(n_samples_test, dtype=int).reshape(-1, 1)

    X = np.concatenate((X_train, X_test), axis=0)
    null_metrics = np.zeros((n_repeats,))

    for idx in range(n_repeats):
        # permute the covariates inplace
        rng.shuffle(test_index_arr)
        perm_X_cov = X_test[test_index_arr, covariate_index]
        X_test[:, covariate_index] = perm_X_cov

        rng.shuffle(train_index_arr)
        perm_X_cov = X_train[train_index_arr, covariate_index]
        X_train[:, covariate_index] = perm_X_cov

        # train a new forest on the permuted data
        # XXX: should there be a train/test split here? even w/ honest forests?
        est.fit(X_train, y_train.ravel())
        y_pred = est.predict(X_test)

        # compute two instances of the metric from the sampled trees
        metric_val = metric_func(y_test, y_pred)

        null_metrics[idx] = metric_val
    return null_metrics


def _compute_null_distribution_coleman(
    y_test: ArrayLike,
    y_pred_proba_normal: ArrayLike,
    y_pred_proba_perm: ArrayLike,
    metric: str = "mse",
    n_repeats: int = 1000,
    seed: int = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute null distribution using Coleman method.

    The null distribution is comprised of two forests.

    Parameters
    ----------
    X_test : ArrayLike of shape (n_samples, n_features)
        The data matrix.
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    y_pred_proba_normal : ArrayLike of shape (n_samples_normal, n_outputs)
        The predicted posteriors from the normal forest.
    y_pred_proba_perm : ArrayLike of shape (n_samples_perm, n_outputs)
        The predicted posteriors from the permuted forest.
    normal_samples : ArrayLike of shape (n_samples_normal,)
        The indices of the normal samples that we have a posterior for.
    perm_samples : ArrayLike of shape (n_samples_perm,)
        The indices of the permuted samples that we have a posterior for.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    metric_star : ArrayLike of shape (n_samples,)
        An array of the metrics computed on half the trees.
    metric_star_pi : ArrayLike of shape (n_samples,)
        An array of the metrics computed on the other half of the trees.
    """
    rng = np.random.default_rng(seed)
    # X_test, y_test = check_X_y(X_test, y_test, copy=True, ensure_2d=True, multi_output=True)

    metric_func = METRIC_FUNCTIONS[metric]

    # sample two sets of equal number of trees from the combined forest these are the posteriors
    all_y_pred = np.concatenate((y_pred_proba_normal, y_pred_proba_perm), axis=0)

    n_samples_test = len(y_test)
    assert len(all_y_pred) == 2 * n_samples_test

    # create two stacked index arrays of y_test resulting in [1, ..., N, 1, ..., N]
    y_test_ind_arr = np.hstack(
        (np.arange(n_samples_test, dtype=int), np.arange(n_samples_test, dtype=int))
    )

    # create index array of [1, ..., 2N] to slice into `all_y_pred`
    y_pred_ind_arr = np.arange((2 * n_samples_test), dtype=int)

    # # get the indices of the samples that we have a posterior for, so each element
    # # is an index into `y_test`
    # all_samples_pred = np.concatenate((normal_samples, perm_samples), axis=0)

    # n_samples_final = len(all_samples_pred)

    # pre-allocate memory for the index array
    # index_arr = np.arange(n_samples_final, dtype=int)

    metric_star = np.zeros((n_repeats,))
    metric_star_pi = np.zeros((n_repeats,))
    for idx in range(n_repeats):
        # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
        rng.shuffle(y_pred_ind_arr)

        first_forest_inds = y_pred_ind_arr[:n_samples_test]
        second_forest_inds = y_pred_ind_arr[:n_samples_test]

        # index into y_test for first half and second half
        first_half_index_test = y_test_ind_arr[first_forest_inds]
        second_half_index_test = y_test_ind_arr[second_forest_inds]

        # now get the pointers to the actual samples used for the metric
        y_test_first_half = y_test[first_half_index_test]
        y_test_second_half = y_test[second_half_index_test]

        # compute two instances of the metric from the sampled trees
        first_half_metric = metric_func(y_test_first_half, all_y_pred[first_forest_inds])
        second_half_metric = metric_func(y_test_second_half, all_y_pred[second_forest_inds])

        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return metric_star, metric_star_pi

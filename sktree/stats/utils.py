from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.utils.validation import check_X_y

from sktree._lib.sklearn.ensemble._forest import ForestClassifier


def _mutual_information(y_true, y_pred):
    H_YX = np.mean(entropy(y_pred, base=np.exp(1)))
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return max(H_Y - H_YX, 0)


METRIC_FUNCTIONS = {"mse": mean_squared_error, "auc": roc_auc_score, "mi": _mutual_information}


def _pvalue(observe_stat: float, permuted_stat: ArrayLike, correction: bool = True) -> float:
    """Compute pvalue with Coleman method.

    Implements the pvalue calculation from Algorithm 1. See
    :footcite:`coleman2022scalable` for full details.

    Parameters
    ----------
    observe_stat : float
        The observed test statistic.
    permuted_stat : ArrayLike of shape (n_repeats,)
        The array of test statistics computed on permutations.
    correction : bool
        Whether to use correction and add 1 to the numerator and denominator, by default True.

    Returns
    -------
    pval : float
        The pvalue.
    """
    n_repeats = len(permuted_stat)
    if correction:
        pval = (1 + (permuted_stat >= observe_stat).sum()) / (1 + n_repeats)
    else:
        pval = (permuted_stat >= observe_stat).sum() / n_repeats
    return pval


def compute_null_distribution_perm(
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
    X_test, y_test = check_X_y(X_test, y_test, ensure_2d=True)
    n_samples = len(y_test)

    metric_func = METRIC_FUNCTIONS[metric]

    # pre-allocate memory for the index array
    index_arr = np.arange(n_samples * 2, dtype=int)

    null_metrics = np.zeros((n_repeats,))

    for idx in range(n_repeats):
        # permute the covariates inplace
        rng.shuffle(index_arr)
        perm_X_cov = X_test[index_arr, covariate_index]
        X_test[:, covariate_index] = perm_X_cov

        # train a new forest on the permuted data
        # XXX: should there be a train/test split here? even w/ honest forests?
        est.fit(X_test, y_test)
        y_pred_proba = est.predict_proba(X_test)

        # compute two instances of the metric from the sampled trees
        metric_val = metric_func(y_true=y_test, y_pred=y_pred_proba)

        null_metrics[idx] = metric_val
    return null_metrics


def _compute_null_distribution_coleman(
    X_test: ArrayLike,
    y_test: ArrayLike,
    y_pred_proba_normal: ArrayLike,
    y_pred_proba_perm: ArrayLike,
    normal_samples: ArrayLike,
    perm_samples: ArrayLike,
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

    # sample two sets of equal number of trees from the combined forest
    all_y_pred = np.concatenate((y_pred_proba_normal, y_pred_proba_perm), axis=0)

    # get the indices of the samples that we have a posterior for, so each element
    # is an index into `y_test`
    all_samples_pred = np.concatenate((normal_samples, perm_samples), axis=0)

    n_samples_final = len(all_samples_pred)

    # pre-allocate memory for the index array
    index_arr = np.arange(n_samples_final, dtype=int)

    metric_star = np.zeros((n_repeats,))
    metric_star_pi = np.zeros((n_repeats,))
    for idx in range(n_repeats):
        # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
        rng.shuffle(index_arr)
        first_half_index = index_arr[: n_samples_final // 2]
        second_half_index = index_arr[n_samples_final // 2 :]

        # now get the pointers to the actual samples used for the metric
        y_test_first_half = y_test[all_samples_pred[first_half_index]]
        y_test_second_half = y_test[all_samples_pred[second_half_index]]

        # compute two instances of the metric from the sampled trees
        first_half_metric = metric_func(y_test_first_half, all_y_pred[first_half_index])
        second_half_metric = metric_func(y_test_second_half, all_y_pred[second_half_index])

        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return metric_star, metric_star_pi

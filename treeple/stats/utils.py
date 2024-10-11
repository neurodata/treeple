import importlib.util
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.validation import check_is_fitted

from treeple._lib.sklearn.ensemble._forest import BaseForest

BOTTLENECK_AVAILABLE = False
if importlib.util.find_spec("bottleneck"):
    import bottleneck as bn

    BOTTLENECK_AVAILABLE = True


BOTTLENECK_WARNING = (
    "Not using bottleneck for calculations involvings nans. Expect slower performance."
)
DISABLE_BN_ENV_VAR = "TREEPLE_NO_BOTTLENECK"

if BOTTLENECK_AVAILABLE and DISABLE_BN_ENV_VAR not in os.environ:

    def nanmean_f(arr: ArrayLike, axis=None) -> ArrayLike:
        return bn.nanmean(arr, axis=axis)

    def anynan_f(arr: ArrayLike) -> ArrayLike:
        return bn.anynan(arr, axis=2)

else:

    def nanmean_f(arr: ArrayLike, axis=None) -> ArrayLike:
        warnings.warn(BOTTLENECK_WARNING)
        return np.nanmean(arr, axis=axis)

    def anynan_f(arr: ArrayLike) -> ArrayLike:
        warnings.warn(BOTTLENECK_WARNING)
        return np.isnan(arr).any(axis=2)


def _mutual_information(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of mutual information for supervised classification setting.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.

    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def _cond_entropy(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of entropy for supervised classification setting.

    H(Y | X)

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels. Not used in computation of the entropy.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.

    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    return H_YX


# Define function to compute the sensitivity at 98% specificity
def _SA98(y_true: ArrayLike, y_pred_proba: ArrayLike, max_fpr=0.02) -> float:
    """Compute the sensitivity at 98% specificity.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.
    max_fpr : float, optional. Default=0.02.

    Returns
    -------
    float :
        The estimated SA98.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    s98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])
    return s98


METRIC_FUNCTIONS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "balanced_accuracy": balanced_accuracy_score,
    "auc": roc_auc_score,
    "mi": _mutual_information,
    "cond_entropy": _cond_entropy,
    "s@98": _SA98,
}

POSTERIOR_FUNCTIONS = ("mi", "auc", "cond_entropy", "s@98")

POSITIVE_METRICS = ("mi", "auc", "balanced_accuracy", "s@98")

REGRESSOR_METRICS = ("mse", "mae")


def _non_nan_samples(posterior_arr: ArrayLike) -> ArrayLike:
    """Determine which samples are not nan in the posterior tree array.

    Parameters
    ----------
    posterior_arr : ArrayLike of shape (n_trees, n_samples, n_outputs)
        The 3D posterior array from the forest.

    Returns
    -------
    nonnan_indices : ArrayLike of shape (n_nonnan_samples,)
        The indices of the samples that are not nan in the posterior array
        along axis=1.
    """
    # Find the row indices with NaN values along the specified axis
    nan_indices = anynan_f(posterior_arr).all(axis=0)

    # Invert the boolean mask to get indices without NaN values
    nonnan_indices = np.where(~nan_indices)[0]
    return nonnan_indices


def _compute_null_distribution_coleman(
    y_test: ArrayLike,
    y_pred_proba_normal: ArrayLike,
    y_pred_proba_perm: ArrayLike,
    metric: str = "mse",
    n_repeats: int = 1000,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    **metric_kwargs,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute null distribution using Coleman method.

    The null distribution is comprised of two forests.

    Parameters
    ----------
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    y_pred_proba_normal : ArrayLike of shape (n_estimators_normal, n_samples, n_outputs)
        The predicted posteriors from the normal forest. Some of the trees
        may have nans predicted in them, which means the tree used these samples
        for training and not for prediction.
    y_pred_proba_perm : ArrayLike of shape (n_estimators_perm, n_samples, n_outputs)
        The predicted posteriors from the permuted forest. Some of the trees
        may have nans predicted in them, which means the tree used these samples
        for training and not for prediction.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.
    metric_kwargs : dict, optional
        Keyword arguments to pass to the metric function.

    Returns
    -------
    metric_star : ArrayLike of shape (n_samples,)
        An array of the metrics computed on half the trees.
    metric_star_pi : ArrayLike of shape (n_samples,)
        An array of the metrics computed on the other half of the trees.
    """
    # sample two sets of equal number of trees from the combined forest these are the posteriors
    # (n_estimators * 2, n_samples, n_outputs)
    all_y_pred = np.concatenate((y_pred_proba_normal, y_pred_proba_perm), axis=0)

    n_estimators, _, _ = y_pred_proba_normal.shape
    n_samples_test = len(y_test)
    if all_y_pred.shape[1] != n_samples_test:
        raise RuntimeError(
            f"The number of samples in `all_y_pred` {len(all_y_pred)} "
            f"is not equal to 2 * n_samples_test {2 * n_samples_test}"
        )

    # create index array of [1, ..., 2N] to slice into `all_y_pred` the stacks of trees
    y_pred_ind_arr = np.arange((2 * n_estimators), dtype=int)

    metric_star = np.zeros((n_repeats,))
    metric_star_pi = np.zeros((n_repeats,))

    # generate the random seeds for the parallel jobs
    ss = np.random.SeedSequence(seed)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_parallel_build_null_forests)(
            y_pred_ind_arr,
            n_estimators,
            all_y_pred,
            y_test,
            seed,
            metric,
            **metric_kwargs,
        )
        for i, seed in zip(range(n_repeats), ss.spawn(n_repeats))
    )

    for idx, (first_half_metric, second_half_metric) in enumerate(out):
        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return metric_star, metric_star_pi


def _parallel_build_null_forests(
    index_arr: ArrayLike,
    n_estimators: int,
    all_y_pred: ArrayLike,
    y_test: ArrayLike,
    seed: int,
    metric: str,
    **metric_kwargs: dict,
):
    """Randomly sample two sets of forests and compute the metric on each."""
    rng = np.random.default_rng(seed)
    metric_func = METRIC_FUNCTIONS[metric]

    # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
    rng.shuffle(index_arr)

    # get random half of the posteriors from two sets of trees
    first_forest_inds = index_arr[: n_estimators // 2]
    second_forest_inds = index_arr[n_estimators // 2 :]

    # get random half of the posteriors as one forest
    first_forest_pred = all_y_pred[first_forest_inds, ...]
    second_forest_pred = all_y_pred[second_forest_inds, ...]

    # determine if there are any nans in the final posterior array, when
    # averaged over the trees
    first_forest_samples = _non_nan_samples(first_forest_pred)
    second_forest_samples = _non_nan_samples(second_forest_pred)

    # todo: is this step necessary?
    # non_nan_samples = np.intersect1d(
    #     first_forest_samples, second_forest_samples, assume_unique=True
    # )
    # now average the posteriors over the trees for the non-nan samples
    # y_pred_first_half = np.nanmean(first_forest_pred[:, non_nan_samples, :], axis=0)
    # y_pred_second_half = np.nanmean(second_forest_pred[:, non_nan_samples, :], axis=0)
    # # compute two instances of the metric from the sampled trees
    # first_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_first_half)
    # second_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_second_half)

    y_pred_first_half = nanmean_f(first_forest_pred[:, first_forest_samples, :], axis=0)
    y_pred_second_half = nanmean_f(second_forest_pred[:, second_forest_samples, :], axis=0)

    # compute two instances of the metric from the sampled trees
    first_half_metric = metric_func(
        y_test[first_forest_samples, :], y_pred_first_half, **metric_kwargs
    )
    second_half_metric = metric_func(
        y_test[second_forest_samples, :], y_pred_second_half, **metric_kwargs
    )
    return first_half_metric, second_half_metric


def get_per_tree_oob_samples(est: BaseForest):
    """The sample indices that are out-of-bag.

    Only utilized if ``bootstrap=True``, otherwise, all samples are "in-bag".
    """
    check_is_fitted(est)
    if est.bootstrap is False:
        raise RuntimeError("Cannot extract out-of-bag samples when bootstrap is False.")
    est._n_samples
    oob_samples = []
    n_samples_bootstrap = _get_n_samples_bootstrap(
        est._n_samples,
        est.max_samples,
    )
    for estimator in est.estimators_:
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state,
            est._n_samples,
            n_samples_bootstrap,
        )
        oob_samples.append(unsampled_indices)
    return oob_samples


def _get_forest_preds_sparse(
    all_y_pred: sp.csc_matrix,  # (n_trees, n_samples)
    all_y_indicator: sp.csc_matrix,  # (n_trees, n_samples)
    forest_indices: ArrayLike,  # (n_trees/2,)
) -> ArrayLike:
    """Get the forest predictions for a set of trees using sparse matrices.

    Parameters
    ----------
    all_y_pred : sp.csc_matrix of shape (n_trees, n_samples)
        The predicted posteriors from the forest.
    all_y_indicator : sp.csc_matrix of shape (n_trees, n_samples)
        The indicator matrix for the predictions.
    forest_indices : ArrayLike of shape (n_trees/2,)
        The indices of the trees in the forest that we are evaluating.

    Returns
    -------
    ArrayLike of shape (n_samples,)
        The averaged predictions for the forest.
    """
    forest_indicator = np.zeros(len(forest_indices) * 2, dtype=np.uint8)
    forest_indicator[forest_indices] = 1

    forest_predictions = forest_indicator @ all_y_pred  # (n_samples)
    forest_counts = forest_indicator @ all_y_indicator  # (n_samples)

    return np.where(forest_counts, forest_predictions / forest_counts, np.nan)  # (n_samples)


def _parallel_build_null_forests_sparse(
    index_arr: ArrayLike,
    all_y_pred: ArrayLike,
    all_y_indicator: ArrayLike,
    y_test: ArrayLike,
    n_outputs: int,
    seed: Optional[int],
    shuffle: bool,
    metric: str,
    **metric_kwargs: dict,
) -> tuple[float, float]:
    """Randomly sample two sets of forests and compute the metric on each using sparse matrices."""
    metric_func = METRIC_FUNCTIONS[metric]

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(index_arr)

    first_forest_inds = index_arr[: len(index_arr) // 2]
    second_forest_inds = index_arr[len(index_arr) // 2 :]

    y_pred_first_half = _get_forest_preds_sparse(
        all_y_pred,
        all_y_indicator,
        first_forest_inds,
    )
    y_pred_second_half = _get_forest_preds_sparse(
        all_y_pred,
        all_y_indicator,
        second_forest_inds,
    )

    first_pred_mask = np.isfinite(y_pred_first_half)
    second_pred_mask = np.isfinite(y_pred_second_half)

    # if this is binary classification, we add the second class
    if n_outputs > 1:
        y_pred_first_half = np.column_stack([y_pred_first_half, 1 - y_pred_first_half])
        y_pred_second_half = np.column_stack([y_pred_second_half, 1 - y_pred_second_half])

    first_half_metric = metric_func(
        y_test[first_pred_mask],
        y_pred_first_half[first_pred_mask],
        **metric_kwargs,
    )
    second_half_metric = metric_func(
        y_test[second_pred_mask],
        y_pred_second_half[second_pred_mask],
        **metric_kwargs,
    )

    return first_half_metric, second_half_metric


def _compute_null_distribution_coleman_sparse(
    y_test: ArrayLike,  # (n_samples, n_outputs)
    y_pred_proba_normal: Tuple[ArrayLike, ArrayLike],  # n_trees, n_oob_samples_tree, n_outputs
    y_pred_proba_perm: Tuple[ArrayLike, ArrayLike],  # n_trees, n_oob_samples_tree, n_outputs
    metric: str = "mse",
    n_repeats: int = 1000,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    **metric_kwargs,
) -> Tuple[ArrayLike, float, float, ArrayLike, ArrayLike]:
    """Compute null distribution using Coleman method.

    The null distribution is comprised of two forests.

    Parameters
    ----------
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    y_pred_proba_normal : Tuple of (ArrayLike, ArrayLike)
        The predicted posteriors from the normal forest. The first element is a
        list of out-of-bag indices for each tree. The second element is a list of
        the predicted posteriors for each tree.
    y_pred_proba_perm : Tuple of (ArrayLike, ArrayLike)
        The predicted posteriors from the permuted forest. The first element is a
        list of out-of-bag indices for each tree. The second element is a list of
        the predicted posteriors for each tree.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.
    n_jobs : int, optional
        The number of parallel jobs to run, by default None.
    metric_kwargs : dict, optional
        Keyword arguments to pass to the metric function.

    Returns
    -------
    oob_predictions : ArrayLike of shape (n_trees, n_samples)
        The out-of-bag predictions. The first n_trees/2 are from the normal forest
        and the second n_trees/2 are from the permuted forest.
    observe_stat : float
        The observed statistic.
    permute_stat : float
        The permuted statistic.
    metric_star : ArrayLike of shape (n_samples,)
        An array of the metrics computed on half the trees.
    metric_star_pi : ArrayLike of shape (n_samples,)
        An array of the metrics computed on the other half of the trees.
    """
    n_trees = len(y_pred_proba_normal[0]) + len(y_pred_proba_perm[0])
    n_outputs = y_pred_proba_normal[1][0].shape[1]

    all_oob_idxs = y_pred_proba_normal[0] + y_pred_proba_perm[0]
    tree_row_ids = np.repeat(
        np.arange(n_trees, dtype=np.int32),
        [len(oob) for oob in all_oob_idxs],
    )  # (n_oob_samples)
    all_oob_idxs = np.concatenate(all_oob_idxs, axis=0)  # (n_oob_samples)

    # XXX: Because we are assuming 2D sparse matrices for binary classification or
    # regression, we can assume that the number of outputs is 1. This is necessary
    # to use scipy's sparse matrix format. However pydata has a general sparse
    # matrix format that can be used for multi-output problems, but currently uses
    # more memory.
    all_oob_values = np.concatenate(
        y_pred_proba_normal[1] + y_pred_proba_perm[1],
        axis=0,
    )[
        :, 0
    ]  # (n_oob_samples)

    oob_predictions = sp.csc_matrix(
        (all_oob_values, (tree_row_ids, all_oob_idxs)),
        shape=(n_trees, len(y_test)),
    )  # (n_trees, n_samples)

    oob_indicators = sp.csc_matrix(
        (np.ones_like(all_oob_values, dtype=np.uint8), (tree_row_ids, all_oob_idxs)),
        shape=(n_trees, len(y_test)),
    )  # (n_trees, n_samples)

    observe_stat, permute_stat = _parallel_build_null_forests_sparse(
        np.arange(n_trees),
        oob_predictions,
        oob_indicators,
        y_test,
        n_outputs,
        seed,
        False,
        metric,
        **metric_kwargs,
    )

    # generate the random seeds for the parallel jobs
    ss = np.random.SeedSequence(seed)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_parallel_build_null_forests_sparse)(
            np.arange(n_trees),
            oob_predictions,
            oob_indicators,
            y_test,
            n_outputs,
            seed,
            True,
            metric,
            **metric_kwargs,
        )
        for _, seed in zip(range(n_repeats), ss.spawn(n_repeats))
    )

    metric_star = np.zeros((n_repeats,))
    metric_star_pi = np.zeros((n_repeats,))

    for idx, (first_half_metric, second_half_metric) in enumerate(out):
        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return oob_predictions, observe_stat, permute_stat, metric_star, metric_star_pi

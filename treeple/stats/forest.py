import threading
from collections import namedtuple
from typing import Callable
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.multiclass import type_of_target

from .._lib.sklearn.ensemble._forest import ForestClassifier
from ..ensemble import HonestForestClassifier
from ..tree import MultiViewDecisionTreeClassifier
from ..tree._classes import DTYPE
from .permuteforest import PermutationHonestForestClassifier
from .utils import (
    METRIC_FUNCTIONS,
    POSITIVE_METRICS,
    _compute_null_distribution_coleman,
    _compute_null_distribution_coleman_sparse,
)


def _parallel_predict_proba_oob(predict_proba, X, out, idx, test_idx, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    # each tree predicts proba with a list of output (n_samples, n_classes[i])
    prediction = predict_proba(X, check_input=False)

    indices = np.zeros(X.shape[0], dtype=bool)
    indices[test_idx] = True
    with lock:
        out[idx, test_idx, :] = prediction[test_idx, :]
    return prediction


def _parallel_predict_proba_oob_sparse(predict_proba, X, test_idx):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there. Different from
    _parallel_predict_proba_oob, this function returns the predictions and
    indices for those values that are finite (not nan or inf).
    """
    # each tree predicts proba with a list of output (n_samples, n_classes[i])
    prediction = predict_proba(X[test_idx, :], check_input=False)
    good_value_mask = np.isfinite(prediction[:, 0])
    return test_idx[good_value_mask], prediction[good_value_mask]


ForestTestResult = namedtuple(
    "ForestTestResult",
    ["observe_test_stat", "permuted_stat", "observe_stat", "pvalue", "null_dist"],
)


def build_coleman_forest(
    est: HonestForestClassifier,
    perm_est: PermutationHonestForestClassifier,
    X,
    y,
    covariate_index=None,
    metric="s@98",
    n_repeats=10_000,
    verbose=False,
    seed=None,
    return_posteriors=True,
    use_sparse=False,
    **metric_kwargs,
):
    """Build a hypothesis testing forest using a two-forest approach.

    The two-forest approach stems from the Coleman et al. 2022 paper, where
    two forests are trained: one on the original dataset, and one on the
    permuted dataset. The dataset is either permuted once, or independently for
    each tree in the permuted forest. The original test statistic is computed by
    comparing the metric on both forests ``(metric_forest - metric_perm_forest)``.
    For full details, see :footcite:`coleman2022scalable`.

    Parameters
    ----------
    est : Forest
        The type of forest to use. Must be enabled with ``bootstrap=True``.
    perm_est : Forest
        The forest to use for the permuted dataset.
    X : ArrayLike of shape (n_samples, n_features)
        Data.
    y : ArrayLike of shape (n_samples, n_outputs)
        Binary target, so ``n_outputs`` should be at most 1.
    covariate_index : ArrayLike, optional of shape (n_covariates,)
        The index array of covariates to shuffle, by default None, which
        defaults to all covariates.
    metric : str, optional
        The metric to compute, by default "s@98", for sensitivity at
        98% specificity.
    n_repeats : int, optional
        Number of times to bootstrap sample the two forests to construct
        the null distribution, by default 10000. The construction of the
        null forests will be parallelized according to the ``n_jobs``
        argument of the ``est`` forest.
    verbose : bool, optional
        Verbosity, by default False.
    seed : int, optional
        Random seed, by default None.
    return_posteriors : bool, optional
        Whether or not to return the posteriors, by default True.
    use_sparse : bool, optional
        Whether or not to use a sparse for the calculation of the permutation
        statistics, by default False. Doesn't affect return values.
    **metric_kwargs : dict, optional
        Additional keyword arguments to pass to the metric function.

    Returns
    -------
    observe_stat : float
        The test statistic. To compute the test statistic, take
        ``permute_stat_`` and subtract ``observe_stat_``.
    pvalue : float
        The p-value of the test statistic.
    orig_forest_proba : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each estimator on their
        out of bag samples.
    perm_forest_proba : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each of the permuted estimators
        on their out of bag samples.
    null_dist : ArrayLike of shape (n_repeats,)
        The null statistic differences from permuted forests.

    References
    ----------
    .. footbibliography::
    """
    metric_func: Callable[[ArrayLike, ArrayLike], float] = METRIC_FUNCTIONS[metric]

    if not isinstance(est, HonestForestClassifier):
        raise RuntimeError(f"Original forest must be a HonestForestClassifier, got {type(est)}")

    # build two sets of forests
    est, orig_forest_proba = build_oob_forest(est, X, y, use_sparse=use_sparse, verbose=verbose)

    if not isinstance(perm_est, PermutationHonestForestClassifier):
        raise RuntimeError(
            f"Permutation forest must be a PermutationHonestForestClassifier, got {type(perm_est)}"
        )

    if covariate_index is None and isinstance(est.tree_estimator, MultiViewDecisionTreeClassifier):
        warn(
            "Covariate index is not defined, but a MultiViewDecisionTreeClassifier is used. "
            "If using CoMIGHT, one should define the covariate index to permute. "
            "Defaulting to use MIGHT."
        )

    perm_est, perm_forest_proba = build_oob_forest(
        perm_est,
        X,
        y,
        use_sparse=use_sparse,
        verbose=verbose,
        covariate_index=covariate_index,
    )

    # get the number of jobs
    n_jobs = est.n_jobs

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if use_sparse:
        y_pred_proba_orig_perm, observe_stat, permute_stat, metric_star, metric_star_pi = (
            _compute_null_distribution_coleman_sparse(
                y,
                orig_forest_proba,
                perm_forest_proba,
                metric,
                n_repeats=n_repeats,
                seed=seed,
                n_jobs=n_jobs,
                **metric_kwargs,
            )
        )

        # if we are returning the posteriors, then we need to replace the
        # sparse indices and values with an array. We convert the sparse data
        # to dense data, so that the function returns results in a consistent format.
        if return_posteriors:
            n_trees = y_pred_proba_orig_perm.shape[0] // 2
            n_samples = y_pred_proba_orig_perm.shape[1]

            to_coords_data = lambda x: (x.row.astype(int), x.col.astype(int), x.data)

            row, col, data = to_coords_data(y_pred_proba_orig_perm[:n_trees, :].tocoo())
            orig_forest_proba = np.full((n_trees, n_samples), np.nan, dtype=np.float64)
            orig_forest_proba[row, col] = data

            row, col, data = to_coords_data(y_pred_proba_orig_perm[n_trees:, :].tocoo())
            perm_forest_proba = np.full((n_trees, n_samples), np.nan, dtype=np.float64)
            perm_forest_proba[row, col] = data

            if y.shape[1] == 2:
                orig_forest_proba = np.column_stack((orig_forest_proba, 1 - orig_forest_proba))
                perm_forest_proba = np.column_stack((perm_forest_proba, 1 - perm_forest_proba))
    else:
        metric_star, metric_star_pi = _compute_null_distribution_coleman(
            y,
            orig_forest_proba,
            perm_forest_proba,
            metric,
            n_repeats=n_repeats,
            seed=seed,
            n_jobs=n_jobs,
            **metric_kwargs,
        )

        y_pred_proba_orig = np.nanmean(orig_forest_proba, axis=0)
        y_pred_proba_perm = np.nanmean(perm_forest_proba, axis=0)
        observe_stat = metric_func(y, y_pred_proba_orig, **metric_kwargs)
        permute_stat = metric_func(y, y_pred_proba_perm, **metric_kwargs)

    # metric^\pi - metric = observed test statistic, which under the
    # null is normally distributed around 0
    observe_test_stat = observe_stat - permute_stat

    # metric^\pi_j - metric_j, which is centered at 0
    null_dist = metric_star_pi - metric_star

    # compute pvalue
    if metric in POSITIVE_METRICS:
        pvalue = (1 + (null_dist >= observe_test_stat).sum()) / (1 + n_repeats)
    else:
        pvalue = (1 + (null_dist <= observe_test_stat).sum()) / (1 + n_repeats)

    forest_result = ForestTestResult(
        observe_test_stat, permute_stat, observe_stat, pvalue, null_dist
    )
    if return_posteriors:
        return forest_result, orig_forest_proba, perm_forest_proba, est, perm_est
    else:
        return forest_result


def build_oob_forest(
    est: ForestClassifier, X, y, use_sparse: bool = False, verbose=False, **est_kwargs
):
    """Build a hypothesis testing forest using oob samples.

    Parameters
    ----------
    est : Forest
        The type of forest to use. Must be enabled with ``bootstrap=True``.
        The forest should have either ``oob_samples_`` or ``estimators_samples_``
        property defined, which will be used to compute the out of bag samples
        per tree.
    X : ArrayLike of shape (n_samples, n_features)
        Data.
    y : ArrayLike of shape (n_samples, n_outputs)
        Binary target, so ``n_outputs`` should be at most 1.
    use_sparse : bool, optional
        Whether or not to use a sparse representation for the posteriors.
    verbose : bool, optional
        Verbosity, by default False.
    **est_kwargs : dict, optional
        Additional keyword arguments to pass to the forest estimator ``fit`` function.

    Returns
    -------
    est : Forest
        Fitted forest.
    all_proba : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each estimator on their
        out of bag samples.
    """
    assert est.bootstrap
    assert type_of_target(y) in ("binary")
    est = clone(est)

    # build forest
    est.fit(X, y.ravel(), **est_kwargs)

    # now evaluate
    X = est._validate_X_predict(X)

    # if we trained a binning tree, then we should re-bin the data
    # XXX: this is inefficient and should be improved to be in line with what
    # the Histogram Gradient Boosting Tree does, where the binning thresholds
    # are passed into the tree itself, thus allowing us to set the node feature
    # value thresholds within the tree itself.
    if hasattr(est, "max_bins") and est.max_bins is not None:
        X = est._bin_data(X, is_training_data=False).astype(DTYPE)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(est.n_estimators, est.n_jobs)

    if use_sparse:
        if hasattr(est, "oob_samples_"):
            oob_samples_list = est.oob_samples_
        else:
            inbag_samples = est.estimators_samples_
            all_samples = np.arange(X.shape[0])
            oob_samples_list = [
                np.setdiff1d(all_samples, inbag_samples[i]) for i in range(len(inbag_samples))
            ]

        oob_preds_test_idxs = Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
            delayed(_parallel_predict_proba_oob_sparse)(e.predict_proba, X, test_idx)
            for e, test_idx in zip(est.estimators_, oob_samples_list)
        )
        all_proba = list(zip(*oob_preds_test_idxs))
    else:
        # avoid storing the output of every estimator by summing them here
        lock = threading.Lock()
        # accumulate the predictions across all trees
        all_proba = np.full(
            (len(est.estimators_), X.shape[0], est.n_classes_), np.nan, dtype=np.float64
        )
        if hasattr(est, "oob_samples_"):
            Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
                delayed(_parallel_predict_proba_oob)(
                    e.predict_proba, X, all_proba, idx, test_idx, lock
                )
                for idx, (e, test_idx) in enumerate(zip(est.estimators_, est.oob_samples_))
            )
        else:
            inbag_samples = est.estimators_samples_
            all_samples = np.arange(X.shape[0])
            oob_samples_list = [
                np.setdiff1d(all_samples, inbag_samples[i]) for i in range(len(inbag_samples))
            ]
            Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
                delayed(_parallel_predict_proba_oob)(
                    e.predict_proba, X, all_proba, idx, test_idx, lock
                )
                for idx, (e, test_idx) in enumerate(zip(est.estimators_, oob_samples_list))
            )

    return est, all_proba

import threading

import numpy as np
from numpy.typing import ArrayLike
from joblib import Parallel, delayed
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._forest import BaseForest

from ._lib.sklearn.tree import _tree as _sklearn_tree

DTYPE = _sklearn_tree.DTYPE


def _parallel_predict_proba(predict_proba, X, out, idx, train_idx, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    # each tree predicts proba with a list of output (n_samples, n_classes[i])
    prediction = predict_proba(X, check_input=False)

    test_idx = np.ones(X.shape[0], dtype=bool)
    test_idx[train_idx] = False
    with lock:
        out[idx, test_idx, :] = prediction[test_idx, :]
    return prediction


def predict_proba_per_tree(
    est: BaseForest, X: ArrayLike, oob: bool = True, n_jobs=None, verbose: bool = False
):
    """Predict posterior probabilities for each tree in a classification forest.

    Parameters
    ----------
    est : BaseForest
        Classification forest.
    X : ArrayLike of shape (n_samples, n_features)
        The samples to predict posteriors for.
    oob : bool, optional
        Whether or not to estimate posteriors only for out-of-bag samples, by default True.
    n_jobs : int, optional
        Number of trees to predict in parallel, by default None.
    verbose : bool, optional
        Verbosity, by default False.

    Returns
    -------
    all_proba : ArrayLike of shape (n_estimators, n_samples, n_classes)
        Output posteriors for each sample across each tree.

    Notes
    -----
    This is a similar function to the ``predict_proba`` method of a forest, but
    it returns the posterior probabilities for each tree separately, rather than
    summing them across all trees. This is useful for computing the posterior
    variance of the forest.
    """
    check_is_fitted(est)
    # Check data
    X = est._validate_X_predict(X)

    # if we trained a binning tree, then we should re-bin the data
    # XXX: this is inefficient and should be improved to be in line with what
    # the Histogram Gradient Boosting Tree does, where the binning thresholds
    # are passed into the tree itself, thus allowing us to set the node feature
    # value thresholds within the tree itself.
    if est.max_bins is not None:
        X = est._bin_data(X, is_training_data=False).astype(DTYPE)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(est.n_estimators, n_jobs)

    # avoid storing the output of every estimator by summing them here
    lock = threading.Lock()
    # accumulate the predictions across all trees
    all_proba = np.full(
        (len(est.estimators_), X.shape[0], est.n_classes_), np.nan, dtype=np.float64
    )
    if oob:
        Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
            delayed(_parallel_predict_proba)(e.predict_proba, X, all_proba, idx, train_idx, lock)
            for idx, (e, train_idx) in enumerate(zip(est.estimators_, est.structure_indices_))
        )
    else:
        Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
            delayed(_parallel_predict_proba)(e.predict_proba, X, all_proba, idx, [], lock)
            for idx, e in enumerate(est.estimators_)
        )
    return all_proba

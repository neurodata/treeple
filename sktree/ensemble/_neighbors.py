import numpy as np


# ported from https://github.com/neurodata/hyppo/blob/main/hyppo/independence/_utils.py
def _sim_matrix(X, n_estimators):
    """
    Compute the proximity matrix of samples in X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_estimators)
        For each datapoint x in X and for each tree in the forest,
        is the index of the leaf x ends up in.
    n_estimators : int
        Number of trees in the forest

    Returns
    -------
    prox_matrix : array-like of shape (n_samples, n_samples)
    """
    aff_matrix = sum(np.equal.outer(X[:, i], X[:, i]) for i in range(n_estimators))

    # normalize by the number of trees
    aff_matrix = np.divide(aff_matrix, n_estimators)
    return aff_matrix

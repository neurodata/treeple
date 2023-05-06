import numpy as np
from sklearn_fork.ensemble._forest import BaseForest


def compute_forest_similarity_matrix(forest, X):
    """Compute the similarity matrix of samples in X using a trained forest.

    As an intermediate calculation, the forest applies the dataset and gets
    the leaves for each sample. Then, the similarity matrix is computed by
    counting the number of times each pair of samples ends up in the same leaf.

    Parameters
    ----------
    forest : sklearn.ensemble._forest.BaseForest or sklearn.tree.BaseDecisionTree
        The fitted forest.
    X : array-like of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    aff_matrix : array-like of shape (n_samples, n_samples)
        The estimated distance matrix.
    """
    if isinstance(forest, BaseForest):
        # apply to the leaves
        X_leaves = forest.apply(X)

        n_est = forest.n_estimators
    else:
        # apply to the leaves for a single tree
        X_leaves = forest.apply(X)[:, np.newaxis]
        n_est = 1

    aff_matrix = sum(np.equal.outer(X_leaves[:, i], X_leaves[:, i]) for i in range(n_est))

    # normalize by the number of trees
    aff_matrix = np.divide(aff_matrix, n_est)
    return aff_matrix


# ported from https://github.com/neurodata/hyppo/blob/main/hyppo/independence/_utils.py
class SimMatrixMixin:
    """Mixin class to calculate similarity and dissimilarity matrices."""

    def compute_similarity_matrix_forest(self, X):
        """
        Compute the similarity matrix of samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        sim_matrix : array-like of shape (n_samples, n_samples)
            The similarity matrix among the samples.
        """
        return compute_forest_similarity_matrix(self, X)

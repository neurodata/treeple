import numpy as np


# ported from https://github.com/neurodata/hyppo/blob/main/hyppo/independence/_utils.py
class SimMatrixMixin:
    """Mixin class to calculate similarity and dissimilarity matrices."""

    def _sim_matrix(self, X):
        """
        Compute the proximity matrix of samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            is the index of the leaf x ends up in.

        Returns
        -------
        prox_matrix : array-like of shape (n_samples, n_samples)
        """
        aff_matrix = sum(np.equal.outer(X[:, i], X[:, i]) for i in range(self.n_estimators))

        # normalize by the number of trees
        aff_matrix = np.divide(aff_matrix, self.n_estimators)
        return aff_matrix

    def compute_similarity_matrix_forest(self, X):
        """
        Compute the similarity matrix of samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            is the index of the leaf x ends up in.

        Returns
        -------
        prox_matrix : array-like of shape (n_samples, n_samples)
        """
        return self._sim_matrix(X)

    def compute_dissimilarity_matrix_forest(self, X):
        """
        Compute the dissimilarity matrix of samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            is the index of the leaf x ends up in.

        Returns
        -------
        prox_matrix : array-like of shape (n_samples, n_samples)
        """
        return 1 - self._sim_matrix(X)

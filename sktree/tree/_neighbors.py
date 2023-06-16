import numbers

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from sktree._lib.sklearn.ensemble._forest import BaseForest


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
    if hasattr(forest, "estimator_"):
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


def _compute_distance_matrix(aff_matrix):
    """Private function to compute distance matrix after `compute_similarity_matrix`."""
    dists = 1.0 - aff_matrix
    return dists


# ported from https://github.com/neurodata/hyppo/blob/main/hyppo/independence/_utils.py
class SimMatrixMixin:
    """Mixin class to calculate similarity and dissimilarity matrices.

    This augments tree/forest models with the sklearn's nearest-neighbors API.
    """

    def compute_similarity_matrix(self, X):
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


class NearestNeighborsMetaEstimator(BaseEstimator, MetaEstimatorMixin):
    """Meta-estimator for nearest neighbors.

    Uses a decision-tree, or forest model to compute distances between samples
    and then uses the sklearn's nearest-neighbors API to compute neighbors.

    Parameters
    ----------
    estimator : BaseDecisionTree, BaseForest
        The estimator to use for computing distances.
    n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries, by default 5.
    algorithm : str, optional
        Algorithm used to compute the nearest-neighbors, by default 'auto'.
        See :class:`sklearn.neighbors.NearestNeighbors` for details.
    radius : float, optional
        Range of parameter space to use by default for radius_neighbors queries, by default 1.0.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors, by default None.
    """

    def __init__(self, estimator, n_neighbors=5, radius=1.0, algorithm="auto", n_jobs=None):
        self.estimator = estimator
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.radius = radius
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        # self._validate_params()
        self.estimator_ = self.estimator
        check_is_fitted(self.estimator_)
        self._fit(X, self.n_neighbors)
        return self

    def _fit(self, X, n_neighbors):
        self.neigh_est_ = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm=self.algorithm,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

        # compute the distance matrix
        aff_matrix = compute_forest_similarity_matrix(self.estimator_, X)
        dists = _compute_distance_matrix(aff_matrix)

        # fit the nearest-neighbors estimator
        self.neigh_est_.fit(dists)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed', default=None
            Not used, present for API consistency by convention.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, enter integer value" % type(n_neighbors)
            )

        if X is not None:
            self._fit(X, n_neighbors)

        return self.neigh_est_.kneighbors(n_neighbors=n_neighbors, return_distance=return_distance)

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Find the neighbors within a given radius of a point or points.

        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.

        The result points are *not* necessarily sorted by distance to their
        query point.

        Parameters
        ----------
        X : {array-like, sparse matrix} of (n_samples, n_features), default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float, or array-like of shape (n_samples,) default=None
            Limiting distance of neighbors to return. The default is the value
            passed to the constructor. If an array-like of shape (n_samples),
            then will query for each sample point with a different radius.

        return_distance : bool, default=True
            Whether or not to return the distances.

        sort_results : bool, default=False
            If True, the distances and indices will be sorted by increasing
            distances before being returned. If False, the results may not
            be sorted. If `return_distance=False`, setting `sort_results=True`
            will result in an error.

            .. versionadded:: 0.22

        Returns
        -------
        neigh_dist : ndarray of shape (n_samples,) of arrays
            Array representing the distances to each point, only present if
            `return_distance=True`. The distance values are computed according
            to the ``metric`` constructor parameter.

        neigh_ind : ndarray of shape (n_samples,) of arrays
            An array of arrays of indices of the approximate nearest points
            from the population matrix that lie within a ball of size
            ``radius`` around the query points.

        Notes
        -----
        Because the number of neighbors of each point is not necessarily
        equal, the results for multiple query points cannot be fit in a
        standard data array.
        For efficiency, `radius_neighbors` returns arrays of objects, where
        each object is a 1D array of indices or distances.
        """
        check_is_fitted(self)

        if X is not None:
            n_samples = X.shape[0]
        else:
            n_samples = self.neigh_est_.n_samples_fit_

        if isinstance(radius, numbers.Number):
            radius = [radius] * n_samples

        # now construct nearest neighbor indices and distances within radius
        nn_ind_data = np.zeros((n_samples,), dtype=object)
        nn_dist_data = np.zeros((n_samples,), dtype=object)
        for idx in range(n_samples):
            nn = self.neigh_est_.radius_neighbors(
                X=X, radius=radius[idx], return_distance=return_distance, sort_results=sort_results
            )

            if return_distance:
                nn_ind_data[idx] = nn[0][idx]
                nn_dist_data[idx] = nn[1][idx]
            else:
                nn_ind_data[idx] = nn

        if return_distance:
            return nn_dist_data, nn_ind_data
        return nn_ind_data

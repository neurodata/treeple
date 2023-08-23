"""
Manifold forest of trees-based ensemble methods.

Those methods include various random forest methods that operate on manifolds.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause
from warnings import warn

import numpy as np
from scipy.sparse import issparse
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble._forest import (
    MAX_INT,
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from sklearn.metrics import calinski_harabasz_score
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_random_state

from .._lib.sklearn.ensemble._forest import BaseForest
from .._lib.sklearn.tree._tree import DTYPE
from ..tree import UnsupervisedDecisionTree, UnsupervisedObliqueDecisionTree
from ..tree._neighbors import SimMatrixMixin


class ForestCluster(SimMatrixMixin, TransformerMixin, ClusterMixin, BaseForest):
    """Unsupervised forest base class."""

    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()

        # Validate or convert input data
        X = self._validate_data(
            X,
            dtype=DTYPE,  # accept_sparse="csc",
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not " "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads",)(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            if callable(self.oob_score):
                self._set_oob_score_and_attributes(X, scoring_function=self.oob_score)
            else:
                self._set_oob_score_and_attributes(X)

        # now compute the similarity/dissimilarity matrix and set it
        sim_mat = self.compute_similarity_matrix(X)

        # compute the labels and set it
        self.labels_ = self._assign_labels(sim_mat)

        return self

    def predict(self, X):
        """Predict clusters for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        X = self._validate_X_predict(X)
        similarity_matrix = self.transform(X)

        # compute the labels and set it
        return self._assign_labels(similarity_matrix)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_samples)
            X transformed in the new space.
        """
        check_is_fitted(self)

        # now compute the affinity matrix and set it
        similarity_matrix = self.compute_similarity_matrix(X)
        return similarity_matrix

    def _assign_labels(self, similarity_matrix):
        """Assign cluster labels given X.

        Parameters
        ----------
        similarity_matrix : ndarray of shape (n_samples, n_samples)
            The affinity matrix.

        Returns
        -------
        predict_labels : ndarray of shape (n_samples,)
            The predicted cluster labels
        """
        if self.clustering_func is None:
            self.clustering_func_ = AgglomerativeClustering
        else:
            self.clustering_func_ = self.clustering_func
        if self.clustering_func_args is None:
            self.clustering_func_args_ = dict()
        else:
            self.clustering_func_args_ = self.clustering_func_args
        cluster = self.clustering_func_(**self.clustering_func_args_)

        # apply agglomerative clustering to obtain cluster labels
        predict_labels = cluster.fit_predict(similarity_matrix)
        return predict_labels

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB transformations for an individual tree.

        Parameters
        ----------
        tree : UnsupervisedDecisionTree object
            A single unsupervised decision tree model.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        tree_prox_matrix : ndarray of shape (n_samples, n_samples)
            The OOB associated proximity matrix.
        """
        # transform X

        # now compute the affinity matrix and set it
        tree_prox_matrix = tree.compute_similarity_matrix_forest(X)

        return tree_prox_matrix

    def _compute_oob_predictions(self, X, y=None):
        """Compute the OOB transformations.

        This only uses the OOB samples per tree to compute the unnormalized
        proximity matrix. These submatrices are then aggregated into the whole
        proximity matrix and normalized based on how many times each sample
        showed up in an OOB tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            Not used.

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_samples)
            The OOB proximity matrix.
        """
        # Prediction requires X to be in CSR format
        if issparse(X):
            X = X.tocsr()

        n_samples = X.shape[0]

        # for clustering, n_classes_ does not exist and we create an empty
        # axis to be consistent with the classification case and make
        # the array operations compatible with the 2 settings
        oob_pred_shape = (n_samples, n_samples)

        oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
        n_oob_pred = np.zeros((n_samples, n_samples), dtype=np.int64)

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples,
            self.max_samples,
        )
        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state,
                n_samples,
                n_samples_bootstrap,
            )

            tree_prox_matrix = self._get_oob_predictions(estimator, X[unsampled_indices, :])
            oob_pred[np.ix_(unsampled_indices, unsampled_indices)] += tree_prox_matrix
            n_oob_pred[np.ix_(unsampled_indices, unsampled_indices)] += 1

        if (n_oob_pred == 0).any():
            warn(
                "Some inputs do not have OOB scores. This probably means "
                "too few trees were used to compute any reliable OOB "
                "estimates.",
                UserWarning,
            )
            n_oob_pred[n_oob_pred == 0] = 1

            # normalize by the number of times each oob sample proximity matrix was computed
            oob_pred /= n_oob_pred

        return oob_pred

    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        scoring_function : callable, default=None
            Scoring function for OOB score. Default is the
            :func:`sklearn.metrics.calinski_harabasz_score`.
            Must not require true ``y_labels``.
        """
        self.oob_decision_function_ = self._compute_oob_predictions(X)

        if scoring_function is None:
            scoring_function = calinski_harabasz_score

        # assign labels
        predict_labels = self._assign_labels(self.oob_decision_function_)

        self.oob_labels_ = predict_labels
        self.oob_score_ = scoring_function(X, predict_labels)


class UnsupervisedRandomForest(ForestCluster):
    """Unsupervised random forest.

    An unsupervised random forest is inherently a clustering algorithm that also
    simultaneously computes an adaptive affinity matrix that is based on the 0-1
    tree distance (i.e. do samples fall within the same leaf).

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees to fit, by default 100.

    criterion : {"twomeans", "fastbic"}, default="twomeans"
        The function to measure the quality of a split. Supported criteria are
        "twomeans" for maximizing the variance and "fastbic" for the
        maximizing the Bayesian Information Criterion (BIC), see
        :ref:`tree_mathematical_formulation`.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        In unsupervised trees, it is recommended by :footcite:`Meghana2019_geodesicrf`
        to use the sqrt of two times the number of samples in the dataset.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    bootstrap : bool, optional
        Whether to bootstrap, by default False.
    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, :func:`~sklearn.metrics.calinski_harabasz_score` is used.
        Provide a callable with signature `metric(X, predicted_labels)` to use a
        custom metric. Only available if `bootstrap=True`. Other supported functions
        from scikit-learn are :func:`sklearn.metrics.silhouette_score`,
        :func:`sklearn.metrics.calinski_harabasz_score`, and
        :func:`sklearn.metrics.davies_bouldin_score`.
    n_jobs : int, optional
        Number of CPUs to use in `joblib` parallelization for constructing trees,
        by default None.
    random_state : int, optional
        Random seed, by default None.
    verbose : int, optional
        Verbosity, by default 0.
    warm_start : bool, optional
        Whether to continue constructing trees from previous instant, by default False.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.
    clustering_func : callable
        Scikit-learn compatible clustering function to take the affinity matrix
        and return cluster labels. By default, :class:`sklearn.cluster.AgglomerativeClustering`.
    clustering_func_args : dict
        Clustering function class keyword arguments. Passed to `clustering_func`.

    Attributes
    ----------
    estimator_ : UnsupervisedDecisionTree
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of UnsupervisedDecisionTree
        The collection of fitted sub-estimators.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    similarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity/similarity matrix used in fit. Note this matrix
        is computed from within-bag and OOB samples.

    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the dissimilarity matrix used in fit. Note this matrix
        is computed from within-bag and OOB samples.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix computed with only out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="twomeans",
        max_depth=None,
        min_samples_split="sqrt",
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        clustering_func=None,
        clustering_func_args=None,
    ) -> None:
        super().__init__(
            estimator=UnsupervisedDecisionTree(),  # type: ignore
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.clustering_func = clustering_func
        self.clustering_func_args = clustering_func_args


class UnsupervisedObliqueRandomForest(ForestCluster):
    """Unsupervised oblique random forest.

    An unsupervised random forest is inherently a clustering algorithm that also
    simultaneously computes an adaptive affinity matrix that is based on the 0-1
    tree distance (i.e. do samples fall within the same leaf).

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees to fit, by default 100.

    criterion : {"twomeans", "fastbic"}, default="twomeans"
        The function to measure the quality of a split. Supported criteria are
        "twomeans" for maximizing the variance and "fastbic" for the
        maximizing the Bayesian Information Criterion (BIC), see
        :ref:`tree_mathematical_formulation`.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        In unsupervised trees, it is recommended by :footcite:`Meghana2019_geodesicrf`
        to use the sqrt of two times the number of samples in the dataset.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    bootstrap : bool, optional
        Whether to bootstrap, by default False.
    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, :func:`~sklearn.metrics.calinski_harabasz_score` is used.
        Provide a callable with signature `metric(X, predicted_labels)` to use a
        custom metric. Only available if `bootstrap=True`. Other supported functions
        from scikit-learn are :func:`sklearn.metrics.silhouette_score`,
        :func:`sklearn.metrics.calinski_harabasz_score`, and
        :func:`sklearn.metrics.davies_bouldin_score`.
    n_jobs : int, optional
        Number of CPUs to use in `joblib` parallelization for constructing trees,
        by default None.
    random_state : int, optional
        Random seed, by default None.
    verbose : int, optional
        Verbosity, by default 0.
    warm_start : bool, optional
        Whether to continue constructing trees from previous instant, by default False.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.
    feature_combinations : float, default=1.5
        The number of features to combine on average at each split
        of the decision trees.
    clustering_func : callable
        Scikit-learn compatible clustering function to take the affinity matrix
        and return cluster labels. By default, :class:`sklearn.cluster.AgglomerativeClustering`.
    clustering_func_args : dict
        Clustering function class keyword arguments. Passed to `clustering_func`.

    Attributes
    ----------
    estimator_ : UnsupervisedDecisionTree
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of UnsupervisedDecisionTree
        The collection of fitted sub-estimators.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    similarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity/similarity matrix used in fit. Note this matrix
        is computed from within-bag and OOB samples.

    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the dissimilarity matrix used in fit. Note this matrix
        is computed from within-bag and OOB samples.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix computed with only out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.
    """

    tree_type = "oblique"

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="twomeans",
        max_depth=None,
        min_samples_split="sqrt",
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        feature_combinations=1.5,
        clustering_func=None,
        clustering_func_args=None,
    ) -> None:
        super().__init__(
            estimator=UnsupervisedObliqueDecisionTree(),  # type: ignore
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "feature_combinations",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_combinations = feature_combinations
        self.clustering_func = clustering_func
        self.clustering_func_args = clustering_func_args

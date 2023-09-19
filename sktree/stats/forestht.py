import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from sklearn.ensemble._forest import ForestClassifier as sklearnForestClassifier
from sklearn.ensemble._forest import ForestRegressor as sklearnForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _is_fitted, check_X_y

from sktree._lib.sklearn.ensemble._forest import (
    BaseForest,
    ForestClassifier,
    ForestRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sktree._lib.sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .utils import (
    METRIC_FUNCTIONS,
    POSITIVE_METRICS,
    POSTERIOR_FUNCTIONS,
    REGRESSOR_METRICS,
    _compute_null_distribution_coleman,
    train_tree,
)


class BaseForestHT(MetaEstimatorMixin):
    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
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
        ccp_alpha=0.0,
        max_samples=None,
        permute_per_tree=True,
        sample_dataset_per_tree=True,
    ):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.permute_per_tree = permute_per_tree
        self.sample_dataset_per_tree = sample_dataset_per_tree

    def reset(self):
        class_attributes = dir(type(self))
        instance_attributes = dir(self)

        for attr_name in instance_attributes:
            if attr_name.endswith("_") and attr_name not in class_attributes:
                delattr(self, attr_name)

    def statistic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="mse",
        return_posteriors: bool = False,
        check_input: bool = True,
        **metric_kwargs,
    ):
        """Compute the test statistic.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike, optional of shape (n_covariates,)
            The index array of covariates to shuffle, by default None.
        metric : str, optional
            The metric to compute, by default "mse".
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.
        check_input : bool, optional
            Whether or not to check the input, by default True.
        **metric_kwargs : dict, optional
            Additional keyword arguments to pass to the metric function.

        Returns
        -------
        stat : float
            The test statistic.
        posterior_final : ArrayLike of shape (n_samples_final, n_outputs), optional
            If ``return_posteriors`` is True, then the posterior probabilities of the
            samples used in the final test. ``n_samples_final`` is equal to ``n_samples``
            if all samples are encountered in the test set of at least one tree in the
            posterior computation.
        samples : ArrayLike of shape (n_samples_final,), optional
            The indices of the samples used in the final test. ``n_samples_final`` is
            equal to ``n_samples`` if all samples are encountered in the test set of at
            least one tree in the posterior computation.
        """
        if check_input:
            X, y = check_X_y(X, y, ensure_2d=True, multi_output=True)
            if y.ndim != 2:
                y = y.reshape(-1, 1)

        if covariate_index is None:
            self.estimator_ = self._get_estimator()
            estimator = self.estimator_
        else:
            self.permuted_estimator_ = clone(self.estimator_)
            estimator = self.permuted_estimator_

        if not is_classifier(self.estimator_) and metric not in REGRESSOR_METRICS:
            raise RuntimeError(f'Metric must be either "mse" or "mae", got {metric}')

        return self._statistic(
            estimator,
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            return_posteriors=return_posteriors,
            **metric_kwargs,
        )

    def test(
        self,
        X,
        y,
        covariate_index: ArrayLike,
        metric: str = "mse",
        test_size: float = 0.2,
        n_repeats: int = 1000,
        return_posteriors: bool = False,
        **metric_kwargs,
    ):
        """Perform hypothesis test using Coleman method.

        X is split into a training/testing split. Optionally, the covariate index
        columns are shuffled.

        On the training dataset, two honest forests are trained and then the posterior
        is estimated on the testing dataset. One honest forest is trained on the
        permuted dataset and the other is trained on the original dataset.

        Finally, resample the posteriors of the two forests to compute the null
        distribution of the statistics.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike, optional of shape (n_covariates,)
            The index array of covariates to shuffle, by default None.
        metric : str, optional
            The metric to compute, by default "mse".
        test_size : float, optional
            Proportion of samples per tree to use for the test set, by default 0.2.
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.
        **metric_kwargs : dict, optional
            Additional keyword arguments to pass to the metric function.

        Returns
        -------
        stat : float
            The test statistic.
        pval : float
            The p-value of the test statistic.
        """
        X, y = check_X_y(X, y, ensure_2d=True, copy=True, multi_output=True)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        indices = np.arange(X.shape[0])
        self.test_size_ = int(test_size * X.shape[0])
        indices_train, indices_test = train_test_split(indices, test_size=test_size, shuffle=True)
        self.indices_train_ = indices_train
        self.indices_test_ = indices_test

        if not hasattr(self, "samples_"):
            # first compute the test statistic on the un-permuted data
            observe_stat, observe_posteriors, observe_samples = self.statistic(
                X,
                y,
                covariate_index=None,
                metric=metric,
                return_posteriors=True,
                check_input=False,
                **metric_kwargs,
            )
        else:
            observe_samples = self.samples_
            observe_posteriors = self.posterior_final_
            observe_stat = self.stat_

        # next permute the data
        permute_stat, permute_posteriors, permute_samples = self.statistic(
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            return_posteriors=True,
            check_input=False,
            **metric_kwargs,
        )

        # Note: at this point, both `estimator` and `permuted_estimator_` should
        # have been fitted already, so we can now compute on the null by resampling
        # the posteriors and computing the test statistic on the resampled posteriors
        if self.sample_dataset_per_tree:
            metric_star, metric_star_pi = _compute_null_distribution_coleman(
                y_test=y[observe_samples, :],
                y_pred_proba_normal=observe_posteriors,
                y_pred_proba_perm=permute_posteriors,
                metric=metric,
                n_repeats=n_repeats,
                seed=self.random_state,
            )
        else:
            if self.permute_per_tree:
                y_test = y
            else:
                y_test = y[self.indices_test_, :]
            print(y.shape, observe_posteriors.shape, permute_posteriors.shape)
            metric_star, metric_star_pi = _compute_null_distribution_coleman(
                y_test=y_test,
                y_pred_proba_normal=observe_posteriors,
                y_pred_proba_perm=permute_posteriors,
                metric=metric,
                n_repeats=n_repeats,
                seed=self.random_state,
            )
        # metric^\pi - metric = observed test statistic, which under the
        # null is normally distributed around 0
        observe_stat = permute_stat - observe_stat

        # metric^\pi_j - metric_j, which is centered at 0
        null_dist = metric_star_pi - metric_star

        # compute pvalue
        if metric in POSITIVE_METRICS:
            pvalue = (1 + (null_dist <= observe_stat).sum()) / (1 + n_repeats)
        else:
            pvalue = (1 + (null_dist >= observe_stat).sum()) / (1 + n_repeats)

        if return_posteriors:
            self.observe_posteriors_ = observe_posteriors
            self.permute_posteriors_ = permute_posteriors
            self.observe_samples_ = observe_samples
            self.permute_samples_ = permute_samples

        self.null_dist_ = null_dist
        return observe_stat, pvalue


class FeatureImportanceForestRegressor(BaseForestHT):
    """Forest hypothesis testing with continuous `y` variable.

    Implements the algorithm described in :footcite:`coleman2022scalable`.

    The dataset is split into a training and testing dataset initially. Then there
    are two forests that are trained: one on the original dataset, and one on the
    permuted dataset. The dataset is either permuted once, or independently for
    each tree in the permuted forest. The original test statistic is computed by
    comparing the metric on both forests ``(metric_forest - metric_perm_forest)``.

    Then the output predictions are randomly sampled to recompute the test statistic
    ``n_repeats`` times. The p-value is computed as the proportion of times the
    null test statistic is greater than the original test statistic.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestRegressor`.

    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

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
          `round(max_features * n_features)` features are considered at each
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

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base tree estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    permute_per_tree : bool, default=True
        Whether to permute the covariate index per tree or per forest.

    sample_dataset_per_tree : bool, default=False
        Whether to sample the dataset per tree or per forest.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_final_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_final_ : ArrayLike of shape (n_samples_final,)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
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
        ccp_alpha=0.0,
        max_samples=None,
        permute_per_tree=True,
        sample_dataset_per_tree=True,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.permute_per_tree = permute_per_tree
        self.sample_dataset_per_tree = sample_dataset_per_tree

    def _get_estimator(self):
        if self.estimator is None:
            estimator_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                ccp_alpha=self.ccp_alpha,
                max_samples=self.max_samples,
            )
        elif not isinstance(self.estimator, (ForestRegressor, sklearnForestRegressor)):
            raise RuntimeError(f"Estimator must be a ForestRegressor, got {type(self.estimator)}")
        else:
            estimator_ = self.estimator
        return estimator_

    def _statistic(
        self,
        estimator: BaseForest,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="mse",
        return_posteriors: bool = False,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func = METRIC_FUNCTIONS[metric]
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        if self.permute_per_tree and not self.sample_dataset_per_tree:
            # first run a dummy fit on the samples to initialize the
            # internal data structure of the forest
            if not _is_fitted(estimator):
                unique_y = np.unique(y)
                X_dummy = np.zeros((unique_y.shape[0], X.shape[1]))
                estimator.fit(X_dummy, unique_y)

            # Fit each tree and compute posteriors with train test splits
            n_samples_test = len(self.indices_test_)

            # now initialize posterior array as (n_trees, n_samples_test, n_outputs)
            posterior_arr = np.zeros((self.n_estimators, n_samples_test, estimator.n_outputs_))
            for idx in range(self.n_estimators):
                tree: DecisionTreeRegressor = estimator.estimators_[idx]
                train_tree(
                    tree, X[self.indices_train_, :], y[self.indices_train_, :], covariate_index
                )

                y_pred = tree.predict(X[self.indices_test_, :]).reshape(-1, tree.n_outputs_)

                # Fill test set posteriors & set rest NaN
                posterior_arr[idx, ...] = y_pred  # posterior

            y_true_final = y[self.indices_test_, :]
            # Average all posteriors
            posterior_final = np.nanmean(posterior_arr, axis=0)
            samples = np.argwhere(~np.isnan(posterior_final).any(axis=1)).squeeze()
        elif self.permute_per_tree and self.sample_dataset_per_tree:
            # first run a dummy fit on the samples to initialize the
            # internal data structure of the forest
            if not _is_fitted(estimator):
                unique_y = np.unique(y)
                X_dummy = np.zeros((unique_y.shape[0], X.shape[1]))
                estimator.fit(X_dummy, unique_y)

            if hasattr(self, "test_size_"):
                test_size = self.test_size_
            else:
                test_size = 0.2  # type: ignore

            # now initialize posterior array as (n_trees, n_samples, n_outputs)
            posterior_arr = np.full((self.n_estimators, n_samples, estimator.n_outputs_), np.nan)
            # Fit each tree and compute posteriors with train test splits
            for idx in range(self.n_estimators):
                # sample train/test dataset for each tree
                indices_train, indices_test = train_test_split(
                    np.arange(n_samples, dtype=int),
                    test_size=test_size,
                    shuffle=True,
                    random_state=rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32),
                )
                tree = estimator.estimators_[idx]
                train_tree(tree, X[indices_train, :], y[indices_train, :], covariate_index)

                y_pred = tree.predict(X[indices_test, :]).reshape(-1, tree.n_outputs_)
                posterior_arr[idx, indices_test, :] = y_pred  # posterior

            # Average all posteriors
            posterior_final = np.nanmean(posterior_arr, axis=0)

            # Find the row indices with NaN values in any column
            nonnan_indices = np.where(~np.isnan(posterior_final).any(axis=1))[0]

            # Ignore all NaN values (samples not tested)
            y_true_final = y[nonnan_indices, :]
            posterior_final = posterior_final[nonnan_indices, :]
            samples = nonnan_indices
        else:
            X_train, X_test = X[self.indices_train_, :], X[self.indices_test_, :]
            y_train, y_test = y[self.indices_train_, :], y[self.indices_test_, :]

            if covariate_index is not None:
                # perform permutation of covariates
                n_samples_train = X_train.shape[0]
                index_arr = rng.choice(
                    np.arange(n_samples_train, dtype=int),
                    size=(n_samples_train, 1),
                    replace=False,
                    shuffle=True,
                )
                X_train[:, covariate_index] = X_train[index_arr, covariate_index]

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            # set variables to compute metric
            samples = self.indices_test_
            y_true_final = y_test
            posterior_final = y_pred

        stat = metric_func(y_true_final, posterior_final, **metric_kwargs)
        if covariate_index is None:
            # Ignore all NaN values (samples not tested) -> (n_samples_final, n_outputs)
            # arrays of y and predicted posterior
            self.samples_ = samples
            self.y_true_final_ = y_true_final
            self.posterior_final_ = posterior_final
            self.stat_ = stat

        if return_posteriors:
            return stat, posterior_final, samples

        return stat


class FeatureImportanceForestClassifier(BaseForestHT):
    """Forest hypothesis testing with categorical `y` variable.

    Implements the algorithm described in :footcite:`coleman2022scalable`.

    The dataset is split into a training and testing dataset initially. Then there
    are two forests that are trained: one on the original dataset, and one on the
    permuted dataset. The dataset is either permuted once, or independently for
    each tree in the permuted forest. The original test statistic is computed by
    comparing the metric on both forests ``(metric_forest - metric_perm_forest)``.

    Then the output predictions are randomly sampled to recompute the test statistic
    ``n_repeats`` times. The p-value is computed as the proportion of times the
    null test statistic is greater than the original test statistic.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestRegressor`.

    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

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
          `round(max_features * n_features)` features are considered at each
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

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base tree estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    permute_per_tree : bool, default=True
        Whether to permute the covariate index per tree or per forest.

    sample_dataset_per_tree : bool, default=False
        Whether to sample the dataset per tree or per forest.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_final_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_final_ : ArrayLike of shape (n_samples_final,)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
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
        ccp_alpha=0.0,
        max_samples=None,
        permute_per_tree=True,
        sample_dataset_per_tree=True,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.permute_per_tree = permute_per_tree
        self.sample_dataset_per_tree = sample_dataset_per_tree

    def _get_estimator(self):
        if self.estimator is None:
            estimator_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                ccp_alpha=self.ccp_alpha,
                max_samples=self.max_samples,
            )
        elif not isinstance(self.estimator, (ForestClassifier, sklearnForestClassifier)):
            raise RuntimeError(f"Estimator must be a ForestClassifier, got {type(self.estimator)}")
        else:
            # self.estimator is an instance of a ForestEstimator, so we should verify that all
            # the parameters are set correctly
            # XXX: implement checks

            estimator_ = self.estimator
        return estimator_

    def _statistic(
        self,
        estimator: BaseForest,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="mse",
        return_posteriors: bool = False,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func = METRIC_FUNCTIONS[metric]
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        if metric in POSTERIOR_FUNCTIONS:
            predict_posteriors = True
        else:
            predict_posteriors = False

        if hasattr(self, "test_size_"):
            test_size = self.test_size_
        else:
            test_size = 0.2  # type: ignore

        if not _is_fitted(estimator):
            unique_y = np.unique(y)
            X_dummy = np.zeros((unique_y.shape[0], X.shape[1]))
            estimator.fit(X_dummy, unique_y)
        if estimator.n_outputs_ > 1 and metric == "auc":
            raise ValueError("AUC metric is not supported for multi-output classification")

        if self.permute_per_tree:
            # first run a dummy fit on the samples to initialize the
            # internal data structure of the forest
            if self.sample_dataset_per_tree:
                # Fit each tree and compute posteriors with train test splits
                n_samples_test = len(self.indices_test_)
            else:
                n_samples_test = n_samples

            if predict_posteriors:
                posterior_arr = np.zeros((self.n_estimators, n_samples_test, estimator.n_classes_))
            else:
                # now initialize posterior array as (n_trees, n_samples_test, n_outputs)
                posterior_arr = np.zeros((self.n_estimators, n_samples_test, estimator.n_outputs_))

            if self.sample_dataset_per_tree:
                for idx in range(self.n_estimators):
                    tree: DecisionTreeClassifier = estimator.estimators_[idx]
                    train_tree(
                        tree, X[self.indices_train_, :], y[self.indices_train_, :], covariate_index
                    )

                    if predict_posteriors:
                        # XXX: currently assumes n_outputs_ == 1
                        y_pred = tree.predict_proba(X[self.indices_test_, :])
                    else:
                        y_pred = tree.predict(X[self.indices_test_, :]).reshape(-1, tree.n_outputs_)

                    # Fill test set posteriors & set rest NaN
                    posterior_arr[idx, ...] = y_pred  # posterior

                y_true_final = y[self.indices_test_, :]
                # Average all posteriors
                posterior_final = np.nanmean(posterior_arr, axis=0)
                samples = np.argwhere(~np.isnan(posterior_final).any(axis=1)).squeeze()
            else:
                # Fit each tree and compute posteriors with train test splits
                for idx in range(self.n_estimators):
                    # sample train/test dataset for each tree
                    indices_train, indices_test = train_test_split(
                        np.arange(n_samples, dtype=int),
                        test_size=test_size,
                        shuffle=True,
                        random_state=rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32),
                    )
                    tree = estimator.estimators_[idx]
                    train_tree(tree, X[indices_train, :], y[indices_train, :], covariate_index)

                    if predict_posteriors:
                        # XXX: currently assumes n_outputs_ == 1
                        y_pred = tree.predict_proba(X[indices_test, :])
                    else:
                        y_pred = tree.predict(X[indices_test, :]).reshape(-1, tree.n_outputs_)

                    posterior_arr[idx, indices_test, :] = y_pred  # posterior

                # Average all posteriors (n_samples, n_outputs)
                posterior_final = np.nanmean(posterior_arr, axis=0)

                # Find the row indices with NaN values in any column
                nonnan_indices = np.where(~np.isnan(posterior_final).any(axis=1))[0]

                # Ignore all NaN values (samples not tested)
                y_true_final = y[nonnan_indices, :]
                posterior_final = posterior_final[nonnan_indices, :]
                samples = nonnan_indices
        else:
            X_train, X_test = X[self.indices_train_, :], X[self.indices_test_, :]
            y_train, y_test = y[self.indices_train_, :], y[self.indices_test_, :]

            if covariate_index is not None:
                # perform permutation of covariates
                n_samples_train = X_train.shape[0]
                index_arr = rng.choice(
                    np.arange(n_samples_train, dtype=int),
                    size=(n_samples_train, 1),
                    replace=False,
                    shuffle=True,
                )
                X_train[:, covariate_index] = X_train[index_arr, covariate_index]

            estimator.fit(X_train, y_train)

            if predict_posteriors:
                # XXX: currently assumes n_outputs_ == 1
                y_pred = estimator.predict_proba(X_test)
            else:
                y_pred = estimator.predict(X_test)

            # set variables to compute metric
            samples = self.indices_test_
            y_true_final = y_test
            posterior_final = y_pred

        if metric == "auc":
            # at this point, posterior_final is the predicted posterior for only the positive class
            # as more than one output is not supported.
            if type_of_target(y_true_final) == "binary":
                posterior_final = posterior_final[:, 1]
            else:
                raise RuntimeError(
                    f"AUC metric is not supported for {type_of_target(y_true_final)} targets."
                )

        stat = metric_func(y_true_final, posterior_final, **metric_kwargs)
        if covariate_index is None:
            # Ignore all NaN values (samples not tested) -> (n_samples_final, n_outputs)
            # arrays of y and predicted posterior
            self.samples_ = samples
            self.y_true_final_ = y_true_final
            self.posterior_final_ = posterior_final
            self.stat_ = stat

        if return_posteriors:
            return stat, posterior_final, samples

        return stat

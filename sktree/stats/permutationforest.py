import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

from sktree._lib.sklearn.ensemble._forest import BaseForest, ForestClassifier, ForestRegressor

from .utils import METRIC_FUNCTIONS, REGRESSOR_METRICS, _compute_null_distribution_perm


class BasePermutationForest(MetaEstimatorMixin):
    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
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
        **estimators_kwargs,
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
        self.estimator_kwargs = estimators_kwargs

    def reset(self):
        class_attributes = dir(type(self))
        instance_attributes = dir(self)

        for attr_name in instance_attributes:
            if attr_name.endswith("_") and attr_name not in class_attributes:
                delattr(self, attr_name)

    def _get_estimator(self):
        pass

    def _statistic(
        self,
        estimator: BaseForest,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="mse",
        return_posteriors: bool = False,
        seed=None,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func = METRIC_FUNCTIONS[metric]
        if seed is None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng(seed)
        n_samples = X.shape[0]
        indices = np.arange(n_samples, dtype=int)

        if covariate_index is not None:
            # perform permutation of covariates
            index_arr = rng.choice(indices, size=(X.shape[0], 1), replace=False, shuffle=False)
            X = X.copy()
            X[:, covariate_index] = X[index_arr, covariate_index]

        X_train, X_test = X[self.indices_train_, :], X[self.indices_test_, :]
        y_train, y_test = y[self.indices_train_, :], y[self.indices_test_, :]

        estimator.fit(X_train, y_train.ravel())

        # Either get the predicted value, or the posterior probabilities
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

    def statistic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike = None,
        metric="mse",
        return_posteriors: bool = False,
        check_input: bool = True,
        seed=None,
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
        test_size : float, optional
            Proportion of samples per tree to use for the test set, by default 0.2.
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.

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

        self.estimator_ = self._get_estimator()

        if is_classifier(self.estimator_):
            if metric not in METRIC_FUNCTIONS:
                raise RuntimeError(
                    f"Metric must be one of {list(METRIC_FUNCTIONS.keys())}, got {metric}"
                )
        else:
            if metric not in REGRESSOR_METRICS:
                raise RuntimeError(f'Metric must be either "mse" or "mae", got {metric}')

        if covariate_index is None:
            estimator = self.estimator_
        else:
            self.permuted_estimator_ = clone(self.estimator_)
            estimator = self.permuted_estimator_

        return self._statistic(
            estimator,
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            return_posteriors=return_posteriors,
            seed=seed,
            **metric_kwargs,
        )

    def test(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike,
        metric: str = "mse",
        test_size: float = 0.2,
        n_repeats: int = 1000,
        return_posteriors: bool = False,
        **metric_kwargs,
    ):
        """Perform hypothesis test using permutation testing.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike of shape (n_covariates,)
            The covariate indices of ``X`` to shuffle.
        metric : str, optional
            Metric to compute, by default "mse".
        test_size : float, optional
            Size of the samples to leave out for each tree to compute posteriors on,
            by default 0.2.
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.

        Returns
        -------
        observe_stat : float
            Observed test statistic.
        pvalue : float
            p-value of the test.
        """
        X, y = check_X_y(X, y, ensure_2d=True, copy=True, multi_output=True)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        indices = np.arange(X.shape[0])

        # train/test split
        # XXX: could add stratifying by y when y is classification
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

        # compute null distribution of the test statistic
        # WARNING: this could take a long time, since it fits a new forest
        null_dist = _compute_null_distribution_perm(
            X_train=X[self.indices_train_, :],
            y_train=y[self.indices_train_, :],
            X_test=X[self.indices_test_, :],
            y_test=y[self.indices_test_, :],
            covariate_index=covariate_index,
            est=self.estimator_,
            metric=metric,
            n_repeats=n_repeats,
            seed=self.random_state,
        )

        if not return_posteriors:
            self.null_dist_ = np.array(null_dist)
        else:
            self.null_dist_ = np.array([x[0] for x in null_dist])
            self.posterior_null_ = np.array([x[1] for x in null_dist])

        n_repeats = len(self.null_dist_)
        pvalue = (1 + (self.null_dist_ < observe_stat).sum()) / (1 + n_repeats)
        return observe_stat, pvalue


class PermutationForestRegressor(BasePermutationForest):
    """Hypothesis testing of covariates with a permutation forest regressor.

    This implements permutation testing of a null hypothesis using a random forest.
    The null hypothesis is generated by permuting ``n_repeats`` times the covariate
    indices and then a random forest is trained for each permuted instance. This
    is compared to the original random forest that was computed on the regular
    non-permuted data.

    .. note:: This does not allow testing on the posteriors.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestRegressor`.

    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

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
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a `joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

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

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

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

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.

    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    tree_estimator : object, default=None
        Type of decision tree classifier to use. By default `None`, which
        defaults to :class:`sklearn.tree.DecisionTreeClassifier`.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_ : ArrayLike of shape (n_samples_final, n_outputs)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    posterior_null_ : ArrayLike of shape (n_samples_final, n_outputs, n_repeats)
        The posterior probabilities of the samples used in the final test for each
        permutation for the null distribution.
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
        max_features=1.0,
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
        **estimators_kwargs,
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
            **estimators_kwargs,
        )

    def _get_estimator(self):
        if not hasattr(self, "estimator_") and self.estimator is None:
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
                **self.estimator_kwargs,
            )
        elif not isinstance(self.estimator_, ForestRegressor):
            raise RuntimeError(f"Estimator must be a ForestRegressor, got {type(self.estimator_)}")
        return estimator_


class PermutationForestClassifier(BasePermutationForest):
    """Hypothesis testing of covariates with a permutation forest classifier.

    This implements permutation testing of a null hypothesis using a random forest.
    The null hypothesis is generated by permuting ``n_repeats`` times the covariate
    indices and then a random forest is trained for each permuted instance. This
    is compared to the original random forest that was computed on the regular
    non-permuted data.

    .. note:: This does not allow testing on the posteriors.
    
    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestClassifier`.

    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

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
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a `joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

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

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

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

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.

    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    tree_estimator : object, default=None
        Type of decision tree classifier to use. By default `None`, which
        defaults to :class:`sklearn.tree.DecisionTreeClassifier`.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_ : ArrayLike of shape (n_samples_final, n_outputs)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    posterior_null_ : ArrayLike of shape (n_samples_final, n_outputs, n_repeats)
        The posterior probabilities of the samples used in the final test for each
        permutation for the null distribution.
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
        **estimators_kwargs,
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
            **estimators_kwargs,
        )

    def _get_estimator(self):
        if not hasattr(self, "estimator_") and self.estimator is None:
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
                **self.estimator_kwargs,
            )
        elif not isinstance(self.estimator_, ForestRegressor):
            raise RuntimeError(f"Estimator must be a ForestRegressor, got {type(self.estimator_)}")
        return estimator_

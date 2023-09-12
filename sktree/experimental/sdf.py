"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import _check_partial_fit_first_call

from .._lib.sklearn.ensemble._forest import (
    RandomForestClassifier,
    _generate_sample_indices,
    _get_n_samples_bootstrap,
)
from .._lib.sklearn.tree import DecisionTreeClassifier


def _partial_fit(tree, X, y, n_samples_bootstrap, classes):
    """Internal function to partially fit a tree."""
    indices = _generate_sample_indices(tree.random_state, X.shape[0], n_samples_bootstrap)
    tree.partial_fit(X[indices, :], y[indices], classes=classes)

    return tree


class StreamDecisionForest(RandomForestClassifier):
    """
    A class used to represent a naive ensemble of
    random stream decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        An integer that represents the number of stream decision trees.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : {"sqrt", "log2"}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    n_swaps : int, default=1
        The number of trees to swap at each partial fitting. The actual
        swaps occur with `1/n_batches_` probability.

    Attributes
    ----------
    estimators_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains all
        sklearn.tree.DecisionTreeClassifier.

    classes_ : list of all unique class labels
        An internal list that stores class labels after the first call
        to `partial_fit`.

    n_batches_ : int
        The number of batches seen with `partial_fit`.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        max_bins=None,
        store_leaf_values=False,
        monotonic_cst=None,
        n_swaps=1,
    ):

        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
            max_bins=max_bins,
            store_leaf_values=store_leaf_values,
        )
        self.n_swaps = n_swaps

    def fit(self, X, y, sample_weight=None, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response data matrix).

        classes : ndarray, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : StreamDecisionForest
            The object itself.
        """
        self.n_batches_ = 1

        return super().fit(X, y, sample_weight=sample_weight, classes=classes)

    def partial_fit(self, X, y, sample_weight=None, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        classes : ndarray, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : StreamDecisionForest
            The object itself.
        """
        self._validate_params()

        # validate input parameters
        first_call = _check_partial_fit_first_call(self, classes=classes)

        # Fit if no tree exists yet
        if first_call:
            self.fit(
                X,
                y,
                sample_weight=sample_weight,
                classes=classes,
            )
            return self
        self.n_batches_ += 1

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(X.shape[0], self.max_samples)
        else:
            n_samples_bootstrap = X.shape[0]

        # Calculate probability of swaps
        swap_prob = 1 / self.n_batches_
        if self.n_swaps > 0 and self.n_batches_ > 2 and np.random.random() <= swap_prob:
            # Evaluate forest performance
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict)(X) for tree in self.estimators_
            )

            # Sort predictions by accuracy
            acc_l = []
            for idx, result in enumerate(results):
                acc_l.append([accuracy_score(result, y), idx])
            acc_l = sorted(acc_l, key=lambda x: x[0])

            # Generate new trees
            new_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(_partial_fit)(
                    DecisionTreeClassifier(
                        criterion=self.criterion,
                        splitter=self.splitter,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                        max_features=self.max_features,
                        max_leaf_nodes=self.max_leaf_nodes,
                        class_weight=self.lass_weight,
                        random_state=self.random_state,
                        min_impurity_decrease=self.min_impurity_decrease,
                        monotonic_cst=self.monotonic_cst,
                        ccp_alpha=self.ccp_alpha,
                        store_leaf_values=self.store_leaf_values,
                    ),
                    X,
                    y,
                    n_samples_bootstrap=n_samples_bootstrap,
                    classes=self.classes_,
                )
                for i in range(self.n_swaps)
            )

            # Swap worst performing trees with new trees
            for i in range(self.n_swaps):
                self.estimators_[acc_l[i][1]] = new_trees[i]

        # Update existing stream decision trees
        super().partial_fit(X, y, classes=classes)

        return self

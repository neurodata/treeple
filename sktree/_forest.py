"""
Manifold forest of trees-based ensemble methods.

Those methods include various random forest methods that operate on manifolds.

The module structure is the following:

Single and multi-output problems are both handled.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

from sklearn.base import TransformerMixin
from sklearn.ensemble._forest import BaseForest

from .tree import UnsupervisedDecisionTree


class UnsupervisedRandomForest(TransformerMixin, BaseForest):
    """Unsupervised random forest.

    Parameters
    ----------
    n_estimators : int, optional
        _description_, by default 100
    estimator_params : _type_, optional
        _description_, by default tuple()
    bootstrap : bool, optional
        _description_, by default False
    oob_score : bool, optional
        _description_, by default False
    n_jobs : _type_, optional
        _description_, by default None
    random_state : _type_, optional
        _description_, by default None
    verbose : int, optional
        _description_, by default 0
    warm_start : bool, optional
        _description_, by default False
    max_samples : _type_, optional
        _description_, by default None
    """

    def __init__(
        self,
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
            estimator=UnsupervisedDecisionTree(),
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

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The 2D data matrix with columns as features and rows as samples.
        y : ArrayLike, optional
            Not used. Passed in for API consistency.
        sample_weight : ArrayLike of shape (n_samples), optional
            The samples weight, by default None.

        Returns
        -------
        output : ArrayLike of shape (n_samples,)
            The transformed output from passing `X` through the forest.
        """
        super().fit(X, y, sample_weight)
        # apply to the leaves
        output = self.apply(X)
        return output

    def fit(self, X, y=None, sample_weight=None):
        """Fit unsupervised forest.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The 2D data matrix with columns as features and rows as samples.
        y : ArrayLike, optional
            Not used. Passed in for API consistency.
        sample_weight : ArrayLike of shape (n_samples), optional
            The samples weight, by default None.

        Returns
        -------
        self : UnsupervisedRandomForest
            The fitted forest.
        """
        # Parameters are validated in fit_transform
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

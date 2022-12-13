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
from sklearn.ensemble._forest import ForestClassifier, BaseForest

from .tree import UnsupervisedDecisionTree


class UnsupervisedRandomForest(TransformerMixin, BaseForest):
    def __init__(self,
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
            max_samples=max_samples)·

    def fit_transform(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
·
        # apply to the leaves
        output = self.apply(X)
        return output

    def fit(self, X, y=None, sample_weight=None):
        # Parameters are validated in fit_transform
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

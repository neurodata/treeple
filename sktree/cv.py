from typing import Any, List, Tuple

from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from sktree._lib.sklearn.ensemble._forest import BaseForest


def cross_fit_forest(
    forest: BaseForest,
    X: ArrayLike,
    y: ArrayLike,
    cv: BaseCrossValidator = None,
    n_splits: int = 5,
    random_state=None,
) -> Tuple[List[Any], List[ArrayLike]]:
    """Perform cross-fitting of a forest estimator.

    Parameters
    ----------
    forest : BaseForest
        Forest.
    X : ArrayLike of shape (n_samples, n_features)
        Input data.
    y : ArrayLike of shape (n_samples, [n_outputs])
        Target data.
    cv : BaseCrossValidator, optional
        Cross validation object, by default None, which defaults to
        :class:`sklearn.model_selection.StratifiedKFold`.
    n_splits : int, optional
        Number of folds to generate, by default 5.
    random_state : int, optional
        Random seed.

    Returns
    -------
    fitted_forests : List[BaseForest]
        List of fitted forests.
    test_indices : List[ArrayLike]
        List of test indices over ``n_samples``.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    fitted_forests = []
    test_indices = []
    for train_index, test_index in cv.split(X, y):
        new_forest = clone(forest)
        new_forest.fit(X[train_index], y[train_index])

        fitted_forests.append(new_forest)
        test_indices.append(test_index)

    return fitted_forests, test_indices

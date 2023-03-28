.. _oblique_forests:

Oblique Random Forests
----------------------

In oblique random forests (see :class:`ObliqueRandomForestClassifier` and
:class:`ObliqueRandomForestRegressor` classes), each tree in the ensemble is built
from a sample drawn with replacement (i.e., a bootstrap sample) from the
training set. The oblique random forest is the same as that of a random forest,
except in how the splits are computed in each tree.

Similar to how random forests achieve a reduced variance by combining diverse trees,
sometimes at the cost of a slight increase in bias, oblique random forests aim to do the same.
They are motivated to construct even more diverse trees, thereby improving model generalization.
In practice the variance reduction is often significant hence yielding an overall better model.

In contrast to the original publication [B2001]_, the scikit-learn
implementation allows the user to control the number of features to combine in computing
candidate splits. This is done via the ``feature_combinations`` parameter. For
more information and intuition, see
:ref:`documentation on oblique decision trees <oblique_trees>`.


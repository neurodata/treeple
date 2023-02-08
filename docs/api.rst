.. _api_documentation:

=================
API Documentation
=================

:py:mod:`sktree`:

.. automodule:: sktree
   :no-members:
   :no-inherited-members:

Unsupervised
------------
Decision-tree models are traditionally used for classification and regression.
However, they are also powerful non-parametric embedding and clustering models.
The :class:`~sklearn.ensemble.RandomTreesEmbedding` is an example of unsupervised
tree model. We implement other state-of-the-art models that explicitly split based
on unsupervised criterion such as variance and BIC.

.. currentmodule:: sktree
.. autosummary::
   :toctree: generated/
   
   UnsupervisedRandomForest
   UnsupervisedObliqueRandomForest

.. autosummary::
   :toctree: generated/
   
   tree.UnsupervisedDecisionTree
   tree.UnsupervisedObliqueDecisionTree

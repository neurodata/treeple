
.. _oblique_trees:

Oblique Trees
=============

Similar to DTs, **Oblique Trees (OTs)** are a non-parametric supervised learning
method used for :ref:`classification <tree_classification>` and :ref:`regression
<tree_regression>`. It was originally described as ``Forest-RC`` in Breiman's
landmark paper on Random Forests [RF]_. Breiman found that combining data features
empirically outperforms DTs on a variety of data sets.

The algorithm implemented in scikit-learn differs from ``Forest-RC`` in that
it allows the user to specify the number of variables to combine to consider
as a split, :math:`\lambda`. If :math:`\lambda` is set to ``n_features``, then
it is equivalent to ``Forest-RC``. :math:`\lambda` presents a tradeoff between
considering dense combinations of features vs sparse combinations of features.

Differences compared to decision trees
--------------------------------------

Compared to DTs, OTs differ in how they compute a candidate split. DTs split
along the passed in data columns in an axis-aligned fashion, whereas OTs split
along oblique curves. Using the Iris dataset, we can similarly construct an OT
as follows:

    >>> from sklearn.datasets import load_iris
    >>> from sktree import tree
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> clf = tree.ObliqueDecisionTreeClassifier()
    >>> clf = clf.fit(X, y)

.. figure:: ../auto_examples/tree/images/sphx_glr_plot_iris_dtc_002.png
   :target: ../auto_examples/tree/plot_iris_dtc.html
   :scale: 75
   :align: center

Another major difference to DTs is that OTs can by definition sample more candidate
splits. The parameter ``max_features`` controls how many splits to sample at each
node. For DTs "max_features" is constrained to be at most "n_features" by default,
whereas OTs can sample possibly up to :math:`2^{n_{features}}` candidate splits
because they are combining features.

Mathematical formulation
------------------------

Given training vectors :math:`x_i \in R^n`, i=1,..., l and a label vector
:math:`y \in R^l`, an oblique decision tree recursively partitions the
feature space such that the samples with the same labels or similar target
values are grouped together. Normal decision trees partition the feature space
in an axis-aligned manner splitting along orthogonal axes based on the dimensions
(columns) of :math:`x_i`. In oblique trees, nodes sample a random projection vector,
:math:`a_i \in R^n`, where the inner-product of :math:`\langle a_i, x_i \rangle`
is a candidate split value. The entries of :math:`a_i` have values
+/- 1 with probability :math:`\lambda / n` with the rest being 0s.

Let the data at node :math:`m` be represented by :math:`Q_m` with :math:`n_m`
samples. For each candidate split :math:`\theta = (a_i, t_m)` consisting of a
(possibly sparse) vector :math:`a_i` and threshold :math:`t_m`, partition the
data into :math:`Q_m^{left}(\theta)` and :math:`Q_m^{right}(\theta)` subsets

.. math::

    Q_m^{left}(\theta) = \{(x, y) | a_i^T x_j \leq t_m\}

    Q_m^{right}(\theta) = Q_m \setminus Q_m^{left}(\theta)

Note that this formulation is a generalization of decision trees, where
:math:`a_i = e_i`, a standard basis vector with a "1" at index "i" and "0"
elsewhere. 

The quality of a candidate split of node :math:`m` is then computed using an
impurity function or loss function :math:`H()`, in the same exact manner as
decision trees.

Limitations compared to decision trees
--------------------------------------

  * There currently does not exist support for pruning OTs, such as with the minimal
    cost-complexity pruning algorithm.
  
  * Moreover, OTs do not have built-in support for missing data, so the recommendation
    by scikit-learn is for users to first impute, or drop their missing data if they
    would like to use OTs.

  * Currently, OTs also does not support sparse inputs for data matrices and labels.

.. topic:: References:

    .. [BRE] L. Breiman, J. Friedman, R. Olshen, and C. Stone. Classification
      and Regression Trees. Wadsworth, Belmont, CA, 1984.
    
    .. [RF] L. Breiman. Random Forests. Machine Learning 45, 5–32 (2001).
      https://doi.org/10.1023/A:1010933404324.
      
    * https://en.wikipedia.org/wiki/Decision_tree_learning

    * https://en.wikipedia.org/wiki/Predictive_analytics

    * J.R. Quinlan. C4. 5: programs for machine learning. Morgan
      Kaufmann, 1993.

    * T. Hastie, R. Tibshirani and J. Friedman. Elements of Statistical
      Learning, Springer, 2009.

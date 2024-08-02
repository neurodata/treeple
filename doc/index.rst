**treeple**
===================
treeple is a package for modern tree-based algorithms for supervised and unsupervised
learning problems. It extends the robust API of `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
for tree algorithms that achieve strong performance in benchmark tasks.

Our package has implemented unsupervised forests (Geodesic Forests
[Madhyastha2020]_), oblique random forests (SPORF [Tomita2020]_, manifold random forests,
MORF [Li2023]_), honest forests [Perry2021]_, extended isolation forests [Hariri2019]_, and more.

For all forests, we also support incremental building of the forests, using the
``partial_fit`` API from scikit-learn [Xu2022]_, and quantile regression by storing
the training samples in the leaves of the trees [Meinshausen2006]_ (Warning: high memory usage
will occur in this setting since predicting quantiles stores the training data within the
leaves of the tree).

We encourage you to use the package for your research and also build on top
with relevant Pull Requests. See our examples for walk-throughs of how to use the package.
Also, see our `contributing guide <https://github.com/neurodata/treeple/blob/main/CONTRIBUTING.md>`_.

We are licensed under PolyForm Noncommercial License (see `License <https://github.com/neurodata/treeple/blob/main/LICENSE>`_).

.. topic:: References

 .. [Hariri2019] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
   "Extended isolation forest." IEEE transactions on knowledge and data
   engineering 33.4 (2019): 1479-1489.

 .. [Meinshausen2006] Meinshausen, Nicolai, and Greg Ridgeway. "Quantile regression forests."
   Journal of machine learning research 7.6 (2006). "Quantile regression forests."

 .. [Madhyastha2020] Madhyastha, Meghana, et al. :doi:`"Geodesic Forests"
    <10.1145/3394486.3403094>`, KDD 2020, 513-523, 2020.

 .. [Tomita2020] Tomita, Tyler M., et al. "Sparse Projection Oblique
    Randomer Forests", The Journal of Machine Learning Research, 21(104),
    1-39, 2020.

 .. [Li2023] Li, Adam, et al. :doi:`"Manifold Oblique Random Forests: Towards
    Closing the Gap on Convolutional Deep Networks" <10.1137/21M1449117>`,
    SIAM Journal on Mathematics of Data Science, 5(1), 77-96, 2023.

 .. [Perry2021] Perry, Ronan, et al. :arxiv:`"Random Forests for Adaptive
    Nearest Neighbor Estimation of Information-Theoretic Quantities"
    <1907.00325>`, arXiv preprint arXiv:1907.00325, 2021.

 .. [Xu2022] Xu, Haoyin, et al. :arxiv:`"Simplest Streaming Trees"
    <2110.08483>`, arXiv preprint arXiv:2110.08483, 2022.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   api
   User Guide<user_guide>
   whats_new
   install
   use

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`

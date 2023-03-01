.. _tree:

===========================
Unsupervised Decision Trees
===========================

.. currentmodule:: sklearn.tree

In unsupervised learning, the goal is to identify patterns 
or structure in data without using labeled examples. Clustering is a common 
unsupervised learning technique that groups similar examples together 
based on their features. In clustering settings, there may be no natural 
notion of “true” class-labels, thus the efficacy of the clustering scheme is 
often measured based on distance based metrics such as :func:`adjusted_rand_score`.

One way to combine clustering with decision trees is to use a clustering 
algorithm to partition the data into groups, and then build a decision tree 
on each group. This approach is known as clustering-based decision trees or 
clustering trees. Each decision tree only needs to consider a subset of 
the data, which can lead to faster and more accurate predictions. Two of 
the most common apporaches used for unsupervised classification are 
K-Means clustering and Mixture Modeling, which use the Expectation 
Maximization Algorithm.

In clustering settings, there may be no natural notion of “true”
class-labels; the efficacy of the clustering scheme is often measured 
in such cases by the economy in description length attained
by a two-step description of the objects by first describing the 
attributes common to the clusters and then describing the differential
attributes of each object within the cluster. k-Means Clustering
and Mixture Modeling using the Expectation Maximization Algorithm 
are examples of techniques used for unsupervised classification.

Another way to use unsupervised learning with decision trees is to use 
clustering to pre-process the data before training a supervised 
decision tree. This can help identify patterns in the data and reduce 
noise, which can improve the performance of the decision tree.

In summary, "unsupervised decision trees" could refer to either 
clustering-based decision trees or decision trees trained on 
pre-processed data using unsupervised learning techniques.


.. _tree_cluster:

Cluster criteria
----------------

Clustering criterion for :class:`UnsupervisedDecisionTree` is a technique that combines clustering 
and decision tree algorithms to produce a set of decision rules. The clustering 
criterion can be used to identify clusters of data that have similar 
characteristics, and then construct decision rules based on the clustering 
results.

One common clustering criterion for decision trees is the :class:`KMeans` algorithm. 
The :class:`KMeans` algorithm partitions the data into K clusters based on the 
distance between the data points. The algorithm then constructs a decision 
tree on each cluster, using the same process as a standard decision tree 
algorithm.

Another clustering criterion for decision trees is the :class:`AgglomerativeClustering`
algorithm. :class:`AgglomerativeClustering` is a bottom-up clustering method that starts 
with each data point as a cluster and then iteratively merges the closest 
pairs of clusters until all the data points are in a single cluster. 
The resulting tree structure can be used to construct a decision tree by 
selecting a threshold to cut the tree at a particular level.

The advantage of using clustering criteria in decision trees is that 
it can help identify subgroups of data with different characteristics, 
which can lead to more accurate and interpretable models. However, the 
choice of clustering criterion and the number of clusters can significantly 
affect the performance of the decision tree. Therefore, it is important to 
carefully evaluate different clustering criteria and parameter settings to 
find the best approach for a particular dataset.

Two-Means
~~~~~~~~~

The two means split finds the cutpoint that minimizes the one-dimensional
2-means objective, which is finding the cutoff point where the total variance
from cluster 1 and cluster 2 are minimal.

.. math:: 
  \min_s \sum_{i=1}^s (x_i - \hat{\mu}_1)^2 + \sum_{i=s+1}^N (x_i - \hat{\mu}_2)^2

where x is a N-dimensional feature vector, N is the number of sample_indices and
the \mu terms are the estimated means of each cluster 1 and 2.

FastBIC
~~~~~~~

The Bayesian Information Criterion (BIC) is a popular model seleciton 
criteria that is based on the log likelihood of the model given data.
:class:`FastBIC` is a method that combines the speed of the :class:`TwoMeans` clustering 
method with the model flexibility of Mclust-BIC. It sorts data for each 
feature and tries all possible splits to assign data points to one of 
two Gaussian distributions based on their position relative to the split.
The parameters for each cluster are estimated using maximum likelihood 
estimation (MLE).The method performs hard clustering rather than soft 
clustering like in GMM, resulting in a simpler calculation of the likelihood.

.. math::

  \hat{L} = \sum_{n=1}^s[\log\hat{\pi}_1+\log{\mathcal{N}(x_n;\hat{\mu}_1,\hat{\sigma}_1^2)}]
  + \sum_{n=s+1}^N[\log\hat{\pi}_2+\log{\mathcal{N}(x_n;\hat{\mu}_2,\hat{\sigma}_2^2)}]

where the prior, mean, and variance are defined as follows, respectively:

.. math::

  \hat{\pi} = \frac{s}{N},\quad\quad
  \hat{\mu} = \frac{1}{s}\sum_{n\le s}{x_n},\quad\quad
  \hat{\sigma}^2 = \frac{1}{s}\sum_{n\le s}{||x_n-\hat{\mu_j}||^2}




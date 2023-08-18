# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport numpy as cnp
import numpy as np
from libc.math cimport log
from libcpp.unordered_map cimport unordered_map

cnp.import_array()

cdef DTYPE_t PI = np.pi


cdef class CriterionTester:
    cpdef init(self,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] samples,
        const DTYPE_t[:] Xf
    ):
        n_classes = np.array([2])
        n_samples = Xf.shape[0]

        # Create instances of FastBIC and FasterBIC with test data
        cdef int n_outputs = 1
        cdef int start = 0
        cdef int end = len(Xf)

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        weighted_n_samples = np.sum(sample_weight)
        feature_values_fast = np.empty(n_samples, dtype=np.float32)
        feature_values_faster = np.empty(n_samples, dtype=np.float32)

        # initialize the two criterion
        fast_bic = FastBIC(n_outputs, n_classes)
        faster_bic = FasterBIC(n_outputs, n_classes)
        fast_bic.init(
            sample_weight,
            weighted_n_samples,
            samples,
            feature_values_fast,
        )
        faster_bic.init(
            sample_weight,
            weighted_n_samples,
            samples,
            feature_values_faster,
        )
        fast_bic.set_sample_pointers(
            start,
            end
        )
        faster_bic.set_sample_pointers(
            start,
            end
        )
        fast_bic.init_feature_vec()
        faster_bic.init_feature_vec()

        self.fast_bic = fast_bic
        self.faster_bic = faster_bic

    cpdef init_feature_vec(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.init_feature_vec()
        faster_bic.init_feature_vec()

    cpdef update(self, int pos):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.update(pos)
        faster_bic.update(pos)

    cpdef node_impurity(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        fast_bic_imp = fast_bic.node_impurity()
        faster_bic_imp = faster_bic.node_impurity()

        return fast_bic_imp, faster_bic_imp

    cpdef children_impurity(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        cdef double fast_bic_left
        cdef double fast_bic_right
        cdef double faster_bic_left
        cdef double faster_bic_right
        fast_bic.children_impurity(&fast_bic_left, &fast_bic_right)
        faster_bic.children_impurity(&faster_bic_left, &faster_bic_right)

        return (fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right)

    cpdef sum_total(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        return fast_bic.sum_total, faster_bic.sum_total

    cpdef sum_left(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        return fast_bic.sum_left, faster_bic.sum_left

    cpdef sum_right(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        return fast_bic.sum_right, faster_bic.sum_right

    cpdef proxy_impurity(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        fast_bic_imp = fast_bic.proxy_impurity_improvement()
        faster_bic_imp = faster_bic.proxy_impurity_improvement()

        return fast_bic_imp, faster_bic_imp

    cpdef reset(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.reset()
        faster_bic.reset()

    cpdef set_sample_pointers(self, int start, int end):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.set_sample_pointers(start, end)
        faster_bic.set_sample_pointers(start, end)

    cpdef weighted_n_samples(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        # for idx in range(len(np.array([0, 1, 2, 3, 5, 10, 20, 200]).reshape(-1, 1))):
        #     print('idx: ', idx)
        #     print(faster_bic.sample_indices[idx])
        #     print(faster_bic.cumsum_weights_map[faster_bic.sample_indices[idx]])
        #     print(faster_bic.cumsum_map[faster_bic.sample_indices[idx]])


        return ((fast_bic.weighted_n_node_samples, fast_bic.weighted_n_left, fast_bic.weighted_n_right),
            (faster_bic.weighted_n_node_samples, faster_bic.weighted_n_left, faster_bic.weighted_n_right))

cdef class UnsupervisedCriterion(BaseCriterion):
    """Abstract criterion for unsupervised learning.

    This object is a copy of the Criterion class of scikit-learn, but is used
    for unsupervised learning. However, ``Criterion`` in scikit-learn was
    designed for supervised learning, where the necessary
    ingredients to compute a split point is solely with y-labels. In
    this object, we subclass and instead rely on the X-data.

    This object stores methods on how to calculate how good a split is using
    different metrics for unsupervised splitting.
    """

    def __cinit__(self):
        """Initialize attributes for unsupervised criterion.
        """
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Initialize count metric for current, going left and going right
        self.sum_total = 0.0
        self.sum_left = 0.0
        self.sum_right = 0.0

    def __reduce__(self):
        return (type(self), (), self.__getstate__())

    cdef int init(
        self,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices,
        const DTYPE_t[:] Xf,
    ) except -1 nogil:
        """Initialize the unsuperivsed criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample (i.e. row of X).
        weighted_n_samples : double
            The total weight of all sample_indices.
        sample_indices : array-like, dtype=SIZE_t
            A mask on the sample_indices, showing which ones we want to use
        Xf : array-like, dtype=DTYPE_t
            The memoryview 1D feature vector with (n_samples,) shape.
        """
        self.sample_weight = sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.sample_indices = sample_indices
        self.Xf = Xf

        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.sum_left = 0.0
        self.sum_right = self.sum_total
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        self.sum_right = 0.0
        self.sum_left = self.sum_total
        return 0

    cdef void init_feature_vec(
        self,
    ) noexcept nogil:
        """Initialize the 1D feature vector, which is used for computing criteria.

        This function is used to set a read-only memoryview of a feature
        vector. The feature vector must be set in order for criteria to be
        computed. It then keeps a running total of the feature vector from
        samples[start:end] so that way it is efficient to compute the right and
        left sums and corresponding metrics.

        Parameters
        ----------
        Xf : array-like, dtype=DTYPE_t
            The read-only memoryview 1D feature vector with (n_samples,) shape.
        """
        # also compute the sum total
        self.sum_total = 0.0
        self.weighted_n_node_samples = 0.0
        cdef SIZE_t s_idx
        cdef SIZE_t p_idx

        cdef DOUBLE_t w = 1.0
        for p_idx in range(self.start, self.end):
            s_idx = self.sample_indices[p_idx]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if self.sample_weight is not None:
                w = self.sample_weight[s_idx]

            self.sum_total += self.Xf[s_idx] * w
            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()

    cdef int update(
        self,
        SIZE_t new_pos
    ) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef const SIZE_t[:] sample_indices = self.sample_indices
        cdef const DOUBLE_t[:] sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                # accumulate the values of the feature vectors weighted
                # by the sample weight
                self.sum_left += self.Xf[i] * w

                # keep track of the weighted count of each sample
                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                self.sum_left -= self.Xf[i] * w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.sum_right = self.sum_total - self.sum_left

        self.pos = new_pos
        return 0

    cdef void node_value(
        self,
        double* dest
    ) noexcept nogil:
        """Set the node value with sum_total and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        # set values at the address pointer is pointing to with the total value
        dest[0] = self.sum_total

    cdef void set_sample_pointers(
        self,
        SIZE_t start,
        SIZE_t end
    ) noexcept nogil:
        """Set sample pointers in the criterion.

        Set given start and end sample_indices. Also will update node statistics,
        such as the `sum_total`, which tracks the total value within the current
        node for sample_indices[start:end].

        Parameters
        ----------
        start : SIZE_t
            The start sample pointer.
        end : SIZE_t
            The end sample pointer.
        """
        self.n_node_samples = end - start
        self.start = start
        self.end = end


cdef class TwoMeans(UnsupervisedCriterion):
    r"""Two means split impurity.

    The two means split finds the cutpoint that minimizes the one-dimensional
    2-means objective, which is finding the cutoff point where the total variance
    from cluster 1 and cluster 2 are minimal.

    The mathematical optimization problem is to find the cutoff index ``s``,
    which is called 'pos' in scikit-learn.

        \min_s \sum_{i=1}^s (x_i - \hat{\mu}_1)^2 + \sum_{i=s+1}^N (x_i - \hat{\mu}_2)^2

    where x is a N-dimensional feature vector, N is the number of sample_indices and
    the \mu terms are the estimated means of each cluster 1 and 2.

    The variance of the node, left child and right child is computed by keeping track of
    `sum_total`, `sum_left` and `sum_right`, which are the sums of a feature vector at
    varying split points.

    Weighted Mean and Variance
    --------------------------
    Since we allow `sample_weights` to be passed, then we optionally allow for us
    to compute a weighted sample mean and weighted sample variance. The weighted
    sample variance has two approaches to compute an unbiased estimate. The first
    is using "frequencies" and the second is using "reliability" weights. Currently,
    we have impmlemented the frequencies approach.

    # TODO: implement reliability weighting

    Node-Wise Feature Generation
    ----------------------------
    URerF doesn't choose split points in the original feature space
    It follows the random projection framework

    \tilde{X}= A^T X'

    where, A is p x d matrix distributed as f_A, where f_A is the
    projection distribution and d is the dimensionality of the
    projected space. A is generated by randomly sampling from
    {-1,+1} lpd times, then distributing these values uniformly
    at random in A. l parameter is used to control the sparsity of A
    and is set to 1/20.

    Each of the d rows \tilde{X}[i; :], i \in {1,2,...d} is then
    inspected for the best split point. The optimal split point and
    splitting dimension are chosen according to which point/dimension
    pair minimizes the splitting criteria described in the following
    section
    """

    cdef double node_impurity(
        self
    ) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the TwoMeans criterion impurity as variance of the current node,
        i.e. the variance of Xf[sample_indices[start:end]]. The smaller the impurity the
        better.
        """
        cdef double mean
        cdef double impurity

        # If calling without setting the
        if self.Xf is None:
            with gil:
                raise MemoryError(
                    'Xf has not been set yet, so one must call init_feature_vec.'
                )

        # first compute mean
        mean = self.sum_total / self.weighted_n_node_samples

        # then compute the impurity as the variance
        impurity = self.sum_of_squares(
            self.start,
            self.end,
            mean
        ) / self.weighted_n_node_samples
        return impurity

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).

        where the impurity of two children nodes are essentially the variances of
        two nodes:

        - left_variance = left_weight * left_impurity / n_sample_of_left_child
        - right_variance = right_weight * right_impurity / n_sample_of_right_child

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        # first compute mean of left and right
        mean_left = self.sum_left / self.weighted_n_left
        mean_right = self.sum_right / self.weighted_n_right

        # set values at the address pointer is pointing to with the variance
        # of the left and right child
        impurity_left[0] = self.sum_of_squares(
            start,
            pos,
            mean_left
        ) / self.weighted_n_left
        impurity_right[0] = self.sum_of_squares(
            pos,
            end,
            mean_right
        ) / self.weighted_n_right

    cdef double sum_of_squares(
        self,
        SIZE_t start,
        SIZE_t end,
        double mean,
    ) noexcept nogil:
        """Computes variance of feature vector from sample_indices[start:end].

        See: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance.  # noqa

        Parameters
        ----------
        start : SIZE_t
            The start pointer
        end : SIZE_t
            The end pointer.
        mean : double
            The precomputed mean.

        Returns
        -------
        ss : double
            Sum of squares
        """
        cdef SIZE_t s_idx, p_idx        # initialize sample and pointer index
        cdef double ss = 0.0            # sum-of-squares
        cdef DOUBLE_t w = 1.0           # optional weight

        # calculate variance for the sample_indices chosen start:end
        for p_idx in range(start, end):
            s_idx = self.sample_indices[p_idx]

            # include optional weighted sum of squares
            if self.sample_weight is not None:
                w = self.sample_weight[s_idx]

            ss += w * (self.Xf[s_idx] - mean) * (self.Xf[s_idx] - mean)
        return ss


cdef class FastBIC(TwoMeans):
    r"""Fast-BIC split criterion

    The Bayesian Information Criterion (BIC) is a popular model seleciton
    criteria that is based on the log likelihood of the model given data.
    MClust-BIC is a method in the R-package 'mclust' that uses BIC as a splitting
    criterion.
    See: https://stats.stackexchange.com/questions/237220/mclust-model-selection

    Fast-BIC is a method that combines the speed of the two-means clustering
    method with the model flexibility of Mclust-BIC. It sorts data for each
    feature and tries all possible splits to assign data points to one of
    two Gaussian distributions based on their position relative to the split.
    The parameters for each cluster are estimated using maximum likelihood
    estimation (MLE). The method performs hard clustering rather than soft
    clustering like in GMM, resulting in a simpler calculation of the likelihood.

    \hat{L} = \sum_{n=1}^s[\log\hat{\pi}_1+\log{\mathcal{N}(x_n;\hat{\mu}_1,\hat{\sigma}_1^2)}]
    + \sum_{n=s+1}^N[\log\hat{\pi}_2+\log{\mathcal{N}(x_n;\hat{\mu}_2,\hat{\sigma}_2^2)}]

    where the prior, mean, and variance are defined as follows, respectively:

    - \hat{\pi} = \frac{s}{N}
    - \hat{\mu} = \frac{1}{s}\sum_{n\le s}{x_n},
    - \hat{\sigma}^2 = \frac{1}{s}\sum_{n\le s}{||x_n-\hat{\mu_j}||^2}

    Fast-BIC is guaranteed to obtain the global maximum likelihood estimator.
    Additionally, Fast-BIC is substantially faster than the traditional BIC method.

    Reference: https://arxiv.org/abs/1907.02844
    """
    cdef double bic_cluster(self, SIZE_t n_samples, double variance) noexcept nogil:
        """Help compute the BIC from assigning to a specific cluster.

        Parameters
        ----------
        n_samples : SIZE_t
            The number of samples assigned cluster.
        variance : double
            The plug-in variance for assigning to specific cluster.

        Notes
        -----
        Computes the following:

        :math:`-2 * (n_i log(w_i) - n_i/2 log(2 \\pi \\sigma_i^2))

        where :math:`n_i` is the number of samples assigned to cluster i,
        :math:`w_i` is the probability of choosing cluster i at random (or also known
        as the prior) and :math:`\\sigma_i^2` is the variance estimate for cluster i.

        Note that :math:`\\sigma_i^2` in the Fast-BIC derivation may be the
        variance of the cluster itself, or the estimated combined variance
        from both clusters.
        """
        cdef SIZE_t n_node_samples = self.n_node_samples

        # chances of choosing the cluster based on how many samples are hard-assigned to cluster
        # i.e. the prior
        # cast to double, so we do not round to integers
        cdef double w_cluster = (n_samples + 0.0) / n_node_samples

        # add to prevent taking log of 0 when there is a degenerate cluster (i.e. single sample, or no variance)
        return -2. * (n_samples * log(w_cluster) + 0.5 * n_samples * log(2. * PI * variance + 1.e-7))

    cdef double node_impurity(
        self
    ) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the FastBIC criterion impurity as estimated maximum log likelihood.
        This is the maximum likelihood given prior, mean, and variance at s number of samples
        Namely, this is the maximum likelihood of Xf[sample_indices[start:end]].
        The smaller the impurity the better.
        """
        cdef double mean
        cdef double variance
        cdef double impurity
        cdef SIZE_t n_node_samples = self.n_node_samples

        # If calling without setting the
        if self.Xf is None:
            with gil:
                raise MemoryError(
                    'Xf has not been set yet, so one must call init_feature_vec.'
                )

        # first compute mean
        mean = self.sum_total / self.weighted_n_node_samples

        # then compute the variance of the cluster
        variance = self.sum_of_squares(
            self.start,
            self.end,
            mean
        ) / self.weighted_n_node_samples

        # Compute the BIC of the current set of samples
        # Note: we do not compute the BIC_diff_var and BIC_same_var because
        # they are equivalent in the single cluster setting
        impurity = self.bic_cluster(n_node_samples, variance)
        return impurity

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t n_samples_left, n_samples_right

        cdef double mean_left, mean_right
        cdef double ss_left, ss_right, variance_left, variance_right, variance_comb
        cdef double BIC_diff_var_left, BIC_diff_var_right
        cdef double BIC_same_var_left, BIC_same_var_right
        cdef double BIC_same_var, BIC_diff_var

        # number of samples of left and right
        n_samples_left = pos - start
        n_samples_right = end - pos

        # first compute mean of left and right
        mean_left = self.sum_left / self.weighted_n_left
        mean_right = self.sum_right / self.weighted_n_right

        # compute the estimated variance of the left and right children
        ss_left = self.sum_of_squares(
            start,
            pos,
            mean_left
        )
        ss_right = self.sum_of_squares(
            pos,
            end,
            mean_right
        )
        variance_left = ss_left / self.weighted_n_left
        variance_right = ss_right / self.weighted_n_right

        # compute the estimated combined variance
        variance_comb = (ss_left + ss_right) / (self.weighted_n_left + self.weighted_n_right)

        # Compute the BIC using different variances for left and right
        BIC_diff_var_left = self.bic_cluster(n_samples_left, variance_left)
        BIC_diff_var_right = self.bic_cluster(n_samples_right, variance_right)

        # Compute the BIC using different variances for left and right
        BIC_same_var_left = self.bic_cluster(n_samples_left, variance_comb)
        BIC_same_var_right = self.bic_cluster(n_samples_right, variance_comb)
        BIC_same_var = BIC_same_var_left - BIC_same_var_right
        BIC_diff_var = BIC_diff_var_left - BIC_diff_var_right

        # choose the BIC formulation that gives us the smallest values
        # (i.e. min of (BIC_diff, BIC_same) in the paper) and then
        # assign the left and right child BIC values by reference
        if BIC_diff_var < BIC_same_var:
            impurity_left[0] = -BIC_diff_var_left
            impurity_right[0] = -BIC_diff_var_right
        else:
            impurity_left[0] = -BIC_same_var_left
            impurity_right[0] = -BIC_same_var_right


cdef class FasterBIC(UnsupervisedCriterion):
    r"""Faster-BIC split criterion

    This utilizes a trick from [2]_ to improve the computation for the variance.
    Since we have an arbitrary segment $X_i, ..., X_j$ with $1 \le i \le j \le n_samples$,
    we can compute the 
    
    Reference: 
    [1] https://arxiv.org/pdf/2110.13883.pdf
    [2] E. Terzi, Problems and algorithms for sequence segmen- tations. Helsingin yliopisto, 2006.
    """
    cdef unordered_map[SIZE_t, DTYPE_t] cumsum_of_squares_map
    cdef unordered_map[SIZE_t, DTYPE_t] cumsum_map
    cdef unordered_map[SIZE_t, DTYPE_t] cumsum_weights_map

    cdef double bic_cluster(self, SIZE_t n_samples, double variance) noexcept nogil:
        """Help compute the BIC from assigning to a specific cluster.

        Parameters
        ----------
        n_samples : SIZE_t
            The number of samples assigned cluster.
        variance : double
            The plug-in variance for assigning to specific cluster.

        Notes
        -----
        Computes the following:

        :math:`-2 * (n_i log(w_i) - n_i/2 log(2 \\pi \\sigma_i^2))

        where :math:`n_i` is the number of samples assigned to cluster i,
        :math:`w_i` is the probability of choosing cluster i at random (or also known
        as the prior) and :math:`\\sigma_i^2` is the variance estimate for cluster i.

        Note that :math:`\\sigma_i^2` in the Fast-BIC derivation may be the
        variance of the cluster itself, or the estimated combined variance
        from both clusters.
        """
        cdef SIZE_t n_node_samples = self.n_node_samples

        # chances of choosing the cluster based on how many samples are hard-assigned to cluster
        # i.e. the prior
        # cast to double, so we do not round to integers
        cdef double w_cluster = (n_samples + 0.0) / n_node_samples

        # add to prevent taking log of 0 when there is a degenerate cluster (i.e. single sample, or no variance)
        return -2. * (n_samples * log(w_cluster) + 0.5 * n_samples * log(2. * PI * variance + 1.e-7))

    cdef double node_impurity(
        self
    ) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the FastBIC criterion impurity as estimated maximum log likelihood.
        This is the maximum likelihood given prior, mean, and variance at s number of samples
        Namely, this is the maximum likelihood of Xf[sample_indices[start:end]].
        The smaller the impurity the better.
        """
        cdef double variance
        cdef double impurity
        cdef SIZE_t n_node_samples = self.n_node_samples

        # then compute the variance of the cluster
        # see Section 5.2 in reference.
        variance = self.fast_total_variance(self.weighted_n_node_samples, self.end)

        # Compute the BIC of the current set of samples
        # Note: we do not compute the BIC_diff_var and BIC_same_var because
        # they are equivalent in the single cluster setting
        impurity = self.bic_cluster(n_node_samples, variance)
        return impurity

    cdef inline DTYPE_t fast_total_variance(self, double weighted_n_node_samples, SIZE_t j) noexcept nogil:
        """Computes variance in O(1).
        
        Computes:

        \sigma_{i,j}^2 = \frac{1}{j-i+1} ((css_j - css_{i-1}) - \frac{1}{j-i+1} (cs_j - cs_{i-1})^2)
        """
        cdef double normalizer = 1. / weighted_n_node_samples
        cdef SIZE_t s_idx = self.sample_indices[j]
        return normalizer * \
            (
                (self.cumsum_of_squares_map[s_idx]) - \
                (normalizer * (self.cumsum_map[s_idx]) * (self.cumsum_map[s_idx]))
            )

    cdef inline DTYPE_t fast_variance(self, double weighted_n_node_samples, SIZE_t j, SIZE_t i) noexcept nogil:
        """Computes variance in O(1).
        
        Computes:

        \sigma_{i,j}^2 = \frac{1}{j-i+1} ((css_j - css_{i-1}) - \frac{1}{j-i+1} (cs_j - cs_{i-1})^2)
        """
        cdef double normalizer = 1. / weighted_n_node_samples
        cdef SIZE_t sj_idx = self.sample_indices[j]
        cdef SIZE_t si_idx = self.sample_indices[i - 1]

        return normalizer * \
            (
                (self.cumsum_of_squares_map[sj_idx] - self.cumsum_of_squares_map[si_idx]) - \
                (normalizer * (self.cumsum_map[sj_idx] - self.cumsum_map[si_idx]) * (self.cumsum_map[sj_idx] - self.cumsum_map[si_idx]))
            )

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexcept nogil:
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t n_samples_left, n_samples_right

        cdef double variance_left, variance_right, variance_comb
        cdef double BIC_diff_var_left, BIC_diff_var_right
        cdef double BIC_same_var_left, BIC_same_var_right
        cdef double BIC_same_var, BIC_diff_var

        # number of samples of left and right
        n_samples_left = pos - start
        n_samples_right = end - pos

        # compute the variance of the node, left, and right child
        variance_left = self.fast_total_variance(self.weighted_n_left, pos)
        variance_right = self.fast_variance(self.weighted_n_right, end, pos)
        variance_comb = self.fast_total_variance(self.weighted_n_node_samples, end)

        # Compute the BIC using different variances for left and right
        BIC_diff_var_left = self.bic_cluster(n_samples_left, variance_left)
        BIC_diff_var_right = self.bic_cluster(n_samples_right, variance_right)

        # Compute the BIC using different variances for left and right
        BIC_same_var_left = self.bic_cluster(n_samples_left, variance_comb)
        BIC_same_var_right = self.bic_cluster(n_samples_right, variance_comb)
        BIC_same_var = BIC_same_var_left - BIC_same_var_right
        BIC_diff_var = BIC_diff_var_left - BIC_diff_var_right

        # choose the BIC formulation that gives us the smallest values
        # (i.e. min of (BIC_diff, BIC_same) in the paper) and then
        # assign the left and right child BIC values by reference
        if BIC_diff_var < BIC_same_var:
            impurity_left[0] = -BIC_diff_var_left
            impurity_right[0] = -BIC_diff_var_right
        else:
            impurity_left[0] = -BIC_same_var_left
            impurity_right[0] = -BIC_same_var_right


    cdef void init_feature_vec(
        self,
    ) noexcept nogil:
        """Compute sufficient statistics from the feature vector at this node.

        This function iterates over the set of samples at this node and computes
        the cumulative sum and cumulative sum-of-suqares, which enables O(1) computation
        of the variance.

        When calling `update` to compute the sum_left and sum_right. It will be fairly straightforward.
        """
        # also compute the sum total
        self.sum_total = 0.0
        self.weighted_n_node_samples = 0.0
        cdef SIZE_t s_idx
        cdef SIZE_t p_idx

        cdef DOUBLE_t w = 1.0

        cdef SIZE_t prev_s_idx
        # = self.sample_indices[self.start] - 1
        # self.cumsum_of_squares_map[self.sample_indices[self.start] - 1] = 0.0
        # self.cumsum_map[self.sample_indices[self.start] - 1] = 0.0
        # self.cumsum_weights_map[self.sample_indices[self.start] - 1] = 0.0

        cdef unordered_map[SIZE_t, DTYPE_t] cumsum_map
        cdef unordered_map[SIZE_t, DTYPE_t] cumsum_of_squares_map
        cdef unordered_map[SIZE_t, DTYPE_t] cumsum_weights_map
        self.cumsum_map = cumsum_map
        self.cumsum_of_squares_map = cumsum_of_squares_map
        self.cumsum_weights_map = cumsum_weights_map

        for p_idx in range(self.start, self.end):
            s_idx = self.sample_indices[p_idx]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if self.sample_weight is not None:
                w = self.sample_weight[s_idx]

            self.sum_total += self.Xf[s_idx] * w
            self.weighted_n_node_samples += w

            if p_idx != self.start:
                self.cumsum_of_squares_map[s_idx] = self.cumsum_of_squares_map[prev_s_idx] + (self.Xf[s_idx] * self.Xf[s_idx] * w * w)
                self.cumsum_map[s_idx] = self.cumsum_map[prev_s_idx] + (self.Xf[s_idx] * w)
                self.cumsum_weights_map[s_idx] = self.cumsum_weights_map[prev_s_idx] + w
            else:
                self.cumsum_of_squares_map[s_idx] = 0.0
                self.cumsum_map[s_idx] = 0.0
                self.cumsum_weights_map[s_idx] = 0.0
            prev_s_idx = s_idx

        # Reset to pos=start
        self.reset()

    cdef int update(
        self,
        SIZE_t new_pos
    ) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Update the split point.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef SIZE_t pos = self.pos

        # infer left-child statistics
        self.weighted_n_left = self.cumsum_weights_map[self.sample_indices[new_pos]]
        self.sum_left = self.cumsum_map[self.sample_indices[new_pos]]

        # Update right part statistics as a result
        self.weighted_n_right = (self.weighted_n_node_samples -
                                self.weighted_n_left)
        self.sum_right = self.sum_total - self.sum_left

        self.pos = new_pos
        return 0


# cdef FasterBICv2(UnsupervisedCriterion):
#     cdef double sumsq_total     # The sum of the weighted count of each feature.
#     cdef double sumsq_left      # Same as above, but for the left side of the split
#     cdef double sumsq_right     # Same as above, but for the right side of the split


#     cdef void init_feature_vec(
#         self,
#     ) noexcept nogil:
#         """Initialize the 1D feature vector, which is used for computing criteria.

#         This function is used to set a read-only memoryview of a feature
#         vector. The feature vector must be set in order for criteria to be
#         computed. It then keeps a running total of the feature vector from
#         samples[start:end] so that way it is efficient to compute the right and
#         left sums and corresponding metrics.

#         Parameters
#         ----------
#         Xf : array-like, dtype=DTYPE_t
#             The read-only memoryview 1D feature vector with (n_samples,) shape.
#         """
#         # also compute the sum total
#         self.sum_total = 0.0
#         self.weighted_n_node_samples = 0.0
#         cdef SIZE_t s_idx
#         cdef SIZE_t p_idx

#         cdef DOUBLE_t w = 1.0
#         for p_idx in range(self.start, self.end):
#             s_idx = self.sample_indices[p_idx]

#             # w is originally set to be 1.0, meaning that if no sample weights
#             # are given, the default weight of each sample is 1.0.
#             if self.sample_weight is not None:
#                 w = self.sample_weight[s_idx]

#             self.sum_total += self.Xf[s_idx] * w
#             self.weighted_n_node_samples += w

#         # Reset to pos=start
#         self.reset()
        
#     cdef int update(
#         self,
#         SIZE_t new_pos
#     ) except -1 nogil:
#         """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

#         Returns -1 in case of failure to allocate memory (and raise MemoryError)
#         or 0 otherwise.

#         Parameters
#         ----------
#         new_pos : SIZE_t
#             The new ending position for which to move sample_indices from the right
#             child to the left child.
#         """
#         cdef SIZE_t pos = self.pos
#         cdef SIZE_t end = self.end

#         cdef const SIZE_t[:] sample_indices = self.sample_indices
#         cdef const DOUBLE_t[:] sample_weight = self.sample_weight

#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef DOUBLE_t w = 1.0

#         # Update statistics up to new_pos
#         #
#         # Given that
#         #   sum_left[x] +  sum_right[x] = sum_total[x]
#         # and that sum_total is known, we are going to update
#         # sum_left from the direction that require the least amount
#         # of computations, i.e. from pos to new_pos or from end to new_pos.
#         if (new_pos - pos) <= (end - new_pos):
#             for p in range(pos, new_pos):
#                 i = sample_indices[p]

#                 if sample_weight is not None:
#                     w = sample_weight[i]

#                 # accumulate the values of the feature vectors weighted
#                 # by the sample weight
#                 self.sum_left += self.Xf[i] * w

#                 # keep track of the weighted count of each sample
#                 self.weighted_n_left += w
#         else:
#             self.reverse_reset()

#             for p in range(end - 1, new_pos - 1, -1):
#                 i = sample_indices[p]

#                 if sample_weight is not None:
#                     w = sample_weight[i]

#                 self.sum_left -= self.Xf[i] * w

#                 self.weighted_n_left -= w

#         # Update right part statistics
#         self.weighted_n_right = (self.weighted_n_node_samples -
#                                  self.weighted_n_left)
#         self.sum_right = self.sum_total - self.sum_left

#         self.pos = new_pos
#         return 0

#     cdef double bic_cluster(self, SIZE_t n_samples, double variance) noexcept nogil:
#         """Help compute the BIC from assigning to a specific cluster.

#         Parameters
#         ----------
#         n_samples : SIZE_t
#             The number of samples assigned cluster.
#         variance : double
#             The plug-in variance for assigning to specific cluster.

#         Notes
#         -----
#         Computes the following:

#         :math:`-2 * (n_i log(w_i) - n_i/2 log(2 \\pi \\sigma_i^2))

#         where :math:`n_i` is the number of samples assigned to cluster i,
#         :math:`w_i` is the probability of choosing cluster i at random (or also known
#         as the prior) and :math:`\\sigma_i^2` is the variance estimate for cluster i.

#         Note that :math:`\\sigma_i^2` in the Fast-BIC derivation may be the
#         variance of the cluster itself, or the estimated combined variance
#         from both clusters.
#         """
#         cdef SIZE_t n_node_samples = self.n_node_samples

#         # chances of choosing the cluster based on how many samples are hard-assigned to cluster
#         # i.e. the prior
#         # cast to double, so we do not round to integers
#         cdef double w_cluster = (n_samples + 0.0) / n_node_samples

#         # add to prevent taking log of 0 when there is a degenerate cluster (i.e. single sample, or no variance)
#         return -2. * (n_samples * log(w_cluster) + 0.5 * n_samples * log(2. * PI * variance + 1.e-7))

#     cdef double node_impurity(
#         self
#     ) noexcept nogil:
#         """Evaluate the impurity of the current node.

#         Evaluate the FastBIC criterion impurity as estimated maximum log likelihood.
#         This is the maximum likelihood given prior, mean, and variance at s number of samples
#         Namely, this is the maximum likelihood of Xf[sample_indices[start:end]].
#         The smaller the impurity the better.
#         """
#         cdef double mean
#         cdef double variance
#         cdef double impurity
#         cdef SIZE_t n_node_samples = self.n_node_samples

#         # then compute the variance of the cluster
#         # see Section 5.2 in reference.
#         variance = self.fast_total_variance(self.weighted_n_node_samples, self.end)

#         # Compute the BIC of the current set of samples
#         # Note: we do not compute the BIC_diff_var and BIC_same_var because
#         # they are equivalent in the single cluster setting
#         impurity = self.bic_cluster(n_node_samples, variance)
#         return impurity


#     cdef void children_impurity(
#         self,
#         double* impurity_left,
#         double* impurity_right
#     ) noexcept nogil:
#         cdef SIZE_t pos = self.pos
#         cdef SIZE_t start = self.start
#         cdef SIZE_t end = self.end
#         cdef SIZE_t n_samples_left, n_samples_right

#         cdef double variance_left, variance_right, variance_comb
#         cdef double BIC_diff_var_left, BIC_diff_var_right
#         cdef double BIC_same_var_left, BIC_same_var_right
#         cdef double BIC_same_var, BIC_diff_var

#         # number of samples of left and right
#         n_samples_left = pos - start
#         n_samples_right = end - pos

#         # compute the variance of the node, left, and right child
#         variance_left = self.fast_total_variance(self.weighted_n_left, pos)
#         variance_right = self.fast_variance(self.weighted_n_right, end, pos)
#         variance_comb = self.fast_total_variance(self.weighted_n_node_samples, end)

#         # Compute the BIC using different variances for left and right
#         BIC_diff_var_left = self.bic_cluster(n_samples_left, variance_left)
#         BIC_diff_var_right = self.bic_cluster(n_samples_right, variance_right)

#         # Compute the BIC using different variances for left and right
#         BIC_same_var_left = self.bic_cluster(n_samples_left, variance_comb)
#         BIC_same_var_right = self.bic_cluster(n_samples_right, variance_comb)
#         BIC_same_var = BIC_same_var_left - BIC_same_var_right
#         BIC_diff_var = BIC_diff_var_left - BIC_diff_var_right

#         # choose the BIC formulation that gives us the smallest values
#         # (i.e. min of (BIC_diff, BIC_same) in the paper) and then
#         # assign the left and right child BIC values by reference
#         if BIC_diff_var < BIC_same_var:
#             impurity_left[0] = -BIC_diff_var_left
#             impurity_right[0] = -BIC_diff_var_right
#         else:
#             impurity_left[0] = -BIC_same_var_left
#             impurity_right[0] = -BIC_same_var_right

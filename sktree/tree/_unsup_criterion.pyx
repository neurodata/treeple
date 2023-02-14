#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport numpy as cnp
from libc.math cimport log

cnp.import_array()


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

    cdef void init_feature_vec(
        self,
        const DTYPE_t[:] Xf,
    ) nogil:
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
        self.Xf = Xf

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

    cdef int init(
        self,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices,
    ) nogil except -1:
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
        """
        self.sample_weight = sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.sample_indices = sample_indices

        return 0

    cdef int reset(self) nogil except -1:
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

    cdef int reverse_reset(self) nogil except -1:
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

    cdef int update(
        self,
        SIZE_t new_pos
    ) nogil except -1:
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
        # of computations, i.e. from pos to new_pos or from end to new_po.
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
    ) nogil:
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
    ) nogil:
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
    ) nogil:
        """Evaluate the impurity of the current node.

        Evaluate the TwoMeans criterion impurity as variance of the current node,
        i.e. the variance of Xf[sample_indices[start:end]]. The smaller the impurity the
        better.
        """
        cdef double mean
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
    ) nogil:
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
        double mean
    ) nogil:
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

    Fast-BIC is a method that combines the speed of the two-means clustering 
    method with the model flexibility of Mclust-BIC. It sorts data for each 
    feature and tries all possible splits to assign data points to one of 
    two Gaussian distributions based on their position relative to the split.
    The parameters for each cluster are estimated using maximum likelihood 
    estimation (MLE).The method performs hard clustering rather than soft 
    clustering like in GMM, resulting in a simpler calculation of the likelihood.
    
    \hat{L} = \sum_{n=1}^s[\log\hat{\pi}_1+\log{\mathcal{N}(x_n;\hat{\mu}_1,\hat{\sigma}_1^2)}]
    + \sum_{n=s+1}^N[\log\hat{\pi}_2+\log{\mathcal{N}(x_n;\hat{\mu}_2,\hat{\sigma}_2^2)}]
    
    where the prior, mean, and variance are defined as follows, respectively:
    \hat{\pi} = \frac{s}{N}
    \hat{\mu} = \frac{1}{s}\sum_{n\le s}{x_n},
    \hat{\sigma}^2 = \frac{1}{s}\sum_{n\le s}{||x_n-\hat{\mu_j}||^2}

    Fast-BIC is gauranteed to obtain the global maximum likelihood estimator,
    where as the Mclust-BIC is liable to find only a local maximum. Additionally,
    Fast-BIC is substantially faster than the traditional BIC method.

    Reference: https://arxiv.org/abs/1907.02844

    """
    cdef double node_impurity(
        self
    ) nogil:
        """Evaluate the impurity of the current node.

        Evaluate the FastBIC criterion impurity as estimated maximum log likelihood.
        This is the maximum likelihood given prior, mean, and variance at s number of samples
        Namely, this is the maximum likelihood of Xf[sample_indices[start:end]].
        The smaller the impurity the better.

        """
        cdef double mean
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
        sig = self.sum_of_squares(
            self.start,
            self.end,
            mean
        ) / self.weighted_n_node_samples

        # simplified equation of maximum log likelihood function at s=0
        impurity = n_node_samples*log(2*sig/mean)

        return impurity

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) nogil:
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
        cdef SIZE_t s_l
        cdef SIZE_t s_r
        cdef double p_l
        cdef double p_r
        cdef double mean_left
        cdef double mean_right
        cdef double sig_left
        cdef double sig_right
        cdef double left_term
        cdef double right_term

        # number of samples of left and right
        s_l = pos - start
        s_r = end - pos

        # compute prior (i.e. \hat{\pi_1} and \hat{\pi_2} in the paper)
        p_l = s_l / self.n_node_samples
        p_r = s_r / self.n_node_samples

        # first compute mean of left and right
        mean_left = self.sum_left / self.weighted_n_left
        mean_right = self.sum_right / self.weighted_n_right

        sig_left = self.sum_of_squares(
            start,
            pos,
            mean_left
        ) / self.weighted_n_left

        sig_right = self.sum_of_squares(
            pos,
            end,
            mean_right
        ) / self.weighted_n_right

        left_term = log(2*p_l*sig_left/mean_left)
        right_term = log(2*p_r*sig_right/mean_right)

        # simplified equation of maximum log likelihood function 
        # at corresponding sample size for left and right child
        impurity_left[0] = s_l*left_term + s_r*right_term
        impurity_right[0] = s_r*left_term + s_l*right_term

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef double _compute_variance(const DTYPE_t[:] Xf,
            double mean,
            SIZE_t* samples,
            DOUBLE_t* sample_weight,
            SIZE_t start, SIZE_t end) nogil:
    """Compute sample variance from vector of feature values.

    Parameters
    ----------
    """
    cdef SIZE_t p
    cdef SIZE_t i
    cdef DOUBLE_t w = 1.0
    cdef double variance = 0.0
    cdef SIZE_t n_samples = end - start

    # compute sample variance
    for p in range(start, end):
        i = samples[p]

        # w is originally set to be 1.0, meaning that if no sample weights
        # are given, the default weight of each sample is 1.0
        if sample_weight != NULL:
            w = sample_weight[i]
        
        # ith sample, and pth feature
        variance += w * (Xf[i] - mean) * (Xf[i] - mean)
    variance /= n_samples
    return variance


cdef class UnsupervisedCriterion(Criterion):
    """Abstract criterion for unsupervised learning.
    
    This object is a copy of the Criterion class of scikit-learn, but is used
    for unsupervised learning. However, ``Criterion`` in scikit-learn was
    designed for supervised learning, where the necessary
    ingredients to compute a split point is solely with y-labels. In
    this object, we subclass and instead rely on the X-data.    

    This object stores methods on how to calculate how good a split is using
    different metrics for unsupervised splitting.
    """
    def __cinit__(self, const DTYPE_t[:] X):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        X : array-like, dtype=DTYPE_t
            The dataset stored as a buffer for memory efficiency of shape
            (n_samples,).
        """
        self.X = X
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Count metric for current, going left and going right
        self.sum_total = 0.0
        self.sum_left = 0.0
        self.sum_right = 0.0

    def __reduce__(self):
        return (type(self),
                (self.n_outputs, np.asarray(self.n_classes)), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  double weighted_n_samples,
                  SIZE_t* samples,
                  SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion.

        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency. Note that
            this is not used, but simply passed as a convenience function.
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample (i.e. row of X).
        weighted_n_samples : double
            The total weight of all samples.
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0
        self.sum_total = 0.0

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]
            
            # ith sample; overall compute the weighted average
            self.sum_total += w * X[i]
            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
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
        self.sum_left = self.sum_total
        self.sum_right = 0.0
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef const [:, :] X = self.X
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left +  sum_right = sum_total
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                self.sum_left += w
                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                self.sum_left -= w
                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.sum_right = self.sum_total - self.sum_left

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        *dest = self.sum_total


cdef class TwoMeans(UnsupervisedCriterion):
    r"""Two means split impurity.

    The two means split finds the cutpoint that minimizes the one-dimensional
    2-means objective, which is finding the cutoff point where the total variance
    from cluster 1 and cluster 2 are minimal. 

    The mathematical optimization problem is to find the cutoff index ``s``,
    which is called 'pos' in scikit-learn.

        \min_s \sum_{i=1}^s (x_i - \hat{\mu}_1)^2 + \sum_{i=s+1}^N (x_i - \hat{\mu}_2)^2

    where x is a N-dimensional feature vector, N is the number of samples and the \mu
    terms are the estimated means of each cluster 1 and 2.
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.

        Evaluate the 2-means criterion as impurity of the current node, by computing
        the sample variance. i.e. the sample variance/impurity of samples[start:end].
        The smaller the impurity the better.
        """
        cdef double variance
        cdef double mean = self.sum_total / (self.end - self.start)

        # compute the variance
        variance = _compute_variance(const DTYPE_t[:] self.X,
            mean,
            self.samples,
            self.sample_weight,
            self.start, self.end)
        return variance

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef double variance_left
        cdef double variance_right

        # compute the means within children
        cdef double mean_left = self.sum_left / (self.pos - self.start)
        cdef double mean_right = self.sum_right / (self.end - self.pos)

        # compute the variance
        variance_left = _compute_variance(const DTYPE_t[:] self.X,
            self.samples,
            self.sample_weight,
            self.start, self.pos)
        variance_right = _compute_variance(const DTYPE_t[:] X,
            self.samples,
            self.sample_weight,
            self.pos, self.end)

        impurity_left[0] = variance_left
        impurity_right[0] = variance_right


cdef class BIC(UnsupervisedCriterion):
    """Fast Bayesian Information Criterion (BIC) impurity.

    The fast BIC criterion analyzes the BIC for splitting samples into two
    clusters.

    The fast-BIC criterion works by defining a split point (i.e. 'pos')
    and then computing the sample mean, sample variance and sample probability
    of cluster 1. This is a total of five parameters, as the sample probability of
    cluster 2 does not need to be computed, once the assignment probability of cluster 1
    is computed.
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.

        Evaluate the BIC criterion as impurity of the current node, by computing
        the sample variance. i.e. the sample variance/impurity of samples[start:end].
        The smaller the impurity the better.
        """
        cdef double variance = 0.0

        # compute sample variance

        return variance

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef double variance_left = 0.0
        cdef double variance_right = 0.0
        cdef double mean_left = 0.0
        cdef double mean_right = 0.0

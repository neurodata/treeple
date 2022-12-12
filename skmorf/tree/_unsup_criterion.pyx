import numpy as np
cimport numpy as cnp
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
    cdef int init(
        self,
        const DOUBLE_t[:, ::1] X,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices
    ) nogil except -1:
        """Initialize the criterion.

        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : array-like, dtype=DOUBLE_t
            The data-feature matrix stored as a buffer for memory efficiency. Note that
            this is not used, but simply passed as a convenience function.
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample (i.e. row of X).
        weighted_n_samples : double
            The total weight of all samples.
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        """
        pass


cdef class TwoMeans(Criterion):
    """Two-Means Criterion.
    """

    def __cinit__(self, const DTYPE_t[:] X):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        X : array-like, dtype=DTYPE_t
            The dataset stored as a buffer for memory efficiency of shape
            (n_samples,).
        """
        # need to implement
        pass
    
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
        # need to implement
        pass

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # need to implement
        pass

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
        # need to implement
        pass

    cdef double node_impurity(self) nogil:
        # need to implement
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        # need to implement
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        # need to implement
        pass

    cdef void set_sample_pointers(
        self,
        SIZE_t start,
        SIZE_t end
    ) nogil:
        # need to implement
        pass
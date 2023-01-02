import numpy as np
cimport numpy as cnp
cnp.import_array()

from .criterion import BaseCriterion

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
            The target stored as a buffer for memory efficiency. Note that
            this is not used, but simply passed as a convenience function.
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample (i.e. row of X).
        weighted_n_samples : double
            The total weight of all samples.
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        """
        # need to implement
        pass
        
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



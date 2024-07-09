# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

from ..._lib.sklearn.tree._criterion cimport BaseCriterion
from ..._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int32_t, intp_t

# Note: This class is an exact copy of scikit-learn's Criterion
# class, with the exception of the type of the internal structure.
# In scikit-learn, they store a buffer for the y-labels, whereas here
# we store a buffer for the X dataset.
#
# In our criterions, we do not store the 'y-labels' because there are none
# in unsupervised learning. We instead store a memview of the dataset 'X'.


cdef class UnsupervisedCriterion(BaseCriterion):
    """Abstract unsupervised criterion.

    Notable Changes
    ---------------
    1. weighted_n_* : This parameter keeps track of the total "weight" of the samples
        in the node, left and right
    """

    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics.

    # Internal structures
    cdef const float32_t[:] feature_values  # 1D memview for the feature vector to compute criterion on

    # Keep running total of Xf[samples[start:end]] and the corresponding sum in
    # the left and right node. For example, this can then efficiently compute the
    # mean of the node, and left/right child by subtracting relevant Xf elements
    # and then dividing by the total number of samples in the node and left/right child.
    cdef float64_t sum_total     # The sum of the weighted count of each feature.
    cdef float64_t sum_left      # Same as above, but for the left side of the split
    cdef float64_t sum_right     # Same as above, but for the right side of the split

    cdef float64_t sumsq_total     # The sum of the weighted count of each feature.
    cdef float64_t sumsq_left      # Same as above, but for the left side of the split
    cdef float64_t sumsq_right     # Same as above, but for the right side of the split

    # Methods
    # -------
    # The 'init' method is copied here with the almost the exact same signature
    # as that of supervised learning criterion in scikit-learn to ensure that
    # Unsupervised criterion can be used with splitter and tree methods.
    cdef intp_t init(
        self,
        const float32_t[:] feature_values,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] samples,
    ) except -1 nogil

    cdef void init_feature_vec(
        self
    ) noexcept nogil

    cdef void set_sample_pointers(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil

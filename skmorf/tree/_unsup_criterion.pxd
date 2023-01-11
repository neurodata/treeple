# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True

from sklearn.tree._tree cimport SIZE_t, DTYPE_t, DOUBLE_t
from sklearn.tree._criterion cimport BaseCriterion

# Note: This class is an exact copy of scikit-learn's Criterion
# class, with the exception of the type of the internal structure.
# In scikit-learn, they store a buffer for the y-labels, whereas here
# we store a buffer for the X dataset. 
#
# In our criterions, we do not store the 'y-labels' because there are none
# in unsupervised learning. We instead store a memview of the dataset 'X'.
#
# Other changes include the removal of "weighted" samples, which is
# not needed since criterion are compuated on the data itself.

cdef class UnsupervisedCriterion(BaseCriterion):
    """Abstract unsupervised criterion."""

    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics.

    # Internal structures
    cdef const DTYPE_T[:, ::1] X # 2D memview for values of X (i.e. feature values)

    # TODO: WIP. Assumed the sum "metric" of node, left and right
    # XXX: this can possibly be defined in downstream classes instead as memoryviews.
    # The sum of the metric stored either at the split, going left, or right
    # of the split. 
    cdef double sum_total   # The sum of the weighted count of each feature.
    cdef double sum_left    # Same as above, but for the left side of the split
    cdef double sum_right   # Same as above, but for the right side of the split

    # Methods
    # -------
    # The 'init' method is copied here with the almost the exact same signature
    # as that of supervised learning criterion in scikit-learn to ensure that
    # Unsupervised criterion can be used with splitter and tree methods.
    cdef int init(
        self,
        const DOUBLE_t[:, ::1] X,
        DOUBLE_t* sample_weight,
        double weighted_n_samples, 
        SIZE_t* samples, 
        SIZE_t start,
        SIZE_t end
    ) nogil except -1

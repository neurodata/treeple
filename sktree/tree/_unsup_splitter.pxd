from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer

from sklearn.tree._splitter cimport BaseSplitter, SplitRecord

from ._unsup_criterion cimport UnsupervisedCriterion


# TODO:
# - check if we need "weighted_n_node_samples"; since this isn't a stopping criterion
# it should probably be an analagous "variance_epsilon" for stopping once the variance
# is less than a certain amount
#           -> Yes, since we won't use something like "min_weight_leaf", because it
#           doesn't make sense, this will be unused.
# - min_weight_leaf should be documented as something else

cdef class UnsupervisedSplitter(BaseSplitter):
    """
    Notable changes wrt scikit-learn:
    1. `weighted_n_node_samples` is used as a stopping criterion and just used to
    keep count of the "number of samples (weighted)". All samples have a default weight
    of '1'.
    """

    # XXX: not sure if this will work since it subclasses
    # an existing extension type attribute
    cdef public UnsupervisedCriterion criterion

    # feature matrix
    cdef const DTYPE_t[:, :] X
    
    cdef SIZE_t n_total_samples

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:] sample_weight
    ) except -1


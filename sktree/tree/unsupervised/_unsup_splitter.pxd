from ..._lib.sklearn.tree._splitter cimport BaseSplitter, SplitRecord
from ..._lib.sklearn.tree._utils cimport UINT32_t
from ..._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._unsup_criterion cimport UnsupervisedCriterion


cdef class UnsupervisedSplitter(BaseSplitter):
    """
    Notable changes wrt scikit-learn:
    1. `weighted_n_node_samples` is used as a stopping criterion and just used to
    keep count of the "number of samples (weighted)". All samples have a default weight
    of '1'.
    2. `X` array instead of `y` array is stored as the criterions are computed over the X
    array.
    3. The feature_values memoryview is a feature vector with shared memory among the splitter
    and the criterion object. This enables the splitter to assign values to it within the
    `node_split` function and then `criterion` automatically can compute relevant statistics
    on the shared memoryview into the array.
    """

    # XXX: requires BaseSplitter to not define "criterion"
    cdef public UnsupervisedCriterion criterion         # criterion computer
    cdef const float32_t[:, :] X                          # feature matrix
    cdef intp_t n_total_samples                         # store the total number of samples

    # Initialization method for unsupervised splitters
    cdef int init(
        self,
        const float32_t[:, :] X,
        const float64_t[:] sample_weight
    ) except -1

    # Overridden Methods from base class
    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil
    cdef int node_split(
        self,
        float64_t impurity,   # Impurity of the node
        SplitRecord* split,
        intp_t* n_constant_features,
        float64_t lower_bound,
        float64_t upper_bound
    ) except -1 nogil
    cdef void node_value(
        self,
        float64_t* dest
    ) noexcept nogil
    cdef float64_t node_impurity(
        self
    ) noexcept nogil

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._tree cimport Node, ParentInfo, Tree
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint32_t


cdef class HonestPruner(Splitter):
    cdef Tree tree          # The tree to be pruned
    cdef intp_t capacity    # The maximum number of nodes in the pruned tree

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1

    # cdef float32_t _compute_feature(
    #     self,
    #     const float32_t[:, :] X_ndarray,
    #     intp_t sample_index,
    #     Node *node
    # ) noexcept nogil

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil


def _build_pruned_tree_honesty(
    Tree tree,
    Tree orig_tree,
    HonestPruner pruner,
    object X,
    const float64_t[:, ::1] y,
    const float64_t[:] sample_weight,
)

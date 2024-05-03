import numpy as np

cimport numpy as cnp

from .._lib.sklearn.tree._tree cimport BaseTree, Node
from .._lib.sklearn.tree._utils cimport UINT32_t
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t


cpdef apply_marginal_tree(
    BaseTree tree,
    object X,
    const intp_t[:] marginal_indices,
    intp_t traversal_method,
    unsigned char use_sample_weight,
    object random_state
)

import numpy as np

cimport numpy as cnp

from .._lib.sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from .._lib.sklearn.tree._tree cimport DTYPE_t  # Type of X
from .._lib.sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from .._lib.sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from .._lib.sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer
from .._lib.sklearn.tree._tree cimport BaseTree, Node


cpdef apply_marginal_tree(
    BaseTree tree,
    object X,
    const SIZE_t[:] marginal_indices,
    int traversal_method,
    unsigned char use_sample_weight,
    object random_state
)

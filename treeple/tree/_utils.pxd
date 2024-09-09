from libcpp.vector cimport vector

import numpy as np

cimport numpy as cnp

cnp.import_array()

from .._lib.sklearn.tree._splitter cimport SplitRecord
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int32_t, intp_t, uint32_t

ctypedef fused vector_or_memview:
    vector[intp_t]
    intp_t[::1]
    intp_t[:]


cdef void fisher_yates_shuffle(
    vector_or_memview indices_to_sample,
    intp_t grid_size,
    uint32_t* random_state,
) noexcept nogil


cdef int rand_weighted_binary(
    float64_t p0,
    uint32_t* random_state
) noexcept nogil

cpdef unravel_index(
    intp_t index,
    cnp.ndarray[intp_t, ndim=1] shape
)

cpdef ravel_multi_index(
    intp_t[:] coords,
    const intp_t[:] shape
)

cdef void unravel_index_cython(
    intp_t index,
    const intp_t[:] shape,
    vector_or_memview coords
) noexcept nogil

cdef intp_t ravel_multi_index_cython(
    vector_or_memview coords,
    const intp_t[:] shape
) noexcept nogil

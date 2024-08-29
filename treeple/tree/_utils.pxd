import numpy as np

cimport numpy as cnp

cnp.import_array()

from libcpp.vector cimport vector

from .._lib.sklearn.tree._splitter cimport SplitRecord
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int32_t, intp_t, uint32_t

ctypedef fused vector_or_memview:
    vector[intp_t]
    intp_t[::1]
    intp_t[:]


cdef inline void fisher_yates_shuffle(
    vector_or_memview indices_to_sample,
    intp_t grid_size,
    uint32_t* random_state,
) noexcept nogil


cdef intp_t rand_weighted_binary(
    float64_t p0,
    uint32_t* random_state
) noexcept nogil

cpdef unravel_index(
    intp_t index, cnp.ndarray[intp_t, ndim=1] shape
)

cpdef ravel_multi_index(intp_t[:] coords, const intp_t[:] shape)

cdef void unravel_index_cython(
    intp_t index,
    const intp_t[:] shape,
    vector_or_memview coords
) noexcept nogil

cdef intp_t ravel_multi_index_cython(
    vector_or_memview coords,
    const intp_t[:] shape
) noexcept nogil

cdef void compute_vectorized_indices_from_cartesian(
    intp_t base_index,
    vector[vector[intp_t]]& index_arrays,
    const intp_t[:] data_dims,
    vector[intp_t]& result
) noexcept nogil

cdef memoryview[float32_t, ndim=3] init_2dmemoryview(
    cnp.ndarray array,
    const intp_t[:] data_dims
)

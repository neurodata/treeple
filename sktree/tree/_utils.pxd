import numpy as np

cimport numpy as cnp

cnp.import_array()

from sktree._lib.sklearn.tree._splitter cimport SplitRecord

from sktree._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t


cdef int rand_weighted_binary(float64_t p0, uint32_t* random_state) noexcept nogil

cpdef unravel_index(
    intp_t index, cnp.ndarray[intp_t, ndim=1] shape
)

cpdef ravel_multi_index(intp_t[:] coords, const intp_t[:] shape)

cdef void unravel_index_cython(intp_t index, const intp_t[:] shape, intp_t[:] coords) noexcept nogil

cdef intp_t ravel_multi_index_cython(intp_t[:] coords, const intp_t[:] shape) noexcept nogil

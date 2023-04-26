import numpy as np

cimport numpy as cnp

cnp.import_array()

from sklearn_fork.tree._splitter cimport SplitRecord
from sklearn_fork.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from sklearn_fork.tree._tree cimport DTYPE_t  # Type of X
from sklearn_fork.tree._tree cimport INT32_t  # Signed 32 bit integer
from sklearn_fork.tree._tree cimport SIZE_t  # Type for indices and counters
from sklearn_fork.tree._tree cimport UINT32_t  # Unsigned 32 bit integer


cpdef unravel_index(
    SIZE_t index, cnp.ndarray[SIZE_t, ndim=1] shape
)

cpdef ravel_multi_index(SIZE_t[:] coords, SIZE_t[:] shape)

cdef void unravel_index_cython(SIZE_t index, const SIZE_t[:] shape, SIZE_t[:] coords) noexcept nogil

cdef SIZE_t ravel_multi_index_cython(SIZE_t[:] coords, SIZE_t[:] shape) nogil

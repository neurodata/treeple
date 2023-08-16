from .._lib.sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from .._lib.sklearn.tree._tree cimport DTYPE_t  # Type of X
from .._lib.sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from .._lib.sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from .._lib.sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer

# This defines c-importable functions for other cython files

# TODO: remove these files when sklearn merges refactor defining these in pxd files
# https://github.com/scikit-learn/scikit-learn/pull/25606
cdef void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) noexcept nogil

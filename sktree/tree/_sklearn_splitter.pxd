from .._lib.sklearn.utils._typedefs cimport float32_t, int32_t, intp_t

# This defines c-importable functions for other cython files

# TODO: remove these files when sklearn merges refactor defining these in pxd files
# https://github.com/scikit-learn/scikit-learn/pull/25606
cdef void sort(float32_t* Xf, intp_t* samples, intp_t n) noexcept nogil

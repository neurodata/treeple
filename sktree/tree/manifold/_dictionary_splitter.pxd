import numpy as np

from libcpp.vector cimport vector

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.tree._utils cimport UINT32_t
from ..._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint8_t
from ._morf_splitter cimport PatchSplitter


cdef class DictionarySplitter(PatchSplitter):
    """A class to hold user-specified kernels in the form of a dictionary.

    The dictionary here refers to the context of dictionary learning.
    """
    # dictionary is stored as a sparse array (n_kernels, n_max_filter_length)
    # However, if the kernel is a small filter, then many of the weights will be 0
    cdef const float32_t[::1] kernel_data
    cdef const int32_t[::1] kernel_indices
    cdef const int32_t[::1] kernel_indptr

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil

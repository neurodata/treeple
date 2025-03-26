import numpy as np

from libcpp.vector cimport vector

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.utils._typedefs cimport (
    float32_t,
    float64_t,
    int32_t,
    intp_t,
    uint8_t,
    uint32_t,
)
from .._oblique_splitter cimport BestObliqueSplitter, ObliqueSplitRecord
from ._morf_splitter cimport PatchSplitter


cdef class UserKernelSplitter(PatchSplitter):
    """A class to hold user-specified kernels."""
    # cdef vector[float32_t[:, ::1]] kernel_dictionary  # A list of C-contiguous 2D kernels
    cdef vector[float32_t*] kernel_dictionary  # A list of C-contiguous 2D kernels
    cdef vector[intp_t*] kernel_dims         # A list of arrays storing the dimensions of each kernel in `kernel_dictionary`


cdef class GaussianKernelSplitter(PatchSplitter):
    """A class to hold Gaussian kernels.

    Overrides the weights that are generated to be sampled from a Gaussian distribution.
    See: https://www.tutorialspoint.com/gaussian-filter-generation-in-cplusplus
    See: https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567
    """

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil

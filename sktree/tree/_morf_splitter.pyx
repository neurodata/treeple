cimport cython

import numpy as np

cimport numpy as cnp

cnp.import_array()

from cython.operator cimport dereference as deref
from libc.stdlib cimport free, malloc, qsort
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._utils cimport RAND_R_MAX, log, rand_int, rand_uniform


cdef class PatchSplitter(ObliqueSplitter):
    # The splitter searches in the input space for a combination of features and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(
        self, 
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) nogil:
        pass

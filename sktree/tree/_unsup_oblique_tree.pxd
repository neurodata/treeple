# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

# See _unsup_oblique_tree.pyx for details.

import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector
from sklearn.tree._oblique_splitter cimport ObliqueSplitRecord
from sklearn.tree._splitter cimport SplitRecord
from sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from sklearn.tree._tree cimport DTYPE_t  # Type of X
from sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer
from sklearn.tree._tree cimport Node

from ._unsup_tree cimport UnsupervisedTree


cdef class UnsupervisedObliqueTree(UnsupervisedTree):
    cdef vector[vector[DTYPE_t]] proj_vec_weights # (capacity, n_features) array of projection vectors
    cdef vector[vector[SIZE_t]] proj_vec_indices  # (capacity, n_features) array of projection vectors

    # overridden methods
    cdef int _resize_c(
        self,
        SIZE_t capacity=*
    ) nogil except -1
    cdef int _set_split_node(
        self,
        SplitRecord* split_node,
        Node *node
    )  nogil except -1
    cdef DTYPE_t _compute_feature(
        self,
        const DTYPE_t[:, :] X_ndarray,
        SIZE_t sample_index,
        Node *node
    ) nogil
    cdef void _compute_feature_importances(
        self,
        DOUBLE_t* importance_data,
        Node* node
    ) nogil

    cpdef cnp.ndarray get_projection_matrix(self)

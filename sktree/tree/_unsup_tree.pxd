# Authors: Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#

# License: BSD 3 clause

# See _unsup_tree.pyx for details.

import numpy as np
cimport numpy as cnp

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer

from sklearn.tree._tree cimport Node
from sklearn.tree._splitter cimport SplitRecord

from ._unsup_splitter cimport UnsupervisedSplitter


cdef class UnsupervisedTree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef public SIZE_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes

    # Methods
    cdef SIZE_t _add_node(
        self,
        SIZE_t parent,
        bint is_left,
        bint is_leaf,
        SIZE_t feature,
        double threshold,
        double impurity,
        SIZE_t n_node_samples,
        double weighted_n_node_samples
    ) nogil except -1
    cdef int _resize(
        self,
        SIZE_t capacity
    ) nogil except -1
    cdef int _resize_c(
        self,
        SIZE_t capacity=*
    ) nogil except -1

    cdef cnp.ndarray _get_value_ndarray(self)
    cdef cnp.ndarray _get_node_ndarray(self)

    cpdef cnp.ndarray apply(self, object X)
    cdef cnp.ndarray _apply_dense(self, object X)
    cdef cnp.ndarray _apply_sparse_csr(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cdef object _decision_path_sparse_csr(self, object X)

    cpdef compute_feature_importances(self, normalize=*)


# =============================================================================
# Tree builder
# =============================================================================

cdef class UnsupervisedTreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef UnsupervisedSplitter splitter  # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(
        self,
        UnsupervisedTree tree,
        object X,
        cnp.ndarray sample_weight=*
    )
    cdef _check_input(
        self,
        object X,
        cnp.ndarray sample_weight
    )

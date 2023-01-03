# Authors: Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#

# License: BSD 3 clause

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
from libcpp.vector cimport vector
from libcpp.algorithm cimport pop_heap
from libcpp.algorithm cimport push_heap
from libcpp cimport bool

import struct

import numpy as np
cimport numpy as cnp
cnp.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)

cdef extern from "<stack>" namespace "std" nogil:
    cdef cppclass stack[T]:
        ctypedef T value_type
        stack() except +
        bint empty()
        void pop()
        void push(T&) except +  # Raise c++ exception for bad_alloc -> MemoryError
        T& top()

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

# Build the corresponding numpy dtype for Node.
# This works by casting `dummy` to an array of Node of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Node dummy;
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

# =============================================================================
# Unsupervised TreeBuilder
# =============================================================================

cdef class UnsupervisedTreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, UnsupervisedTree tree, object X, cnp.ndarray sample_weight=None):
        """Build a decision tree from the training set X."""
        pass

    cdef inline _check_input(self, object X, cnp.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, sample_weight

# Best first builder ----------------------------------------------------------
cdef struct FrontierRecord:
    # Record of information of a Node, the frontier for a split. Those records are
    # maintained in a heap to access the Node with the best improvement in impurity,
    # allowing growing trees greedily on this improvement.
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t pos
    SIZE_t depth
    bint is_leaf
    double impurity
    double impurity_left
    double impurity_right
    double improvement

cdef inline bool _compare_records(
    const FrontierRecord& left,
    const FrontierRecord& right,
):
    return left.improvement < right.improvement

cdef inline void _add_to_frontier(
    FrontierRecord rec,
    vector[FrontierRecord]& frontier,
) nogil:
    """Adds record `rec` to the priority queue `frontier`."""
    frontier.push_back(rec)
    push_heap(frontier.begin(), frontier.end(), &_compare_records)


cdef class BestFirstTreeBuilder(UnsupervisedTreeBuilder):
    """Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    """
    cdef SIZE_t max_leaf_nodes

    def __cinit__(self, UnsupervisedSplitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf,  min_weight_leaf,
                  SIZE_t max_depth, SIZE_t max_leaf_nodes,
                  double min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, UnsupervisedTree tree, object X,
                cnp.ndarray sample_weight=None):
        """Build a decision tree from the training set X."""

        # check input
        X, sample_weight = self._check_input(X, sample_weight)

        # Parameters
        cdef UnsupervisedSplitter splitter = self.splitter
        cdef SIZE_t max_leaf_nodes = self.max_leaf_nodes

        # Recursive partition (without actual recursion)
        splitter.init(X, sample_weight)

        cdef vector[FrontierRecord] frontier
        cdef FrontierRecord record
        cdef FrontierRecord split_node_left
        cdef FrontierRecord split_node_right

        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        # Initial capacity
        cdef SIZE_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = self._add_split_node(splitter, tree, 0, n_node_samples,
                                      INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
                                      &split_node_left)
            if rc >= 0:
                _add_to_frontier(split_node_left, frontier)

            while not frontier.empty():
                pop_heap(frontier.begin(), frontier.end(), &_compare_records)
                record = frontier.back()
                frontier.pop_back()

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    rc = self._add_split_node(splitter, tree,
                                              record.start, record.pos,
                                              record.impurity_left,
                                              IS_NOT_FIRST, IS_LEFT, node,
                                              record.depth + 1,
                                              &split_node_left)
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    rc = self._add_split_node(splitter, tree, record.pos,
                                              record.end,
                                              record.impurity_right,
                                              IS_NOT_FIRST, IS_NOT_LEFT, node,
                                              record.depth + 1,
                                              &split_node_right)
                    if rc == -1:
                        break

                    # Add nodes to queue
                    _add_to_frontier(split_node_left, frontier)
                    _add_to_frontier(split_node_right, frontier)

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(self, UnsupervisedSplitter splitter, UnsupervisedTree tree,
                                    SIZE_t start, SIZE_t end, double impurity,
                                    bint is_first, bint is_left, Node* parent,
                                    SIZE_t depth,
                                    FrontierRecord* res) nogil except -1:
        """Adds node w/ partition ``[start, end)`` to the frontier. """
        cdef SplitRecord split
        cdef SIZE_t node_id
        cdef SIZE_t n_node_samples
        cdef SIZE_t n_constant_features = 0
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double weighted_n_node_samples
        cdef bint is_leaf

        splitter.node_reset(start, end, &weighted_n_node_samples)

        if is_first:
            impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (depth >= self.max_depth or
                   n_node_samples < self.min_samples_split or
                   n_node_samples < 2 * self.min_samples_leaf or
                   weighted_n_node_samples < 2 * self.min_weight_leaf or
                   impurity <= EPSILON  # impurity == 0 with tolerance
                   )

        if not is_leaf:
            splitter.node_split(impurity, &split, &n_constant_features)
            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                       split.improvement + EPSILON < min_impurity_decrease)

        node_id = tree._add_node(parent - tree.nodes
                                 if parent != NULL
                                 else _TREE_UNDEFINED,
                                 is_left, is_leaf,
                                 split.feature, split.threshold, impurity, n_node_samples,
                                 weighted_n_node_samples)
        if node_id == SIZE_MAX:
            return -1

        # compute values also for split nodes (might become leafs later).
        splitter.node_value(tree.value + node_id * tree.value_stride)

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = impurity

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = impurity
            res.impurity_right = impurity

        return 0


# =============================================================================
# Unsupervised Tree
# =============================================================================

cdef class UnsupervisedTree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of double, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, cnp.ndarray n_classes, int n_outputs):
        """Constructor."""
        cdef SIZE_t dummy = 0
        size_t_dtype = np.array(dummy).dtype

        n_classes = _check_n_classes(n_classes, size_t_dtype)

        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (UnsupervisedTree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)

        node_ndarray = _check_node_ndarray(node_ndarray, expected_dtype=NODE_DTYPE)
        value_ndarray = _check_value_ndarray(
            value_ndarray,
            expected_dtype=np.dtype(np.float64),
            expected_shape=value_shape
        )

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<cnp.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<cnp.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples) nogil except -1:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef cnp.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef cnp.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef inline cnp.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef cnp.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out

    cdef inline cnp.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef cnp.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef cnp.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef cnp.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef cnp.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                        dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef cnp.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef cnp.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                    (1 + self.max_depth),
                                                    dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef cnp.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                                dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_sparse_csr(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef cnp.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef cnp.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef cnp.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef cnp.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef cnp.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                    (1 + self.max_depth),
                                                    dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef cnp.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                                dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out


    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef cnp.ndarray[cnp.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef cnp.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef cnp.npy_intp shape[3]
        shape[0] = <cnp.npy_intp> self.node_count
        shape[1] = <cnp.npy_intp> self.n_outputs
        shape[2] = <cnp.npy_intp> self.max_n_classes
        cdef cnp.ndarray arr
        arr = cnp.PyArray_SimpleNewFromData(3, shape, cnp.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    cdef cnp.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.node_count
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef cnp.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> cnp.ndarray,
                                   <cnp.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   cnp.NPY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    def compute_partial_dependence(self, DTYPE_t[:, ::1] X,
                                   int[::1] target_features,
                                   double[::1] out):
        """Partial dependence of the response on the ``target_feature`` set.

        For each sample in ``X`` a tree traversal is performed.
        Each traversal starts from the root with weight 1.0.

        At each non-leaf node that splits on a target feature, either
        the left child or the right child is visited based on the feature
        value of the current sample, and the weight is not modified.
        At each non-leaf node that splits on a complementary feature,
        both children are visited and the weight is multiplied by the fraction
        of training samples which went to each child.

        At each leaf, the value of the node is multiplied by the current
        weight (weights sum to 1 for all visited terminal nodes).

        Parameters
        ----------
        X : view on 2d ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : view on 1d ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        out : view on 1d ndarray, shape (n_samples)
            The value of the partial dependence function on each grid
            point.
        """
        cdef:
            double[::1] weight_stack = np.zeros(self.node_count,
                                                dtype=np.float64)
            SIZE_t[::1] node_idx_stack = np.zeros(self.node_count,
                                                  dtype=np.intp)
            SIZE_t sample_idx
            SIZE_t feature_idx
            int stack_size
            double left_sample_frac
            double current_weight
            double total_weight  # used for sanity check only
            Node *current_node  # use a pointer to avoid copying attributes
            SIZE_t current_node_idx
            bint is_target_feature
            SIZE_t _TREE_LEAF = TREE_LEAF  # to avoid python interactions

        for sample_idx in range(X.shape[0]):
            # init stacks for current sample
            stack_size = 1
            node_idx_stack[0] = 0  # root node
            weight_stack[0] = 1  # all the samples are in the root node
            total_weight = 0

            while stack_size > 0:
                # pop the stack
                stack_size -= 1
                current_node_idx = node_idx_stack[stack_size]
                current_node = &self.nodes[current_node_idx]

                if current_node.left_child == _TREE_LEAF:
                    # leaf node
                    out[sample_idx] += (weight_stack[stack_size] *
                                        self.value[current_node_idx])
                    total_weight += weight_stack[stack_size]
                else:
                    # non-leaf node

                    # determine if the split feature is a target feature
                    is_target_feature = False
                    for feature_idx in range(target_features.shape[0]):
                        if target_features[feature_idx] == current_node.feature:
                            is_target_feature = True
                            break

                    if is_target_feature:
                        # In this case, we push left or right child on stack
                        if X[sample_idx, feature_idx] <= current_node.threshold:
                            node_idx_stack[stack_size] = current_node.left_child
                        else:
                            node_idx_stack[stack_size] = current_node.right_child
                        stack_size += 1
                    else:
                        # In this case, we push both children onto the stack,
                        # and give a weight proportional to the number of
                        # samples going through each branch.

                        # push left child
                        node_idx_stack[stack_size] = current_node.left_child
                        left_sample_frac = (
                            self.nodes[current_node.left_child].weighted_n_node_samples /
                            current_node.weighted_n_node_samples)
                        current_weight = weight_stack[stack_size]
                        weight_stack[stack_size] = current_weight * left_sample_frac
                        stack_size += 1

                        # push right child
                        node_idx_stack[stack_size] = current_node.right_child
                        weight_stack[stack_size] = (
                            current_weight * (1 - left_sample_frac))
                        stack_size += 1

            # Sanity check. Should never happen.
            if not (0.999 < total_weight < 1.001):
                raise ValueError("Total weight should be 1.0 but was %.9f" %
                                 total_weight)


def _check_n_classes(n_classes, expected_dtype):
    if n_classes.ndim != 1:
        raise ValueError(
            f"Wrong dimensions for n_classes from the pickle: "
            f"expected 1, got {n_classes.ndim}"
        )

    if n_classes.dtype == expected_dtype:
        return n_classes

    # Handles both different endianness and different bitness
    if n_classes.dtype.kind == "i" and n_classes.dtype.itemsize in [4, 8]:
        return n_classes.astype(expected_dtype, casting="same_kind")

    raise ValueError(
        "n_classes from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {n_classes.dtype}"
    )


def _check_value_ndarray(value_ndarray, expected_dtype, expected_shape):
    if value_ndarray.shape != expected_shape:
        raise ValueError(
            "Wrong shape for value array from the pickle: "
            f"expected {expected_shape}, got {value_ndarray.shape}"
        )

    if not value_ndarray.flags.c_contiguous:
        raise ValueError(
            "value array from the pickle should be a C-contiguous array"
        )

    if value_ndarray.dtype == expected_dtype:
        return value_ndarray

    # Handles different endianness
    if value_ndarray.dtype.str.endswith('f8'):
        return value_ndarray.astype(expected_dtype, casting='equiv')

    raise ValueError(
        "value array from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {value_ndarray.dtype}"
    )


def _dtype_to_dict(dtype):
    return {name: dt.str for name, (dt, *rest) in dtype.fields.items()}


def _dtype_dict_with_modified_bitness(dtype_dict):
    # field names in Node struct with SIZE_t types (see sklearn/tree/_tree.pxd)
    indexing_field_names = ["left_child", "right_child", "feature", "n_node_samples"]

    expected_dtype_size = str(struct.calcsize("P"))
    allowed_dtype_size = "8" if expected_dtype_size == "4" else "4"

    allowed_dtype_dict = dtype_dict.copy()
    for name in indexing_field_names:
        allowed_dtype_dict[name] = allowed_dtype_dict[name].replace(
            expected_dtype_size, allowed_dtype_size
        )

    return allowed_dtype_dict


def _all_compatible_dtype_dicts(dtype):
    # The Cython code for decision trees uses platform-specific SIZE_t
    # typed indexing fields that correspond to either i4 or i8 dtypes for
    # the matching fields in the numpy array depending on the bitness of
    # the platform (32 bit or 64 bit respectively).
    #
    # We need to cast the indexing fields of the NODE_DTYPE-dtyped array at
    # pickle load time to enable cross-bitness deployment scenarios. We
    # typically want to make it possible to run the expensive fit method of
    # a tree estimator on a 64 bit server platform, pickle the estimator
    # for deployment and run the predict method of a low power 32 bit edge
    # platform.
    #
    # A similar thing happens for endianness, the machine where the pickle was
    # saved can have a different endianness than the machine where the pickle
    # is loaded

    dtype_dict = _dtype_to_dict(dtype)
    dtype_dict_with_modified_bitness = _dtype_dict_with_modified_bitness(dtype_dict)
    dtype_dict_with_modified_endianness = _dtype_to_dict(dtype.newbyteorder())
    dtype_dict_with_modified_bitness_and_endianness = _dtype_dict_with_modified_bitness(
        dtype_dict_with_modified_endianness
    )

    return [
        dtype_dict,
        dtype_dict_with_modified_bitness,
        dtype_dict_with_modified_endianness,
        dtype_dict_with_modified_bitness_and_endianness,
    ]


def _check_node_ndarray(node_ndarray, expected_dtype):
    if node_ndarray.ndim != 1:
        raise ValueError(
            "Wrong dimensions for node array from the pickle: "
            f"expected 1, got {node_ndarray.ndim}"
        )

    if not node_ndarray.flags.c_contiguous:
        raise ValueError(
            "node array from the pickle should be a C-contiguous array"
        )

    node_ndarray_dtype = node_ndarray.dtype
    if node_ndarray_dtype == expected_dtype:
        return node_ndarray

    node_ndarray_dtype_dict = _dtype_to_dict(node_ndarray_dtype)
    all_compatible_dtype_dicts = _all_compatible_dtype_dicts(expected_dtype)

    if node_ndarray_dtype_dict not in all_compatible_dtype_dicts:
        raise ValueError(
            "node array from the pickle has an incompatible dtype:\n"
            f"- expected: {expected_dtype}\n"
            f"- got     : {node_ndarray_dtype}"
        )

    return node_ndarray.astype(expected_dtype, casting="same_kind")


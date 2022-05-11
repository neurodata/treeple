#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3
#cython: binding=True

import numpy as np
cimport numpy as np
from cython.operator import dereference
np.import_array()

from sklearn import datasets
from sklearn.utils import check_random_state

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from oblique_forests.tree._criterion import Gini, MSE
from oblique_forests.tree._criterion cimport Criterion
from oblique_forests.tree._oblique_splitter import ObliqueSplitter
from oblique_forests.tree._oblique_splitter cimport BaseObliqueSplitter, ObliqueSplitRecord
from oblique_forests.tree._oblique_tree import ObliqueDepthFirstTreeBuilder
from oblique_forests.tree._oblique_tree cimport ObliqueTree, ObliqueTreeBuilder

TREE_UNDEFINED = -2
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef double EPSILON = np.finfo('double').eps


# Toy example
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

X = np.asfortranarray(X, dtype=DTYPE)
y = np.ascontiguousarray(y, dtype=DOUBLE)

# Diabetes dataset
random_state = check_random_state(0)
diabetes = datasets.load_diabetes()
perm = random_state.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

# Random labels
y_random = random_state.randint(0, 4, size=(20, ))

DATASETS = {
    "toy": {"X": X, "y": y},
    "diabetes": {"X": diabetes.data, "y": diabetes.target},
    "zeros": {"X": np.zeros((20, 3)), "y": y_random}
}

cdef SIZE_t max_depth = np.iinfo(np.int32).max
cdef SIZE_t min_samples_split = 2
cdef SIZE_t min_samples_leaf = 1
cdef double min_weight_leaf = 0.
cdef SIZE_t max_features = 2
cdef double feature_combinations = 1.5
cdef double min_impurity_split = 0.
cdef double min_impurity_decrease = 0.
cdef DOUBLE_t* null_sample_weight = NULL


def prep_X_y(X, y):
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    n_outputs = y.shape[1]
    y_encoded = np.zeros(y.shape, dtype=int)
    for k in range(n_outputs):
        classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                return_inverse=True)
    y = y_encoded

    X = np.asfortranarray(X, dtype=DTYPE)
    y = np.ascontiguousarray(y, dtype=DOUBLE)
    return X, y

# =============================================================================
# Cython-level tests
# =============================================================================

def test_cinit():
    X = DATASETS["toy"]["X"]
    y = DATASETS["toy"]["y"]
    X, y = prep_X_y(X, y)
    n_samples, n_features = X.shape

    n_outputs = y.shape[1]
    n_classes = []
    for k in range(n_outputs):
        classes_k = np.unique(y[:, k])
        n_classes.append(classes_k.shape[0])
    n_classes = np.array(n_classes, dtype=np.intp)

    cdef Criterion criterion = Gini(n_outputs, n_classes)
    cdef ObliqueTree tree = ObliqueTree(n_features, n_classes, n_outputs)
    
    assert tree.n_features == n_features
    assert tree.n_outputs == n_outputs
    for k in range(n_outputs):
        assert tree.n_classes[k] == n_classes[k]
    assert tree.max_n_classes == np.max(n_classes)
    assert tree.max_depth == 0
    assert tree.node_count == 0
    assert tree.capacity == 0
    assert tree.value == NULL
    assert tree.nodes == NULL


def test_add_node():
    X = DATASETS["toy"]["X"]
    y = DATASETS["toy"]["y"]
    X, y = prep_X_y(X, y)
    n_samples, n_features = X.shape

    n_outputs = y.shape[1]
    n_classes = []
    for k in range(n_outputs):
        classes_k = np.unique(y[:, k])
        n_classes.append(classes_k.shape[0])
    n_classes = np.array(n_classes, dtype=np.intp)

    cdef Criterion criterion = Gini(n_outputs, n_classes)
    cdef ObliqueTree tree = ObliqueTree(n_features, n_classes, n_outputs)
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    splitter.init(X, y, null_sample_weight)
    
    cdef double weighted_n_node_samples
    cdef SIZE_t n_node_samples = splitter.n_samples
    cdef SIZE_t start = 0
    cdef SIZE_t end = n_node_samples
    cdef SIZE_t depth = 0
    cdef SIZE_t parent = _TREE_UNDEFINED
    cdef bint is_left = 0
    cdef double impurity
    cdef double min_impurity_decrease = 0
    cdef SIZE_t n_constant_features = 0
    cdef ObliqueSplitRecord split
    splitter.node_reset(start, end, &weighted_n_node_samples)
    is_leaf = (depth >= max_depth or
               n_node_samples < min_samples_split or
               n_node_samples < 2 * min_samples_leaf or
               weighted_n_node_samples < 2 * min_weight_leaf)
    impurity = splitter.node_impurity()

    if not is_leaf:
        splitter.node_split(impurity, &split, &n_constant_features)
        is_leaf = (is_leaf or split.pos >= end or
                   (split.improvement + EPSILON <
                   min_impurity_decrease))


def test_build_tree():
    X = DATASETS["toy"]["X"]
    y = DATASETS["toy"]["y"]
    X, y = prep_X_y(X, y)
    n_samples, n_features = X.shape

    n_outputs = y.shape[1]
    n_classes = []
    for k in range(n_outputs):
        classes_k = np.unique(y[:, k])
        n_classes.append(classes_k.shape[0])
    n_classes = np.array(n_classes, dtype=np.intp)

    cdef Criterion criterion = Gini(n_outputs, n_classes)
    cdef ObliqueTree tree = ObliqueTree(n_features, n_classes, n_outputs)
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef ObliqueTreeBuilder builder = ObliqueDepthFirstTreeBuilder(splitter, min_samples_split,
                                                     min_samples_leaf,
                                                     min_weight_leaf,
                                                     max_depth,
                                                     min_impurity_decrease,
                                                     min_impurity_split)
    builder.build(tree, X, y)
    assert tree.node_count > 0


def test_diabetes():
    X = DATASETS["diabetes"]["X"]
    y = DATASETS["diabetes"]["y"]
    X, y = prep_X_y(X, y)
    n_samples, n_features = X.shape

    n_outputs = y.shape[1]
    n_classes = []
    for k in range(n_outputs):
        classes_k = np.unique(y[:, k])
        n_classes.append(classes_k.shape[0])
    n_classes = np.array(n_classes, dtype=np.intp)

    cdef Criterion criterion = MSE(n_outputs, n_samples)
    cdef ObliqueTree tree = ObliqueTree(n_features, n_classes, n_outputs)
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef ObliqueTreeBuilder builder = ObliqueDepthFirstTreeBuilder(splitter, min_samples_split,
                                                     min_samples_leaf,
                                                     min_weight_leaf,
                                                     max_depth,
                                                     min_impurity_decrease,
                                                     min_impurity_split)
    builder.build(tree, X, y)
    assert tree.node_count > 0

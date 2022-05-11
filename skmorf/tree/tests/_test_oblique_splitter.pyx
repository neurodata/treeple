#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3
#cython: binding=True

import functools
import inspect
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
np.import_array()

from sklearn.utils import check_random_state

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from oblique_forests.tree._criterion import Gini
from oblique_forests.tree._criterion cimport Criterion
from oblique_forests.tree._oblique_splitter import ObliqueSplitter
from oblique_forests.tree._oblique_splitter cimport BaseObliqueSplitter, ObliqueSplitRecord
from oblique_forests.tree._utils cimport safe_realloc


# Toy example
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

X = np.asfortranarray(X, dtype=DTYPE)
y = np.ascontiguousarray(y, dtype=DOUBLE)

# Random labels
random_state = check_random_state(0)
y_random = random_state.randint(0, 4, size=(20, ))

DATASETS = {
    "toy": {"X": X, "y": y},
    "zeros": {"X": np.zeros((20, 3)), "y": y_random}
}

cdef SIZE_t max_features = 2
cdef SIZE_t min_samples_leaf = 1
cdef double min_weight_leaf = 0.
cdef double feature_combinations = 1.5
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

def test_init():
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
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)

    assert splitter.proj_mat_weights.size() == max_features
    assert splitter.proj_mat_indices.size() == max_features
    assert all([splitter.proj_mat_weights[i].size() == 0
                for i in range(splitter.proj_mat_weights.size())])
    assert all([splitter.proj_mat_indices[i].size() == 0
                for i in range(splitter.proj_mat_indices.size())])
    assert splitter.init(X, y, null_sample_weight) == 0


def test_sample_proj_mat():
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
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    splitter.init(X, y, null_sample_weight)

    # Projection matrix of splitter initializes to all zeros
    assert all([splitter.proj_mat_weights[i].size() == 0 
                for i in range(splitter.proj_mat_weights.size())])
    assert all([splitter.proj_mat_indices[i].size() == 0 
                for i in range(splitter.proj_mat_indices.size())])

    # Sample projections in place using proj_mat pointer
    splitter.sample_proj_mat(splitter.proj_mat_weights, splitter.proj_mat_indices)

    # Projection matrix of splitter now has at least one nonzero
    assert any([splitter.proj_mat_weights[i].size() > 0 
                for i in range(splitter.proj_mat_weights.size())])
    assert any([splitter.proj_mat_indices[i].size() > 0 
                for i in range(splitter.proj_mat_indices.size())])


def test_node_reset():
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
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef SIZE_t n_node_samples
    cdef double weighted_n_node_samples
    splitter.init(X, y, null_sample_weight)
    n_node_samples = splitter.n_samples

    splitter.sample_proj_mat(splitter.proj_mat_weights, splitter.proj_mat_indices)

    assert splitter.node_reset(0, n_node_samples, &weighted_n_node_samples) == 0
    assert weighted_n_node_samples == n_samples

    assert all([splitter.proj_mat_weights[i].size() == 0 
                for i in range(splitter.proj_mat_weights.size())])
    assert all([splitter.proj_mat_indices[i].size() == 0 
                for i in range(splitter.proj_mat_indices.size())])


def test_node_impurity():
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
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef double impurity
    cdef double weighted_n_node_samples
    splitter.init(X, y, null_sample_weight)
    splitter.node_reset(0, splitter.n_samples, &weighted_n_node_samples)
    impurity = splitter.node_impurity()
    assert impurity == 0.5


def test_node_split():
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
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    
    cdef double impurity
    cdef ObliqueSplitRecord split
    cdef double weighted_n_node_samples
    cdef SIZE_t n_constant_features = 0
    splitter.init(X, y, null_sample_weight)
    splitter.node_reset(0, splitter.n_samples, &weighted_n_node_samples)
    impurity = splitter.node_impurity()
    
    assert splitter.node_split(impurity, &split, &n_constant_features) == 0

    assert 0 <= split.feature <= splitter.max_features
    assert 0 <= split.pos <= n_samples
    assert 0 <= split.improvement <= 1
    assert 0 <= split.impurity_left <= 1
    assert 0 <= split.impurity_right <= 1

    # Check that split proj_vec matches a vector in splitter.proj_mat
    cdef vector[DTYPE_t] proj_vec_weights = deref(split.proj_vec_weights)
    cdef vector[SIZE_t] proj_vec_indices = deref(split.proj_vec_indices)

    assert proj_vec_weights.size() > 0
    assert proj_vec_indices.size() > 0
    assert (proj_vec_weights.size() 
            == splitter.proj_mat_weights[split.feature].size())
    assert (proj_vec_indices.size() 
            == splitter.proj_mat_indices[split.feature].size())
    assert all(proj_vec_weights[i] == splitter.proj_mat_weights[split.feature][i]
               for i in range(proj_vec_weights.size()))
    assert all(proj_vec_indices[i] == splitter.proj_mat_indices[split.feature][i]
               for i in range(proj_vec_indices.size()))


def test_node_value():
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
    cdef BaseObliqueSplitter splitter = ObliqueSplitter(criterion, max_features, 
                                                        min_samples_leaf, min_weight_leaf, 
                                                        feature_combinations, random_state)
    cdef double impurity
    cdef ObliqueSplitRecord split
    cdef double weighted_n_node_samples
    cdef SIZE_t n_constant_features = 0
    splitter.init(X, y, null_sample_weight)
    splitter.node_reset(0, splitter.n_samples, &weighted_n_node_samples)
    impurity = splitter.node_impurity()
    splitter.node_split(impurity, &split, &n_constant_features)

    cdef double* dest = NULL
    cdef SIZE_t capacity = 1
    cdef SIZE_t value_stride = n_outputs * max(n_classes)

    safe_realloc(&dest, capacity * value_stride)
    splitter.node_value(dest)

    for i in range(n_classes[0]):
        assert dest[i] == criterion.sum_total[i]

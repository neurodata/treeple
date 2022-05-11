#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

cimport cython

import numpy as np

from libcpp.unordered_map cimport unordered_map
from cython.operator import dereference, postincrement

from libcpp.algorithm cimport sort as stdsort

from libcpp.vector cimport vector
from libcpp.pair cimport pair

from cython.parallel import prange

# TODO: rand not thread safe, replace with sklearn's utils when merging code
#from libc.stdlib cimport rand, srand, RAND_MAX 

from ..tree._utils cimport rand_int, rand_uniform

# DTYPE_t = type of X
# DOUBLE_t = type of y
# SIZE_t = indices type
from ..tree._utils cimport DTYPE_t, DOUBLE_t, SIZE_t, INT32_t, UINT32_t

# 0 < t < len(y)

cdef void argsort(DTYPE_t[:] x, SIZE_t[:] idx) nogil:

    cdef SIZE_t length = x.shape[0]
    cdef SIZE_t i = 0
    cdef pair[DOUBLE_t, SIZE_t] p
    cdef vector[pair[DOUBLE_t, SIZE_t]] v
        
    for i in range(length):
        p.first = x[i]
        p.second = i
        v.push_back(p)

    stdsort(v.begin(), v.end())

    for i in range(length):
        idx[i] = v[i].second

cdef (SIZE_t, SIZE_t) argmin(DOUBLE_t[:, :] A) nogil:
    cdef SIZE_t N = A.shape[0]
    cdef SIZE_t M = A.shape[1]
    cdef SIZE_t i = 0
    cdef SIZE_t j = 0
    cdef SIZE_t min_i = 0
    cdef SIZE_t min_j = 0
    cdef DOUBLE_t minimum = A[0, 0]

    for i in range(N):
        for j in range(M):

            if A[i, j] < minimum:
                minimum = A[i, j]
                min_i = i
                min_j = j

    return (min_i, min_j)

cdef void matmul(DTYPE_t[:, :] A, DTYPE_t[:, :] B, DTYPE_t[:, :] res) nogil:

    cdef SIZE_t i, j, k
    cdef SIZE_t m, n, p

    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[1]

    for i in range(m):
        for j in range(p):

            res[i, j] = 0
            for k in range(n):
                res[i, j] += A[i, k] * B[k, j]
 
cdef class BaseObliqueSplitter:

    cdef DTYPE_t[:, :] X
    cdef DOUBLE_t[:] y
    cdef DTYPE_t max_features
    cdef DTYPE_t feature_combinations
    cdef UINT32_t random_state
    
    cdef SIZE_t n_samples
    cdef SIZE_t n_features
    cdef SIZE_t proj_dims
    cdef SIZE_t n_non_zeros


    def __cinit__(self, DTYPE_t[:, :] X, 
            DOUBLE_t[:] y, DTYPE_t max_features, 
            DTYPE_t feature_combinations, UINT32_t random_state):

        self.X = X
        self.y = y
        self.max_features = max_features
        self.feature_combinations = feature_combinations
        self.random_state = random_state

        """
        classes = np.array(np.unique(y), dtype=np.intp)
        self.n_classes = len(classes)
        self.class_indices = np.indices(self.y.shape)[0]
        """

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        #self.root_impurity = self.impurity(self.y)

        self.proj_dims = max(np.ceil(max_features * X.shape[1]), 1)
        self.n_non_zeros = max(np.ceil(self.proj_dims * feature_combinations), 1)

    cdef DOUBLE_t impurity(self, DOUBLE_t[:] y) nogil:
        cdef SIZE_t length = y.shape[0]
        cdef DOUBLE_t dlength = y.shape[0]
        cdef DOUBLE_t temp = 0
        cdef DOUBLE_t gini = 1.0
        
        cdef unordered_map[DOUBLE_t, DOUBLE_t] counts
        cdef unordered_map[DOUBLE_t, DOUBLE_t].iterator it = counts.begin()

        if length == 0:
            return 0

        # Count all unique elements
        for i in range(0, length):
            temp = y[i]
            counts[temp] += 1

        it = counts.begin()
        while it != counts.end():
            temp = dereference(it).second
            temp = temp / dlength
            temp = temp * temp
            gini -= temp

            postincrement(it)

        return gini

    cdef DOUBLE_t score(self, DOUBLE_t[:] y, SIZE_t t) nogil:
        cdef DOUBLE_t length = y.shape[0]
        cdef DOUBLE_t left_gini = 1.0
        cdef DOUBLE_t right_gini = 1.0
        cdef DOUBLE_t gini = 0
    
        cdef DOUBLE_t[:] left = y[:t]
        cdef DOUBLE_t[:] right = y[t:]

        cdef DOUBLE_t l_length = left.shape[0]
        cdef DOUBLE_t r_length = right.shape[0]

        left_gini = self.impurity(left)
        right_gini = self.impurity(right)

        gini = (l_length / length) * left_gini + (r_length / length) * right_gini
        return gini
    
    # TODO
    """
    C's rand function is not thread safe, so this block is currently with GIL.
    When merging this code with sklearn, we can use their random number generator from their utils
    But since I don't have that here with me, I'm using C's rand function for now.

    proj_mat & proj_X should be np.zeros()

    """
    cdef void sample_proj_mat(self, DTYPE_t[:, :] X, DTYPE_t[:, :] proj_mat, DTYPE_t[:, :] proj_X) nogil:
        
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t proj_dims = proj_X.shape[1]

        cdef SIZE_t i, feat, pdim

        # Draw n non zeros & put SIZE_to proj_mat
        for i in range(self.n_non_zeros):
            feat = rand_int(0, n_features, &self.random_state)
            pdim = rand_int(0, proj_dims, &self.random_state)
            weight = 1 if (rand_int(0, 1, &self.random_state) % 2 == 1) else -1
            
            proj_mat[feat, pdim] = weight 
        
        matmul(X, proj_mat, proj_X)

    # X, y are X/y relevant samples. sample_inds only passed in for sorting
    # Will need to change X to not be proj_X rn
    cpdef best_split(self, DTYPE_t[:, :] X, DOUBLE_t[:] y, SIZE_t[:] sample_inds):

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t proj_dims = X.shape[1]
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef SIZE_t temp = 0;
        cdef DOUBLE_t node_impurity = 0;

        cdef SIZE_t thresh_i = 0
        cdef SIZE_t feature = 0
        cdef DOUBLE_t best_gini = 0
        cdef DOUBLE_t threshold = 0
        cdef DOUBLE_t improvement = 0
        cdef DOUBLE_t left_impurity = 0
        cdef DOUBLE_t right_impurity = 0

        Q = np.zeros((n_samples, proj_dims), dtype=np.float64)
        cdef DOUBLE_t[:, :] Q_view = Q

        idx = np.zeros(n_samples, dtype=np.intp)
        cdef SIZE_t[:] idx_view = idx

        y_sort = np.zeros(n_samples, dtype=np.float64)
        cdef DOUBLE_t[:] y_sort_view = y_sort
        
        feat_sort = np.zeros(n_samples, dtype=np.float32)
        cdef DTYPE_t[:] feat_sort_view = feat_sort

        si_return = np.zeros(n_samples, dtype=np.intp)
        cdef SIZE_t[:] si_return_view = si_return
        
        # No split or invalid split --> node impurity
        node_impurity = self.impurity(y)
        Q_view[:, :] = node_impurity
        
        # loop over columns of the matrix (projected feature dimensions)
        for j in range(0, proj_dims):
            # get the sorted indices along the rows (sample dimension)
            argsort(X[:, j], idx_view)

            for i in range(0, n_samples):
                temp = idx_view[i]
                y_sort_view[i] = y[temp]
                feat_sort_view[i] = X[temp, j]

            for i in prange(1, n_samples, nogil=True):
                
                # Check if the split is valid!
                if feat_sort_view[i-1] < feat_sort_view[i]:
                    Q_view[i, j] = self.score(y_sort_view, i)

        # Identify best split
        (thresh_i, feature) = argmin(Q_view)
      
        best_gini = Q_view[thresh_i, feature]
        # Sort samples by split feature
        argsort(X[:, feature], idx_view)
        for i in range(0, n_samples):
            temp = idx_view[i]

            # Sort X so we can get threshold
            feat_sort_view[i] = X[temp, feature]
            
            # Sort y so we can get left_y, right_y
            y_sort_view[i] = y[temp]
            
            # Sort true sample inds
            si_return_view[i] = sample_inds[temp]
        
        # Get threshold, split samples SIZE_to left and right
        if (thresh_i == 0):
            threshold = node_impurity #feat_sort_view[thresh_i]
        else:
            threshold = 0.5 * (feat_sort_view[thresh_i] + feat_sort_view[thresh_i - 1])

        left_idx = si_return_view[:thresh_i]
        right_idx = si_return_view[thresh_i:]
        
        # Evaluate improvement
        improvement = node_impurity - best_gini

        # Evaluate impurities for left and right children
        left_impurity = self.impurity(y_sort_view[:thresh_i])
        right_impurity = self.impurity(y_sort_view[thresh_i:])

        return feature, threshold, left_impurity, left_idx, right_impurity, right_idx, improvement 

    """
    Python wrappers for cdef functions.
    Only to be used for testing purposes.
    """

    def test_argsort(self, y):
        idx = np.zeros(len(y), dtype=np.intp)
        argsort(y, idx)
        return idx

    def test_argmin(self, M):
        return argmin(M)

    def test_impurity(self, y):
        return self.impurity(y)

    def test_score(self, y, t):
        return self.score(y, t)

    def test_best_split(self, X, y, idx):
        return self.best_split(X, y, idx)

    def test_matmul(self, A, B):
        res = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
        matmul(A, B, res)
        return res

    def test(self):

        # Test score
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        s = [self.score(y, i) for i in range(10)]
        print(s)

        # Test splitter
        # This one worked
        X = np.array([[0, 0, 0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        si = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.intp)

        (f, t, li, lidx, ri, ridx, imp) = self.best_split(X, y, si)
        print(f, t)

        

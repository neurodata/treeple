import functools
import importlib
import inspect
import sys


cython_test_modules = ["_test_oblique_splitter"]

def cytest(func):
    """
    Wraps `func` in a plain Python function.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        return func(*bound.args, **bound.kwargs)

    return wrapped


for mod in cython_test_modules:
    try:
        # For each callable in `mod` with name `test_*`,
        # set the result as an attribute of this module.
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                setattr(sys.modules[__name__], name, cytest(item))
    except ImportError:
        pass


# class TestBaseSplitter:
#     def test_argsort(self):
#         b = BOS()

#         # Ascending array
#         y = np.array([0, 1, 2, 3, 4], dtype=np.float64)
#         idx = b.test_argsort(y)
#         assert_allclose(y, idx)

#         # Descending array
#         y = np.array([4, 3, 2, 1, 0], dtype=np.float64)
#         idx = b.test_argsort(y)
#         assert_allclose(y, idx)

#         # Array with repeated values
#         y = np.array([1, 1, 1, 0, 0], dtype=np.float64)
#         idx = b.test_argsort(y)
#         assert_allclose([3, 4, 0, 1, 2], idx)

#     def test_argmin(self):

#         b = BOS()

#         X = np.ones((5, 5), dtype=np.float64)
#         X[3, 4] = 0
#         (i, j) = b.test_argmin(X)
#         assert 3 == i
#         assert 4 == j

#     def test_matmul(self):
        
#         b = BOS()

#         A = np.zeros((3, 3), dtype=np.float64)
#         B = np.ones((3, 3), dtype=np.float64)
        
#         for i in range(3):
#             for j in range(3):
#                 A[i, j] = 3*i + j + 1
        
#         res = b.test_matmul(A, B)

#         C = np.ones((3, 3), dtype=np.float64)
#         C[0] = 6
#         C[1] = 15
#         C[2] = 24

#         assert_allclose(C, res)


#     def test_impurity(self):

#         """
#         First 2
#         Taken from SPORF's fpGiniSplitTest.h
#         """

#         b = BOS()

#         y = np.ones(6, dtype=np.float64) * 4
#         imp = b.test_impurity(y)
#         assert 0 == imp

#         y[:3] = 2
#         imp = b.test_impurity(y)
#         assert 0.5 == imp

#         y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.float64)
#         imp = b.test_impurity(y)
#         assert_almost_equal((2 / 3), imp)

#     def test_score(self):

#         b = BOS()

#         y = np.ones(10, dtype=np.float64)
#         y[:5] = 0
#         s = b.test_score(y, 5)
#         assert 0 == s

#         y = np.ones(9, dtype=np.float64)
#         y[:3] = 0
#         y[6:] = 2
#         s = b.test_score(y, 3)
#         assert_almost_equal((1 / 3), s)

#     def test_halfSplit(self):

#         b = BOS()

#         y = np.ones(100, dtype=np.float64) * 4
#         y[:50] = 2

#         X = np.ones((100, 1), dtype=np.float64) * 10
#         X[:50] = 5

#         idx = np.array([i for i in range(100)], dtype=np.intc)

#         (
#             feat,
#             thresh,
#             left_imp,
#             left_idx,
#             right_imp,
#             right_idx,
#             improvement,
#         ) = b.best_split(X, y, idx)
#         assert thresh == 7.5
#         assert left_imp == 0
#         assert right_imp == 0
#         assert feat == 0

#     def test_oneOffEachEnd(self):

#         b = BOS()

#         y = np.ones(6, dtype=np.float64) * 4
#         y[0] = 1

#         X = np.ones((6, 1), dtype=np.float64) * 10
#         X[0] = 5

#         idx = np.array([i for i in range(6)], dtype=np.intc)

#         (
#             feat1,
#             thresh1,
#             left_imp1,
#             left_idx,
#             right_imp1,
#             right_idx,
#             improvement,
#         ) = b.best_split(X, y, idx)

#         y = np.ones(6, dtype=np.float64) * 4
#         y[-1] = 1

#         X = np.ones((6, 1), dtype=np.float64) * 10
#         X[-1] = 5

#         (
#             feat2,
#             thresh2,
#             left_imp2,
#             left_idx,
#             right_imp2,
#             right_idx,
#             improvement,
#         ) = b.best_split(X, y, idx)
#         assert feat1 == feat2
#         assert thresh1 == thresh2
#         assert left_imp1 + right_imp1 == left_imp2 + right_imp2

#     def test_secondFeature(self):

#         b = BOS()

#         y = np.ones(6, dtype=np.float64) * 4
#         y[:3] = 2
#         X = np.array([[10, 5, 10, 5, 10, 5]], dtype=np.float64).T
#         idx = np.array([i for i in range(6)], dtype=np.intc)

#         (
#             feat,
#             thresh,
#             left_imp,
#             left_idx,
#             right_imp,
#             right_idx,
#             improvement,
#         ) = b.best_split(X, y, idx)

#         assert 7.5 == thresh
#         assert 0 < left_imp
#         assert_almost_equal(left_imp, right_imp)

#         X[:] = 8
#         X[:3] = 4

#         (
#             feat,
#             thresh,
#             left_imp,
#             left_idx,
#             right_imp,
#             right_idx,
#             improvement,
#         ) = b.best_split(X, y, idx)

#         assert 6 == thresh
#         assert 0 == left_imp
#         assert 0 == right_imp

#     def test_largeSplit(self):

#         b = BOS()

#         y = np.ones(100, dtype=np.float64) * 4
#         y[:50] = 2
#         y[20:25] = 4

#         X = np.array([[i for i in range(100)]], dtype=np.float64).T
#         idx = np.array([i for i in range(100)], dtype=np.intc)

#         (
#             feat,
#             thresh,
#             left_imp,
#             left_idx,
#             right_imp,
#             right_idx,
#             improvement,
#         ) = b.best_split(X, y, idx)

#         # Expect a split down the middle
#         assert 49.5 == thresh
#         assert 0 == right_imp
#         assert_almost_equal(0.18, left_imp)

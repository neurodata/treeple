from itertools import product

import numpy as np
from numpy.testing import assert_equal

from .._utils import ravel_multi_index, unravel_index


def test_unravel_index():
    # Test with 1D array
    indices = np.array([0, 1, 2, 3, 4])
    shape = np.asarray((5,))
    expected_output = [(0,), (1,), (2,), (3,), (4,)]
    for idx, index in enumerate(indices):
        assert unravel_index(index, shape) == expected_output[idx]

    # Test with 2D array
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    shape = np.array((3, 3))
    expected_output = list(product(range(3), repeat=2))
    for idx, index in enumerate(indices):
        assert_equal(unravel_index(index, shape), expected_output[idx])

    # Test with 3D array
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    shape = np.array((2, 2, 2, 2))
    expected_output = list(product(range(2), repeat=4))
    for idx, index in enumerate(indices):
        assert_equal(unravel_index(index, shape), expected_output[idx])

    # Test with empty array
    indices = np.array([])
    shape = (2, 2)
    expected_output = []
    for idx, index in enumerate(indices):
        assert unravel_index(index, shape) == expected_output[idx]

    # Test with out-of-bounds index
    # indices = np.array([0, 1, 2, 3, 4])
    # shape = np.array((2, 2))
    # try:
    #     for idx, index in enumerate(indices):
    #         ind = unravel_index(index, shape)
    #         print(ind)
    #     assert False
    # except ValueError as e:
    #     assert str(e) == "Invalid index"
    # else:
    #     print(e)
    #     assert False, "Expected IndexError but no exception was raised"


def test_ravel_multi_index():
    # Test 1D array
    index = np.array([3], dtype=np.intp)
    shape = np.array([6], dtype=np.intp)
    assert_equal(ravel_multi_index(index, shape), 3)

    # Test 2D array
    index = np.array([1, 2], dtype=np.intp)
    shape = np.array([2, 3], dtype=np.intp)
    assert_equal(ravel_multi_index(index, shape), 5)

    # Test invalid index
    # index = np.array([2, 4], dtype=np.intp)
    # shape = np.array([2, 3], dtype=np.intp)
    # try:
    #     ind = ravel_multi_index(index, shape)
    #     print(ind)
    #     assert False
    # except ValueError as e:
    #     assert str(e) == "Invalid index"
    # else:
    #     assert False, "Expected ValueError"

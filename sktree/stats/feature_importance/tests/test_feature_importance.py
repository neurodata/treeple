import numpy as np
import pytest
from numpy.testing import assert_array_less, assert_almost_equal, assert_raises, assert_warns

from .. import PermutationTest

class TestFeatureImportance:

    def test_null(self):
        # matches p-value for the null.
        np.random.seed(123456789)

        X = np.ones((100, 10), dtype=float)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        p_val = [0.05]*10
        _, calculated_p_val = PermutationTest(n_estimators=10).test(X, y)
        assert_array_less(p_val, calculated_p_val)
    
    def test_alternate(self):
        # matches p-value for the alternate hypothesis.
        np.random.seed(123456789)

        X = np.concatenate((np.zeros((50, 10)), np.ones((50, 10))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)
        
        p_val = [0.05]*10
        _, calculated_p_val = PermutationTest(n_estimators=10).test(X, y)
        assert_array_less(calculated_p_val, p_val)
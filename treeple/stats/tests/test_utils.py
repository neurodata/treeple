import importlib
import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import treeple.stats.utils as utils
from treeple import HonestForestClassifier
from treeple.stats.utils import get_per_tree_oob_samples

seed = 1234
rng = np.random.default_rng(seed)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_get_per_tree_oob_samples(bootstrap):
    n_estimators = 5
    est = HonestForestClassifier(n_estimators=n_estimators, random_state=0, bootstrap=bootstrap)

    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    samples = np.arange(len(y))
    est.fit(X, y)

    if bootstrap:
        inbag_samples = est.estimators_samples_
        oob_samples = [
            [idx for idx in samples if idx not in inbag_samples[jdx]] for jdx in range(n_estimators)
        ]
        oob_samples_ = get_per_tree_oob_samples(est)
        for itree in range(n_estimators):
            assert len(oob_samples[itree]) > 1
            assert_array_equal(oob_samples_[itree], oob_samples[itree])
    else:
        with pytest.raises(RuntimeError, match="Cannot extract out-of-bag samples"):
            get_per_tree_oob_samples(est)


@pytest.mark.parametrize("use_bottleneck", [True, False])
def test_non_nan_samples(use_bottleneck: bool):

    if not use_bottleneck:
        os.environ[utils.DISABLE_BN_ENV_VAR] = "1"
        importlib.reload(utils)

    posterior_array = np.array(
        [
            # tree 1
            [
                [0, 1],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
            # tree 2
            [
                [0, 1],
                [np.nan, np.nan],
                [1, 0],
            ],
        ]
    )  # [2, 3, 2]

    expected = np.array([0, 2])
    actual = utils._non_nan_samples(posterior_array)
    np.testing.assert_array_equal(expected, actual)

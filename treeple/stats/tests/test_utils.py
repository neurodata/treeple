import importlib
import os

import numpy as np
import pytest
import scipy.sparse as sp
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
    if use_bottleneck and utils.DISABLE_BN_ENV_VAR in os.environ:
        del os.environ[utils.DISABLE_BN_ENV_VAR]
        importlib.reload(utils)
    else:
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


@pytest.mark.parametrize("use_bottleneck", [True, False])
def test_nanmean_f(use_bottleneck: bool):
    if use_bottleneck and utils.DISABLE_BN_ENV_VAR in os.environ:
        del os.environ[utils.DISABLE_BN_ENV_VAR]
        importlib.reload(utils)
    else:
        os.environ[utils.DISABLE_BN_ENV_VAR] = "1"
        importlib.reload(utils)

    posterior_array = np.array(
        [
            [1, 2, np.nan],
            [3, 4, np.nan],
        ]
    )

    expected = np.array([1.5, 3.5])
    actual = utils.nanmean_f(posterior_array, axis=1)
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    ("forest_indices", "expected"),
    [
        (np.arange(3), np.array([0.375, 0.75, 0.25])),
        (np.arange(3) + 2, np.array([0.10, 0.05, 0.25])),
        (np.arange(3) + 3, np.array([0.10, 0.45, np.nan])),
    ],
)
def test_get_forest_preds_sparse(
    forest_indices,
    expected,
):

    all_y_pred = sp.csc_matrix(
        np.array(
            [
                [0.50, 0.00, 0.00],
                [0.25, 0.75, 0.00],
                [0.00, 0.00, 0.25],
                [0.10, 0.00, 0.00],
                [0.00, 0.05, 0.00],
                [0.00, 0.85, 0.00],
            ]
        )
    )

    all_y_indicator = sp.csc_matrix(
        np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
            ]
        )
    )

    np.testing.assert_array_equal(
        utils._get_forest_preds_sparse(all_y_pred, all_y_indicator, forest_indices),
        expected,
    )

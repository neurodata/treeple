import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sktree.stats.utils import get_per_tree_oob_samples
from sktree import HonestForestClassifier

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

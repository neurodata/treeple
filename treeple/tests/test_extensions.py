import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification

from treeple import (
    ExtraObliqueRandomForestClassifier,
    ExtraObliqueRandomForestRegressor,
    HonestForestClassifier,
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    PatchObliqueRandomForestClassifier,
    PatchObliqueRandomForestRegressor,
)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize(
    "Forest",
    [
        HonestForestClassifier,
        ExtraObliqueRandomForestClassifier,
        ObliqueRandomForestClassifier,
        PatchObliqueRandomForestClassifier,
    ],
)
def test_predict_proba_per_tree(Forest, n_classes):
    # Assuming forest_model is an instance of a forest model with ForestMixin
    # You may need to adjust the actual implementation according to your specific model
    X, y = make_classification(
        n_samples=100, n_features=50, n_informative=20, n_classes=n_classes, random_state=0
    )

    # Call the method being tested
    if Forest == HonestForestClassifier:
        est = Forest(n_estimators=10, bootstrap=True, random_state=0, honest_prior="empirical")
    else:
        est = Forest(n_estimators=10, bootstrap=True, random_state=0)
    est.fit(X, y)
    proba_per_tree = est.predict_proba_per_tree(X)

    # Perform assertions to check the correctness of the output
    assert proba_per_tree.shape[0] == est.n_estimators
    assert proba_per_tree.shape[1] == X.shape[0]
    assert proba_per_tree.shape[2] == est.n_classes_
    assert not np.isnan(proba_per_tree).any()

    proba_per_tree = est.predict_proba_per_tree(X, est.oob_samples_)
    # Perform assertions to check the correctness of the output
    assert proba_per_tree.shape[0] == est.n_estimators
    assert proba_per_tree.shape[1] == X.shape[0]
    assert proba_per_tree.shape[2] == est.n_classes_
    assert np.isnan(proba_per_tree).any()


@pytest.mark.parametrize(
    "Forest",
    [
        HonestForestClassifier,
        ExtraObliqueRandomForestClassifier,
        ObliqueRandomForestClassifier,
        PatchObliqueRandomForestClassifier,
        ObliqueRandomForestRegressor,
        PatchObliqueRandomForestRegressor,
        ExtraObliqueRandomForestRegressor,
    ],
)
@pytest.mark.parametrize("bootstrap", [True, False])
@pytest.mark.parametrize("random_state", [None, 0])
def test_forest_has_deterministic_sampling_for_oob_structure_and_leaves(
    Forest, bootstrap, random_state
):
    """Test that forest models can produce the oob and inbag samples deterministically.

    When bootstrap is True, oob should be exclusive from in bag samples.
    When bootstrap is False, there is no oob.
    """
    rng = np.random.default_rng(0)

    n_estimators = 5
    est = Forest(
        n_estimators=n_estimators,
        random_state=random_state,
        bootstrap=bootstrap,
    )
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    samples = np.arange(len(y))

    est.fit(X, y)

    inbag_samples = est.estimators_samples_
    oob_samples = [
        [idx for idx in samples if idx not in inbag_samples[jdx]] for jdx in range(n_estimators)
    ]
    if not bootstrap:
        assert all(oob_list_ == [] for oob_list_ in oob_samples)

        with pytest.raises(RuntimeError, match="Cannot extract out-of-bag samples"):
            est.oob_samples_
    else:
        oob_samples_ = est.oob_samples_
        for itree in range(n_estimators):
            assert len(oob_samples[itree]) > 1, oob_samples[itree]
            assert set(inbag_samples[itree]).intersection(set(oob_samples_[itree])) == set()
            assert set(inbag_samples[itree]).union(set(oob_samples_[itree])) == set(samples)
            assert_array_equal(oob_samples_[itree], oob_samples[itree])

import numpy as np
import pytest
from sklearn.datasets import make_classification

from sktree import (
    ExtraObliqueRandomForestClassifier,
    HonestForestClassifier,
    ObliqueRandomForestClassifier,
    PatchObliqueRandomForestClassifier,
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
    X, y = make_classification(n_classes=n_classes, random_state=0)

    # Call the method being tested
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

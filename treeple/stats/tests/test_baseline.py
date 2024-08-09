import numpy as np
import pytest
from numpy.testing import assert_array_equal

from treeple import HonestForestClassifier
from treeple.stats import (
    PermutationHonestForestClassifier,
    build_cv_forest,
    build_permutation_forest,
)

seed = 12345
rng = np.random.default_rng(seed)


@pytest.mark.parametrize("bootstrap, max_samples", [(True, 1.6), (False, None)])
def test_build_cv_honest_forest(bootstrap, max_samples):
    n_estimators = 100
    est = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        bootstrap=bootstrap,
        max_samples=max_samples,
        honest_fraction=0.5,
        stratify=True,
    )
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = np.array([0, 1] * 50)
    samples = np.arange(len(y))

    est_list, proba_list, train_idx_list, test_idx_list = build_cv_forest(
        est,
        X,
        y,
        return_indices=True,
        seed=seed,
        cv=3,
    )

    assert isinstance(est_list, list)
    assert isinstance(proba_list, list)

    for est, proba, train_idx, test_idx in zip(est_list, proba_list, train_idx_list, test_idx_list):
        assert len(train_idx) + len(test_idx) == len(samples)
        structure_samples = est.structure_indices_
        leaf_samples = est.honest_indices_

        if not bootstrap:
            oob_samples = [[] for _ in range(est.n_estimators)]
        else:
            oob_samples = est.oob_samples_

        # compared to oob samples, now the train samples are comprised of the entire dataset
        # seen over the entire forest. The test dataset is completely disjoint
        for tree_idx in range(est.n_estimators):
            n_samples_in_tree = len(structure_samples[tree_idx]) + len(leaf_samples[tree_idx])
            assert n_samples_in_tree + len(oob_samples[tree_idx]) == len(train_idx), (
                f"For tree: "
                f"{tree_idx} {len(structure_samples[tree_idx])} + "
                f"{len(leaf_samples[tree_idx])} + {len(oob_samples[tree_idx])} "
                f"!= {len(train_idx)} {len(test_idx)}"
            )


def test_build_permutation_forest():
    """Simple test for building a permutation forest."""
    n_estimators = 30
    n_samples = 100
    n_features = 3
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 10
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, honest_fraction=0.5, bootstrap=True
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, honest_fraction=0.5, bootstrap=True
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_permutation_forest(clf, clf, X, y, seed=seed)

    forest_result, orig_forest_proba, perm_forest_proba = build_permutation_forest(
        clf, perm_clf, X, y, metric="s@98", n_repeats=20, seed=seed
    )
    assert forest_result.observe_test_stat > 0.1, f"{forest_result.observe_stat}"
    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _ = build_permutation_forest(
        clf, perm_clf, X, y, metric="s@98", n_repeats=10, seed=seed
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_test_stat < 0.05, f"{forest_result.observe_test_stat}"

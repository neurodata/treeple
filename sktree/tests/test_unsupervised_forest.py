import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree import UnsupervisedObliqueRandomForest, UnsupervisedRandomForest

CLUSTER_CRITERIONS = ("twomeans", "fastbic")

FOREST_CLUSTERS = {
    "UnsupervisedRandomForest": UnsupervisedRandomForest,
    "UnsupervisedObliqueRandomForest": UnsupervisedObliqueRandomForest,
}

# load iris dataset
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@parametrize_with_checks(
    [
        UnsupervisedRandomForest(random_state=12345, n_estimators=50),
        UnsupervisedObliqueRandomForest(random_state=12345, n_estimators=50),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in [
        # Cannot apply agglomerative clustering on < 2 samples
        "check_methods_subset_invariance",
        # # sample weights do not necessarily imply a sample is not used in clustering
        "check_sample_weights_invariance",
        # # sample order is not preserved in predict
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize("name, forest", FOREST_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_simulation(name, forest, criterion):
    n_samples = 100
    n_classes = 2

    if name == "UnsupervisedRandomForest":
        n_features = 5
        n_estimators = 50
        if criterion == 'twomeans':
            expected_score = 0.45
        elif criterion == 'fastbic':
            expected_score = 0.95
    elif name == 'UnsupervisedObliqueRandomForest':
        n_features = 20
        n_estimators = 20
        # This should be > 0.9 according to the UReRF
        if criterion == 'twomeans':
            expected_score = 0.95
        elif criterion == 'fastbic':
            expected_score = 0.95
    else:
        raise ValueError(f'{name} does not exist')

    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=12345
    )

    clf = forest(n_estimators=n_estimators, criterion=criterion, random_state=12345)
    clf.fit(X)
    sim_mat = clf.affinity_matrix_

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)
    print(f'For forest on simulation data: {name} - {criterion}: {score}')

    assert (
        score > expected_score
    ), f"{name}-blobs failed with criterion {criterion} and score = {score}"


@pytest.mark.parametrize("name, forest", FOREST_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_iris(name, forest, criterion):
    # Check consistency on dataset iris.
    if name == 'UnsupervisedRandomForest':
        if criterion == 'twomeans':
            expected_score = 0.2
        elif criterion == 'fastbic':
            expected_score = 0.35
    elif name == 'UnsupervisedObliqueRandomForest':
        if criterion == 'twomeans':
            expected_score = 0.2
        elif criterion == 'fastbic':
            expected_score = 0.55
    else:
        raise ValueError(f'{name} does not exist')

    n_classes = 3
    est = forest(criterion=criterion, random_state=12345)
    est.fit(iris.data, iris.target)
    sim_mat = est.affinity_matrix_

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(iris.target, predict_labels)
    print(f'For forest on iris: {name} - {criterion}: {score}')

    assert score > expected_score, f"{name}-iris failed with criterion {criterion} and score = {score}"

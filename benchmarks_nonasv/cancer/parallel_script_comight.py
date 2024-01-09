import sys
import os

import numpy as np

from hyppo.conditional import ConditionalDcorr
from sklearn.model_selection import StratifiedShuffleSplit
from sktree.stats import (
    FeatureImportanceForestClassifier,
)
from sktree import HonestForestClassifier
from sktree.tree import MultiViewDecisionTreeClassifier
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree


def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")


def mi_ksg(x, y, z=None, k=3, base=2):
    """Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = (
            avgdigamma(x, dvec),
            avgdigamma(y, dvec),
            digamma(k),
            digamma(len(x)),
        )
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            digamma(k),
        )
    return (-a - b + c + d) / log(base)


seed = 12345
rng = np.random.default_rng(seed)

# hard-coded parameters
n_estimators = 500
max_features = 0.3
test_size = 0.2


def _run_parallel_comight(
    idx,
    n_samples,
    seed,
    n_features_2,
    test_size,
    sim_type,
    rootdir,
):
    """Run parallel job on pre-generated data.

    Parameters
    ----------
    idx : int
        The index of the pre-generated dataset, stored as npz file.
    n_samples : int
        The number of samples to keep.
    seed : int
        The random seed.
    n_features_2 : int
        The number of dimensions to keep in feature set 2.
    test_size : float
        The size of the test set to use for predictive-model based tests.
    sim_type : str
        The simulation type. Either 'independent', 'collider', 'confounder',
        or 'direct-indirect'.
    rootdir : str
        The root directory where 'data/' and 'output/' will be.
    run_cdcorr : bool, optional
        Whether or not to run conditional dcorr, by default True.
    """
    n_jobs = 1
    n_features_ends = [100, None]

    # set output directory to save npz files
    output_dir = os.path.join(rootdir, f"output/varying-dimensionality/{sim_type}/")
    os.makedirs(output_dir, exist_ok=True)

    # load data
    npy_data = np.load(os.path.join(rootdir, f"data/{sim_type}/{sim_type}_{idx}.npz"))

    X = npy_data["X"]
    y = npy_data["y"]

    X = X[:, : 100 + n_features_2]
    if n_samples < X.shape[0]:
        cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
        for train_idx, _ in cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
    assert len(X) == len(y)
    assert len(y) == n_samples
    n_features_ends[1] = X.shape[1]

    est = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=n_estimators,
            tree_estimator=MultiViewDecisionTreeClassifier(
                max_features=[max_features, min(n_features_2, max_features * 100)],
                feature_set_ends=n_features_ends,
                apply_max_features_per_feature_set=True,
            ),
            random_state=seed,
            honest_fraction=0.5,
            n_jobs=n_jobs,
        ),
        random_state=seed,
        test_size=test_size,
        sample_dataset_per_tree=False,
    )

    # now compute the pvalue when shuffling X2
    covariate_index = np.arange(n_features_ends[0], n_features_ends[1])

    # Estimate CMI with
    mi_rf, pvalue = est.test(
        X,
        y,
        covariate_index=covariate_index,
        return_posteriors=True,
        metric="mi",
    )
    comight_posteriors_x2 = est.observe_posteriors_
    comight_null_posteriors_x2 = est.permute_posteriors_

    samples = est.observe_samples_
    permute_samples = est.permute_samples_

    assert np.isnan(comight_posteriors_x2[:, samples, :]).sum() == 0

    np.savez(
        os.path.join(output_dir, f"comight_{n_samples}_{n_features_2}_{idx}.npz"),
        n_samples=n_samples,
        n_features_2=n_features_2,
        y_true=y,
        comight_pvalue=pvalue,
        comight_mi=mi_rf,
        comight_posteriors_x2=comight_posteriors_x2,
        comight_null_posteriors_x2=comight_null_posteriors_x2,
    )


def _run_parallel_cond_dcorr(
    idx,
    n_samples,
    seed,
    n_features_2,
    test_size,
    sim_type,
    rootdir,
):
    """Run parallel job on pre-generated data.

    Parameters
    ----------
    idx : int
        The index of the pre-generated dataset, stored as npz file.
    n_samples : int
        The number of samples to keep.
    seed : int
        The random seed.
    n_features_2 : int
        The number of dimensions to keep in feature set 2.
    test_size : float
        The size of the test set to use for predictive-model based tests.
    sim_type : str
        The simulation type. Either 'independent', 'collider', 'confounder',
        or 'direct-indirect'.
    rootdir : str
        The root directory where 'data/' and 'output/' will be.
    run_cdcorr : bool, optional
        Whether or not to run conditional dcorr, by default True.
    """
    n_jobs = 1
    n_features_ends = [100, None]

    # set output directory to save npz files
    output_dir = os.path.join(rootdir, f"output/varying-dimensionality/{sim_type}/")
    os.makedirs(output_dir, exist_ok=True)

    # load data
    npy_data = np.load(os.path.join(rootdir, f"data/{sim_type}/{sim_type}_{idx}.npz"))

    X = npy_data["X"]
    y = npy_data["y"]

    X = X[:, : 100 + n_features_2]
    if n_samples < X.shape[0]:
        cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
        for train_idx, _ in cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
    assert len(X) == len(y)
    assert len(y) == n_samples
    n_features_ends[1] = X.shape[1]

    # now compute the pvalue when shuffling X2
    covariate_index = np.arange(n_features_ends[0], n_features_ends[1])

    cdcorr = ConditionalDcorr(bandwidth="silverman")
    Z = X[:, covariate_index]
    mask_array = np.ones(X.shape[1])
    mask_array[covariate_index] = 0
    mask_array = mask_array.astype(bool)

    X_minus_Z = X[:, mask_array]
    cdcorr_stat, cdcorr_pvalue = cdcorr.test(X_minus_Z.copy(), y.copy(), Z.copy())

    np.savez(
        os.path.join(output_dir, f"conddcorr_{n_samples}_{n_features_2}_{idx}.npz"),
        n_samples=n_samples,
        n_features_2=n_features_2,
        y_true=y,
        cdcorr_pvalue=cdcorr_pvalue,
        cdcorr_stat=cdcorr_stat,
    )


if __name__ == "__main__":
    # Ensure proper number of arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python script.py idx n_samples n_features_2 sim_type")
        sys.exit(1)

    # Extract arguments from terminal input
    idx = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    n_features_2 = int(sys.argv[3])
    sim_type = sys.argv[4]
    rootdir = sys.argv[5]

    # Call your function with the extracted arguments
    _run_parallel_comight(idx, n_samples, seed, n_features_2, test_size, sim_type, rootdir)

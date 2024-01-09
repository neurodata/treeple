import numpy as np

from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_sparse_spd_matrix,
    make_spd_matrix,
)


def _skewed_unimodal(n_dims):
    """Generate skewed unimodal density.

    #2 in Marron/Wand:

    X ~ 1/5 N(0, 1) + 1/5 N(1/2, 4/9) + 3/5 N(13/12, 25/81)
    """
    means = np.zeros((n_dims,))
    cov = np.zeros((n_dims, n_dims))

    _mean = 1./5 * 1/2 + 3./5 * 13/12
    _var = (1./5)**2 + (1./5)**2 * (2./3)**2 + (3./5)**2 * (5./9)**2

    means[:] = _mean
    np.fill_diagonal(cov, _var)
    return means, cov

def _strongly_skewed(n_dims):
    """Generate strongly skewed density.

    #3 in Marron/Wand:

    X ~ \sum_{i=0}^7 1/8 N(3[(2/3)^i - 1], (2/3)^(2l))
    """
    means = np.zeros((n_dims,))
    cov = np.zeros((n_dims, n_dims))

    _mean = 1./5 * 1/2 + 3./5 * 13/12
    _var = (1./5)**2 + (1./5)**2 * (2./3)**2 + (3./5)**2 * (5./9)**2
    means[:] = _mean
    np.fill_diagonal(cov, _var)
    return means, cov



def _strongly_skewed(n_dims):
    """Generate strongly skewed density.

    #3 in Marron/Wand:

    X ~ \sum_{i=0}^7 1/8 N(3[(2/3)^i - 1], (2/3)^(2l))
    """
    means = np.zeros((n_dims,))
    cov = np.zeros((n_dims, n_dims))

    _mean = 0.
    _var = 0.
    for idx in range(8):
        _mean += 1./8 * (3 * (2./3)**idx - 1) + 2./5 ** (2*idx)
        _var += (1./8)**2 * (2./3)**(2*idx)
    means[:] = _mean
    np.fill_diagonal(cov, _var)
    return means, cov



def _make_two_view_cov(
    n_features_1,
    n_features_2,
    view_relations,
    rng,
):
    # first setup the covariance matrix
    n_features = n_features_1 + n_features_2
    joint_cov = np.zeros((n_features, n_features))

    # now generate random PD matrix for the Covariance
    if view_relations == 'independent':
        cov_1 = make_spd_matrix(n_dim=n_features_1, random_state=rng.integers(0, 1e6)),
        joint_cov[:n_features_1, :n_features_1] = cov_1

        cov_2 = make_spd_matrix(n_dim=n_features_2, random_state=rng.integers(0, 1e6))
        joint_cov[:n_features_1, :n_features_1] = cov_2
    else:
        joint_cov = make_spd_matrix(n_dim=n_features, random_state=rng.integers(0, 1e6))

    return joint_cov

def make_twoview_classification(
    n_samples=100,
    n_features_1=10,
    n_features_2=100,
    noise_dims_1=90,
    noise_dims_2=3900,
    view_relations='independent',
    transform_1=None,
    transform_2=None,
    flip_y=0.01,
    seed=None,
):
    rng = np.random.default_rng(seed)

    # first setup the covariance matrix for first-view, where each covariance is separate
    n_features = n_features_1 + n_features_2
    joint_cov = _make_two_view_cov(
        n_features_1,
        n_features_2,
        view_relations,
        rng,
    )
    means = rng.uniform(-5, 5, size=(n_features,))

    # now sample two-views
    X = rng.multivariate_normal(means, joint_cov, size=n_samples)

    # now take half the samples and apply a transformation
    if transform_1 is not None:

        

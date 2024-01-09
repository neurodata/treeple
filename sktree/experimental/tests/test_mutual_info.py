import numpy as np
from numpy.testing import assert_almost_equal
from sktree.experimental.mutual_info import mi_gaussian, mutual_info_ksg

from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree


# Define a set of functions here to test our mutual information KSG implementation against


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


def mi(x, y, z=None, k=3, base=2, alpha=0):
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
        # if alpha > 0:
        #     d += lnc_correction(tree, points, k, alpha)
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


def nonlinear_gaussian_with_additive_noise():
    """Nonlinear no-noise function with additive Gaussian noise.

    See: https://github.com/BiuBiuBiLL/NPEET_LNC/issues/4
    """
    # first simulate multivariate Gaussian without noise

    # then add the noise

    # compute MI by computing the H(Y|X) and H(X)
    # H(Y|X) = np.log(noise_std)
    # H(X) = kNN K-L estimate with large # of samples
    pass


def main():
    d1 = [1, 1, 0]
    d2 = [1, 0, 1]
    d3 = [0, 1, 1]
    mat = [d1, d2, d3]
    tmat = np.transpose(mat)
    diag = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    # mean = np.array([0, 0, 0])
    cov = np.dot(tmat, np.dot(diag, mat))
    print("covariance matrix")
    print(cov)
    print(tmat)


def test_ksg_mi_estimate():
    seed = 1234
    rng = np.random.default_rng(seed)
    n_samples = 100

    d1 = [1, 1, 0]
    d2 = [1, 0, 1]
    d3 = [0, 1, 1]
    mat = [d1, d2, d3]
    tmat = np.transpose(mat)
    diag = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    # mean = np.array([0, 0, 0])
    cov = np.dot(tmat, np.dot(diag, mat))
    mean = np.zeros((3,))

    # generate random data according to multivariate Gaussian
    data = rng.multivariate_normal(mean, cov, size=(n_samples,))
    X, Y = data[:, -2:-1], data[:, -1:]

    print(X.shape, Y.shape, cov)
    true_mi = mi_gaussian(cov[-2:, -2:])
    est_mi = mutual_info_ksg(X, Y, k=0.2, random_seed=seed)
    print(true_mi, est_mi)
    print(mi(X, Y, k=int(0.2 * n_samples)))
    assert_almost_equal(true_mi, est_mi)


def test_mi():
    d1 = [1, 1, 0]
    d2 = [1, 0, 1]
    d3 = [0, 1, 1]
    mat = [d1, d2, d3]
    tmat = np.transpose(mat)
    diag = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    # mean = np.array([0, 0, 0])
    cov = np.dot(tmat, np.dot(diag, mat))
    print("covariance matrix")
    print(cov)
    trueent = -0.5 * (3 + np.log(8.0 * np.pi * np.pi * np.pi * np.linalg.det(cov)))
    trueent += -0.5 * (1 + np.log(2.0 * np.pi * cov[2][2]))  # z sub
    trueent += 0.5 * (
        2
        + np.log(
            4.0 * np.pi * np.pi * np.linalg.det([[cov[0][0], cov[0][2]], [cov[2][0], cov[2][2]]])
        )
    )  # xz sub
    trueent += 0.5 * (
        2
        + np.log(
            4.0 * np.pi * np.pi * np.linalg.det([[cov[1][1], cov[1][2]], [cov[2][1], cov[2][2]]])
        )
    )  # yz sub
    print("true CMI(x:y|x)", trueent / np.log(2))

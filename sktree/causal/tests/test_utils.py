import numpy as np 

from sktree.causal._utils import matinv, lstsq, pinv, fast_max_eigv, fast_min_eigv


def test_fast_eigv(self):
    rng = np.random.default_rng(123)

    n = 4
    for _ in range(10):
        A = rng.normal(0, 1, size=(n, n))
        A = np.asfortranarray(A @ A.T)
        apx = fast_min_eigv(A, 5, 123)
        opt = np.min(np.linalg.eig(A)[0])
        np.testing.assert_allclose(apx, opt, atol=.01, rtol=.3)
        apx = fast_max_eigv(A, 10, 123)
        opt = np.max(np.linalg.eig(A)[0])
        np.testing.assert_allclose(apx, opt, atol=.5, rtol=.2)


def test_linalg():
    rng = np.random.default_rng(1234)

    for n, m, nrhs in [(3, 3, 3), (3, 2, 1), (3, 1, 2), (1, 4, 2), (3, 4, 5)]:
        for _ in range(100):
            A = rng.normal(0, 1, size=(n, m))
            y = rng.normal(0, 1, size=(n, nrhs))
            yf = y
            if m > n:
                yf = np.zeros((m, nrhs))
                yf[:n] = y
            ours = np.asfortranarray(np.zeros((m, nrhs)))
            lstsq(np.asfortranarray(A), np.asfortranarray(yf.copy()), ours, copy_b=True)
            true = np.linalg.lstsq(A, y, rcond=np.finfo(np.float64).eps * max(n, m))[0]
            np.testing.assert_allclose(ours, true, atol=.00001, rtol=.0)

            ours = np.asfortranarray(np.zeros(A.T.shape, dtype=np.float64))
            pinv(np.asfortranarray(A), ours)
            true = np.linalg.pinv(A)
            np.testing.assert_allclose(ours, true, atol=.00001, rtol=.0)

            if n == m:
                ours = np.asfortranarray(np.zeros(A.T.shape, dtype=np.float64))
                matinv(np.asfortranarray(A), ours)
                true = np.linalg.inv(A)
                np.testing.assert_allclose(ours, true, atol=.00001, rtol=.0)

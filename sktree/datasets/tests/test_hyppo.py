from sktree.datasets import make_quadratic_classification


def test_make_quadratic_classification_v():
    n_samples = 100
    n_features = 5
    x, v = make_quadratic_classification(n_samples, n_features)
    assert all(val in [0, 1] for val in v)
    assert x.shape == (n_samples * 2, n_features)
    assert len(x) == len(v)

import numpy as np


def gaussian_kernel(shape, sigma=1.0, mu=0.0):
    """N-dimensional gaussian kernel for the given shape.

    See: https://gist.github.com/liob/e784775e882b83749cb3bbcef480576e
    """
    m = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape])
    d = np.sqrt(np.sum([x * x for x in m], axis=0))
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
    return g / np.sum(g)

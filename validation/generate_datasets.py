import numpy as np
from sporfdata import orthant, sparse_parity

# Sparse parity
for n in [1000, 5000, 10000]:
    X, y = sparse_parity(n)
    df = np.zeros((n, 21))
    df[:, :-1] = X
    df[:, -1] = y

    np.save("data/sparse_parity_train_" + str(n), df)

X, y = sparse_parity(10000)
df = np.zeros((10000, 21))
df[:, :-1] = X
df[:, -1] = y

np.save("data/sparse_parity_test", df)

# Orthant
for n in [400, 2000, 4000]:
    X, y = orthant(n)
    df = np.zeros((n, 7))
    df[:, :-1] = X
    df[:, -1] = y

    np.save("data/orthant_train_" + str(n), df)

X, y = orthant(10000)
df = np.zeros((10000, 7))
df[:, :-1] = X
df[:, -1] = y

np.save("data/orthant_test", df)

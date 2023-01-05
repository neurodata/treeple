import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load true values
# test_df = pd.read_csv("Sparse_parity_test.csv", header=None).to_numpy()
test_df = np.load("data/sparse_parity_test.npy")
y = test_df[:, -1]

ofpreds1k = np.load("output/of_sparse_parity_preds_1000.npy")
ofpreds5k = np.load("output/of_sparse_parity_preds_5000.npy")
ofpreds10k = np.load("output/of_sparse_parity_preds_10000.npy")

reps = ofpreds1k.shape[0]

rerfpreds1k = np.load("output/rerf_sparse_parity_preds_1000.npy")
rerfpreds5k = np.load("output/rerf_sparse_parity_preds_5000.npy")
rerfpreds10k = np.load("output/rerf_sparse_parity_preds_10000.npy")

oferr1k = np.sum(y == ofpreds1k, axis=1) / 10000
oferr5k = np.sum(y == ofpreds5k, axis=1) / 10000
oferr10k = np.sum(y == ofpreds10k, axis=1) / 10000

rerferr1k = np.sum(y == rerfpreds1k, axis=1) / 10000
rerferr5k = np.sum(y == rerfpreds5k, axis=1) / 10000
rerferr10k = np.sum(y == rerfpreds10k, axis=1) / 10000

ofmeans = np.array([np.mean(oferr1k), np.mean(oferr5k), np.mean(oferr10k)])
rerfmeans = np.array([np.mean(rerferr1k), np.mean(rerferr5k), np.mean(rerferr10k)])
diff_means = ofmeans - rerfmeans

n = [1000, 5000, 10000]
logn = np.log(n)
plt.figure(1)
plt.plot(logn, diff_means)
plt.xticks(logn, n)
plt.title("PySPORF vs RerF: Sparse Parity")
plt.xlabel("Number of training samples")
plt.ylabel("Difference in Accuracy")
plt.legend(["PySPORF - RerF"])

plt.savefig("figures/sparse_parity_diff")

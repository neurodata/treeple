import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load true values
# test_df = pd.read_csv("orthant_test.csv", header=None).to_numpy()
test_df = np.load("data/orthant_test.npy")
y = test_df[:, -1]
print(y.shape)

ofpreds400 = np.load("output/of_orthant_preds_400.npy")
ofpreds2k = np.load("output/of_orthant_preds_2000.npy")
ofpreds4k = np.load("output/of_orthant_preds_4000.npy")

reps = ofpreds400.shape[0]
print(ofpreds400.shape)

rerfpreds400 = np.load("output/rerf_orthant_preds_400.npy")
rerfpreds2k = np.load("output/rerf_orthant_preds_2000.npy")
rerfpreds4k = np.load("output/rerf_orthant_preds_4000.npy")

oferr400 = np.sum(y == ofpreds400, axis=1) / 10000
oferr2k = np.sum(y == ofpreds2k, axis=1) / 10000
oferr4k = np.sum(y == ofpreds4k, axis=1) / 10000

rerferr400 = np.sum(y == rerfpreds400, axis=1) / 10000
rerferr2k = np.sum(y == rerfpreds2k, axis=1) / 10000
rerferr4k = np.sum(y == rerfpreds4k, axis=1) / 10000

ofmeans = np.array([np.mean(oferr400), np.mean(oferr2k), np.mean(oferr4k)])
rerfmeans = np.array([np.mean(rerferr400), np.mean(rerferr2k), np.mean(rerferr4k)])

diff_means = ofmeans - rerfmeans

n = [400, 2000, 4000]
logn = np.log(n)
plt.figure(1)
plt.plot(logn, diff_means)
plt.xticks(logn, n)
plt.title("PySPORF vs RerF: Orthant")
plt.xlabel("Number of training samples")
plt.ylabel("Difference in Accuracy")
plt.legend(["PySPORF - RerF"])

plt.savefig("figures/orthant_diff")

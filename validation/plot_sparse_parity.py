import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load true values
# test_df = pd.read_csv("Sparse_parity_test.csv", header=None).to_numpy()
test_df = np.load("data/sparse_parity_test.npy")
y = test_df[:, -1]

pysporfpreds1k = np.load("output/of_sparse_parity_preds_1000.npy")
pysporfpreds5k = np.load("output/of_sparse_parity_preds_5000.npy")
pysporfpreds10k = np.load("output/of_sparse_parity_preds_10000.npy")

reps = pysporfpreds1k.shape[0]

sporfpreds1k = np.load("output/sporf_sparse_parity_preds_1000.npy")
sporfpreds5k = np.load("output/sporf_sparse_parity_preds_5000.npy")
sporfpreds10k = np.load("output/sporf_sparse_parity_preds_10000.npy")

rfpreds1k = np.load("output/rf_sparse_parity_preds_1000.npy")
rfpreds5k = np.load("output/rf_sparse_parity_preds_5000.npy")
rfpreds10k = np.load("output/rf_sparse_parity_preds_10000.npy")

rerfpreds1k = np.load("output/rerf_sparse_parity_preds_1000.npy")
rerfpreds5k = np.load("output/rerf_sparse_parity_preds_5000.npy")
rerfpreds10k = np.load("output/rerf_sparse_parity_preds_10000.npy")

oferr1k = 1 - np.sum(y == pysporfpreds1k, axis=1) / 10000
oferr5k = 1 - np.sum(y == pysporfpreds5k, axis=1) / 10000
oferr10k = 1 - np.sum(y == pysporfpreds10k, axis=1) / 10000

sporferr1k = 1 - np.sum(y == sporfpreds1k, axis=1) / 10000
sporferr5k = 1 - np.sum(y == sporfpreds5k, axis=1) / 10000
sporferr10k = 1 - np.sum(y == sporfpreds10k, axis=1) / 10000

rferr1k = 1 - np.sum(y == rfpreds1k, axis=1) / 10000
rferr5k = 1 - np.sum(y == rfpreds5k, axis=1) / 10000
rferr10k = 1 - np.sum(y == rfpreds10k, axis=1) / 10000

rerferr1k = 1 - np.sum(y == rerfpreds1k, axis=1) / 10000
rerferr5k = 1 - np.sum(y == rerfpreds5k, axis=1) / 10000
rerferr10k = 1 - np.sum(y == rerfpreds10k, axis=1) / 10000

ofmeans = [np.mean(oferr1k), np.mean(oferr5k), np.mean(oferr10k)]
ofsterr = [np.std(oferr1k), np.std(oferr5k), np.std(oferr10k)] / np.sqrt(reps)

sporfmeans = [np.mean(sporferr1k), np.mean(sporferr5k), np.mean(sporferr10k)]
sporfsterr = [np.std(sporferr1k), np.std(sporferr5k), np.std(sporferr10k)] / np.sqrt(reps)

rfmeans = [np.mean(rferr1k), np.mean(rferr5k), np.mean(rferr10k)]
rfsterr = [np.std(rferr1k), np.std(rferr5k), np.std(rferr10k)] / np.sqrt(reps)

rerfmeans = [np.mean(rerferr1k), np.mean(rerferr5k), np.mean(rerferr10k)]
rerfsterr = [np.std(rerferr1k), np.std(rerferr5k), np.std(rerferr10k)] / np.sqrt(reps)


n = [1000, 5000, 10000]
logn = np.log(n)
plt.figure(1)
plt.errorbar(logn, ofmeans, yerr=ofsterr)
plt.errorbar(logn, sporfmeans, yerr=sporfsterr)
plt.errorbar(logn, rfmeans, yerr=rfsterr)
plt.errorbar(logn, rerfmeans, yerr=rerfsterr)
plt.ylim(ymax=0.5, ymin=0)
plt.xticks(logn, n)
plt.title("Sparse Parity")
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.legend(["PySPORF", "CySPORF", "RF", "RerF"])

plt.savefig("figures/sparse_parity_experiment2")

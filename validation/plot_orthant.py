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

rfpreds400 = np.load("output/rf_orthant_preds_400.npy")
rfpreds2k = np.load("output/rf_orthant_preds_2000.npy")
rfpreds4k = np.load("output/rf_orthant_preds_4000.npy")

rerfpreds400 = np.load("output/rerf_orthant_preds_400.npy")
rerfpreds2k = np.load("output/rerf_orthant_preds_2000.npy")
rerfpreds4k = np.load("output/rerf_orthant_preds_4000.npy")

oferr400 = 1 - np.sum(y == ofpreds400, axis=1) / 10000
oferr2k = 1 - np.sum(y == ofpreds2k, axis=1) / 10000
oferr4k = 1 - np.sum(y == ofpreds4k, axis=1) / 10000

rferr400 = 1 - np.sum(y == rfpreds400, axis=1) / 10000
rferr2k = 1 - np.sum(y == rfpreds2k, axis=1) / 10000
rferr4k = 1 - np.sum(y == rfpreds4k, axis=1) / 10000

rerferr400 = 1 - np.sum(y == rerfpreds400, axis=1) / 10000
rerferr2k = 1 - np.sum(y == rerfpreds2k, axis=1) / 10000
rerferr4k = 1 - np.sum(y == rerfpreds4k, axis=1) / 10000

ofmeans = [np.mean(oferr400), np.mean(oferr2k), np.mean(oferr4k)]
ofsterr = [np.std(oferr400), np.std(oferr2k), np.std(oferr4k)] / np.sqrt(reps)

rfmeans = [np.mean(rferr400), np.mean(rferr2k), np.mean(rferr4k)]
rfsterr = [np.std(rferr400), np.std(rferr2k), np.std(rferr4k)] / np.sqrt(reps)

rerfmeans = [np.mean(rerferr400), np.mean(rerferr2k), np.mean(rerferr4k)]
rerfsterr = [np.std(rerferr400), np.std(rerferr2k), np.std(rerferr4k)] / np.sqrt(reps)


n = [400, 2000, 4000]
logn = np.log(n)
plt.figure(1)
plt.errorbar(logn, ofmeans, yerr=ofsterr)
plt.errorbar(logn, rfmeans, yerr=rfsterr)
plt.errorbar(logn, rerfmeans, yerr=rerfsterr)
plt.ylim(ymax=0.1, ymin=0)
plt.xticks(logn, n)
plt.title("Orthant")
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.legend(["PySPORF", "RF", "RerF"])

plt.savefig("figures/orthant_experiment")

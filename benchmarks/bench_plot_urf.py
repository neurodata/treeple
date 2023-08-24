from collections import defaultdict
from time import time

import numpy as np
from numpy import random as nr

from sktree import UnsupervisedObliqueRandomForest, UnsupervisedRandomForest


def compute_bench(samples_range, features_range):
    it = 0
    results = defaultdict(lambda: [])

    est_params = {"min_samples_split": 5, "criterion": "fastbic", "n_jobs": None}

    max_it = len(samples_range) * len(features_range)
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1

            print("==============================")
            print("Iteration %03d of %03d" % (it, max_it))
            print("==============================")
            print()
            print(f"n_samples: {n_samples} and n_features: {n_features}")
            data = nr.randint(-50, 51, (n_samples, n_features))

            print("Unsupervised RF")
            tstart = time()
            est = UnsupervisedRandomForest(**est_params).fit(data)

            delta = time() - tstart
            max_depth = max(tree.get_depth() for tree in est.estimators_)
            print("Speed: %0.3fs" % delta)
            print("Max depth: %d" % max_depth)
            print()

            results["unsup_rf_speed"].append(delta)
            results["unsup_rf_depth"].append(max_depth)

            print("Unsupervised Oblique RF")
            # let's prepare the data in small chunks
            est = UnsupervisedObliqueRandomForest(**est_params)
            tstart = time()
            est.fit(data)
            delta = time() - tstart
            max_depth = max(tree.get_depth() for tree in est.estimators_)
            print("Speed: %0.3fs" % delta)
            print("Max depth: %d" % max_depth)
            print()
            print()

            results["unsup_obliquerf_speed"].append(delta)
            results["unsup_obliquerf_depth"].append(max_depth)

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d  # noqa register the 3d projection

    samples_range = np.linspace(50, 150, 5).astype(int)
    features_range = np.linspace(150, 50000, 5).astype(int)
    chunks = np.linspace(500, 10000, 15).astype(int)

    results = compute_bench(samples_range, features_range)

    max_time = max([max(i) for i in [t for (label, t) in results.items() if "speed" in label]])
    max_inertia = max(
        [max(i) for i in [t for (label, t) in results.items() if "speed" not in label]]
    )

    fig = plt.figure("scikit-learn Unsupervised (Oblique and Axis) RF benchmark results")
    for c, (label, timings) in zip("brcy", sorted(results.items())):
        if "speed" in label:
            ax = fig.add_subplot(2, 1, 1, projection="3d")
            ax.set_zlim3d(0.0, max_time * 1.1)
        else:
            ax = fig.add_subplot(2, 1, 2, projection="3d")
            ax.set_zlim3d(0.0, max_inertia * 1.1)

        X, Y = np.meshgrid(samples_range, features_range)
        Z = np.asarray(timings).reshape(samples_range.shape[0], features_range.shape[0])
        ax.plot_surface(X, Y, Z.T, cstride=1, rstride=1, color=c, alpha=0.5)
        ax.set_title(f"{label}")
        ax.set_xlabel("n_samples")
        ax.set_ylabel("n_features")

    plt.show()

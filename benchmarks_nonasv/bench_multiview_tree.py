"""
To run this, you'll need to have installed.

  * scikit-learn
  * scikit-tree

Does two benchmarks

First, we fix a training set, increase the number of
samples to classify and plot number of classified samples as a
function of time.

In the second benchmark, we increase the number of dimensions of the
training set, classify a sample and plot the time taken as a function
of the number of dimensions.
"""

import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from sktree.tree import HonestTreeClassifier

# to store the results
scikit_classifier_results = []
sklearn_classifier_results = []
honest_classifier_results = []

mu_second = 0.0 + 10**6  # number of microseconds in a second


def bench_scikitlearn_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sklearn.tree import DecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeClassifier(max_features=0.3)
    clf.fit(X, Y)
    delta = datetime.now() - tstart
    # stop time

    sklearn_classifier_results.append(delta.seconds + delta.microseconds / mu_second)


def bench_oblique_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sktree.tree import MultiViewDecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = MultiViewDecisionTreeClassifier(max_features=0.3)
    clf.fit(X, Y)
    delta = datetime.now() - tstart
    # stop time

    # tstart = datetime.now()
    # clf.predict(X)
    # delta = datetime.now() - tstart

    scikit_classifier_results.append(delta.seconds + delta.microseconds / mu_second)


def bench_honest_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sktree.tree import MultiViewDecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = HonestTreeClassifier(
        max_features=0.3, honest_fraction=0.5, tree_estimator=MultiViewDecisionTreeClassifier()
    )
    clf.fit(X, Y)
    delta = datetime.now() - tstart
    # stop time

    # tstart = datetime.now()
    # clf.predict(X)
    # delta = datetime.now() - tstart

    honest_classifier_results.append(delta.seconds + delta.microseconds / mu_second)


if __name__ == "__main__":
    print("============================================")
    print("Warning: this is going to take a looong time")
    print("============================================")

    n = 10
    step = 1000
    n_samples = 100
    dim = 100
    n_classes = 2
    for i in range(n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_oblique_tree_classifier(X, Y)
        bench_scikitlearn_tree_classifier(X, Y)
        bench_honest_tree_classifier(X, Y)

    xx = range(0, n * step, step)
    plt.figure("scikit-tree oblique tree benchmark results")
    plt.subplot(211)
    plt.title("Learning with varying number of samples")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, sklearn_classifier_results, "o--", label="sklearn-classification")
    plt.plot(xx, honest_classifier_results, "r-", label="honest-classification")
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")

    scikit_classifier_results = []
    sklearn_classifier_results = []
    honest_classifier_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 2
    n_samples = 500

    dim = start_dim
    for i in range(0, n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        dim += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_oblique_tree_classifier(X, Y)
        bench_scikitlearn_tree_classifier(X, Y)
        bench_honest_tree_classifier(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title("Learning in high dimensional spaces")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, sklearn_classifier_results, "o--", label="sklearn-classification")
    plt.plot(xx, honest_classifier_results, "r-", label="honest-classification")
    plt.legend(loc="upper left")
    plt.xlabel("number of dimensions")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()

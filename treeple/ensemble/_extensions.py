import threading

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .._lib.sklearn.ensemble._forest import _get_n_samples_bootstrap
from ..tree._classes import DTYPE


def _parallel_predict_proba_per_tree(predict_proba, X, out, idx, test_idx, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    # each tree predicts proba with a list of output (n_samples, n_classes[i])
    prediction = predict_proba(X, check_input=False)

    indices = np.zeros(X.shape[0], dtype=bool)
    indices[test_idx] = True
    with lock:
        out[idx, test_idx, :] = prediction[test_idx, :]
    return prediction


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap, bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(
        random_state, n_samples, n_samples_bootstrap, bootstrap
    )
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap, bootstrap):
    """
    Private function used to _parallel_build_trees function.

    XXX: this is copied over from scikit-learn and modified to allow sampling with
    and without replacement given ``bootstrap``.
    """
    random_instance = check_random_state(random_state)
    n_sample_idx = np.arange(0, n_samples, dtype=np.int32)
    sample_indices = random_instance.choice(n_sample_idx, n_samples_bootstrap, replace=bootstrap)
    return sample_indices


class ForestMixin:
    """A set of mixin methods to extend forests models API."""

    @property
    def oob_samples_(self):
        """The sample indices that are out-of-bag.

        Only utilized if ``bootstrap=True``, otherwise, all samples are "in-bag".
        """
        if self.bootstrap is False and (
            self._n_samples_bootstrap is None or (self._n_samples_bootstrap == self._n_samples)
        ):
            raise RuntimeError(
                "Cannot extract out-of-bag samples when bootstrap is False and "
                "n_samples == n_samples_bootstrap"
            )
        check_is_fitted(self)

        oob_samples = []

        n_samples_bootstrap = _get_n_samples_bootstrap(
            self._n_samples,
            self.max_samples,
        )
        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state,
                self._n_samples,
                n_samples_bootstrap,
                self.bootstrap,
            )
            oob_samples.append(unsampled_indices)
        return oob_samples


class ForestClassifierMixin:
    """A set of mixin methods to extend forests models API."""

    def predict_proba_per_tree(self, X, indices=None):
        """
        Compute the probability estimates for each tree in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        indices : list of ``n_estimators`` length of array-like of shape (n_samples,), optional
            The indices of the samples used to compute the probability estimates
            for each tree in the forest. If None, the indices are every
            sample in the input data.

        Returns
        -------
        proba_per_tree : array-like of shape (n_estimators, n_samples, n_classes)
            The probability estimates for each tree in the forest.
        """
        # now evaluate
        X = self._validate_X_predict(X)

        if indices is None:
            indices = [np.arange(X.shape[0]) for _ in range(len(self.estimators_))]

        if len(indices) != len(self.estimators_):
            raise ValueError(
                "The length of the indices list must be equal to the number of estimators."
            )

        # if we trained a binning tree, then we should re-bin the data
        # XXX: this is inefficient and should be improved to be in line with what
        # the Histogram Gradient Boosting Tree does, where the binning thresholds
        # are passed into the tree itself, thus allowing us to set the node feature
        # value thresholds within the tree itself.
        if self.max_bins is not None:
            X = self._bin_data(X, is_training_data=False).astype(DTYPE)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        lock = threading.Lock()
        # accumulate the predictions across all trees
        proba_per_tree = np.full(
            (len(self.estimators_), X.shape[0], self.n_classes_), np.nan, dtype=np.float64
        )
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_parallel_predict_proba_per_tree)(
                e.predict_proba, X, proba_per_tree, idx, test_idx, lock
            )
            for idx, (e, test_idx) in enumerate(zip(self.estimators_, indices))
        )

        return proba_per_tree

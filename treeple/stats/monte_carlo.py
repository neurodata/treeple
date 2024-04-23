import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

from .utils import _compute_null_distribution_perm


# XXX: This is an experimental abstraction for hypothesis testing using
# purely monte-carlo methods for any scikit-learn estimator that supports
# the `predict_proba` method.
class PermutationTest(MetaEstimatorMixin):
    def __init__(self, estimator, n_repeats=100, test_size=0.2, random_state=None) -> None:
        """Permutation tester for hypothesis testing.

        This approaches the problem of conditional hypothesis testing using permutations
        of the covariate of interest. The test statistic is computed on the original data,
        and then on the permuted data. The p-value is the proportion of times the test
        statistic on the permuted data is more extreme than the test statistic on the
        original data.

        For example, one can use kNN as the estimator, or a logistic regression model.

        Parameters
        ----------
        estimator : _type_
            _description_
        n_repeats : int, optional
            _description_, by default 100
        test_size : float, optional
            _description_, by default 0.2
        random_state : _type_, optional
            _description_, by default None
        """
        self.estimator = estimator
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.random_state = random_state

    @property
    def train_test_samples_(self):
        """
        The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        indices = np.arange(self._n_samples_, dtype=int)

        # Get drawn indices along both sample and feature axes
        indices_train, indices_test = train_test_split(
            indices, test_size=self.test_size, shuffle=True, random_state=self.random_state
        )
        return indices_train, indices_test

    def test(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike,
        metric: str = "mse",
        n_repeats: int = 1000,
        return_posteriors: bool = False,
        **metric_kwargs,
    ):
        """Perform hypothesis test using permutation testing.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike of shape (n_covariates,)
            The covariate indices of ``X`` to shuffle.
        metric : str, optional
            Metric to compute, by default "mse".
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.
        **metric_kwargs : dict, optional
            Keyword arguments to pass to the metric function.

        Returns
        -------
        observe_stat : float
            Observed test statistic.
        pvalue : float
            Pvalue of the test.
        """
        X, y = check_X_y(X, y, ensure_2d=True, copy=True, multi_output=True)
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        self._n_samples_ = X.shape[0]

        self.estimator_ = clone(self.estimator)

        indices_train, indices_test = self.train_test_samples_

        # train/test split
        self.estimator_.fit(X[indices_train, :], y[indices_train, :])
        posterior = self.estimator_.predict_proba(X[indices_test, :])
        pauc = roc_auc_score(y[indices_test, :], posterior[:, 1], max_fpr=0.1)
        self.observe_stat_ = pauc

        # compute null distribution of the test statistic
        # WARNING: this could take a long time, since it fits a new forest
        null_dist = _compute_null_distribution_perm(
            X_train=X[indices_train, :],
            y_train=y[indices_train, :],
            X_test=X[indices_test, :],
            y_test=y[indices_test, :],
            covariate_index=covariate_index,
            est=self.estimator_,
            metric=metric,
            n_repeats=n_repeats,
            seed=self.random_state,
        )

        if not return_posteriors:
            self.null_dist_ = np.array(null_dist)
        else:
            self.null_dist_ = np.array([x[0] for x in null_dist])
            self.posterior_null_ = np.array([x[1] for x in null_dist])

        n_repeats = len(self.null_dist_)
        pvalue = (1 + (self.null_dist_ < self.observe_stat_).sum()) / (1 + n_repeats)
        return self.observe_stat_, pvalue

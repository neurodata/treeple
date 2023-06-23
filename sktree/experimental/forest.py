from copy import copy

import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from .ksg import entropy_continuous


class SupervisedInfoForest(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, y_categorical: bool = False, n_jobs=None):
        self.estimator = estimator
        self.y_categorical = y_categorical
        self.n_jobs = n_jobs

    def fit(self, X, y, Z=None):
        X, y = self._validate_data(X, y, accept_sparse="csc")

        print(X.shape, y.shape)
        self.estimator_yxz_ = copy(self.estimator)
        self.estimator_yz_ = copy(self.estimator)

        if Z is not None:
            XZ = np.hstack((X, Z))
        else:
            XZ = X

        # compute the entropy of H(Y | X, Z)
        self.estimator_yxz_.fit(XZ, y)

        # compute the entropy of H(Y | Z)
        if Z is None:
            # compute entropy using empirical distribution
            if self.y_categorical:
                _, self.counts_ = np.unique(y, return_counts=True)
            else:
                self.H_yz_ = entropy_continuous(y, k=5, n_jobs=self.n_jobs)
        else:
            self.estimator_yz_.fit(Z, y)

        return self

    def predict_cmi(self, X, Z=None):
        if Z is not None:
            X, Z = self._validate_data(X, Z, accept_sparse="csc")
        else:
            X = self._validate_data(X, accept_sparse="csc")

        if Z is not None:
            XZ = np.hstack((X, Z))
        else:
            XZ = X

        # compute the entropy of H(Y | X, Z)
        H_yxz = self.estimator_yxz_.predict_proba(XZ)

        # compute the entropy of H(Y | Z)
        if Z is None:
            if self.y_categorical:
                # compute entropy using empirical distribution
                H_yz = entropy(self.counts_, base=np.exp(1))
            else:
                H_yz = self.H_yz_
        else:
            H_yz = self.estimator_yz_.predict_proba(Z)

        return H_yxz - H_yz

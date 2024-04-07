from .base import FeatureImportanceTest
from ...ensemble import ObliqueRandomForestClassifier, PatchObliqueRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy.typing import ArrayLike
import numpy as np
from sklearn.utils.validation import check_X_y
import scipy.stats as ss
from numba import jit
from joblib import Parallel, delayed

class PermutationTest(FeatureImportanceTest):
    r"""
    Feature importance test statistic and p-value.
    """

    def __init__(self, n_estimators, classifier='RandomForest'):
        FeatureImportanceTest.__init__(self)
        self.n_estimators = n_estimators
        self.feature_importance = None
        
        if classifier=='RandomForest':
            self.model = RandomForestClassifier(n_estimators=self.n_estimators)
            self.permuted_model = RandomForestClassifier(n_estimators=self.n_estimators)
        elif classifier=='ObliqueRandomForest':
            self.model = ObliqueRandomForestClassifier(n_estimators=self.n_estimators)
            self.permuted_model = ObliqueRandomForestClassifier(n_estimators=self.n_estimators)
        elif classifier=='PatchObliqueRandomForest':
            self.model = PatchObliqueRandomForestClassifier(n_estimators=self.n_estimators)
            self.permuted_model = PatchObliqueRandomForestClassifier(n_estimators=self.n_estimators)
        else:
            raise ValueError('Classifier not recognized!')


    def _fit(self, X, y):
        r"""
        Helper function that is used to fit a particular random 
        forest.
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
            """
        check_X_y(X, y)

        self.model.fit(X,y)
        feature_importance = self.model.feature_importances_
        del self.model

        np.random.shuffle(y)
        self.permuted_model.fit(X,y)
        permuted_feature_importance = self.model.feature_importances_
        del self.permuted_model

        self.feature_importance = np.concatenate(
            (
                feature_importance,
                permuted_feature_importance
            )
        )

    @jit(nopython=True, cache=True)
    def _statistics(self, idx):
        r"""
        Helper function that calulates the feature importance 
        test statistic.
        """
        stat = np.zeros(len(self.feature_importance[0]))
        for ii in range(self.n_estimators):
            r = ss.rankdata(
                    1-self.feature_importance[idx[ii]], method='max'
                )
            r_0 = ss.rankdata(
                    1-self.feature_importance[idx[self.n_estimators+ii]], method='max'
                )

            stat += (r_0 > r)*1

        stat /= self.n_estimators

        return stat
    
    def _perm_stat(self):
        r"""
        Helper function that calulates the null distribution.
        """

        idx = list(range(2*self.n_estimators))
        np.random.shuffle(idx)

        return self._statistics(idx)
    
    def test(
            self, 
            X: ArrayLike,
            y: ArrayLike,
            n_repeats: int = 1000,
            n_jobs:int = -1
    ):
        r"""
        Calculates p values for fearture imprtance test.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.
        n_jobs : int, optional
            Number of workers to use, by default 1000.

        Returns
        -------
        stat : float
            The computed discriminability statistic.
        pvalue : float
            The computed one sample test p-value.
            """
        
        self._fit(X, y)
        stat = self._statistics(list(range(2*self.n_estimators)))
        null_stat = Parallel(n_jobs=n_jobs)(
                                delayed(self._perm_stat)() \
                                    for _ in range(n_repeats)
                    ) 
        count = np.sum((null_stat>=stat)*1,axis=0)
        p_val = (1 + count)/(1+n_repeats) 

        return stat, p_val
        
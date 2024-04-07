from abc import ABC, abstractmethod

class FeatureImportanceTest(ABC):
    r"""
    A base class for a feature importance test.
    """

    def __init__(self):
        self.p_value = None 
        super().__init__()
    
    @abstractmethod
    def _fit(self, X, y):
       r"""
       Helper function that is used to fit a particular random 
       forest.
        X : ArrayLike of shape (n_samples, n_features)
           The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
           The target matrix.
        """

    @abstractmethod
    def _statistics(self, idx):
        r"""
        Helper function that calulates the feature importance test statistic.
        """

    def _perm_stat(self):
        r"""
        Helper function that is used to calculate parallel permuted test
        statistics.

        Returns
        -------
        perm_stat : float
            Test statistic for each value in the null distribution.
        """
    
    @abstractmethod
    def testtest(self, X, y, n_repeats, n_jobs):
        r"""
        Calculates the feature importance test statistic and p-value.
        """
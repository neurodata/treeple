from abc import ABC, abstractmethod

class FeatureImportanceTest(ABC):
    r"""
    A base class for a feature importance test.
    """

    def __init__(self):
        self.p_value = None 
        super().__init__()
    
    @abstractmethod
    def statistics(self, F1, F2):
        r"""
        Calulates the feature importance test statistic.

        Parameters
        ----------
        F1 : ndarray of float
             Feature importance matrix from the unshuffled 
             trees. It must have shape ``(T, P)`` where `T` 
             is the number of trees and `P` is the number of 
             features.
        F2 : ndarray of float
             Feature importance matrix from the shuffled 
             trees. It must have shape ``(T, P)`` where `T` 
             is the number of trees and `P` is the number of 
             features.
        """
        pass

    def _perm_stat(self, index):
        r"""
        Helper function that is used to calculate parallel permuted test
        statistics.

        Parameters
        ----------
        index : int
            Iterator used for parallel statistic calculation

        Returns
        -------
        perm_stat : float
            Test statistic for each value in the null distribution.
        """
        pass
    
    @abstractmethod
    def test(self, *args, **kwargs):
        r"""
        Calculates the feature importance test statistic and p-value.
        """
        pass
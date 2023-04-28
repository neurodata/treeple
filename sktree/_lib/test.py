# from sktree._lib.criterion import BaseCriterion
import sys
import os
sys.path.append(os.getcwd() + './sklearn/sklearn/')
print(sys.path)

from .sklearn.sklearn.tree._splitter import Splitter
from .sklearn.sklearn.tree._tree import Tree
from .sklearn.sklearn.tree._criterion import Criterion
from .sklearn.sklearn.tree._utils import weighted_percentile

def test():
    clf = BaseDecisionTree()
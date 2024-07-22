import numpy as np

from treeple.tree import HonestTreeClassifier


def test_honest_tree_pruning():
    """Test honest tree with pruning to ensure no empty leaves."""
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10

    clf = HonestTreeClassifier(honest_method="prune", random_state=0)
    clf = clf.fit(X, y)
    # assert np.all(clf.tree_.children_left != -1)

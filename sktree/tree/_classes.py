from sklearn.tree import BaseDecisionTree

from ._unsup_criterion import UnsupervisedCriterion

class UnsupervisedDecisionTree(BaseDecisionTree):
    """Unsupervised decision tree.

    TODO: need to implement the logic fit/fit_transform/etc.

    Parameters
    ----------
    criterion : _type_
        _description_
    splitter : _type_
        _description_
    max_depth : _type_
        _description_
    min_samples_split : _type_
        _description_
    min_samples_leaf : _type_
        _description_
    min_weight_fraction_leaf : _type_
        _description_
    max_features : _type_
        _description_
    max_leaf_nodes : _type_
        _description_
    random_state : _type_
        _description_
    min_impurity_decrease : _type_
        _description_
    """

    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

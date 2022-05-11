class BaseManifoldSplitter:
    """Abstract base class for oblique splitters."""

    def sample_proj_mat(self, sample_inds):
        """Sample a projection matrix.

        Parameters
        ----------
        sample_inds : [type]
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError("All oblique splitters must implement this function.")

    def split(self, sample_inds):
        """Perform the splitting of a node.

        Parameters
        ----------
        sample_inds : [type]
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError("")


class StackRecord:
    """
    A class used to keep track of a node's parent and other information about the node and its split.

    Parameters
    ----------
    parent : int
        The index of the parent node.
    depth : int
        The depth at which this node is.
    is_left : bool
        Represents if the node is a left child or not.
    impurity : float
        This is Gini impurity of this node.
    sample_idx : array of shape [n_samples]
        This is the indices of the nodes that are in this node.
    n_samples : int
        The number of samples in this node.

    Methods
    -------
    None
    """

    def __init__(self, parent, depth, is_left, impurity, sample_idx, n_samples):

        self.parent = parent
        self.depth = depth
        self.is_left = is_left
        self.impurity = impurity
        self.sample_idx = sample_idx
        self.n_samples = n_samples


class Node:
    """
    A class used to represent an oblique node.

    Parameters
    ----------
    None

    Methods
    -------
    None
    """

    def __init__(self):
        self.node_id = None
        self.is_leaf = None
        self.parent = None
        self.left_child = None
        self.right_child = None

        self.feature = None
        self.threshold = None
        self.impurity = None
        self.n_samples = None

        self.label = None
        self.proba = None

        # oblique manifold trees
        self.proj_vec = None
        self.transform_params = None


class SplitInfo:
    """
    A class used to store information about a certain split.

    Parameters
    ----------
    feature : int
        The feature which is used for the particular split.
    threshold : float
        The feature value which defines the split, if an example has a value less
        than this threshold for the feature of this split then it will go to the
        left child, otherwise it will go the right child where these children are
        the children nodes of the node for which this split defines.
    left_impurity : float
        This is Gini impurity of left side of the split.
    left_idx : array of shape [left_n_samples]
        This is the indices of the nodes that are in the left side of this split.
    left_n_samples : int
        The number of samples in the left side of this split.
    right_impurity : float
        This is Gini impurity of right side of the split.
    right_idx : array of shape [right_n_samples]
        This is the indices of the nodes that are in the right side of this split.
    right_n_samples : int
        The number of samples in the right side of this split.
    no_split : bool
        A boolean specifying if there is a valid split or not. Here an invalid
        split means all of the samples would go to one side.
    improvement : float
        A metric to determine if the split improves the decision tree.
    proj_vec : array of shape [n_features]
        The vector of the sparse random projection matrix relevant for the split.
    """

    def __init__(
        self,
        feature,
        threshold,
        left_impurity,
        left_idx,
        left_n_samples,
        right_impurity,
        right_idx,
        right_n_samples,
        no_split,
        improvement,
        proj_vec,
        transform_params=None,
    ):

        self.feature = feature
        self.threshold = threshold
        self.left_impurity = left_impurity
        self.left_idx = left_idx
        self.left_n_samples = left_n_samples
        self.right_impurity = right_impurity
        self.right_idx = right_idx
        self.right_n_samples = right_n_samples
        self.no_split = no_split
        self.improvement = improvement

        # Oblique/manifold nodes
        self.proj_vec = proj_vec
        self.transform_params = transform_params

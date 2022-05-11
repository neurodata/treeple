import numpy as np
import numpy.random as rng
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._split import BaseObliqueSplitter
from .oblique_base import BaseManifoldSplitter, Node, SplitInfo, StackRecord


class ObliqueSplitter(BaseManifoldSplitter):
    """
    A class used to represent an oblique splitter.

    Splits are done on the linear combination of the features according
    to a specific basis function class.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    n_jobs : int
        The number of cores to parallelize the calculation of Gini impurity.
        Supply -1 to use all cores available to the Process.

    Methods
    -------
    sample_proj_mat(sample_inds)
        This gets the projection matrix and it fits the transform to the samples of interest.
    leaf_label_proba(idx)
        This calculates the label and the probability for that label for a particular leaf
        node.
    score(y_sort, t)
        Finds the Gini impurity for a split.
    _score(self, proj_X, y_sample, i, j)
        Handles array indexing before calculating Gini impurity.
    impurity(idx)
        Finds the impurity for a certain set of samples.
    split(sample_inds)
        Determines the best possible split for the given set of samples.
    """

    def __init__(self, X, y, max_features, feature_combinations, random_state):
        self.X = np.array(X, dtype=np.float64)

        # y must be 1D for now
        self.y = np.array(y, dtype=np.float64).squeeze()

        classes = np.array(np.unique(y), dtype=int)
        self.n_classes = len(classes)
        self.class_indices = {j: i for i, j in enumerate(classes)}

        self.indices = np.indices(self.y.shape)[0]

        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

        self.random_state = random_state
        rng.seed(random_state)

        # Compute root impurity
        unique, count = np.unique(y, return_counts=True)
        count = count / len(y)
        self.root_impurity = 1 - np.sum(np.power(count, 2))

        # proj_dims = d = mtry
        self.proj_dims = max(int(max_features * self.n_features), 1)

        # feature_combinations = mtry_mult
        # In processingNodeBin.h, mtryDensity = int(mtry * mtry_mult)
        self.n_non_zeros = max(int(self.proj_dims * feature_combinations), 1)

        # Base oblique splitter in cython
        self.BOS = BaseObliqueSplitter(X, y, max_features, feature_combinations, random_state)

        # Temporary debugging parameter, turns off oblique splits
        self.debug = False

    def sample_proj_mat(self, sample_inds):
        """
        Get the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.
        """
        if self.debug:
            return self.X[sample_inds, :], np.eye(self.n_features)

        # This is the way its done in the C++
        # create a d X d' matrix, where d' < d is the projected dimension
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # sample non-zero indices along the feature and projection dimension
        rand_feat = rng.randint(self.n_features, size=self.n_non_zeros)
        rand_dim = rng.randint(self.proj_dims, size=self.n_non_zeros)
        weights = [1 if rng.rand() > 0.5 else -1 for i in range(self.n_non_zeros)]
        proj_mat[rand_feat, rand_dim] = weights

        # Need to remove zero vectors from projmat
        proj_mat = proj_mat[:, np.unique(rand_dim)]

        # apply transformation of sample data points
        proj_X = self.X[sample_inds, :] @ proj_mat

        return (
            proj_X,
            proj_mat,
        )

    def leaf_label_proba(self, idx):
        """
        Label the probability of the test sample based on which leaves it falls in.

        Find the most common label and probability of this label from the samples at
        the leaf node for which this is used on.

        Parameters
        ----------
        idx : array of shape [n_samples]
            The indices of the samples that are at the leaf node for which the label
            and probability need to be found.

        Returns
        -------
        label : int
            The label for any sample that is predicted to be at this node.
        proba : float
            The probability of the predicted sample to have this node's label.
        """
        samples = self.y[idx]
        n = len(samples)
        labels, count = np.unique(samples, return_counts=True)

        proba = np.zeros(self.n_classes)
        for i, l in enumerate(labels):
            class_idx = self.class_indices[l]
            proba[class_idx] = count[i] / n

        most = np.argmax(count)
        label = labels[most]
        # max_proba = count[most] / n

        return label, proba

    # Finds the best split
    def split(self, sample_inds):
        """
        Find the optimal split for a set of samples.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The indices of the nodes in the set for which the best split is found.

        Returns
        -------
        split_info : SplitInfo
            Class holding information about the split.
        """
        # ensure that sample indices are 1D
        sample_inds = sample_inds.squeeze()

        if not self.y.squeeze().ndim == 1:
            raise RuntimeError("Does not support multivariate output yet.")

        # Project the data
        projection_data = self.sample_proj_mat(sample_inds)

        # extract the projection metadata
        if len(projection_data) == 2:
            proj_X, proj_mat = projection_data
            transform_params = None
        elif len(projection_data) == 3:
            proj_X, proj_mat, transform_params = projection_data

        y_sample = self.y[sample_inds]
        n_samples = len(sample_inds)

        # Assign types to everything
        # TODO: assign types when making the splitter class. This is silly.
        proj_X = np.array(proj_X, dtype=np.float32)
        y_sample = np.array(y_sample, dtype=np.float64)
        sample_inds = np.array(sample_inds, dtype=np.intp)

        # Call cython splitter
        (
            feature_idx,
            threshold,
            left_impurity,
            left_idx,
            right_impurity,
            right_idx,
            improvement,
        ) = self.BOS.best_split(proj_X, y_sample, sample_inds)

        # TODO: These are coming out as memory views. Fix this.
        left_idx = np.asarray(left_idx)
        right_idx = np.asarray(right_idx)

        left_n_samples = len(left_idx)
        right_n_samples = len(right_idx)

        no_split = left_n_samples == 0 or right_n_samples == 0

        # select the index of the projection we want to use
        proj_vec = proj_mat[:, feature_idx]
        if transform_params is not None:
            if len(transform_params) == 1:
                transform_params = transform_params[0]
            else:
                transform_params = transform_params[feature_idx]

        # store the split info, which in this case
        # also stores the projection vector selected with
        # best split criterion (i.e. Gini impurity)
        split_info = SplitInfo(
            feature_idx,
            threshold,
            left_impurity,
            left_idx,
            left_n_samples,
            right_impurity,
            right_idx,
            right_n_samples,
            no_split,
            improvement,
            proj_vec=proj_vec,
            transform_params=transform_params,
        )

        return split_info


class ObliqueTree:
    """
    A class used to represent a tree with oblique splits.

    Parameters
    ----------
    splitter : class
        The type of splitter for this tree, should be an ObliqueSplitter.
    min_samples_split : int
        Minimum number of samples possible at a node.
    min_samples_leaf : int
        Minimum number of samples possible at a leaf.
    max_depth : int
        Maximum depth allowed for the tree.
    min_impurity_split : float
        Minimum Gini impurity value that must be achieved for a split to occur on the node.
    min_impurity_decrease : float
        Minimum amount Gini impurity value must decrease by for a split to be valid.

    Methods
    -------
    add_node(parent, is_left, impurity, n_samples, is_leaf, feature, threshold, proj_mat, label, proba)
        Adds a node to the existing tree
    build()
        This is what is initially called on to completely build the oblique tree.
    predict(X)
        Finds the final node for each input sample as it passes through the decision tree.
    """

    def __init__(
        self,
        splitter,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        min_impurity_split,
        min_impurity_decrease,
    ):
        # Tree parameters
        self.depth = 0
        self.node_count = 0
        self.nodes = []

        # Build parameters
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        if self.max_depth is None:
            self.max_depth = np.inf

        self.min_impurity_split = min_impurity_split
        self.min_impurity_decrease = min_impurity_decrease

    def add_node(
        self,
        parent,
        is_left,
        impurity,
        n_samples,
        is_leaf,
        feature,
        threshold,
        label,
        proba,
        proj_vec=None,
        transform_params=None,
    ):
        """
        Add a node to the existing oblique tree.

        Parameters
        ----------
        parent : int
            The index of the parent node for the new node being added.
        is_left : bool
            Determines if this new node being added is a left or right child.
        impurity : float
            Impurity of this new node.
        n_samples : int
            Number of samples at this new node.
        is_leaf : bool
            Determines if this new node is a leaf of the tree or an internal node.
        feature : int
            Index of feature on which the split occurs at this node.
        threshold : float
            The threshold feature value for this node determining if a sample will go
            to this node's left of right child. If a sample has a value less than the
            threshold (for the feature of this node) it will go to the left childe,
            otherwise it will go the right child.
        label : int
            The label a sample will be given if it is predicted to be at this node.
        proba : float
            The probability a predicted sample has of being the node's label.
        proj_vec : {ndarray, sparse matrix} of shape (n_features)
            Projection vector for this new node.
        transform_params : dict
            A dictionary of keys and values that parametrize a kernel that is specified with a ``name``.


        Returns
        -------
        node_id : int
            Index of the new node just added.
        """
        node = Node()
        node.node_id = self.node_count
        node.impurity = impurity
        node.n_samples = n_samples

        # If not the root node, set parents
        if self.node_count > 0:
            node.parent = parent
            if is_left:
                self.nodes[parent].left_child = node.node_id
            else:
                self.nodes[parent].right_child = node.node_id

        # Set node parameters
        if is_leaf:
            node.is_leaf = True
            node.label = label
            node.proba = proba
        else:
            node.is_leaf = False
            node.feature = feature
            node.threshold = threshold
            node.proj_vec = proj_vec
            node.transform_params = transform_params

        self.node_count += 1
        self.nodes.append(node)

        return node.node_id

    def build(self):
        """
        Build the oblique tree.

        The build process will generate a ``StackRecord``
        to run the build process. At every node that the tree
        is built, a corresponding ``Node`` is stored to
        store the state of the split.
        """
        # Initialize, add root node
        stack = []
        root = StackRecord(
            0,
            1,
            False,
            self.splitter.root_impurity,
            self.splitter.indices,
            self.splitter.n_samples,
        )
        stack.append(root)

        # Build tree
        while len(stack) > 0:

            # Pop a record off the stack
            cur = stack.pop()

            # Evaluate if it is a leaf
            is_leaf = (
                cur.depth >= self.max_depth
                or cur.n_samples < self.min_samples_split
                or cur.n_samples < 2 * self.min_samples_leaf
                or cur.impurity <= self.min_impurity_split
            )

            # Split if not
            if not is_leaf:
                split = self.splitter.split(cur.sample_idx)

                is_leaf = (
                    is_leaf
                    or split.no_split
                    or split.improvement <= self.min_impurity_decrease
                )

            # Add the node to the tree
            if is_leaf:

                label, proba = self.splitter.leaf_label_proba(cur.sample_idx)

                node_id = self.add_node(
                    cur.parent,
                    cur.is_left,
                    cur.impurity,
                    cur.n_samples,
                    is_leaf,
                    None,
                    None,
                    label,
                    proba,
                    proj_vec=None,
                    transform_params=None,
                )

            else:
                node_id = self.add_node(
                    cur.parent,
                    cur.is_left,
                    cur.impurity,
                    cur.n_samples,
                    is_leaf,
                    split.feature,
                    split.threshold,
                    None,
                    None,
                    proj_vec=split.proj_vec,
                    transform_params=split.transform_params,
                )

            # Push the right and left children to the stack if applicable
            if not is_leaf:

                right_child = StackRecord(
                    node_id,
                    cur.depth + 1,
                    False,
                    split.right_impurity,
                    split.right_idx,
                    split.right_n_samples,
                )
                stack.append(right_child)

                left_child = StackRecord(
                    node_id,
                    cur.depth + 1,
                    True,
                    split.left_impurity,
                    split.left_idx,
                    split.left_n_samples,
                )
                stack.append(left_child)

            if cur.depth > self.depth:
                self.depth = cur.depth

    def _transform_data(self, X, proj_vec):
        proj_X = X @ proj_vec
        return proj_X

    def predict(self, X, check_input=True):
        """
        Predicts final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input array for which predictions are made.

        Returns
        -------
        predictions : array of shape [n_samples]
            Array of the final node index for each input prediction sample.
        """
        from .conv import _apply_convolution

        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            cur = self.nodes[0]
            while cur is not None and not cur.is_leaf:
                # get the projected point
                proj_X = self._transform_data(X[i], cur.proj_vec)

                if proj_X < cur.threshold:
                    id = cur.left_child
                    cur = self.nodes[id]
                else:
                    id = cur.right_child
                    cur = self.nodes[id]

            predictions[i] = cur.node_id

        return predictions


class ObliqueTreeClassifier(BaseEstimator):
    """
    A class used to represent a classifier that uses an oblique decision tree.

    Parameters
    ----------
    max_depth : int
        Maximum depth allowed for oblique tree.
    min_samples_split : int
        Minimum number of samples possible at a node.
    min_samples_leaf : int
        Minimum number of samples possible at a leaf.
    random_state : int
        Maximum depth allowed for the tree.
    min_impurity_decrease : float
        Minimum amount Gini impurity value must decrease by for a split to be valid.
    min_impurity_split : float
        Minimum Gini impurity value that must be achieved for a split to occur on the node.
    feature_combinations : float
        The feature combinations to use for the oblique split.
    max_features : float
        Output dimension = max_features * dimension
    n_jobs : int, optional (default: -1)
        The number of cores to parallelize the calculation of Gini impurity.
        Supply -1 to use all cores available to the Process.

    Methods
    -------
    fit(X,y)
        Fits the oblique tree to the training samples.
    apply(X)
        Calls on the predict function from the oblique tree for the test samples.
    predict(X)
        Gets the prediction labels for the test samples.
    predict_proba(X)
        Gets the probability of the prediction labels for the test samples.
    predict_log_proba(X)
        Gets the log of the probability of the prediction labels for the test samples.
    """

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        min_impurity_decrease=0,
        min_impurity_split=0,
        feature_combinations=1.5,
        max_features=1,
        n_jobs=1,
        bootstrap=False,
        warm_start=False,
        verbose=False
    ):

        # RF parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # OF parameters
        # Feature combinations = L
        self.feature_combinations = feature_combinations

        # Max features
        self.max_features = max_features

        self.n_classes = None
        self.n_jobs = n_jobs

        # passed in parameters
        # TODO: implement these functionality similar to sklearn.
        self.bootstrap = bootstrap
        self.warm_start = warm_start
        self.verbose = verbose

    def _tree_class(self):
        """Instantiate the tree to use."""
        return ObliqueTree

    # TODO: sklearn params do nothing
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        """
        Predict final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The training samples.
        y : array of shape [n_samples]
            Labels for the training samples.

        Returns
        -------
        ObliqueTreeClassifier
            The fit classifier.
        """
        splitter = BaseObliqueSplitter(
            X, y.ravel(), self.max_features, self.feature_combinations, self.random_state
        )
        self.n_classes = splitter.n_classes

        # get the tree class
        tree_func = self._tree_class()

        # instantiate the tree and build it
        self.tree = tree_func(
            splitter,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_depth,
            self.min_impurity_split,
            self.min_impurity_decrease,
        )
        self.tree.build()

        return self

    def apply(self, X):
        """
        Get predictions form the oblique tree for the test samples.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        pred_nodes : array of shape[n_samples]
            The indices for each test sample's final node in the oblique tree.
        """
        pred_nodes = self.tree.predict(X).astype(int)
        return pred_nodes

    def predict(self, X, check_input=True):
        """
        Determine final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The predictions (labels) for each testing sample.
        """
        X = np.array(X, dtype=np.float64)

        preds = np.zeros(X.shape[0])
        pred_nodes = self.apply(X)
        for k in range(len(pred_nodes)):
            id = pred_nodes[k]
            preds[k] = self.tree.nodes[id].label

        return preds

    def predict_proba(self, X, check_input=True):
        """
        Determine probabilities of the final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The probabilities of the predictions (labels) for each testing sample.
        """
        X = np.array(X, dtype=np.float64)

        preds = np.zeros((X.shape[0], self.n_classes))
        pred_nodes = self.apply(X)
        for k in range(len(preds)):
            id = pred_nodes[k]
            preds[k] = self.tree.nodes[id].proba

        return preds

    def predict_log_proba(self, X, check_input=True):
        """
        Determine log of the probabilities of the final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The log of the probabilities of the predictions (labels) for each testing sample.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    # TODO: Actually do this function
    def _validate_X_predict(self, X, check_input=True):
        return X

import numpy as np

cimport numpy as cnp

cnp.import_array()


def _build_pruned_tree_honesty(
    Tree tree,
    Tree orig_tree,
    HonestPruner pruner,
    object X,
    const float64_t[:, ::1] y,
    const float64_t[:] sample_weight,
):
    cdef:
        intp_t n_nodes = orig_tree.node_count
        unsigned char[:] leaves_in_subtree = np.zeros(
            shape=n_nodes, dtype=np.uint8)

    # initialize the pruner/splitter
    pruner.init(X, y, sample_weight, orig_tree.missing_values_in_feature_mask)

    # apply pruning to the tree
    _honest_prune(leaves_in_subtree, orig_tree, pruner)

    _build_pruned_tree(tree, orig_tree, leaves_in_subtree,
                       pruner.capacity)


cdef class HonestPruner(Splitter):
    """Pruning to enforce honest splits are non-degenerate."""

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        Tree orig_tree,
        *argv
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        orig_tree : Tree
            The original tree to be pruned.
        """
        self.tree = orig_tree
        self.capacity = 0

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        """Split nodes using the already constructed tree.

        Returns 0 if a split cannot be done, 1 if a split can be done
        and -1 in case of failure to allocate memory (and raise MemoryError).
        """
        pass
        # Extract input
        # cdef float32_t X_i_node_feature

        # # Initialize auxiliary data-structure
        # cdef Node* node = NULL
        # cdef intp_t i = 0
        # with nogil:
        #     for i in range(self.n_samples):
        #         node = self.nodes

        #         # While node not a leaf
        #         while node.left_child != _TREE_LEAF:
        #             X_i_node_feature = self._compute_feature(X_ndarray, i, node)
        #             # ... and node.right_child != _TREE_LEAF:
        #             if isnan(X_i_node_feature):
        #                 if node.missing_go_to_left:
        #                     node = &self.nodes[node.left_child]
        #                 else:
        #                     node = &self.nodes[node.right_child]
        #             elif X_i_node_feature <= node.threshold:
        #                 node = &self.nodes[node.left_child]
        #             else:
        #                 node = &self.nodes[node.right_child]

        #         out[i] = <intp_t>(node - self.nodes)  # node offset

        # return np.asarray(out)


cdef _honest_prune(
    unsigned char[:] leaves_in_subtree,
    Tree orig_tree,
    HonestPruner pruner,
):
    """Perform honest pruning of the tree.

    Iterates through the original tree in a BFS fashion using the pruner
    and tracks at each node:

    - the parent node id
    - the number of samples in the parent node
    - the number of samples in the node

    Until one of two stopping conditions are met:

    1. The orig_node is a leaf node.
    2. The orig_node is a non-leaf node and the split is degenerate.

    Parameters
    ----------
    leaves_in_subtree : array of shape (n_nodes,), dtype=np.uint8
        Array of booleans indicating whether the node is in the subtree.
    orig_tree : Tree
        The original tree.
    pruner : HonestPruner
        The input samples to be used for computing the split of samples
        in the nodes.
    """
    cdef:
        intp_t i
        intp_t n_nodes = orig_tree.node_count
        # prior probability using weighted samples
        float64_t[:] weighted_n_node_samples = orig_tree.weighted_n_node_samples
        float64_t total_sum_weights = weighted_n_node_samples[0]
        float64_t[:] impurity = orig_tree.impurity
        # weighted impurity of each node
        float64_t[:] r_node = np.empty(shape=n_nodes, dtype=np.float64)

        # Initialize output
        # intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

    # out = orig_tree.apply(X)

    # find parent node ids and leaves
    with nogil:
        # initialize the weights of each node
        for i in range(r_node.shape[0]):
            r_node[i] = (
                weighted_n_node_samples[i] * impurity[i] / total_sum_weights)
        # Push the root node

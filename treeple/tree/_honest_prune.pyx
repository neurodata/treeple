# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

cimport numpy as cnp

cnp.import_array()

from libc.math cimport isnan
from libc.stdlib cimport free, malloc
from libcpp.stack cimport stack
from sklearn.tree._tree cimport ParentInfo

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef float64_t INFINITY = np.inf

cdef inline void _init_parent_record(ParentInfo* record) noexcept nogil:
    record.n_constant_features = 0
    record.impurity = INFINITY
    record.lower_bound = -INFINITY
    record.upper_bound = INFINITY


def _build_pruned_tree_honesty(
    Tree tree,
    Tree orig_tree,
    HonestPruner pruner,
    object X,
    const float64_t[:, ::1] y,
    const float64_t[:] sample_weight,
    const uint8_t[::1] missing_values_in_feature_mask=None,
):
    """Prune an existing tree with honest splits.

    Parameters
    ----------
    tree : Tree
        The tree to be pruned.
    orig_tree : Tree
        The original tree to be pruned.
    pruner : HonestPruner
        The pruner to enforce honest splits.
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    sample_weight : array-like of shape (n_samples,)
        The sample weights.
    missing_values_in_feature_mask : array-like of shape (n_features,)
        The mask of missing values in the features.
    """
    cdef:
        intp_t n_nodes = orig_tree.node_count
        uint8_t[:] leaves_in_subtree = np.zeros(
            shape=n_nodes, dtype=np.uint8)

    # initialize the pruner/splitter
    pruner.init(X, y, sample_weight, missing_values_in_feature_mask)

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
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.X = X

    cdef int partition_samples(
        self,
        intp_t node_idx,
    ) noexcept nogil:
        """Partition samples for X at the threshold and feature index of `orig_tree`.

        If missing values are present, this method partitions `samples`
        so that the `best_n_missing` missing values' indices are in the
        right-most end of `samples`, that is `samples[end_non_missing:end]`.
        """
        cdef float64_t threshold = self.tree.nodes[node_idx].threshold
        cdef intp_t feature = self.tree.nodes[node_idx].feature
        cdef intp_t n_missing = 0
        cdef intp_t pos = self.start
        cdef intp_t p
        cdef intp_t sample_idx
        cdef intp_t current_end = self.end - n_missing
        cdef const float32_t[:, :] X_ndarray = self.X

        # partition the samples one by one by swapping them to the left or right
        # of pos depending on the feature value compared to the orig_tree threshold
        for p in range(self.start, self.end):
            sample_idx = self.samples[p]

            # missing-values are always placed at the right-most end
            if isnan(X_ndarray[sample_idx, feature]):
                self.samples[p], self.samples[current_end] = \
                    self.samples[current_end], self.samples[p]

                n_missing += 1
                current_end -= 1
            elif p > pos and X_ndarray[sample_idx, feature] <= threshold:
                self.samples[p], self.samples[pos] = \
                    self.samples[pos], self.samples[p]
                pos += 1

        # this is the split point for left/right children
        self.pos = pos
        self.n_missing = n_missing
        self.missing_go_to_left = self.tree.nodes[node_idx].missing_go_to_left

    cdef bint check_node_partition_conditions(
        self,
        SplitRecord* current_split,
        float64_t lower_bound,
        float64_t upper_bound
    ) noexcept nogil:
        """Check that the current node satisfies paritioning conditions.

        Parameters
        ----------
        current_split : SplitRecord pointer
            A pointer to a memory-allocated SplitRecord object which will be filled with the
            split chosen.
        """
        # update the criterion if we are checking split conditions
        self.criterion.init_missing(self.n_missing)
        self.criterion.reset()
        self.criterion.update(self.pos)

        current_split.pos = self.pos
        current_split.n_missing = self.n_missing
        current_split.missing_go_to_left = self.missing_go_to_left

        # first check the presplit conditions
        cdef bint invalid_split = self.check_presplit_conditions(
            current_split,
            self.n_missing,
            self.missing_go_to_left
        )

        if invalid_split:
            return 0

        # TODO: make work with lower/upper bound. This will require passing
        # lower/upper bound from the parent node into check_node_partition_conditions
        # Reject if monotonicity constraints are not satisfied
        if (
            self.with_monotonic_cst and
            self.monotonic_cst[current_split.feature] != 0 and
            not self.criterion.check_monotonicity(
                self.monotonic_cst[current_split.feature],
                lower_bound,
                upper_bound,
            )
        ):
            return 0

        # Note this is called after pre-split condition checks
        # shift missing values to left if required, so we can check
        # the split conditions
        shift_missing_values_to_left_if_required(
            current_split,
            self.samples,
            self.end
        )

        # next check the postsplit conditions that leverages the criterion
        invalid_split = self.check_postsplit_conditions()
        return invalid_split

    cdef inline intp_t n_left_samples(
        self
    ) noexcept nogil:
        """Number of samples to send to the left child."""
        cdef intp_t n_left

        if self.missing_go_to_left:
            n_left = self.pos - self.start + self.n_missing
        else:
            n_left = self.pos - self.start
        return n_left

    cdef inline intp_t n_right_samples(
        self
    ) noexcept nogil:
        """Number of samples to send to the right child."""
        cdef intp_t n_right
        cdef intp_t end_non_missing = self.end - self.n_missing
        if self.missing_go_to_left:
            n_right = end_non_missing - self.pos
        else:
            n_right = end_non_missing - self.pos + self.n_missing
        return n_right

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        """Split nodes using the already constructed tree.

        This is a simpler version of splitting nodes during the construction of a tree.
        Here, we only need to split the samples in the node based on the feature and
        threshold of the node in the original tree. In addition, we track the relevant
        information from the parent node, such as lower/upper bounds, and the parent's
        impurity, and n_constant_features.

        Returns 0 if a split cannot be done, 1 if a split can be done
        and -1 in case of failure to allocate memory (and raise MemoryError).
        """
        pass


cdef _honest_prune(
    uint8_t[:] leaves_in_subtree,
    Tree orig_tree,
    HonestPruner pruner,
):
    """Perform honest pruning of the tree.

    Iterates through the original tree in a BFS fashion using the pruner
    and tracks at each node (orig_node):

    - the number of samples in the node
    - the number of samples that would be sent to the left and right children

    Until one of three stopping conditions are met:

    1. The orig_node is a leaf node in the original tree.
        Thus we keep the node as a leaf in the pruned tree.
    2. The orig_node is a non-leaf node and the split is degenerate.
        Thus we would prune the subtree, and assign orig_node as a leaf
        node in the pruned tree.
    3. Stopping criterion is met based on the samples that reach the node.
        These are the stopping conditions implemented in a Splitter/Pruner.
        Thus we would prune the subtree, and assign orig_node as a leaf
        node in the pruned tree.

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
        # get the left child, right child and parents of every node
        intp_t[:] child_l = orig_tree.children_left
        intp_t[:] child_r = orig_tree.children_right
        # intp_t[:] parents = np.zeros(shape=n_nodes, dtype=np.intp)

        # stack to keep track of the nodes to be pruned such that BFS is done
        stack[PruningRecord] pruning_stack
        PruningRecord stack_record

        intp_t node_idx
        SplitRecord* split_ptr = <SplitRecord *>malloc(pruner.pointer_size())

        bint is_leaf_in_origtree
        bint invalid_split
        bint split_is_degenerate

        intp_t start = 0
        intp_t end = 0
        float64_t weighted_n_node_samples

        float64_t lower_bound, upper_bound
        float64_t left_child_min, left_child_max, right_child_min, right_child_max, middle_value

    cdef bint first = 0
    cdef ParentInfo parent_record
    _init_parent_record(&parent_record)

    # find parent node ids and leaves
    with nogil:
        # Push the root node
        pruning_stack.push({
            "node_idx": 0,
            "start": 0,
            "end": pruner.n_samples,
            "impurity": INFINITY,
            "lower_bound": -INFINITY,
            "upper_bound": INFINITY,
        })

        # Note: this DFS building strategy differs from scikit-learn in that
        # we check stopping conditions (and leaf candidacy) after a split occurs.
        # If we don't hit a leaf, then we will add the children to the stack, but otherwise
        # we will halt the split, and mark the node to be a new leaf node in the pruned tree.
        while not pruning_stack.empty():
            stack_record = pruning_stack.top()
            pruning_stack.pop()
            start = stack_record.start
            end = stack_record.end
            parent_record.impurity = stack_record.impurity
            parent_record.lower_bound = stack_record.lower_bound
            parent_record.upper_bound = stack_record.upper_bound
            lower_bound = stack_record.lower_bound
            upper_bound = stack_record.upper_bound

            # node index of actual node within the orig_tree
            node_idx = stack_record.node_idx

            # reset which samples indices are considered at this split node
            pruner.node_reset(start, end, &weighted_n_node_samples)

            # get the impurity to initialize passing into its children
            if first:
                parent_record.impurity = pruner.node_impurity()
                first = 0

            # partition samples into left/right child based on the
            # current node split in the orig_tree
            pruner.partition_samples(node_idx)

            # check end conditions
            split_ptr.feature = orig_tree.nodes[node_idx].feature
            invalid_split = pruner.check_node_partition_conditions(
                split_ptr,
                lower_bound,
                upper_bound
            )
            split_is_degenerate = (
                pruner.n_left_samples() == 0 or pruner.n_right_samples() == 0
            )
            is_leaf_in_origtree = child_l[node_idx] == _TREE_LEAF
            if invalid_split or split_is_degenerate or is_leaf_in_origtree:
                # ... and child_r[node_idx] == _TREE_LEAF:
                #
                # 1) if node is not degenerate, that means there are still honest-samples in
                # both left/right children of the proposed split, or the node itself is a leaf
                # or 2) there are still nodes to split on, but the honest-samples have been
                # used up so the "parent" should be the new leaf
                leaves_in_subtree[node_idx] = 1
            else:
                if (
                    not pruner.with_monotonic_cst or
                    pruner.monotonic_cst[split_ptr.feature] == 0
                ):
                    # Split on a feature with no monotonicity constraint

                    # Current bounds must always be propagated to both children.
                    # If a monotonic constraint is active, bounds are used in
                    # node value clipping.
                    left_child_min = right_child_min = parent_record.lower_bound
                    left_child_max = right_child_max = parent_record.upper_bound
                elif pruner.monotonic_cst[split_ptr.feature] == 1:
                    # Split on a feature with monotonic increase constraint
                    left_child_min = parent_record.lower_bound
                    right_child_max = parent_record.upper_bound

                    # Lower bound for right child and upper bound for left child
                    # are set to the same value.
                    middle_value = pruner.criterion.middle_value()
                    right_child_min = middle_value
                    left_child_max = middle_value
                else:  # i.e. pruner.monotonic_cst[split.feature] == -1
                    # Split on a feature with monotonic decrease constraint
                    right_child_min = parent_record.lower_bound
                    left_child_max = parent_record.upper_bound

                    # Lower bound for left child and upper bound for right child
                    # are set to the same value.
                    middle_value = pruner.criterion.middle_value()
                    left_child_min = middle_value
                    right_child_max = middle_value

                pruning_stack.push({
                    "node_idx": child_l[node_idx],
                    "start": pruner.start,
                    "end": pruner.pos,
                    "impurity": split_ptr.impurity_left,
                    "lower_bound": left_child_min,
                    "upper_bound": left_child_max,
                })
                pruning_stack.push({
                    "node_idx": child_r[node_idx],
                    "start": pruner.pos,
                    "end": pruner.end,
                    "impurity": split_ptr.impurity_right,
                    "lower_bound": right_child_min,
                    "upper_bound": right_child_max,
                })

    # free the memory created for the SplitRecord pointer
    free(split_ptr)


from libc.stdint cimport INTPTR_MAX
from libc.string cimport memcpy


cdef struct BuildPrunedRecord:
    intp_t start
    intp_t depth
    intp_t parent
    bint is_left


cdef _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const uint8_t[:] leaves_in_subtree,
    intp_t capacity
):
    """Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : uint8_t memoryview, shape=(node_count, )
        Boolean mask for leaves to include in subtree
    capacity : intp_t
        Number of nodes to initially allocate in pruned tree
    """
    tree._resize(capacity)

    cdef:
        intp_t orig_node_id
        intp_t new_node_id
        intp_t depth
        intp_t parent
        bint is_left
        bint is_leaf

        # value_stride for original tree and new tree are the same
        intp_t value_stride = orig_tree.value_stride
        intp_t max_depth_seen = -1
        intp_t rc = 0
        Node* node
        float64_t* orig_value_ptr
        float64_t* new_value_ptr

        stack[BuildPrunedRecord] prune_stack
        BuildPrunedRecord stack_record

        SplitRecord split

    with nogil:
        # push root node onto stack
        prune_stack.push({"start": 0, "depth": 0, "parent": _TREE_UNDEFINED, "is_left": 0})

        while not prune_stack.empty():
            stack_record = prune_stack.top()
            prune_stack.pop()

            orig_node_id = stack_record.start
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left

            is_leaf = leaves_in_subtree[orig_node_id]
            node = &orig_tree.nodes[orig_node_id]

            # redefine to a SplitRecord to pass into _add_node
            split.feature = node.feature
            split.threshold = node.threshold

            # protect against an infinite loop as a runtime error, when leaves_in_subtree
            # are improperly set where a node is not marked as a leaf, but is a node
            # in the original tree. Thus, it violates the assumption that the node
            # is a leaf in the pruned tree, or has a descendant that will be pruned.
            if (not is_leaf and node.left_child == _TREE_LEAF
                    and node.right_child == _TREE_LEAF):
                raise ValueError(
                    "Node has reached a leaf in the original tree, but is not "
                    "marked as a leaf in the leaves_in_subtree mask."
                )

            new_node_id = tree._add_node(
                parent, is_left, is_leaf, &split,
                node.impurity, node.n_node_samples,
                node.weighted_n_node_samples, node.missing_go_to_left)

            if new_node_id == INTPTR_MAX:
                rc = -1
                break

            # copy value from original tree to new tree
            orig_value_ptr = orig_tree.value + value_stride * orig_node_id
            new_value_ptr = tree.value + value_stride * new_node_id
            memcpy(new_value_ptr, orig_value_ptr, sizeof(float64_t) * value_stride)

            if not is_leaf:
                # Push right child on stack
                prune_stack.push({"start": node.right_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 0})
                # push left child on stack
                prune_stack.push({"start": node.left_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 1})

            if depth > max_depth_seen:
                max_depth_seen = depth

        if rc >= 0:
            tree.max_depth = max_depth_seen
    if rc == -1:
        raise MemoryError("pruning tree")

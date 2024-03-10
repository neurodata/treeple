from libcpp.numeric cimport accumulate


cdef class MultiViewSplitter(Splitter):
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const cnp.int8_t[:] monotonic_cst,
        const intp_t[:] feature_set_ends,
        intp_t n_feature_sets,
        const intp_t[:] max_features_per_set,
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

        monotonic_cst : const cnp.int8_t[:]
            Monotonicity constraints

        """
        self.feature_set_ends = feature_set_ends

        # infer the number of feature sets
        self.n_feature_sets = n_feature_sets

        # replaces usage of max_features
        self.max_features_per_set = max_features_per_set

        self.vec_n_visited_features = np.zeros(n_feature_sets, dtype=np.intp)
        self.vec_n_drawn_constants = np.zeros(n_feature_sets, dtype=np.intp)
        self.vec_n_found_constants = np.zeros(n_feature_sets, dtype=np.intp)

    def __reduce__(self):
        return (type(self),
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.random_state,
                    self.monotonic_cst.base if self.monotonic_cst is not None else None,
                    self.feature_set_ends.base if self.feature_set_ends is not None else None,
                    self.n_feature_sets,
                    self.max_features_per_set.base if self.max_features_per_set is not None else None,
                ), self.__getstate__())


cdef class BestMultiViewSplitter(MultiViewSplitter):
    """Splitter for finding the best split on dense data."""

    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
        self,
        float64_t impurity,   # Impurity of the node
        SplitRecord* split,
        float64_t lower_bound,
        float64_t upper_bound,
    ) except -1 nogil:
        """Find the best split on node samples[start:end]

        Note: this implementation differs from scikit-learn because

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # typecast the pointer to an ObliqueSplitRecord
        cdef MultiViewSplitRecord* split = <MultiViewSplitRecord*>(split)

        # Find the best split
        cdef intp_t start = splitter.start
        cdef intp_t end = splitter.end
        cdef intp_t end_non_missing
        cdef intp_t n_missing = 0
        cdef bint has_missing = 0
        cdef intp_t n_searches
        cdef intp_t n_left, n_right
        cdef bint missing_go_to_left

        cdef intp_t[::1] samples = splitter.samples
        cdef intp_t[::1] features = splitter.features
        cdef intp_t[::1] constant_features = splitter.constant_features

        cdef float32_t[::1] feature_values = splitter.feature_values
        cdef intp_t max_features = splitter.max_features
        cdef intp_t min_samples_leaf = splitter.min_samples_leaf
        cdef float64_t min_weight_leaf = splitter.min_weight_leaf
        cdef UINT32_t* random_state = &splitter.rand_r_state

        cdef SplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY

        # pointer in the feature set
        cdef intp_t f_i

        cdef intp_t f_j
        cdef intp_t p
        cdef intp_t p_prev

        cdef intp_t ifeature
        cdef intp_t feature_set_begin = 0

        # Number of features discovered to be constant during the split search
        cdef intp_t[:] vec_n_found_constants = self.vec_n_found_constants
        # Number of features known to be constant and drawn without replacement
        cdef intp_t[:] vec_n_drawn_constants = self.vec_n_drawn_constants
        cdef intp_t[:] vec_n_visited_features = self.vec_n_visited_features

        # We reset the number of visited features, drawn constants and found constants
        # for each feature set to 0 at the beginning of the split search.
        for ifeature in range(self.n_feature_sets):
            vec_n_found_constants[ifeature] = 0
            vec_n_drawn_constants[ifeature] = 0
            vec_n_visited_features[ifeature] = 0

        cdef vector[intp_t] n_known_constants_vec = split.vec_n_constant_features
        if n_known_constants_vec.size() != self.n_feature_sets:
            with gil:
                raise ValueError(
                    "n_known_constants_vec.size() != self.n_feature_sets inside MultiViewSplitter"
                )

        # n_total_constants = n_known_constants + n_found_constants
        cdef vector[intp_t] n_total_constants_vec = n_known_constants_vec

        _init_split(&best_split, end)

        partitioner.init_node_split(start, end)

        for ifeature in range(self.n_feature_sets):
            # get the max-features for this feature-set
            max_features = self.max_features_per_set[ifeature]
            f_i = self.feature_set_ends[ifeature]

            # Sample up to max_features without replacement using a
            # Fisher-Yates-based algorithm (using the local variables `f_i` and
            # `f_j` to compute a permutation of the `features` array).
            #
            # Skip the CPU intensive evaluation of the impurity criterion for
            # features that were already detected as constant (hence not suitable
            # for good splitting) by ancestor nodes and save the information on
            # newly discovered constant features to spare computation on descendant
            # nodes.
            while ((f_i - feature_set_begin) > n_total_constants_vec[ifeature] and  # Stop early if remaining features
                                                                                    # are constant within this feature set
                    (vec_n_visited_features[ifeature] < max_features or  # At least one drawn features must be non constant
                     vec_n_visited_features[ifeature] <= vec_n_found_constants[ifeature] + vec_n_drawn_constants[ifeature])):

                vec_n_visited_features[ifeature] += 1

                # The following is loop invariant per feature set:
                # [ --- view-one ---, --- view-two --- ]
                # within each view, the features are ordered as follows:
                # [constant, known constant, newly found constant, non-constant]

                # Loop invariant: elements of features in
                # - [:n_drawn_constant[ holds drawn and known constant features;
                # - [n_drawn_constant:n_known_constant[ holds known constant
                #   features that haven't been drawn yet;
                # - [n_known_constant:n_total_constant[ holds newly found constant
                #   features;
                # - [n_total_constant:f_i[ holds features that haven't been drawn
                #   yet and aren't constant apriori.
                # - [f_i:n_features[ holds features that have been drawn
                #   and aren't constant.

                # Draw a feature at random from the feature-set
                f_j = rand_int(vec_n_drawn_constants[ifeature], f_i - vec_n_found_constants[ifeature],
                               random_state) + feature_set_begin

                # If the drawn feature is known to be constant, swap it with the
                # last known constant feature and update the number of drawn constant features
                if f_j < n_known_constants_vec[ifeature]:
                    # f_j in the interval [n_drawn_constants, n_known_constants[
                    features[vec_n_drawn_constants[ifeature]], features[f_j] = features[f_j], features[vec_n_drawn_constants[ifeature]]

                    vec_n_drawn_constants[ifeature] += 1
                    continue

                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += vec_n_found_constants[ifeature]
                # f_j in the interval [n_total_constants, f_i[
                current_split.feature = features[f_j]
                partitioner.sort_samples_and_feature_values(current_split.feature)
                n_missing = partitioner.n_missing
                end_non_missing = end - n_missing

                if (
                    # All values for this feature are missing, or
                    end_non_missing == start or
                    # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
                    feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
                ):
                    # We consider this feature constant in this case.
                    # Since finding a split among constant feature is not valuable,
                    # we do not consider this feature for splitting.
                    features[f_j], features[n_total_constants_vec[ifeature]] = features[n_total_constants_vec[ifeature]], features[f_j]

                    vec_n_found_constants[ifeature] += 1
                    n_total_constants_vec[ifeature] += 1
                    continue

                f_i -= 1
                features[f_i], features[f_j] = features[f_j], features[f_i]
                has_missing = n_missing != 0
                criterion.init_missing(n_missing)  # initialize even when n_missing == 0

                # Evaluate all splits
                # If there are missing values, then we search twice for the most optimal split.
                # The first search will have all the missing values going to the right node.
                # The second search will have all the missing values going to the left node.
                # If there are no missing values, then we search only once for the most
                # optimal split.
                n_searches = 2 if has_missing else 1

                for i in range(n_searches):
                    missing_go_to_left = i == 1
                    criterion.missing_go_to_left = missing_go_to_left
                    criterion.reset()

                    p = start

                    while p < end_non_missing:
                        partitioner.next_p(&p_prev, &p)

                        if p >= end_non_missing:
                            continue

                        current_split.pos = p

                        # Reject if monotonicity constraints are not satisfied
                        if (
                            with_monotonic_cst and
                            monotonic_cst[current_split.feature] != 0 and
                            not criterion.check_monotonicity(
                                monotonic_cst[current_split.feature],
                                lower_bound,
                                upper_bound,
                            )
                        ):
                            continue

                        # Reject if min_samples_leaf is not guaranteed
                        if missing_go_to_left:
                            n_left = current_split.pos - splitter.start + n_missing
                            n_right = end_non_missing - current_split.pos
                        else:
                            n_left = current_split.pos - splitter.start
                            n_right = end_non_missing - current_split.pos + n_missing
                        if splitter.check_presplit_conditions(&current_split, n_missing, missing_go_to_left) == 1:
                            continue

                        criterion.update(current_split.pos)

                        # Reject if monotonicity constraints are not satisfied
                        if (
                            with_monotonic_cst and
                            monotonic_cst[current_split.feature] != 0 and
                            not criterion.check_monotonicity(
                                monotonic_cst[current_split.feature],
                                lower_bound,
                                upper_bound,
                            )
                        ):
                            continue

                        # Reject if min_weight_leaf is not satisfied
                        if splitter.check_postsplit_conditions() == 1:
                            continue

                        current_proxy_improvement = criterion.proxy_impurity_improvement()

                        if current_proxy_improvement > best_proxy_improvement:
                            best_proxy_improvement = current_proxy_improvement
                            # sum of halves is used to avoid infinite value
                            current_split.threshold = (
                                feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                            )

                            if (
                                current_split.threshold == feature_values[p] or
                                current_split.threshold == INFINITY or
                                current_split.threshold == -INFINITY
                            ):
                                current_split.threshold = feature_values[p_prev]

                            current_split.n_missing = n_missing
                            if n_missing == 0:
                                current_split.missing_go_to_left = n_left > n_right
                            else:
                                current_split.missing_go_to_left = missing_go_to_left

                            best_split = current_split  # copy

                # Evaluate when there are missing values and all missing values goes
                # to the right node and non-missing values goes to the left node.
                if has_missing:
                    n_left, n_right = end - start - n_missing, n_missing
                    p = end - n_missing
                    missing_go_to_left = 0

                    if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                        criterion.missing_go_to_left = missing_go_to_left
                        criterion.update(p)

                        if not ((criterion.weighted_n_left < min_weight_leaf) or
                                (criterion.weighted_n_right < min_weight_leaf)):
                            current_proxy_improvement = criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                current_split.threshold = INFINITY
                                current_split.missing_go_to_left = missing_go_to_left
                                current_split.n_missing = n_missing
                                current_split.pos = p
                                best_split = current_split

            # update the feature_set_begin for the next iteration
            feature_set_begin = self.feature_set_ends[i_feature]

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partitioner.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing
            )
            criterion.init_missing(best_split.n_missing)
            criterion.missing_go_to_left = best_split.missing_go_to_left

            criterion.reset()
            criterion.update(best_split.pos)
            criterion.children_impurity(
                &best_split.impurity_left, &best_split.impurity_right
            )
            best_split.improvement = criterion.impurity_improvement(
                impurity,
                best_split.impurity_left,
                best_split.impurity_right
            )

            shift_missing_values_to_left_if_required(&best_split, samples, end)

        # Reorganize constant features per feature view
        feature_set_begin = 0
        for ifeature in range(self.n_feature_sets):
            # Respect invariant for constant features: the original order of
            # element in features[:n_known_constants] must be preserved for sibling
            # and child nodes
            memcpy(&features[feature_set_begin], &constant_features[feature_set_begin], sizeof(intp_t) * n_known_constants_vec[ifeature])

            # Copy newly found constant features starting from [n_known_constants:n_found_constants]
            # for ifeature in range(self.n_feature_sets):
            #     n_known_constants = n_known_constants_vec[ifeature]
            memcpy(&constant_features[n_known_constants_vec[ifeature]],
                   &features[n_known_constants_vec[ifeature]],
                   sizeof(intp_t) * vec_n_found_constants[ifeature])

            feature_set_begin = self.feature_set_ends[ifeature]

        # Return values
        split[0] = best_split
        split.n_total_constants = accumulate(
            vec_n_constant_features.begin(),
            vec_n_constant_features.end(),
            0
        )
        split.vec_n_constant_features = vec_n_constant_features
        # n_constant_features[0] = n_total_constants
        return 0

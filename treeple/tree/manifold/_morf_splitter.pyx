# cython: cdivision=True
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

cimport numpy as cnp

import numpy as np

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from ..._lib.sklearn.tree._criterion cimport Criterion
from ..._lib.sklearn.tree._tree cimport ParentInfo
from ..._lib.sklearn.tree._utils cimport rand_int
from .._sklearn_splitter cimport sort
from .._utils cimport (
    compute_vectorized_indices_from_cartesian,
    fisher_yates_shuffle,
    init_2dmemoryview,
    ravel_multi_index_cython,
    unravel_index_cython,
)


cdef float64_t INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

cdef inline void _init_split(ObliqueSplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0


cdef class PatchSplitter(BestObliqueSplitter):
    """Patch splitter.

    A convolutional 2D patch splitter.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        # store a view to the (n_samples, n_features) dataset
        BestObliqueSplitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

        # store a reshaped view of (n_samples, height, width) dataset
        self.X_reshaped = init_2dmemoryview(X, self.data_dims)
        return 0

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil:
        """ Sample the projection vector.

        This is a placeholder method.

        """
        pass

    cdef intp_t sample_top_left_seed(self) noexcept nogil:
        pass


cdef class BestPatchSplitter(PatchSplitter):
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        float64_t feature_combinations,
        const intp_t[:] min_patch_dims,
        const intp_t[:] max_patch_dims,
        const uint8_t[:] dim_contiguous,
        const intp_t[:] data_dims,
        bytes boundary,
        const float32_t[:, :] feature_weight,
        *argv
    ):
        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        # Max features = output dimensionality of projection vectors
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[float32_t]](self.max_features)
        self.proj_mat_indices = vector[vector[intp_t]](self.max_features)

        # initialize state to allow generalization to higher-dimensional tensors
        self.ndim = data_dims.shape[0]
        self.data_dims = data_dims

        # create a buffer for storing the patch dimensions sampled per projection matrix
        self.patch_sampled_size = np.zeros(data_dims.shape[0], dtype=np.intp)
        self.unraveled_patch_point = np.zeros(data_dims.shape[0], dtype=np.intp)

        # store the min and max patch dimension constraints
        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous

        # create a memoryview to store the n-dimensional indices
        self.patch_nd_indices = <memoryview> np.zeros(self.max_patch_dims[:], dtype=np.intp)

        # store random indices for discontiguous sampling if necessary
        # these are initialized here to allow fisher-yates shuffling without having
        # to re-allocate memory
        self.random_indices = vector[vector[intp_t]](self.ndim)
        for idx in range(self.ndim):
            if not self.dim_contiguous[idx]:
                self.random_indices[idx].reserve(self.max_patch_dims[idx])
                for jdx in range(self.max_patch_dims[idx]):
                    self.random_indices[idx].push_back(jdx)

        # initialize a buffer to allow for Fisher-Yates
        self._index_patch_buffer = np.zeros(np.max(self.max_patch_dims), dtype=np.intp)
        self._index_data_buffer = np.zeros(np.max(self.data_dims), dtype=np.intp)

        # whether or not to perform some discontinuous sampling
        if not all(self.dim_contiguous):
            self._discontiguous = True
        else:
            self._discontiguous = False

        self.boundary = boundary
        self.feature_weight = feature_weight

    def __reduce__(self):
        """Enable pickling the splitter."""
        return (
            type(self),
            (
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.min_weight_leaf,
                self.random_state,
                self.monotonic_cst.base if self.monotonic_cst is not None else None,
                self.feature_combinations,
                self.min_patch_dims.base if self.min_patch_dims is not None else None,
                self.max_patch_dims.base if self.max_patch_dims is not None else None,
                self.dim_contiguous.base if self.dim_contiguous is not None else None,
                self.data_dims.base if self.data_dims is not None else None,
                self.boundary,
                self.feature_weight.base if self.feature_weight is not None else None,
            ), self.__getstate__())

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        """Find the best_split split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # typecast the pointer to an ObliqueSplitRecord
        cdef ObliqueSplitRecord* oblique_split = <ObliqueSplitRecord*>(split)

        # Draw random splits and pick the best_split
        cdef intp_t[::1] samples = self.samples
        cdef intp_t start = self.start
        cdef intp_t end = self.end

        # pointer array to store feature values to split on
        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t max_features = self.max_features
        cdef intp_t min_samples_leaf = self.min_samples_leaf

        # keep track of split record for current_split node and the best_split split
        # found among the sampled projection vectors
        cdef ObliqueSplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY

        cdef float64_t impurity = parent_record.impurity

        cdef intp_t feat_i, p       # index over computed features and start/end
        cdef intp_t partition_end
        cdef float32_t temp_d         # to compute a projection feature value

        # instantiate the split records
        _init_split(&best_split, end)

        # For every vector in the projection matrix
        for feat_i in range(max_features):
            # Projection vector has no nonzeros
            if self.proj_mat_weights[feat_i].empty():
                continue

            # XXX: 'feature' is not actually used in oblique split records
            # Just indicates which split was sampled
            current_split.feature = feat_i
            
            # sample the projection vector
            self.sample_proj_vec(self.proj_mat_weights[feat_i], self.proj_mat_indices[feat_i])
            current_split.proj_vec_weights = &self.proj_mat_weights[feat_i]
            current_split.proj_vec_indices = &self.proj_mat_indices[feat_i]

            # Compute linear combination of features and then
            # sort samples according to the feature values.
            self.compute_features_over_samples(
                start,
                end,
                samples,
                feature_values,
                &self.proj_mat_weights[feat_i],
                &self.proj_mat_indices[feat_i]
            )

            # Sort the samples
            sort(&feature_values[start], &samples[start], end - start)

            # Evaluate all splits
            self.criterion.reset()
            p = start
            while p < end:
                while (p + 1 < end and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD):
                    p += 1

                p += 1

                if p < end:
                    current_split.pos = p

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current_split.pos - start) < min_samples_leaf) or
                            ((end - current_split.pos) < min_samples_leaf)):
                        continue

                    self.criterion.update(current_split.pos)
                    # Reject if min_weight_leaf is not satisfied
                    if self.check_postsplit_conditions() == 1:
                        continue

                    current_proxy_improvement = \
                        self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        # sum of halves is used to avoid infinite value
                        current_split.threshold = feature_values[p - 1] / 2.0 + feature_values[p] / 2.0

                        if (
                            (current_split.threshold == feature_values[p]) or
                            (current_split.threshold == INFINITY) or
                            (current_split.threshold == -INFINITY)
                        ):
                            current_split.threshold = feature_values[p - 1]

                        best_split = current_split  # copy

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                # Account for projection vector
                temp_d = 0.0
                for j in range(best_split.proj_vec_indices.size()):
                    temp_d += self.X[samples[p], deref(best_split.proj_vec_indices)[j]] *\
                                deref(best_split.proj_vec_weights)[j]

                if temp_d <= best_split.threshold:
                    p += 1

                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = \
                        samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(&best_split.impurity_left,
                                             &best_split.impurity_right)
            best_split.improvement = self.criterion.impurity_improvement(
                impurity, best_split.impurity_left, best_split.impurity_right)

        # Return values
        deref(oblique_split).proj_vec_indices = best_split.proj_vec_indices
        deref(oblique_split).proj_vec_weights = best_split.proj_vec_weights
        deref(oblique_split).feature = best_split.feature
        deref(oblique_split).pos = best_split.pos
        deref(oblique_split).threshold = best_split.threshold
        deref(oblique_split).improvement = best_split.improvement
        deref(oblique_split).impurity_left = best_split.impurity_left
        deref(oblique_split).impurity_right = best_split.impurity_right

        # XXX: Fix when we can track constants
        parent_record.n_constant_features = 0
        return 0
    
    cdef inline intp_t sample_top_left_seed(
        self
    ) noexcept nogil:
        """Sample the top-left seed, and patch size for the n-dim patch.

        Returns
        -------
        top_left_seed : intp_t
            The top-left seed vectorized (i.e. raveled) for the n-dim patch.
        """
        # now get the top-left seed that is used to then determine the top-left
        # position in patch
        # compute top-left seed for the multi-dimensional patch
        cdef intp_t top_left_patch_seed

        cdef uint32_t* random_state = &self.rand_r_state

        # define parameters for the random patch
        cdef intp_t patch_dim
        cdef intp_t delta_patch_dim
        cdef intp_t dim
        cdef intp_t idx
        
        # create a vector to store the unraveled patch top left seed point
        cdef vector[intp_t] unraveled_patch_point = vector[intp_t](self.ndim)

        for idx in range(self.ndim):
            # compute random patch width and height
            # Note: By constraining max patch height/width to be at least the min
            # patch height/width this ensures that the minimum value of
            # patch_height and patch_width is 1
            patch_dim = rand_int(
                self.min_patch_dims[idx],
                self.max_patch_dims[idx] + 1,
                random_state
            )

            # sample the top-left index and patch size for this dimension based on boundary effects
            if self.boundary is None:
                # compute the difference between the image dimensions and the current
                # random patch dimensions for sampling
                delta_patch_dim = (self.data_dims[idx] - patch_dim) + 1
                top_left_patch_seed = rand_int(0, delta_patch_dim, random_state)

                # write to buffer
                self.patch_sampled_size[idx] = patch_dim
            elif self.boundary == "wrap":
                # add circular boundary conditions
                delta_patch_dim = self.data_dims[idx] + 2 * (patch_dim - 1)

                # sample the top left index for this dimension
                top_left_patch_seed = rand_int(0, delta_patch_dim, random_state)

                # resample the patch dimension due to padding
                dim = top_left_patch_seed % delta_patch_dim

                # resample the patch dimension due to padding
                patch_dim = min(patch_dim, min(dim+1, self.data_dims[idx] + patch_dim - dim - 1))
                self.patch_sampled_size[idx] = patch_dim

                # TODO: make this work
                # Convert the top-left-seed value to it's appropriate index in the full image.
                top_left_patch_seed = max(0, dim - patch_dim + 1)

            # unraveled_patch_point[idx] = top_left_patch_seed
            self.unraveled_patch_point[idx] = top_left_patch_seed
        
        # top_left_patch_seed = ravel_multi_index_cython(
        #     unraveled_patch_point,
        #     self.data_dims
        # )
        top_left_patch_seed = ravel_multi_index_cython(
            self.unraveled_patch_point,
            self.data_dims
        )
        return top_left_patch_seed


    cdef inline intp_t sample_top_left_seed_v2(
        self
    ) noexcept nogil:
        """Sample the top-left seed, and patch size for the n-dim patch.

        Returns
        -------
        top_left_seed : intp_t
            The top-left seed vectorized (i.e. raveled) for the n-dim patch.
        """
        # the top-left position in the patch for each dimension
        cdef intp_t top_left_patch_seed

        cdef uint32_t* random_state = &self.rand_r_state

        # define parameters for the random patch
        cdef intp_t patch_dim
        cdef intp_t delta_patch_dim
        cdef intp_t dim
        
        # pre-allocated buffer to store the unraveled patch top left seed point
        cdef intp_t[:] unraveled_patch_point = self.unraveled_patch_point

        for dim in range(self.ndim):
            # compute random patch width and height
            # Note: By constraining max patch height/width to be at least the min
            # patch height/width this ensures that the minimum value of
            # patch_height and patch_width is 1
            patch_dim = rand_int(
                self.min_patch_dims[dim],
                self.max_patch_dims[dim] + 1,
                random_state
            )

            if not self.dim_contiguous[dim]:
                # fisher-yates shuffle the discontiguous dimension, so we can randomly sample
                # without replacement, the indices of the patch in that dimension
                fisher_yates_shuffle(self.random_indices[dim], self.random_indices[dim].size(), random_state)

            # sample the top-left index and patch size for this dimension based on boundary effects
            if self.boundary is None:
                # compute the difference between the image dimensions and the current
                # random patch dimensions for sampling
                delta_patch_dim = (self.data_dims[dim] - patch_dim) + 1
                top_left_patch_seed = rand_int(0, delta_patch_dim, random_state)

                # write to buffer
                self.patch_sampled_size[dim] = patch_dim
            elif self.boundary == "wrap":
                pass


            # now add the relevant indices for each element of the patch
            for jdx in range(patch_dim):
                if self.dim_contiguous[dim]:
                    self.patch_nd_indices[dim][jdx] = top_left_patch_seed + jdx
                else:
                    # if discontiguous, we will perform random sampling
                    self.patch_nd_indices[dim][jdx] = self.random_indices[dim][jdx]

            unraveled_patch_point[dim] = top_left_patch_seed
        
        # get the vectorized index of the top-left seed
        top_left_patch_seed = ravel_multi_index_cython(
            unraveled_patch_point,
            self.data_dims
        )
        return top_left_patch_seed

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample projection matrix using a contiguous patch.

        Randomly sample patches with weight of 1.
        """
        cdef intp_t max_features = self.max_features
        cdef intp_t proj_i

        # define parameters for vectorized points in the original data shape
        # and top-left seed
        cdef intp_t top_left_patch_seed

        for proj_i in range(0, max_features):
            # now get the top-left seed that is used to then determine the top-left
            # position in patch
            # compute top-left seed for the multi-dimensional patch
            top_left_patch_seed = self.sample_top_left_seed()

            # sample a projection vector given the top-left seed point in n-dimensional space
            self.sample_proj_vec(
                proj_mat_weights,
                proj_mat_indices,
                proj_i,
                top_left_patch_seed,
                self.patch_sampled_size
            )

    cdef void sample_proj_vec(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices,
        intp_t proj_i,
        intp_t top_left_patch_seed,
        const intp_t[:] patch_dims,
    ) noexcept nogil:
        cdef uint32_t* random_state = &self.rand_r_state
        # iterates over the size of the patch
        cdef intp_t patch_idx

        cdef intp_t i

        # size of the sampled patch, which is just the size of the n-dim patch
        # (\prod_i self.patch_sampled_size[i])
        cdef intp_t patch_size = 1
        for i in range(0, self.ndim):
            patch_size *= patch_dims[i]

        # stores how many patches we have iterated so far
        cdef intp_t vectorized_patch_offset
        cdef intp_t vectorized_point_offset
        cdef intp_t vectorized_point

        cdef intp_t dim_idx

        # weights are default to 1
        cdef float32_t weight = 1.

        # XXX: still unsure if it works yet
        # XXX: THIS ONLY WORKS FOR THE FIRST DIMENSION THAT IS DISCONTIGUOUS.
        cdef intp_t other_dims_offset
        cdef intp_t row_index

        cdef intp_t num_rows = self.data_dims[0]
        if self._discontiguous:
            # fill with values 0, 1, ..., dimension - 1
            for i in range(0, self.data_dims[0]):
                self._index_data_buffer[i] = i
            # then shuffle indices using Fisher-Yates
            for i in range(num_rows):
                j = rand_int(0, num_rows - i, random_state)
                self._index_data_buffer[i], self._index_data_buffer[j] = \
                    self._index_data_buffer[j], self._index_data_buffer[i]
            # now select the first `patch_dims[0]` indices
            for i in range(num_rows):
                self._index_patch_buffer[i] = self._index_data_buffer[i]

        for patch_idx in range(patch_size):
            # keep track of which dimensions of the patch we have iterated over
            vectorized_patch_offset = 1

            # Once the vectorized top-left-seed is unraveled, you can add the patch
            # points in the array structure and compute their vectorized (unraveled)
            # points, which are added to the projection vector
            unravel_index_cython(top_left_patch_seed, self.data_dims, self.unraveled_patch_point)

            for dim_idx in range(self.ndim):
                # compute the offset from the top-left patch seed based on:
                # 1. the current patch index
                # 2. the patch dimension indexed by `dim_idx`
                # 3. and the vectorized patch dimensions that we have seen so far
                # the `vectorized_point_offset` is the offset from the top-left vectorized seed for this dimension
                vectorized_point_offset = (patch_idx // (vectorized_patch_offset)) % patch_dims[dim_idx]

                # then we compute the actual point in the original data shape
                self.unraveled_patch_point[dim_idx] = self.unraveled_patch_point[dim_idx] + vectorized_point_offset
                vectorized_patch_offset *= patch_dims[dim_idx]

            # if any dimensions are discontiguous, we want to migrate the entire axis a fixed amount
            # based on the shuffling
            if self._discontiguous is True:
                for dim_idx in range(self.ndim):
                    if self.dim_contiguous[dim_idx] is True:
                        continue

                    # determine the "row" we are currently on
                    # other_dims_offset = 1
                    # for idx in range(dim_idx + 1, self.ndim):
                    #     other_dims_offset *= self.data_dims[idx]
                    # row_index = self.unraveled_patch_point[dim_idx] % other_dims_offset
                    # determine the "row" we are currently on
                    other_dims_offset = 1
                    for idx in range(dim_idx + 1, self.ndim):
                        if not self.dim_contiguous[idx]:
                            other_dims_offset *= self.data_dims[idx]

                    row_index = 0
                    for idx in range(dim_idx + 1, self.ndim):
                        if not self.dim_contiguous[idx]:
                            row_index += (
                                (self.unraveled_patch_point[idx] // other_dims_offset) %
                                self.patch_sampled_size[idx]
                            ) * other_dims_offset
                            other_dims_offset //= self.data_dims[idx]

                    # assign random row index now
                    self.unraveled_patch_point[dim_idx] = self._index_patch_buffer[row_index]

            # ravel the patch point into the original data dimensions
            vectorized_point = ravel_multi_index_cython(self.unraveled_patch_point, self.data_dims)
            proj_mat_indices[proj_i].push_back(vectorized_point)
            proj_mat_weights[proj_i].push_back(weight)

    cdef void sample_proj_vec_v2(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices,
        vector[intp_t]& top_left_seed,
        intp_t proj_i,
        const intp_t[:] patch_dims,
    ) noexcept nogil:
        cdef uint32_t* random_state = &self.rand_r_state
        
        # iterates over the size of the patch
        cdef intp_t patch_idx

        # stores how many patches we have iterated so far
        cdef intp_t vectorized_patch_offset
        cdef intp_t vectorized_point_offset
        cdef intp_t vectorized_point

        cdef intp_t dim_idx

        # weights are default to 1
        cdef float32_t weight = 1.

        # XXX: still unsure if it works yet
        # XXX: THIS ONLY WORKS FOR THE FIRST DIMENSION THAT IS DISCONTIGUOUS.
        cdef intp_t other_dims_offset
        cdef intp_t row_index

        cdef intp_t i
        cdef intp_t num_rows = self.data_dims[0]
        
        # iterate over the patch and store the vectorized index
        cdef vector[intp_t] index_arr = vector[intp_t](self.ndim)
        for dim in range(top_left_seed.size()):
            index_arr[dim] = top_left_seed[dim]

        # fisher_yates_shuffle(dim_index_arr, patch_dims[dim], random_state)

        # iterate over the indices of the patch, and get the resulting
        # raveled index at each patch-point, resulting in now a 1D vector
        # of indices representing each patch-point.
        while True:
            # Note: we iterate over the last axis first assuming that data
            # is stored in a C-contiguous array.
            for dim in range(self.ndim - 1, -1, -1):
                if self.dim_contiguous[dim]:
                    index_arr[dim] += 1

                    # check we have not reached the boundaries of the patch. If we have not,
                    # we can break out of the loop and continue iterating in this dimension
                    if index_arr[dim] < top_left_seed[dim] + patch_dims[dim]:
                        break

                    # if we have reached the boundary, we reset the index and continue
                    index_arr[dim] = top_left_seed[dim]
                else:
                    random_index = random_indices[dim][r_idx]
                    # dimension is assumed discontiguous, and thus is randomly chosen
                    index_arr[dim] = rand_int(0, self.data_dims[dim], random_state)

                    discontiguous_index += 1

                    if r_idx < patch_dims[dim]:
                        idx[dim] = random_indices[dim][r_idx]
                        r_idx += 1
                        break

                    r_idx = 0
                    # if we have reached the boundary, we reset the index and continue
                    index_arr[dim] = top_left_seed[dim]

                # if self.dim_contiguous[dim] and 
                    # break
                # elif not self.dim_contiguous[dim] and index > patch_dims[dim]:
                #     break
            else:
                break
            
        #     # get the vectorized version of this index
        #     # dim_idx = ravel_multi_index_cython(
        #     #     index_arr,
        #     #     self.data_dims
        #     # )

        #     proj_mat_indices[proj_i].push_back(vectorized_point)
        #     proj_mat_weights[proj_i].push_back(weight)

            # idx[dim] = start[dim]
                # TODO: iterate correctly over multi-dim array
                # for idx in range(len(self.data_dims[dim])):
                #         unvectorized_idx[dim] = top_left_seed[dim] + idx
                #     else:
                #         unvectorized_idx[dim] = top_left_seed[dim] + rand_int(0, patch_dims[dim], random_state)

                #     # get the vectorized version of this index
                #     dim_idx = ravel_multi_index_cython(
                #         unvectorized_idx,
                #         self.data_dims
                #     )

                    # proj_mat_indices[proj_i].push_back(vectorized_point)
                    # proj_mat_weights[proj_i].push_back(weight)


        # for patch_idx in range(patch_size):
        #     # keep track of which dimensions of the patch we have iterated over
        #     vectorized_patch_offset = 1

        #     # Once the vectorized top-left-seed is unraveled, you can add the patch
        #     # points in the array structure and compute their vectorized (unraveled)
        #     # points, which are added to the projection vector
        #     unravel_index_cython(top_left_patch_seed, self.data_dims, self.unraveled_patch_point)

        #     for dim_idx in range(self.ndim):
        #         # compute the offset from the top-left patch seed based on:
        #         # 1. the current patch index
        #         # 2. the patch dimension indexed by `dim_idx`
        #         # 3. and the vectorized patch dimensions that we have seen so far
        #         # the `vectorized_point_offset` is the offset from the top-left vectorized seed for this dimension
        #         vectorized_point_offset = (patch_idx // (vectorized_patch_offset)) % patch_dims[dim_idx]

        #         # then we compute the actual point in the original data shape
        #         self.unraveled_patch_point[dim_idx] = self.unraveled_patch_point[dim_idx] + vectorized_point_offset
        #         vectorized_patch_offset *= patch_dims[dim_idx]

        #     # if any dimensions are discontiguous, we want to migrate the entire axis a fixed amount
        #     # based on the shuffling
        #     if self._discontiguous is True:
        #         for dim_idx in range(self.ndim):
        #             if self.dim_contiguous[dim_idx] is True:
        #                 continue

        #             # determine the "row" we are currently on
        #             # other_dims_offset = 1
        #             # for idx in range(dim_idx + 1, self.ndim):
        #             #     other_dims_offset *= self.data_dims[idx]
        #             # row_index = self.unraveled_patch_point[dim_idx] % other_dims_offset
        #             # determine the "row" we are currently on
        #             other_dims_offset = 1
        #             for idx in range(dim_idx + 1, self.ndim):
        #                 if not self.dim_contiguous[idx]:
        #                     other_dims_offset *= self.data_dims[idx]

        #             row_index = 0
        #             for idx in range(dim_idx + 1, self.ndim):
        #                 if not self.dim_contiguous[idx]:
        #                     row_index += (
        #                         (self.unraveled_patch_point[idx] // other_dims_offset) %
        #                         self.patch_sampled_size[idx]
        #                     ) * other_dims_offset
        #                     other_dims_offset //= self.data_dims[idx]

        #             # assign random row index now
        #             self.unraveled_patch_point[dim_idx] = self._index_patch_buffer[row_index]

        #     # ravel the patch point into the original data dimensions
        #     vectorized_point = ravel_multi_index_cython(self.unraveled_patch_point, self.data_dims)
        #     proj_mat_indices[proj_i].push_back(vectorized_point)
        #     proj_mat_weights[proj_i].push_back(weight)

    cdef void compute_features_over_samples(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]* proj_vec_weights,  # weights of the vector (max_features,)
        vector[intp_t]* proj_vec_indices      # indices of the features (max_features,)
    ) noexcept nogil:
        """Compute the feature values for the samples[start:end] range.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : int
            The start index of the samples.
        end : int
            The end index of the samples.
        samples : array-like, shape (n_samples,)
            The indices of the samples.
        feature_values : array-like, shape (n_samples,)
            The pre-allocated buffer to store the feature values.
        proj_vec_weights : array-like, shape (max_features,)
            The weights of the projection vector.
        proj_vec_indices : array-like, shape (max_features,)
            The indices of the projection vector. This will only store the vectorized
            index of the top-left-seed.
        """
        cdef intp_t idx, jdx

        # initialize feature weight to normalize across patch
        cdef float32_t patch_weight

        if proj_vec_indices.size() != 1:
            with gil:
                raise ValueError("proj_vec_indices should only have one element corresponding to the top left seed.")

        # patch dimensions
        cdef intp_t[:] patch_dims = self.patch_sampled_size
        cdef vector[intp_t] top_left_seed = vector[intp_t](self.ndim)

        cdef intp_t volume_of_patch = 1
        # Calculate the total number of Cartesian products
        for i in range(self.ndim):
            volume_of_patch *= len(self.patch_nd_indices[i])

        # create a buffer to store the raveled indices of the patch, which will be used
        # to compute the relevant feature values
        cdef vector[intp_t] raveled_patch_indices = vector[intp_t](volume_of_patch)
        for jdx in range(0, proj_vec_indices.size()):
            # get the index of the top-left patch
            top_left_patch_index = deref(proj_vec_indices)[jdx]

            # compute the raveled index of all the points in the patch
            compute_vectorized_indices_from_cartesian(
                top_left_patch_index,
                self.patch_nd_indices,
                self.data_dims,
                raveled_patch_indices
            )

        # Compute linear combination of features and then
        # sort samples according to the feature values.
        for idx in range(start, end):
            patch_weight = 0.0

            # initialize the feature value to 0
            feature_values[idx] = 0
            for kdx in range(raveled_patch_indices.size()):
                feature_index = raveled_patch_indices[kdx]

                feature_values[idx] += self.X[
                    samples[idx], feature_index
                ] * deref(proj_vec_weights)[jdx]

            if self.feature_weight is not None:
                # gets the feature weight for this specific column from X
                # the default of feature_weights[i] is (1/n_features) for all i
                patch_weight += self.feature_weight[samples[idx], deref(proj_vec_indices)[jdx]]

            if self.feature_weight is not None:
                feature_values[idx] /= patch_weight


cdef class BestPatchSplitterTester(BestPatchSplitter):
    """A class to expose a Python interface for testing."""
    cpdef sample_top_left_seed_cpdef(self):
        top_left_patch_seed, patch_size = self.sample_top_left_seed()
        patch_dims = np.array(self.patch_sampled_size, dtype=np.intp)
        return top_left_patch_seed, patch_size, patch_dims

    cpdef sample_projection_vector(
        self,
        intp_t proj_i,
        intp_t patch_size,
        intp_t top_left_patch_seed,
        intp_t[:] patch_dims,
    ):
        cdef vector[vector[float32_t]] proj_mat_weights = vector[vector[float32_t]](self.max_features)
        cdef vector[vector[intp_t]] proj_mat_indices = vector[vector[intp_t]](self.max_features)
        cdef intp_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_vec(
            proj_mat_weights,
            proj_mat_indices,
            proj_i,
            top_left_patch_seed,
            patch_dims
        )

        # convert the projection matrix to something that can be used in Python
        proj_vecs = np.zeros((1, self.n_features), dtype=np.float64)
        for i in range(0, 1):
            for j in range(0, proj_mat_weights[i].size()):
                weight = proj_mat_weights[i][j]
                feat = proj_mat_indices[i][j]
                proj_vecs[i, feat] = weight
        return proj_vecs

    cpdef sample_projection_matrix_py(self):
        """Sample projection matrix using a patch.

        Used for testing purposes.

        Randomly sample patches with weight of 1.
        """
        cdef vector[vector[float32_t]] proj_mat_weights = vector[vector[float32_t]](self.max_features)
        cdef vector[vector[intp_t]] proj_mat_indices = vector[vector[intp_t]](self.max_features)
        cdef intp_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_mat(proj_mat_weights, proj_mat_indices)

        # convert the projection matrix to something that can be used in Python
        proj_vecs = np.zeros((self.max_features, self.n_features), dtype=np.float64)
        for i in range(0, self.max_features):
            for j in range(0, proj_mat_weights[i].size()):
                weight = proj_mat_weights[i][j]
                feat = proj_mat_indices[i][j]

                proj_vecs[i, feat] = weight

        return proj_vecs

    cpdef init_test(self, X, y, sample_weight, missing_values_in_feature_mask=None):
        """Initializes the state of the splitter.

        Used for testing purposes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape (n_samples,)
            Sample weights.
        missing_values_in_feature_mask : array-like, shape (n_features,)
            Whether or not a feature has missing values.
        """
        self.init(X, y, sample_weight, missing_values_in_feature_mask)

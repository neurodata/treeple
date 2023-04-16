# cython: cdivision=True

cimport numpy as cnp

cnp.import_array()

from libc.math cimport floor
from libcpp.vector cimport vector
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._utils cimport rand_int


cdef extern from 


cdef class PatchSplitter(BaseObliqueSplitter):
    """Patch splitter.

    A convolutional 2D patch splitter.
    """
    def __cinit__(
        self,
        Criterion criterion,
        SIZE_t max_features,
        SIZE_t min_samples_leaf,
        double min_weight_leaf,
        object random_state,
        SIZE_t min_patch_height,
        SIZE_t max_patch_height,
        SIZE_t min_patch_width,
        SIZE_t max_patch_width,
        SIZE_t data_height,
        SIZE_t data_width,
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

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[DTYPE_t]](self.max_features)
        self.proj_mat_indices = vector[vector[SIZE_t]](self.max_features)

        self.min_patch_height = min_patch_height
        self.max_patch_height = max_patch_height
        self.min_patch_width = min_patch_width
        self.max_patch_width = max_patch_width
        self.data_height = data_height
        self.data_width = data_width

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight
    ) except -1:
        BaseObliqueSplitter.init(self, X, y, sample_weight)

        return 0

    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples)
        self.criterion.set_sample_pointers(start, end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

        # Clear all projection vectors
        for i in range(self.max_features):
            self.proj_mat_weights[i].clear()
            self.proj_mat_indices[i].clear()
        return 0

    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil:
        """ Sample the projection vector.

        This is a placeholder method.

        """
        pass

    cdef int pointer_size(self) noexcept nogil:
        """Get size of a pointer to record for ObliqueSplitter."""

        return sizeof(ObliqueSplitRecord)


cdef class BaseDensePatchSplitter(PatchSplitter):
    # XXX: currently BaseOblique class defines this, which shouldn't be the case
    # cdef const DTYPE_t[:, :] X

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  const DOUBLE_t[:] sample_weight) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Call parent init
        PatchSplitter.init(self, X, y, sample_weight)

        self.X = X
        return 0

cdef class BestPatchSplitter(BaseDensePatchSplitter):
    def __reduce__(self):
        """Enable pickling the splitter."""
        return (
            BestPatchSplitter,
            (
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.min_weight_leaf,
                self.random_state,
                self.min_patch_height,
                self.max_patch_height,
                self.min_patch_width,
                self.max_patch_width,
                self.data_height,
                self.data_width,
            ), self.__getstate__())

    # NOTE: vectors are passed by value, so & is needed to pass by reference
    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample projection matrix using a contiguous patch.

        Randomly sample patches with weight of 1.
        """
        cdef SIZE_t max_features = self.max_features
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef int feat_i, proj_i

        # weights are default to 1
        cdef DTYPE_t weight = 1.

        cdef SIZE_t data_width = self.data_width
        cdef SIZE_t data_height = self.data_height
        cdef SIZE_t min_patch_height = self.min_patch_height
        cdef SIZE_t max_patch_height = self.max_patch_height
        cdef SIZE_t min_patch_width = self.min_patch_width
        cdef SIZE_t max_patch_width = self.max_patch_width

        # define parameters for the random patch
        cdef SIZE_t delta_width
        cdef SIZE_t delta_height
        cdef SIZE_t patch_height
        cdef SIZE_t patch_width

        # define parameters for vectorized points in the original data shape
        # and top-left seed
        cdef SIZE_t vectorized_point
        cdef SIZE_t top_left_seed
        cdef SIZE_t patch_end_seed

        for proj_i in range(0, max_features):
            # compute random patch width and height
            # Note: By constraining max patch height/width to be at least the min patch
            # height/width this ensures that the minimum value of patch_height and
            # patch_width is 1
            patch_height = rand_int(min_patch_height, max_patch_height + 1,
                                    random_state)
            patch_width = rand_int(min_patch_width, max_patch_width + 1, random_state)

            # compute the difference between the image dimensions and the current random
            # patch dimensions for sampling
            delta_width = data_width - patch_width + 1
            delta_height = data_height - patch_height + 1

            # now get the top-left seed that is used to then determine the top-left
            # position in patch
            top_left_seed = rand_int(0, delta_width * delta_height, random_state)

            # Get the end-point of the patch
            # Note: The end-point of the dataset might be less than the patch.
            # This occurs if we sample a seed point at the edge of the "image".
            # Therefore, we take the minimum between the end-point, or the last
            # index in the vectorized image.
            patch_end_seed = min(
                top_left_seed + delta_width * delta_height,
                self.n_features
            )

            for feat_i in range(top_left_seed, patch_end_seed):
                # now compute top-left point
                vectorized_point = (feat_i % delta_width) + \
                                    (data_width * <SIZE_t>floor(feat_i / delta_width))

<<<<<<< Updated upstream
                # store the non-zero indices and non-zero weights of the data
                proj_mat_indices[proj_i].push_back(vectorized_point)
                proj_mat_weights[proj_i].push_back(weight)
=======
        # stores how many patches we have iterated so far
        cdef int vectorized_patch_offset
        cdef SIZE_t vectorized_point_offset
        cdef SIZE_t vectorized_point

        cdef SIZE_t dim_idx
    
        # weights are default to 1
        cdef DTYPE_t weight = 1.

        # XXX: still unsure if it works yet
        # XXX: THIS ONLY WORKS FOR THE FIRST DIMENSION THAT IS DISCONTIGUOUS.
        cdef SIZE_t other_dims_offset
        cdef SIZE_t row_index

        cdef SIZE_t i
        cdef int num_rows = self.data_dims[0]
        if self._discontiguous:
            # fill with values 0, 1, ..., dimension - 1
            for i in range(0, self.data_dims[0]):
                self._index_data_buffer[i] = i

            shuffle()
            # then shuffle indices using Fisher-Yates
            # for i in range(num_rows):
            #     j = rand_int(0, num_rows - i, random_state)
            #     self._index_data_buffer[i], self._index_data_buffer[j] = \
            #         self._index_data_buffer[j], self._index_data_buffer[i]
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
                    other_dims_offset = 1
                    for idx in range(dim_idx + 1, self.ndim):
                        other_dims_offset *= self.data_dims[idx]
                    row_index = self.unraveled_patch_point[dim_idx] % other_dims_offset

                    # assign random row index now
                    self.unraveled_patch_point[dim_idx] = self._index_patch_buffer[row_index]

            # ravel the patch point into the original data dimensions
            vectorized_point = ravel_multi_index_cython(self.unraveled_patch_point, self.data_dims)
            proj_mat_indices[proj_i].push_back(vectorized_point)
            proj_mat_weights[proj_i].push_back(weight)

cdef class BestPatchSplitterTester(BestPatchSplitter):
    """A class to expose a Python interface for testing."""
    cpdef sample_top_left_seed_cpdef(self):
        top_left_patch_seed, patch_size = self.sample_top_left_seed()
        patch_dims = np.array(self.patch_dims_buff, dtype=np.intp)
        return top_left_patch_seed, patch_size, patch_dims

    cpdef sample_projection_vector(self,
        SIZE_t proj_i,
        SIZE_t patch_size,
        SIZE_t top_left_patch_seed,
        SIZE_t[:] patch_dims,
        ):
        cdef vector[vector[DTYPE_t]] proj_mat_weights = vector[vector[DTYPE_t]](self.max_features)
        cdef vector[vector[SIZE_t]] proj_mat_indices = vector[vector[SIZE_t]](self.max_features)
        cdef SIZE_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_vec(
            proj_mat_weights,
            proj_mat_indices,
            proj_i,
            patch_size,
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

    cpdef sample_projection_matrix(self):
        """Sample projection matrix using a patch.

        Used for testing purposes.

        Randomly sample patches with weight of 1.
        """
        cdef vector[vector[DTYPE_t]] proj_mat_weights = vector[vector[DTYPE_t]](self.max_features)
        cdef vector[vector[SIZE_t]] proj_mat_indices = vector[vector[SIZE_t]](self.max_features)
        cdef SIZE_t i, j

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

    cpdef init_test(self, X, y, sample_weight):
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
        """
        self.init(X, y, sample_weight)
>>>>>>> Stashed changes

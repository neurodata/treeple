# cython: cdivision=True
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

import numpy as np

cimport numpy as cnp

cnp.import_array()

from libcpp.vector cimport vector
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._utils cimport rand_int

from ._utils cimport ravel_multi_index_cython, unravel_index_cython


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
        SIZE_t[:] min_patch_dims,
        SIZE_t[:] max_patch_dims,
        cnp.uint8_t[::1] dim_contiguous,
        SIZE_t[:] data_dims,
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

        # initialize state to allow generalization to higher-dimensional tensors
        self.ndim = data_dims.shape[0]
        self.data_dims = data_dims

        # create a buffer for storing the patch dimensions sampled per projection matrix
        self.patch_dims_buff = np.zeros(self.ndim, dtype=np.intp)
        self.unraveled_patch_point = np.zeros(self.ndim, dtype=np.intp)

        # store the min and max patch dimension constraints
        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous

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
                self.min_patch_dims,
                self.max_patch_dims,
                self.dim_contiguous,
                self.data_dims
            ), self.__getstate__())

    cdef (SIZE_t, SIZE_t) sample_top_left_seed(self) noexcept nogil:
        # now get the top-left seed that is used to then determine the top-left
        # position in patch
        # compute top-left seed for the multi-dimensional patch
        cdef SIZE_t top_left_patch_seed = 1
        cdef SIZE_t patch_size = 1

        cdef UINT32_t* random_state = &self.rand_r_state

        # define parameters for the random patch
        cdef SIZE_t patch_dim
        cdef SIZE_t delta_patch_dim

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

            # write to buffer
            self.patch_dims_buff[idx] = patch_dim
            patch_size *= patch_dim
            
            # compute the difference between the image dimensions and the current
            # random patch dimensions for sampling
            delta_patch_dim = self.data_dims[idx] - patch_dim + 1
            top_left_patch_seed *= delta_patch_dim
        top_left_patch_seed = rand_int(0, top_left_patch_seed, random_state)
        return top_left_patch_seed, patch_size

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

        cdef int proj_i, idx, jdx
        cdef SIZE_t patch_idx
        cdef SIZE_t dim_idx
    
        # weights are default to 1
        cdef DTYPE_t weight = 1.

        # size of the sampled patch
        cdef SIZE_t patch_size

        # define parameters for vectorized points in the original data shape
        # and top-left seed
        cdef SIZE_t vectorized_point
        cdef SIZE_t top_left_patch_seed
        cdef SIZE_t vectorized_offset

        # stores how many patches we have iterated so far
        cdef int patch_dim_sofar

        for proj_i in range(0, max_features):
            # now get the top-left seed that is used to then determine the top-left
            # position in patch
            # compute top-left seed for the multi-dimensional patch
            top_left_patch_seed, patch_size = self.sample_top_left_seed()

            # push the first point onto the vector
            vectorized_point = top_left_patch_seed
            proj_mat_indices[proj_i].push_back(vectorized_point)
            proj_mat_weights[proj_i].push_back(weight)

            for patch_idx in range(patch_size):
                # keep track of which dimensions of the patch we have iterated over
                patch_dim_sofar = 1

                # Once the vectorized top-left-seed is unraveled, you can add the patch
                # points in the array structure and compute their vectorized (unraveled)
                # points, which are added to the projection vector
                unravel_index_cython(top_left_patch_seed, self.data_dims, self.unraveled_patch_point)

                for dim_idx in range(self.ndim):
                    # compute the offset from the top-left patch seed
                    vectorized_offset = (patch_idx // (patch_dim_sofar)) % self.patch_dims_buff[dim_idx]

                    # then we compute the actual point in the original data shape
                    self.unraveled_patch_point[dim_idx] = self.unraveled_patch_point[dim_idx] + vectorized_offset
                    patch_dim_sofar *= self.patch_dims_buff[dim_idx]

                # ravel the patch point into the original data dimensions
                vectorized_point = ravel_multi_index_cython(self.unraveled_patch_point, self.data_dims)
                proj_mat_indices[proj_i].push_back(vectorized_point)
                proj_mat_weights[proj_i].push_back(weight)

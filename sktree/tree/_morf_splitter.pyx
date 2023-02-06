import numpy as np

cimport numpy as cnp

cnp.import_array()

from libcpp.vector cimport vector
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._utils cimport rand_int


cdef class PatchSplitter(BaseObliqueSplitter):
    """Patch splitter.

    TBD.
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

        # create a helper array for allowing efficient Fisher-Yates
        self.indices_to_sample = np.arange(self.max_features * self.n_features,
                                           dtype=np.intp)
        return 0

    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) nogil except -1:
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
    ) nogil:
        """ Sample the projection vector.

        This is a placeholder method.

        """
        pass

    cdef int pointer_size(self) nogil:
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
                self.min_patch_height,
                self.max_patch_height,
                self.min_patch_width,
                self.max_patch_width,
                self.data_height,
                self.data_width,
                self.random_state
            ), self.__getstate__())

    # NOTE: vectors are passed by value, so & is needed to pass by reference
    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) nogil:
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
            patch_height = rand_int(min_patch_height, max_patch_height, random_state)
            patch_width = rand_int(min_patch_width, max_patch_width, random_state)

            # compute the difference between the image dimensions and the current random
            # patch dimensions for sampling
            delta_width = data_width - patch_width + 1
            delta_height = data_height - patch_height + 1

            # now get the top-left seed that is used to then determine the top-left
            # position in patch
            top_left_seed = rand_int(0, delta_width * delta_height, random_state)

            # now we want to set the weights and indices for the patch in the
            # vectorized position
            patch_end_seed = top_left_seed + delta_width * delta_height

            for feat_i in range(top_left_seed, patch_end_seed):
                # now compute top-left point
                vectorized_point = (feat_i % delta_width) + \
                                    (data_width * feat_i // delta_width)

                # store the non-zero indices and non-zero weights of the data
                proj_mat_indices[proj_i].push_back(vectorized_point)
                proj_mat_weights[proj_i].push_back(weight)

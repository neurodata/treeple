

cdef class DictionarySplitter(PatchSplitter):
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
        object filter_dictionary,
        *argv
    ):
        # initialize state to allow generalization to higher-dimensional tensors
        self.ndim = data_dims.shape[0]
        self.data_dims = data_dims

        # create a buffer for storing the patch dimensions sampled per projection matrix
        self.patch_dims_buff = np.zeros(data_dims.shape[0], dtype=np.intp)
        self.unraveled_patch_point = np.zeros(data_dims.shape[0], dtype=np.intp)

        # store the min and max patch dimension constraints
        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous

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

        # extract the filter dictionary
        self.n_filters = filter_dictionary.shape[0]
        self.kernel_data = filter_dictionary.data
        self.kernel_indices = filter_dictionary.indices
        self.kernel_indptr = filter_dictionary.indptr

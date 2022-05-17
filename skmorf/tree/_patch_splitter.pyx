#distutils: language=c++
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True


# TODO: implement the sampling procedure for each of the MORF splitters
cdef class ImagePatchSplitter(ObliqueSplitter):
    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  double feature_combinations, 
                  SIZE_t min_patch_height, SIZE_t max_patch_height,
                  SIZE_t min_patch_width, SIZE_t max_patch_width,
                  SIZE_t image_height, SIZE_t image_width,
                  object random_state):

        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.sample_weight = NULL

        # Max features = output dimensionality of projection vectors
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        # Oblique tree parameters
        self.feature_combinations = feature_combinations
        self.image_width = image_width
        self.image_height = image_height
        self.min_patch_height = min_patch_height
        self.max_patch_height = max_patch_height
        self.min_patch_width = min_patch_width
        self.max_patch_width = max_patch_width

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[DTYPE_t]](self.max_features)
        self.proj_mat_indices = vector[vector[SIZE_t]](self.max_features)

    cdef void sample_proj_vec(self, 
                              vector[DTYPE_t]& proj_mat_weights,
                              vector[SIZE_t]& proj_mat_indices) nogil:
        cdef UINT32_t* random_state = &self.rand_r_state
        cdef SIZE_t image_height = self.image_height
        cdef SIZE_t image_width = self.image_width
        cdef bint wrap = False

        # get a patch height and width randomly
        cdef SIZE_t patch_height = rand_int(self.min_patch_height, self.max_patch_height, random_state)
        cdef SIZE_t patch_width = rand_int(self.min_patch_width, self.max_patch_width, random_state)
        
        # sample the top left position of the patch
        cdef SIZE_t delta_width, delta_height
        cdef SIZE_t top_left_seed, col, row
        cdef SIZE_t top_left_position
        if wrap:
            delta_width = image_height - patch_height + 1
            delta_height = image_width - patch_width + 1

            top_left_position = rand_int(0, delta_width*delta_height, random_state)
        else:
            delta_width = image_height + 2 * (patch_height - 1)
            delta_height = image_width + 2 * (patch_width - 1)
            top_left_seed = rand_int(0, delta_height * delta_width)
            col = top_left_seed % delta_width
            row = top_left_seed / delta_width


        cdef int irow, jcol
        cdef SIZE_t pixel_index
        
        # in this case, we use the summation operator, so
        # all weights are 1
        cdef SIZE_t weight = 1

        for irow in range(0, patch_height):
            for jcol in range(0, patch_width):
                # get the top left index
                pixel_index = top_left_position + jcol + (image_width * irow)

                proj_mat_indices.push_back(pixel_index)
                proj_mat_weights.push_back(weight)

cdef class MtsPatchSplitter(ObliqueSplitter):
    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  double feature_combinations, 
                  SIZE_t min_patch_signals, SIZE_t max_patch_signals,
                  SIZE_t min_patch_time, SIZE_t max_patch_time,
                  SIZE_t n_signals, SIZE_t n_time_points,
                  object random_state):
        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.sample_weight = NULL

        # Max features = output dimensionality of projection vectors
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        # Oblique tree parameters
        self.feature_combinations = feature_combinations
        self.n_time_points = n_time_points
        self.n_signals = n_signals
        self.min_patch_signals = min_patch_signals
        self.max_patch_signals = max_patch_signals
        self.min_patch_time = min_patch_time
        self.max_patch_time = max_patch_time

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[DTYPE_t]](self.max_features)
        self.proj_mat_indices = vector[vector[SIZE_t]](self.max_features)

    cdef void sample_proj_vec(self, 
                              vector[DTYPE_t]& proj_mat_weights,
                              vector[SIZE_t]& proj_mat_indices) nogil:
        cdef UINT32_t* random_state = &self.rand_r_state
        cdef SIZE_t image_height = self.image_height
        cdef SIZE_t image_width = self.image_width
        cdef bint wrap = False

        # get a patch height and width randomly
        cdef SIZE_t patch_height = rand_int(self.min_patch_height, self.max_patch_height, random_state)
        cdef SIZE_t patch_width = rand_int(self.min_patch_width, self.max_patch_width, random_state)
        
        # sample the top left position of the patch
        cdef SIZE_t delta_width, delta_height
        cdef SIZE_t top_left_seed, col, row
        cdef SIZE_t top_left_position
        if wrap:
            delta_width = image_height - patch_height + 1
            delta_height = image_width - patch_width + 1

            top_left_position = rand_int(0, delta_width*delta_height, random_state)
        else:
            delta_width = image_height + 2 * (patch_height - 1)
            delta_height = image_width + 2 * (patch_width - 1)
            top_left_seed = rand_int(0, delta_height * delta_width)
            col = top_left_seed % delta_width
            row = top_left_seed / delta_width


        cdef int irow, jcol
        cdef SIZE_t pixel_index
        
        # in this case, we use the summation operator, so
        # all weights are 1
        cdef SIZE_t weight = 1

        for irow in range(0, patch_height):
            for jcol in range(0, patch_width):
                # get the top left index
                pixel_index = top_left_position + jcol + (image_width * irow)

                proj_mat_indices.push_back(pixel_index)
                proj_mat_weights.push_back(weight)
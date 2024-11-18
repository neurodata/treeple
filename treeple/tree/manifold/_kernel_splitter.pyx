import numpy as np

cimport numpy as cnp
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from ..._lib.sklearn.tree._criterion cimport Criterion
from ..._lib.sklearn.tree._utils cimport RAND_R_MAX, rand_int
from ..._lib.sklearn.utils._typedefs cimport float32_t, int32_t
from .._utils cimport ravel_multi_index_cython, unravel_index_cython

ctypedef float32_t* float32_t_ptr
ctypedef intp_t* intp_t_ptr


cdef class Kernel2D:
    cdef const intp_t[:] kernels_size
    cdef const float32_t[:] kernel_data
    cdef const int32_t[:] kernel_indices
    cdef const int32_t[:] kernel_indptr
    cdef uint32_t rand_r_state           # sklearn_rand_r random number state
    cdef object random_state             # Random state

    cdef intp_t n_dims_largest
    cdef intp_t n_kernels

    def __cinit__(
        self,
        object kernel_matrix,
        cnp.ndarray kernel_sizes,
        object random_state,
        *argv
    ):
        """
        Initialize the class with a CSR sparse matrix and a 1D array of kernel sizes.
        """
        self.kernels_size = kernel_sizes
        self.kernel_data = kernel_matrix.data
        self.kernel_indices = kernel_matrix.indices
        self.kernel_indptr = kernel_matrix.indptr
        self.n_kernels, self.n_dims_largest = kernel_matrix.shape
        self.random_state = random_state
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)

    cdef inline float32_t apply_kernel(
        self,
        float32_t[:, ::1] image,
        intp_t kernel_idx
    ) noexcept nogil: 
        """
        Apply a kernel to a random point on a 2D image and return the resulting value.
        
        Parameters:
        -----------
        image : double[:, :]
            The 2D image array (image_height, image_width).
        kernel_idx : intp_t
            The index of the kernel to apply.
        
        Returns:
        --------
        result : double
            The resulting value after applying the kernel.
        """
        cdef uint32_t* random_state = &self.rand_r_state
        cdef intp_t img_height = image.shape[0]
        cdef intp_t img_width = image.shape[1]
        cdef intp_t size = self.kernels_size[kernel_idx]
        
        # Ensure the kernel fits within the image dimensions
        cdef intp_t max_x = img_width - 1
        cdef intp_t max_y = img_height - 1
        
        # Sample a random top-left point to apply the kernel
        cdef intp_t start_x = rand_int(0, max_x + 1, random_state)
        cdef intp_t start_y = rand_int(0, max_y + 1, random_state)
        
        # Compute the result
        cdef float32_t result = 0.0
        cdef intp_t idx, row, col
        cdef float32_t image_value, kernel_value
        
        cdef intp_t start = self.kernel_indptr[kernel_idx]
        cdef intp_t end = self.kernel_indptr[kernel_idx + 1]
        
        for idx in range(start, end):
            row = self.kernel_indices[idx] // size
            col = self.kernel_indices[idx] % size
            
            # Only process the kernel if it's within the image bounds
            if image_value and kernel_value and (start_y + row <= max_y) and (start_x + col <= max_x):
                image_value = image[start_y + row, start_x + col]
                kernel_value = self.kernel_data[idx]
                result += image_value * kernel_value
        
        return result

    def apply_kernel_py(self, float32_t[:, ::1] image, intp_t kernel_idx):
        """
        Python-accessible wrapper for apply_kernel_nogil.
        """
        return self.apply_kernel(image, kernel_idx)




# cdef class UserKernelSplitter(PatchSplitter):
#     def __cinit__(
#         self,
#         criterion: Criterion,
#         max_features: intp_t,
#         min_samples_leaf: intp_t,
#         min_weight_leaf: double,
#         random_state: object,
#         min_patch_dims: intp_t,
#         max_patch_dims: intp_t,
#         dim_contiguous: cnp.uint8_t,
#         data_dims: intp_t,
#         boundary: str,
#         feature_weight: float32_t,
#         kernel_dictionary: Kernel2D,
#         *argv
#     ):
#         # initialize the kernel dictionary into a vector to allow Cython to see it
#         # see: https://stackoverflow.com/questions/46240207/passing-list-of-numpy-arrays-to-c-using-cython
#         cdef intp_t n_arrays = len(kernel_dictionary)
#         self.kernel_dictionary = vector[float32_t_ptr](n_arrays)  # A list of C-contiguous 2D kernels
#         self.kernel_dims = vector[intp_t_ptr](n_arrays)         # A list of arrays storing the dimensions of each kernel in `kernel_dictionary`

#         # buffers to point to each element in the list
#         cdef float32_t[:] kernel
#         cdef intp_t[:] kernel_dim

#         cdef intp_t i
#         for i in range(n_arrays):
#             kernel = kernel_dictionary[i]
#             kernel_dim = kernel_dims[i]

#             # store a pointer to the data
#             self.kernel_dictionary.push_back(&kernel[0])
#             self.kernel_dims.push_back(&kernel_dim[0])

#     cdef inline void compute_features_over_samples(
#         self,
#         intp_t start,
#         intp_t end,
#         const intp_t[:] samples,
#         float32_t[:] feature_values,
#         vector[float32_t]* proj_vec_weights,  # weights of the vector (max_features,)
#         vector[intp_t]* proj_vec_indices      # indices of the features (max_features,)
#     ) noexcept nogil:
#         """Compute the feature values for the samples[start:end] range.

#         Returns -1 in case of failure to allocate memory (and raise MemoryError)
#         or 0 otherwise.
#         """
#         cdef intp_t idx, jdx
#         cdef intp_t col_idx
#         cdef float32_t col_weight

#         # Compute linear combination of features and then
#         # sort samples according to the feature values.
#         for jdx in range(0, proj_vec_indices.size()):
#             col_idx = deref(proj_vec_indices)[jdx]
#             col_weight = deref(proj_vec_weights)[jdx]

#             for idx in range(start, end):
#                 # initialize the feature value to 0
#                 if jdx == 0:
#                     feature_values[idx] = 0.0
#                 feature_values[idx] += self.X[samples[idx], col_idx] * col_weight


#     cdef void sample_proj_mat(
#         self,
#         vector[vector[float32_t]]& proj_mat_weights,
#         vector[vector[intp_t]]& proj_mat_indices
#     ) noexcept nogil:
#         """Sample projection matrix using a contiguous patch.
#         Randomly sample patches with weight of 1.
#         """
#         cdef intp_t max_features = self.max_features
#         cdef intp_t proj_i

#         # define parameters for vectorized points in the original data shape
#         # and top-left seed
#         cdef intp_t top_left_patch_seed

#         # size of the sampled patch, which is just the size of the n-dim patch
#         # (\prod_i self.patch_dims_buff[i])
#         cdef intp_t patch_size

#         cdef float32_t[:] kernel
#         cdef intp_t[:] kernel_dim

#         for proj_i in range(0, max_features):
#             # now get the top-left seed that is used to then determine the top-left
#             # position in patch
#             # compute top-left seed for the multi-dimensional patch
#             top_left_patch_seed, patch_size = self.sample_top_left_seed()

#             # sample a random index in the kernel library
#             kernel_idx = rand_int(0, self.kernel_dictionary.size(), &self.rand_r_state)

#             # get that kernel and add it to the projection vector indices and weights
#             kernel = self.kernel_dictionary[kernel_idx]
#             kernel_dim = self.kernel_dims[kernel_idx]

#             # convert top-left-patch-seed to unraveled indices
#             # get the top-left index in the original data
#             top_left_idx = self.unravel_index(top_left_patch_seed, self.data_dims_buff, self.ndim)

#             # loop over the kernel and add the weights and indices to the projection
#             for idim in range(self.ndim):
#                 # get the dimension of the kernel
#                 kernel_dim = self.kernel_dims[kernel_idx][idim]

#                 # get the top-left index in the kernel
#                 top_left_kernel_idx = self.unravel_index(top_left_patch_seed, kernel_dim, self.ndim)

                # loop over the kernel and add the weights and indices to the projection
                # for i in range(kernel_dim):
                #     # get the index in the original data
                #     idx = self.ravel_multi_index(top_left_idx, self.data_dims_buff, self.ndim)

                #     # get the index in the kernel
                #     kernel_idx = self.ravel_multi_index(top_left_kernel_idx, kernel_dim, self.ndim)

                #     # add the weight and index to the projection matrix
                #     proj_mat_weights[proj_i].push_back(kernel[kernel_idx])
                #     proj_mat_indices[proj_i].push_back(idx)

                #     # increment the top-left index in the original data
                #     top_left_idx[idim] += 1

                #     # increment the top-left index in the kernel
                #     top_left_kernel_idx[idim] += 1

                # # increment the top-left index in the original data
                # top_left_idx[idim] += self.patch_dims_buff[idim] - kernel_dim

                # # increment the top-left index in the kernel
                # top_left_kernel_idx[idim] += self.patch_dims_buff[idim] - kernel_dim


# cdef class UserKernelSplitter(PatchSplitter):
#     def __cinit__(
#         self,
#         criterion: Criterion,
#         max_features: intp_t,
#         min_samples_leaf: intp_t,
#         min_weight_leaf: double,
#         random_state: object,
#         min_patch_dims: intp_t,
#         max_patch_dims: intp_t,
#         dim_contiguous: cnp.uint8_t,
#         data_dims: intp_t,
#         boundary: str,
#         feature_weight: float32_t,
#         kernel_dictionary: Kernel2D,
#         *argv
#     ):
#         # initialize the kernel dictionary into a vector to allow Cython to see it
#         # see: https://stackoverflow.com/questions/46240207/passing-list-of-numpy-arrays-to-c-using-cython
#         cdef intp_t n_arrays = len(kernel_dictionary)
#         self.kernel_dictionary = vector[float32_t_ptr](n_arrays)  # A list of C-contiguous 2D kernels
#         self.kernel_dims = vector[intp_t_ptr](n_arrays)         # A list of arrays storing the dimensions of each kernel in `kernel_dictionary`

#         # buffers to point to each element in the list
#         cdef float32_t[:] kernel
#         cdef intp_t[:] kernel_dim

#         cdef intp_t i
#         for i in range(n_arrays):
#             kernel = kernel_dictionary[i]
#             kernel_dim = kernel_dims[i]

#             # store a pointer to the data
#             self.kernel_dictionary.push_back(&kernel[0])
#             self.kernel_dims.push_back(&kernel_dim[0])

#     cdef void sample_proj_mat(
#         self,
#         vector[vector[float32_t]]& proj_mat_weights,
#         vector[vector[intp_t]]& proj_mat_indices
#     ) noexcept nogil:
#         """Sample projection matrix using a contiguous patch.
#         Randomly sample patches with weight of 1.
#         """
#         cdef intp_t max_features = self.max_features
#         cdef intp_t proj_i

#         # define parameters for vectorized points in the original data shape
#         # and top-left seed
#         cdef intp_t top_left_patch_seed

#         # size of the sampled patch, which is just the size of the n-dim patch
#         # (\prod_i self.patch_dims_buff[i])
#         cdef intp_t patch_size

#         cdef float32_t[:] kernel
#         cdef intp_t[:] kernel_dim

#         for proj_i in range(0, max_features):
#             # now get the top-left seed that is used to then determine the top-left
#             # position in patch
#             # compute top-left seed for the multi-dimensional patch
#             top_left_patch_seed, patch_size = self.sample_top_left_seed()

#             # sample a random index in the kernel library
#             # kernel_idx = 

#             # get that kernel and add it to the projection vector indices and weights
#             kernel = self.kernel_dictionary[kernel_idx]
#             kernel_dim = self.kernel_dims[kernel_idx]

#             # convert top-left-patch-seed to unraveled indices
#             # get the top-left index in the original data
#             top_left_idx = self.unravel_index(top_left_patch_seed, self.data_dims_buff, self.ndim)

#             # loop over the kernel and add the weights and indices to the projection
#             for idim in range(self.ndim):
#                 # get the dimension of the kernel
#                 kernel_dim = self.kernel_dims[kernel_idx][idim]

#                 # get the top-left index in the kernel
#                 top_left_kernel_idx = self.unravel_index(top_left_patch_seed, kernel_dim, self.ndim)

#                 # loop over the kernel and add the weights and indices to the projection
#                 # for i in range(kernel_dim):
#                 #     # get the index in the original data
#                 #     idx = self.ravel_multi_index(top_left_idx, self.data_dims_buff, self.ndim)

#                 #     # get the index in the kernel
#                 #     kernel_idx = self.ravel_multi_index(top_left_kernel_idx, kernel_dim, self.ndim)

#                 #     # add the weight and index to the projection matrix
#                 #     proj_mat_weights[proj_i].push_back(kernel[kernel_idx])
#                 #     proj_mat_indices[proj_i].push_back(idx)

#                 #     # increment the top-left index in the original data
#                 #     top_left_idx[idim] += 1

#                 #     # increment the top-left index in the kernel
#                 #     top_left_kernel_idx[idim] += 1

#                 # # increment the top-left index in the original data
#                 # top_left_idx[idim] += self.patch_dims_buff[idim] - kernel_dim

#                 # # increment the top-left index in the kernel
#                 # top_left_kernel_idx[idim] += self.patch_dims_buff[idim] - kernel_dim

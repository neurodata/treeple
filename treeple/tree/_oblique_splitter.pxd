# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#          Chester Huynh <chester.huynh924@gmail.com>
#          Parth Vora <pvora4@jhu.edu>
#
# License: BSD 3 clause

# See _oblique_splitter.pyx for details.

import numpy as np

from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._tree cimport ParentInfo
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint8_t, uint32_t
from ._sklearn_splitter cimport sort


cdef struct ObliqueSplitRecord:
    # Data to track sample split
    intp_t feature              # Which feature to split on.
    intp_t pos                  # Split samples array at the given position,
    #                           # i.e. count of samples below threshold for feature.
    #                           # pos is >= end if the node is a leaf.
    float64_t threshold            # Threshold to split at.
    float64_t improvement          # Impurity improvement given parent node.
    float64_t impurity_left        # Impurity of the left split.
    float64_t impurity_right       # Impurity of the right split.
    uint8_t missing_go_to_left  # Controls if missing values go to the left node.
    intp_t n_missing            # Number of missing values for the feature being split on

    vector[float32_t]* proj_vec_weights  # weights of the vector (max_features,)
    vector[intp_t]* proj_vec_indices     # indices of the features (max_features,)


cdef class BaseObliqueSplitter(Splitter):
    # Base class for oblique splitting, where additional data structures and API is defined.
    #

    # Oblique Splitting extra parameters (mtry, n_dims) matrix
    cdef vector[vector[float32_t]] proj_mat_weights       # nonzero weights of sparse proj_mat matrix
    cdef vector[vector[intp_t]] proj_mat_indices        # nonzero indices of sparse proj_mat matrix

    # TODO: assumes all oblique splitters only work with dense data
    cdef const float32_t[:, :] X

    # feature weights across (n_dims,)
    cdef float32_t[:] feature_weights

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil

    # Redefined here since the new logic requires calling sample_proj_mat
    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil

    cdef void compute_features_over_samples(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]* proj_vec_weights,  # weights of the vector (max_features,)
        vector[intp_t]* proj_vec_indices    # indices of the features (max_features,)
    ) noexcept nogil

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class ObliqueSplitter(BaseObliqueSplitter):
    # The splitter searches in the input space for a linear combination of features and a threshold
    # to split the samples samples[start:end].

    # Oblique Splitting extra parameters
    cdef public float64_t feature_combinations             # Number of features to combine
    cdef intp_t n_non_zeros                             # Number of non-zero features
    cdef intp_t[::1] indices_to_sample                  # an array of indices to sample of size mtry X n_features

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil


cdef class BestObliqueSplitter(ObliqueSplitter):
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class RandomObliqueSplitter(ObliqueSplitter):
    cdef void find_min_max(
        self,
        float32_t[::1] feature_values,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil

    cdef intp_t partition_samples(
        self,
        float64_t current_threshold
    ) noexcept nogil

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


# XXX: This splitter is experimental. Expect changes frequently.
cdef class MultiViewSplitter(BestObliqueSplitter):
    cdef const intp_t[:] feature_set_ends   # an array indicating the column indices of the end of each feature set
    cdef intp_t n_feature_sets                  # the number of feature sets is the length of feature_set_ends + 1

    cdef const intp_t[:] max_features_per_set  # the maximum number of features to sample from each feature set

    cdef vector[vector[intp_t]] multi_indices_to_sample

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil


# XXX: This splitter is experimental. Expect changes frequently.
cdef class MultiViewObliqueSplitter(BestObliqueSplitter):
    cdef const intp_t[:] feature_set_ends   # an array indicating the column indices of the end of each feature set
    cdef intp_t n_feature_sets                  # the number of feature sets is the length of feature_set_ends + 1

    # whether or not to uniformly sample feature-sets into each projection vector
    # if True, then sample from each feature set for each projection vector
    cdef bint uniform_sampling

    cdef vector[vector[intp_t]] multi_indices_to_sample

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil

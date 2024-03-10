import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._utils cimport UINT32_t
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._sklearn_splitter cimport sort


cdef struct MultiViewSplitRecord:
    # Data to track sample split
    intp_t feature              # Which feature to split on.
    intp_t pos                  # Split samples array at the given position,
    #                           # i.e. count of samples below threshold for feature.
    #                           # pos is >= end if the node is a leaf.
    float64_t threshold            # Threshold to split at.
    float64_t improvement          # Impurity improvement given parent node.
    float64_t impurity_left        # Impurity of the left split.
    float64_t impurity_right       # Impurity of the right split.
    intp_t n_constant_features     # Number of constant features in the split.

    # could maybe be optimized
    vector[intp_t] vec_n_constant_features     # Number of constant features in the split for each feature view


# XXX: This splitter is experimental. Expect changes frequently.
cdef class MultiViewSplitter(Splitter):
    cdef const intp_t[:] feature_set_ends   # an array indicating the column indices of the end of each feature set
    cdef intp_t n_feature_sets                  # the number of feature sets is the length of feature_set_ends + 1

    cdef const intp_t[:] max_features_per_set  # the maximum number of features to sample from each feature set

    # The following are used to track per feature set:
    # - the number of visited features
    # - the number of found constants in this split search
    # - the number of drawn constants in this split search
    cdef intp_t[:] vec_n_visited_features
    cdef intp_t[:] vec_n_found_constants
    cdef intp_t[:] vec_n_drawn_constants

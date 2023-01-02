# Authors: Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#
# License: BSD 3 clause

# See _unsup_splitter.pyx for details.

from ._splitter cimport BaseSplitter

cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.

cdef class UnsupervisedSplitter(BaseSplitter):
    """Abstract interface for supervised splitter."""

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:] sample_weight
    ) except -1
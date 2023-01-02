# Authors: Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#
# License: BSD 3 clause

# See _unsup_splitter.pyx for details.

from ._splitter cimport BaseSplitter

cdef class UnsupervisedSplitter(BaseSplitter):
    """Abstract interface for supervised splitter."""

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:] sample_weight
    ) except -1
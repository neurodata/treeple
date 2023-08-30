# distutils: language = c++
# distutils: sources = _vector_hash.hpp
from libcpp.vector cimport vector
from ..._lib.sklearn.tree._tree cimport SIZE_t  # Type for indices and counters


cdef extern from "_vector_hash.hpp":
    cdef cppclass VectorHash:
        VectorHash() except +
        size_t operator()(const vector[SIZE_t]& v)

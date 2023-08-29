cdef extern from "VectorHash.hpp":
    cdef cppclass VectorHash:
        VectorHash() except +
        size_t operator()(const std::vector[int]& v)

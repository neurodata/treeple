# distutils: language=c++
# cython: cdivision=True
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

cimport numpy as cnp

cnp.import_array()

from .._lib.sklearn.tree._utils cimport rand_int, rand_uniform


cdef inline void fisher_yates_shuffle(
    vector_or_memview indices_to_sample,
    intp_t grid_size,
    uint32_t* random_state,
) noexcept nogil:
    """Shuffle the indices in place using the Fisher-Yates algorithm.
    Parameters
    ----------
    indices_to_sample : A C++ vector or 1D memoryview
        The indices to shuffle.
    grid_size : intp_t
        The size of the grid to shuffle. This is explicitly passed in
        to support the templated `vector_or_memview` type, which allows
        for both C++ vectors and Cython memoryviews. Getitng the length
        of both types uses different API.
    random_state : uint32_t*
        The random state.
    """
    cdef intp_t i, j

    # XXX: should this be `i` or `i+1`? for valid Fisher-Yates?
    for i in range(0, grid_size - 1):
        j = rand_int(i, grid_size, random_state)
        indices_to_sample[j], indices_to_sample[i] = \
            indices_to_sample[i], indices_to_sample[j]


cdef inline int rand_weighted_binary(
    float64_t p0,
    uint32_t* random_state
) noexcept nogil:
    """Sample from integers 0 and 1 with different probabilities.

    Parameters
    ----------
    p0 : float64_t
        The probability of sampling 0.
    random_state : uint32_t*
        The random state.
    """
    cdef float64_t random_value = rand_uniform(0.0, 1.0, random_state)

    if random_value < p0:
        return 0
    else:
        return 1

cpdef unravel_index(
    intp_t index,
    cnp.ndarray[intp_t, ndim=1] shape
):
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Purely used for testing purposes.

    Parameters
    ----------
    index : intp_t
        A flat index.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the flat indices should be converted.

    Returns
    -------
    numpy.ndarray[intp_t, ndim=1]
        A coordinate array having the same shape as the input `shape`.
    """
    index = np.intp(index)
    shape = np.array(shape)
    coords = np.empty(shape.shape[0], dtype=np.intp)
    cdef const intp_t[:] shape_memview = shape
    cdef intp_t[:] coords_memview = coords
    unravel_index_cython(index, shape_memview, coords_memview)
    return coords


cpdef ravel_multi_index(intp_t[:] coords, const intp_t[:] shape):
    """Converts a tuple of coordinate arrays into a flat index.

    Purely used for testing purposes.

    Parameters
    ----------
    coords : numpy.ndarray[intp_t, ndim=1]
        An array of coordinate arrays to be converted.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the coordinates should be converted.

    Returns
    -------
    intp_t
        The resulting flat index.

    Raises
    ------
    ValueError
        If the input `coords` have invalid indices.
    """
    return ravel_multi_index_cython(coords, shape)


cdef inline void unravel_index_cython(
    intp_t index,
    const intp_t[:] shape,
    vector_or_memview coords
) noexcept nogil:
    """Converts a flat index into a tuple of coordinate arrays.

    Parameters
    ----------
    index : intp_t
        The flat index to be converted.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the flat index should be converted.
    coords : intp_t[:] or vector[intp_t]
        A preinitialized array of coordinates to store the result of the
        unraveled `index`.
    """
    cdef intp_t ndim = shape.shape[0]
    cdef intp_t j, size

    for j in range(ndim - 1, -1, -1):
        size = shape[j]
        coords[j] = index % size
        index //= size


cdef inline intp_t ravel_multi_index_cython(
    vector_or_memview coords,
    const intp_t[:] shape
) noexcept nogil:
    """Converts a tuple of coordinate arrays into a flat index in the vectorized dimension.

    Parameters
    ----------
    coords : intp_t[:] or vector[intp_t]
         An array of coordinates to be converted and vectorized into a sinlg
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the coordinates should be converted.

    Returns
    -------
    intp_t
        The resulting flat index.

    Raises
    ------
    ValueError
        If the input `coords` have invalid indices.
    """
    cdef intp_t i, ndim
    cdef intp_t flat_index, index

    ndim = len(shape)

    # Compute flat index
    flat_index = 0
    for i in range(ndim):
        index = coords[i]
        # if index < 0 or index >= shape[i]:
        #     raise ValueError("Invalid index")
        flat_index += index
        if i < ndim - 1:
            flat_index *= shape[i + 1]

    return flat_index

# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libc.stdlib cimport qsort
from libc.string cimport memcpy
from sklearn.tree._utils cimport log


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) noexcept nogil:
    if n == 0:
        return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(
    DTYPE_t* Xf,
    SIZE_t* samples,
    SIZE_t i,
    SIZE_t j
) noexcept nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) noexcept nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n // 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) noexcept nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) noexcept nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) noexcept nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef int compare_SIZE_t(const void* a, const void* b) noexcept nogil:
    """Comparison function for sort."""
    return <int>((<SIZE_t*>a)[0] - (<SIZE_t*>b)[0])


cdef inline void binary_search(INT32_t[::1] sorted_array,
                               INT32_t start, INT32_t end,
                               SIZE_t value, SIZE_t* index,
                               INT32_t* new_start) noexcept nogil:
    """Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    """
    cdef INT32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) // 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(INT32_t[::1] X_indices,
                                              DTYPE_t[::1] X_data,
                                              INT32_t indptr_start,
                                              INT32_t indptr_end,
                                              SIZE_t[::1] samples,
                                              SIZE_t start,
                                              SIZE_t end,
                                              SIZE_t[::1] index_to_samples,
                                              DTYPE_t[::1] Xf,
                                              SIZE_t* end_negative,
                                              SIZE_t* start_positive) noexcept nogil:
    """Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    """
    cdef INT32_t k
    cdef SIZE_t index
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(INT32_t[::1] X_indices,
                                           DTYPE_t[::1] X_data,
                                           INT32_t indptr_start,
                                           INT32_t indptr_end,
                                           SIZE_t[::1] samples,
                                           SIZE_t start,
                                           SIZE_t end,
                                           SIZE_t[::1] index_to_samples,
                                           DTYPE_t[::1] Xf,
                                           SIZE_t* end_negative,
                                           SIZE_t* start_positive,
                                           SIZE_t[::1] sorted_samples,
                                           bint* is_samples_sorted) noexcept nogil:
    """Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef SIZE_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(&sorted_samples[start], &samples[start],
               n_samples * sizeof(SIZE_t))
        qsort(&sorted_samples[start], n_samples, sizeof(SIZE_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef SIZE_t p = start
    cdef SIZE_t index
    cdef SIZE_t k
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
            # If k != -1, we have found a non zero value
            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(SIZE_t[::1] index_to_samples, SIZE_t[::1] samples,
                             SIZE_t pos_1, SIZE_t pos_2) noexcept nogil:
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    samples[pos_1], samples[pos_2] = samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2

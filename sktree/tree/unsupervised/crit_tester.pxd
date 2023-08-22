

cdef class CriterionTester:
    cdef public object fast_bic
    cdef public object faster_bic

    cpdef init(self,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] samples,
        const DTYPE_t[:] Xf)
    cpdef init_feature_vec(self)
    cpdef update(self, int pos)
    cpdef reset(self)
    cpdef node_impurity(self)
    cpdef children_impurity(self)
    cpdef proxy_impurity(self)
    cpdef weighted_n_samples(self)

    cpdef sum_right(self)
    cpdef sum_left(self)
    cpdef sum_total(self)
    cpdef set_sample_pointers(self, int start, int end)
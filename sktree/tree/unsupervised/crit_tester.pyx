

cdef class CriterionTester:
    cpdef init(self,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] samples,
        const DTYPE_t[:] Xf
    ):
        n_classes = np.array([2])
        n_samples = Xf.shape[0]

        # Create instances of FastBIC and FasterBIC with test data
        cdef int n_outputs = 1
        cdef int start = 0
        cdef int end = len(Xf)

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        weighted_n_samples = np.sum(sample_weight)
        feature_values_fast = np.empty(n_samples, dtype=np.float32)
        feature_values_faster = np.empty(n_samples, dtype=np.float32)

        # initialize the two criterion
        fast_bic = FastBIC(n_outputs, n_classes)
        faster_bic = FasterBIC(n_outputs, n_classes)
        fast_bic.init(
            sample_weight,
            weighted_n_samples,
            samples,
            feature_values_fast,
        )
        faster_bic.init(
            sample_weight,
            weighted_n_samples,
            samples,
            feature_values_faster,
        )
        fast_bic.set_sample_pointers(
            start,
            end
        )
        faster_bic.set_sample_pointers(
            start,
            end
        )
        fast_bic.init_feature_vec()
        faster_bic.init_feature_vec()

        self.fast_bic = fast_bic
        self.faster_bic = faster_bic

    cpdef init_feature_vec(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.init_feature_vec()
        faster_bic.init_feature_vec()

    cpdef update(self, int pos):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.update(pos)
        faster_bic.update(pos)

    cpdef node_impurity(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        fast_bic_imp = fast_bic.node_impurity()
        faster_bic_imp = faster_bic.node_impurity()

        return fast_bic_imp, faster_bic_imp

    cpdef children_impurity(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        cdef double fast_bic_left
        cdef double fast_bic_right
        cdef double faster_bic_left
        cdef double faster_bic_right
        fast_bic.children_impurity(&fast_bic_left, &fast_bic_right)
        faster_bic.children_impurity(&faster_bic_left, &faster_bic_right)

        return (fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right)

    cpdef sum_total(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        return fast_bic.sum_total, faster_bic.sum_total

    cpdef sum_left(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        return fast_bic.sum_left, faster_bic.sum_left

    cpdef sum_right(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        return fast_bic.sum_right, faster_bic.sum_right

    cpdef proxy_impurity(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        fast_bic_imp = fast_bic.proxy_impurity_improvement()
        faster_bic_imp = faster_bic.proxy_impurity_improvement()

        return fast_bic_imp, faster_bic_imp

    cpdef reset(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.reset()
        faster_bic.reset()

    cpdef set_sample_pointers(self, int start, int end):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic
        fast_bic.set_sample_pointers(start, end)
        faster_bic.set_sample_pointers(start, end)

    cpdef weighted_n_samples(self):
        cdef FastBIC fast_bic = self.fast_bic
        cdef FasterBIC faster_bic = self.faster_bic

        # for idx in range(len(np.array([0, 1, 2, 3, 5, 10, 20, 200]).reshape(-1, 1))):
        #     print('idx: ', idx)
        #     print(faster_bic.sample_indices[idx])
        #     print(faster_bic.cumsum_weights_map[faster_bic.sample_indices[idx]])
        #     print(faster_bic.cumsum_map[faster_bic.sample_indices[idx]])


        return ((fast_bic.weighted_n_node_samples, fast_bic.weighted_n_left, fast_bic.weighted_n_right),
            (faster_bic.weighted_n_node_samples, faster_bic.weighted_n_left, faster_bic.weighted_n_right))

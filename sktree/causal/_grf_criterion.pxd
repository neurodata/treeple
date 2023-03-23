# Licensed under the MIT License.
# Original Authors:
#   - EconML
#   - Vasilis Syrgkanis
#
# Modified by: Adam Li

import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

from sklearn.tree._criterion cimport Criterion, RegressionCriterion


cdef class GeneralizedMomentCriterion(RegressionCriterion):
# The A random vector of the linear moment equation for each sample of size (n_samples, n_outputs)
    # these are the "weights" that are pre-computed.
    cdef const DOUBLE_t[:, ::1] alpha       

    # the approximate Jacobian evaluated at every single sample point
    # random vector of the linear moment equation
    # size (n_samples, n_outputs, n_outputs)
    cdef const DOUBLE_t[:, :, ::1] pointJ   

    cdef DOUBLE_t[:, ::1] rho                  # Proxy heterogeneity label: rho = E[J | X in Node]^{-1} m(J, A; theta(Node)) of shape (`n_samples`, `n_outputs`)
    cdef DOUBLE_t[:, ::1] moment               # Moment for each sample: m(J, A; theta(Node)) of shape (`n_samples`, `n_outputs`)
    cdef DOUBLE_t[:, ::1] J                    # Node average jacobian: J(Node) = E[J | X in Node] of shape (n_outputs, n_outputs)
    cdef DOUBLE_t[:, ::1] invJ                 # Inverse of node average jacobian: J(Node)^{-1} of shape (n_outputs, n_outputs)
    cdef DOUBLE_t[:] parameter                 # Estimated node parameter: theta(Node) = E[J|X in Node]^{-1} E[A|X in Node]
    cdef DOUBLE_t[:] parameter_pre             # Preconditioned node parameter: theta_pre(Node) = E[A | X in Node]

    cdef SIZE_t n_relevant_outputs

    cdef int compute_sample_preparameter(
        self,
        DOUBLE_t[:] parameter_pre,
        const DOUBLE_t[:, ::1] alpha,
        DOUBLE_t weight,
        SIZE_t sample_index,
    ) nogil except -1
    cdef int compute_sample_parameter(
        self,
        DOUBLE_t[:] parameter,
        DOUBLE_t[:] parameter_pre,
        DOUBLE_t[:, ::1] invJ
    ) nogil except -1
    cdef int compute_sample_moment(
        self,
        DOUBLE_t[:, ::1] moment,
        DOUBLE_t[:, ::1] alpha,
        DOUBLE_t[:] parameter,
        const DOUBLE_t[:, :, ::1] pointJ,
        SIZE_t sample_index
    ) except -1 nogil
    cdef int compute_sample_rho(
        self,
        DOUBLE_t[:, ::1] moment,
        DOUBLE_t[:, ::1] invJ,
        SIZE_t sample_index
    ) except -1 nogil
    cdef int compute_sample_jacobian(
        self,
        DOUBLE_t[:, ::1] J,
        const DOUBLE_t[:, :, ::1] pointJ,
        DOUBLE_t weight,
        SIZE_t sample_index,
    ) except -1 nogil
    cdef int compute_node_inv_jacobian(
        self,
        DOUBLE_t[:, ::1] J,
        DOUBLE_t[:, ::1] invJ,
    ) except -1 nogil

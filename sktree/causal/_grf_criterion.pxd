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


cdef class CausalCriterion(RegressionCriterion):
    

cdef class MomentCriterion(RegressionCriterion):
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

cdef class LinearMomentCriterion(RegressionCriterion):
    """ A criterion class that estimates local parameters defined via linear moment equations
    of the form:

    E[ m(J, A; theta(x)) | X=x] = E[ J * theta(x) - A | X=x] = 0

    Calculates impurity based on heterogeneity induced on the estimated parameters, based on the proxy score
    defined in the Generalized Random Forest paper:
        Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized random forests."
        The Annals of Statistics 47.2 (2019): 1148-1178
        https://arxiv.org/pdf/1610.01271.pdf

    **How do we utilize the abstract Criterion data structure?**
    - `sum_left`, `sum_right` represent for a sample i and each output k, the rho[i, k] terms per output weighted
      by sample_weight[i, k].
    - `sum_total` keeps track of the previous sums.
    - `sq_sum_total` keeps track of the square sum total ``sample_weight[i, k] * rho[i, k] * rho[i, k]``.

    - `n_revelant_outputs` should now store the relevant outputs, which should correspond to the dimensionality
        of the treatment vector.
    
    """
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

    cdef int compute_sample_preparameter(
        self,
        DOUBLE_t[:] parameter,
        DOUBLE_t[:] parameter_pre,
        DOUBLE_t[:, ::1] invJ,
        SIZE_t[:] samples_index,
        DOUBLE_t[:] sample_weight,
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
        DOUBLE_t[:, ::1] parameter,
        const DOUBLE_t[:, :, ::1] pointJ,
        SIZE_t sample_idx,
    ) except -1 nogil

    cdef int node_reset_jacobian(self, DOUBLE_t* J, DOUBLE_t* invJ, double* weighted_n_node_samples,
                                  const DOUBLE_t[:, ::1] pointJ,
                                  DOUBLE_t* sample_weight,
                                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1
    cdef int node_reset_parameter(self, DOUBLE_t* parameter, DOUBLE_t* parameter_pre,
                                   DOUBLE_t* invJ,
                                   const DOUBLE_t[:, ::1] alpha,
                                   DOUBLE_t* sample_weight, double weighted_n_node_samples,
                                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1
    cdef int node_reset_rho(self, DOUBLE_t* rho, DOUBLE_t* moment, SIZE_t* node_index_mapping,
                       DOUBLE_t* parameter, DOUBLE_t* invJ, double weighted_n_node_samples,
                       const DOUBLE_t[:, ::1] pointJ, const DOUBLE_t[:, ::1] alpha,
                       DOUBLE_t* sample_weight, SIZE_t* samples, 
                       SIZE_t start, SIZE_t end) nogil except -1
    cdef int node_reset_sums(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* rho,
                             DOUBLE_t* J,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             DOUBLE_t* sum_total, DOUBLE_t* var_total,
                             DOUBLE_t* sq_sum_total, DOUBLE_t* y_sq_sum_total,
                             SIZE_t start, SIZE_t end) nogil except -1

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

#
# This code contains a refactoring of the EconML GRF code,
# published under the following license and copyright:
# BSD 3-Clause License
#
# All rights reserved.

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs, sqrt

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport matinv_, pinv_

cdef double INFINITY = np.inf

# TODO: OPEN QUESTIONS
# 1. why is alpha = y * T and y * Z in CATE and iV forests?
# 2. why is pointJ = T x T and T x Z?

cdef class GeneralizedMomentCriterion(RegressionCriterion):
    """A generalization of the regression criterion in scikit-learn.

    Generalized criterion with moment equations was introduced in the generalized
    random forest paper, which shows how many common forests are trained using this
    framework of criterion.

    A criterion class that estimates local parameters defined via moment equations
    of the form::

        E[ m(J, A; theta(x)) | X=x]

    where our specific instance is a linear moment equation::
        
        E[ J * theta(x) - A | X=x] = 0

    where:

    - m(J, A; theta(x)) is the moment equation that is specified per modeling setup.
    - J is the Jacobian array of shape (n_outputs, n_outputs) per sample
    - A is the alpha weights of shape (n_outputs) per sample
    - theta(x) is the parameters we are interested in of shape (n_outputs) per sample
    - X is the data matrix of shape (n_samples, n_features)

    We have the following given points:

    - alpha is the weights per sample of shape (n_samples, n_outputs)
    - pointJ is the pointwise Jacobian array per sample of shape (n_samples, n_outputs, n_outputs)
    
    Then we have the following estimating equations:

    J(Node) := E[J[i] | X[i] in Node] = sum_{i in Node} w[i] J[i]
    moment[i] := J[i] * theta(Node) - A[i]
    rho[i] := - J(Node)^{-1} (J[i] * theta(Node) - A[i])
    theta_pre(node) := E[A[i] | X[i] in Node] weight(node) = sum_{i in Node} w[i] A[i]
    theta(Node) := J(Node)^{-1} E[A[i] | X[i] in Node] = J(node)^{-1} theta_pre(node)

    Notes
    -----
    Calculates impurity based on heterogeneity induced on the estimated parameters, based on
    the proxy score defined in :footcite:`Athey2016GeneralizedRF`.

    Specifically, we compute node, proxy and children impurity similarly to a
    mean-squared error setting, where these are in general "proxies" for the true
    criterion:

        n_{C1} * n_{C2} / n_P^2 (\hat{\\theta}_{C1} - \hat{\\theta}_{C2})^2

    as specified in Equation 5 of the paper :footcite:`Athey2016GeneralizedRF`.
    The proxy is computed with Equation 9 of the paper:

        1/n_{C1} (\sum_{i \in C1} \\rho_i)^2 + 1/n_{C2} (\sum_{i \in C2} \\rho_i)^2

    where :math:`\\rho_i` is the pseudo-label for the ith sample.
    """
    def __cinit__(
        self,
        SIZE_t n_outputs,
        SIZE_t n_samples,
        SIZE_t n_relevant_outputs,
        SIZE_t n_y,
    ):
        """Initialize parameters for this criterion. Parent `__cinit__` is always called before children.
        So we only perform extra initializations that were not perfomed by the parent classes.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of parameters/values to be estimated
        n_samples : SIZE_t
            The total number of rows in the 2d matrix y.
            The total number of samples to fit on.
        n_relevant_outputs : SIZE_t
            We only care about the first n_relevant_outputs of these parameters/values
        n_y : SIZE_t
            The first n_y columns of the 2d matrix y, contain the raw labels y_{ik}, the rest are auxiliary variables
        """
        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        if n_y > 1:
            raise AttributeError("LinearMomentGRFCriterion currently only supports a scalar y")

        self.proxy_children_impurity = True     # The children_impurity() only returns an approximate proxy

        self.n_outputs = 
        self.n_relevant_outputs = n_relevant_outputs
        self.n_y = n_y

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.rho = np.zeros((n_samples, n_outputs), dtype=np.float64)
        self.moment = np.zeros((n_samples, n_outputs), dtype=np.float64)
        self.J = np.zeros((n_outputs, n_outputs), dtype=np.float64)
        self.invJ = np.zeros((n_outputs, n_outputs), dtype=np.float64)
        self.parameter = np.zeros(n_outputs, dtype=np.float64)
        self.parameter_pre = np.zeros(n_outputs, dtype=np.float64)

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices
    ) nogil except -1:
        """Initialize the criterion object with data.
        
        Parameters
        ----------
        y : DOUBLE_t 2D memoryview of shape (n_samples, n_y + n_outputs + n_outputs * n_outputs)
            The input y array.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.

        Notes
        -----
        For generalized criterion, the y array has columns associated with the
        actual 'y' in the first `n_y` columns, and the next `n_outputs` columns are associated
        with the alpha vectors and the next `n_outputs * n_outputs` columns are
        associated with the estimated sample Jacobian vectors.

        `n_relevant_outputs` is a number less than or equal to `n_outputs`, which
        tracks the relevant outputs.
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_y = self.n_y

        self.y = y[:, :n_y]                     # The first n_y columns of y are the original raw outcome
        self.alpha = y[:, n_y:(n_y + n_outputs)]        # A[i] part of the moment is the next n_outputs columns
        # J[i] part of the moment is the next n_outputs * n_outputs columns, stored in Fortran contiguous format
        self.pointJ = y[:, (n_y + n_outputs):(n_y + n_outputs + n_outputs * n_outputs)]
        self.sample_weight = sample_weight      # Store the sample_weight locally
        self.samples = samples                  # Store the sample index structure used and updated by the splitter
        self.weighted_n_samples = weighted_n_samples    # Store total weight of all samples computed by splitter

        return 0

    cdef double node_impurity(
        self
    ) noexcept nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end].
        
        This sums up the relevant metric over all "relevant outputs" (`n_relevant_outputs`)
        for samples in the node and then normalizes to get the average impurity of the node.
        Similarly, `sum_total` stores an (`n_outputs`,) array and should have been computed apriori.

        This distinctly generalizes the scikit-learn Regression Criterion, as the `n_outputs`
        contains both `n_relevant_outputs` and extra outputs that are nuisance parameters. But
        otherwise, the node impurity calculation follows that of a regression.
        """

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_relevant_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_relevant_outputs

    cdef double proxy_impurity_improvement(
        self
    ) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split. It is a proxy quantity such that the
        split that maximizes this value also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split. The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        This sums up the relevant metric over all "relevant outputs" (`n_relevant_outputs`)
        for samples in the node and then normalizes to get the average impurity of the node.
        Similarly, `sum_left` and `sum_right` stores an (`n_outputs`,) array and should have
        been computed apriori.

        This distinctly generalizes the scikit-learn Regression Criterion, as the `n_outputs`
        contains both `n_relevant_outputs` and extra outputs that are nuisance parameters. But
        otherwise, the node impurity calculation follows that of a regression.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_relevant_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexept nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
        left child (samples[start:pos]) and the impurity the right child
        (samples[pos:end]). Here we use the proxy child impurity:
            impurity_child[k] = sum_{i in child} w[i] rho[i, k]^2 / weight(child)
                                - (sum_{i in child} w[i] * rho[i, k] / weight(child))^2
            impurity_child = sum_{k in n_relevant_outputs} impurity_child[k] / n_relevant_outputs
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, p, k, offset
        cdef DOUBLE_t w = 1.0

        # We calculate: sq_sum_left = sum_{i in child} w[i] rho[i, k]^2
        for p in range(start, pos):
            i = self.sample_indices[p]

            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_relevant_outputs):
                y_ik = self.rho[i, k]
                sq_sum_left += w * y_ik * y_ik
        # We calculate sq_sum_right = sq_sum_total - sq_sum_left
        sq_sum_right = self.sq_sum_total - sq_sum_left

        # We normalize each sq_sum_child by the weight of that child
        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        # We subtract from the impurity_child, the quantity:
        # sum_{k in n_relevant_outputs} (sum_{i in child} w[i] * rho[i, k] / weight(child))^2
        #   = sum_{k in n_relevant_outputs} (sum_child[k] / weight(child))^2
        for k in range(self.n_relevant_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_relevant_outputs
        impurity_right[0] /= self.n_relevant_outputs

    cdef int compute_sample_preparameter(
        self,
        DOUBLE_t[:] parameter_pre,
        const DOUBLE_t[:, ::1] alpha,
        DOUBLE_t weight,
        SIZE_t sample_index,
    ) nogil except -1:
        """Calculate the un-normalized pre-conditioned parameter.

            theta_pre(node) := E[A[i] | X[i] in Node] weight(node) = sum_{i in Node} w[i] A[i]
            theta(node) := J(node)^{-1} theta_pre(node)

        Parameters
        ----------
        parameter_pre : DOUBLE_t memoryview of size (n_outputs,)
            To be computed node un-normalized pre-conditioned parameter theta_pre(node).
        alpha : DOUBLE_t 2D memoryview of size (n_samples, n_outputs)
            The memory view that contains the A[i] for each sample i
        weight : DOUBLE_t
            The weight of the sample.
        sample_index : SIZE_t
            The index of the sample to be used on computation of criteria of the current node.
        """
        cdef SIZE_t i, j, p
        cdef DOUBLE_t w = 1.0

        # compute parameter_pre per output dimension, j as the
        # \sum_{i} w[i] * alpha[i, j]
        for j in range(self.n_outputs):
            parameter_pre[j] += weight * alpha[sample_index, j]
        return 0

    cdef int compute_sample_parameter(
        self,
        DOUBLE_t[:] parameter,
        DOUBLE_t[:] parameter_pre,
        DOUBLE_t[:, ::1] invJ
    ) nogil except -1:
        """Calculate the parameter.

            theta_pre(node) := E[A[i] | X[i] in Node] weight(node) = sum_{i in Node} w[i] A[i]
            theta(node) := J(node)^{-1} theta_pre(node)

        Parameters
        ----------
        parameter : DOUBLE_t memoryview of size (n_outputs,)
            To be computed node parameter theta(node).
        parameter_pre : DOUBLE_t memoryview of size (n_outputs,)
            Already computed node un-normalized pre-conditioned parameter theta_pre(node)
        invJ : DOUBLE_t 2D memoryview of size (n_outputs, n_outputs)
            Unnormalized node jacobian inverse J(node)^{-1} in C-contiguous format.
        """
        cdef SIZE_t i, j

        for j in range(self.n_outputs):
            for i in range(self.n_outputs):
                parameter[i] += invJ[i, j] * parameter_pre[j]
        return 0

    cdef int compute_sample_moment(
        self,
        DOUBLE_t[:, ::1] moment,
        DOUBLE_t[:, ::1] alpha,
        DOUBLE_t[:] parameter,
        const DOUBLE_t[:, :, ::1] pointJ,
        SIZE_t sample_index
    ) except -1 nogil:
        """Calculate the linear moment and rho for each sample i in the node.

            moment[i] := J[i] * theta(Node) - A[i]
            rho[i] := - (J(Node) / weight(node))^{-1} * moment[i]

        Parameters
        ----------
        moment : DOUBLE_t 2D memoryview of shape (`n_samples`, `n_outputs`)
            Array of moments per sample in node.
        alpha : DOUBLE_t[:, ::1] of size (n_samples, n_outputs)
            The memory view that contains the A[i] for each sample i
        parameter : DOUBLE_t memoryview of shape (`n_outputs`)
            Array of computed parameters theta(x) per sample in node.
        pointJ : DOUBLE_t[:, :, ::1] 3D memory of size (n_samples, n_outputs, n_outputs)
            The memory view that contains the J[i] for each sample i, where J[i] is the
            Jacobian array.
        sample_index : SIZE_t
            The index of the sample to be used on computation of criteria of the current node.
        """
        cdef SIZE_t j, k

        # compute the moment for each output
        for j in range(self.n_outputs):
            moment[sample_index, j] = - alpha[sample_index, j]
            for k in range(self.n_outputs):
                moment[sample_index, j] += pointJ[sample_index, j, k] * parameter[k]
        return 0

    cdef int compute_sample_rho(
        self,
        DOUBLE_t[:, ::1] moment,
        DOUBLE_t[:, ::1] invJ,
        SIZE_t sample_index
    ) except -1 nogil:
        """Calculate the rho for each sample i in the node.

            moment[i] := J[i] * theta(Node) - A[i]
            rho[i] := - (J(Node) / weight(node))^{-1} * moment[i]

        Parameters
        ----------
        moment : DOUBLE_t 2D memoryview of shape (`n_samples`, `n_outputs`)
            Array of moments per sample in node.
        invJ : DOUBLE_t 2D memoryview of size (n_outputs, n_outputs)
            Unnormalized node jacobian inverse J(node)^{-1} in C-contiguous format.
        sample_index : SIZE_t
            The index of the sample to be used on computation of criteria of the current node.
        """
        cdef SIZE_t j, k

        # compute the moment for each output
        for j in range(self.n_outputs):
            rho[sample_index, j] = 0.0
            for k in range(self.n_outputs):
                rho[sample_index, j] -= (
                    invJ[j, k] * \
                    moment[sample_index, k] * \
                    self.weighted_n_node_samples
                )
        return 0

    cdef int compute_sample_jacobian(
        self,
        DOUBLE_t[:, ::1] J,
        const DOUBLE_t[:, :, ::1] pointJ,
        DOUBLE_t weight,
        SIZE_t sample_index,
    ) except -1 nogil:
        """Calculate the un-normalized jacobian for a sample::

            J(node) := E[J[i] | X[i] in Node] weight(node) = sum_{i in Node} w[i] J[i]

        and its inverse J(node)^{-1} (or revert to pseudo-inverse if the matrix is not invertible). For
        dimensions n_outputs={1, 2}, we also add a small constant of 1e-6 on the diagonals of J, to ensure
        invertibility. For larger dimensions we revert to pseudo-inverse if the matrix is not invertible.

        Parameters
        ----------
        J : DOUBLE_t 2D memoryview (n_outputs, n_outputs)
            Un-normalized jacobian J(node) in C-contiguous format
        pointJ : DOUBLE_t[:, :, ::1] 3D memory of size (n_samples, n_outputs, n_outputs)
            The memory view that contains the J[i] for each sample i, where J[i] is the
            Jacobian array.
        weight : DOUBLE_t
            The weight of the sample.
        sample_index : SIZE_t
            The index of the sample to be used on computation of criteria of the current node.
        """
        cdef SIZE_t j, k

        # Calculate un-normalized empirical jacobian
        for j in range(self.n_outputs):
            for k in range(self.n_outputs):
                J[j, k] += w * pointJ[sample_index, j, k]

        return 0

    cdef int compute_node_inv_jacobian(
        self,
        DOUBLE_t[:, ::1] J,
        DOUBLE_t[:, ::1] invJ,
    ) except -1 nogil:
        """Calculate the node inverse un-normalized jacobian::

        Its inverse J(node)^{-1} (or revert to pseudo-inverse if the matrix is not invertible). For
        dimensions n_outputs={1, 2}, we also add a small constant of 1e-6 on the diagonals of J, to ensure
        invertibility. For larger dimensions we revert to pseudo-inverse if the matrix is not invertible.

        Parameters
        ----------
        J : DOUBLE_t 2D memoryview (n_outputs, n_outputs)
            Un-normalized jacobian J(node) in C-contiguous format
        invJ : DOUBLE_t 2D memoryview of size (n_outputs, n_outputs)
            Unnormalized node jacobian inverse J(node)^{-1} in C-contiguous format.
        """
        cdef SIZE_t k

        # Calculae inverse and store it in invJ
        if self.n_outputs <=2:
            # Fast closed form inverse calculation
            _fast_invJ(J, invJ, self.n_outputs, clip=1e-6)
        else:
            for k in range(self.n_outputs):
                J[k, k] += 1e-6    # add small diagonal constant for invertibility

            # Slower matrix inverse via lapack package
            if not matinv_(J, invJ, self.n_outputs):     # if matrix is invertible use the inverse
                pinv_(J, invJ, self.n_outputs, self.n_outputs)    # else calculate pseudo-inverse

            for k in range(self.n_outputs):
                J[k, k] -= 1e-6    # remove the invertibility constant
        return 0

    cdef void set_sample_pointers(
        self,
        SIZE_t start,
        SIZE_t end
    ) noexcept nogil:
        """Set sample pointers in the criterion and update statistics.
        
        The dataset array that we compute criteria on is assumed to consist of 'N' 
        ordered samples or rows (i.e. sorted). Since we pass this by reference, we 
        use sample pointers to move the start and end around to consider only a subset of data. 
        This function should also update relevant statistics that the class uses to compute the final criterion.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to be used on computation of criteria of the current node.
        end : SIZE_t
            The last sample used on this node
        """
        self.start = start
        self.end = end

        self.n_node_samples = end - start

        self.sq_sum_total = 0.0
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))

        # Init jacobian matrix to zero
        self.J[:] = 0.0

        # init parameter parameter to zero
        self.parameter_pre[:] = 0.0
        self.parameter[:] = 0.0

        # perform the first loop to aggregate
        for p in range(start, end):
            i = self.sample_indices[p]

            if self.sample_weight is not None:
                w = self.sample_weight[i]
            self.weighted_n_node_samples += w
            
            # compute the Jacobian for this sample
            self.compute_sample_jacobian(
                self.J, self.pointJ, w, i)

            # compute the pre-conditioned parameter
            self.compute_sample_preparameter(
                self.parameter_pre,
                self.alpha,
                w,
                i
            )

        # compute the inverse-Jacobian
        self.compute_node_inv_jacobian(self.J, self.invJ)

        for p in range(start, end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            self.compute_sample_parameter(
                DOUBLE_t[:] parameter,
                DOUBLE_t[:] parameter_pre,
                DOUBLE_t[:, ::1] invJ
            )
            # next compute the moment for this sample
            self.compute_sample_moment(
                self.moment,
                self.alpha,
                self.parameter,
                self.pointJ,
                i
            )

            # compute the pseudo-label, rho
            self.compute_sample_rho(
                self.moment,
                self.invJ,
                i
            )

            # compute statistics for the current total sum and square sum
            # of the metrics for criterion, in this case the pseudo-label
            for k in range(self.n_outputs):
                y_ik = self.rho[i, k]
                w_y_ik = w * y_ik

                self.sum_total[k] += w_y_ik

                if k < self.n_relevant_outputs
                self.sq_sum_total += w_y_ik * y_ik

        # Reset to pos=start
        self.reset()

    cdef int update(
        self,
        SIZE_t new_pos
    ) except -1 nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] + sum_right[x] = sum_total[x]
        #           var_left[x] + var_right[x] = var_total[x]
        # and that sum_total and var_total are known, we are going to update
        # sum_left and var_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        
        # The invariance of the update is that:
        #   sum_left[k] = sum_{i in Left} w[i] rho[i, k]
        #   var_left[k] = sum_{i in Left} w[i] pointJ[i, k, k]
        # and similarly for the right child. Notably, the second is un-normalized,
        # so to be used for further calculations it needs to be normalized by the child weight.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = self.sample_indices[p]

                if self.sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # we add w[i] * rho[i, k] to sum_left[k]
                    self.sum_left[k] += w * self.rho[i, k]
                    # we add w[i] * J[i, k, k] to var_left[k]
                    # self.var_left[k] += w * self.pointJ[i, k, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = self.sample_indices[p]

                if self.sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # we subtract w[i] * rho[i, k] from sum_left[k]
                    self.sum_left[k] -= w * self.rho[i, k]
                    # we subtract w[i] * J[i, k, k] from var_left[k]
                    # self.var_left[k] -= w * self.pointJ[i, k, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
            self.var_right[k] = self.var_total[k] - self.var_left[k]

        self.pos = new_pos
        return 0


    cdef void node_value(self, double* dest) nogil:
        """Return the estimated node parameter of samples[start:end] into dest."""
        memcpy(dest, self.parameter, self.n_outputs * sizeof(double))

    cdef void node_jacobian(self, double* dest) nogil:
        """Return the node normalized Jacobian of samples[start:end] into dest in a C contiguous format."""
        cdef SIZE_t i, j 
        # Jacobian is stored in f-contiguous format for fortran. We translate it to c-contiguous for
        # user interfacing. Moreover, we normalize by weight(node).
        cdef SIZE_t n_outputs = self.n_outputs
        for i in range(n_outputs):
            for j in range(n_outputs):
                dest[i * n_outputs + j] = self.J[i, j] / self.weighted_n_node_samples
        
    cdef void node_precond(self, double* dest) nogil:
        """Return the normalized node preconditioned value of samples[start:end] into dest."""
        cdef SIZE_t i
        for i in range(self.n_outputs):
            dest[i] = self.parameter_pre[i] / self.weighted_n_node_samples

    
    cdef double min_eig_left(self) nogil:
        """ Calculate proxy for minimum eigenvalue of jacobian of left child. Here we simply
        use the minimum absolute value of the diagonals of the jacobian. This proxy needs to be
        super fast as this calculation happens for every candidate split. So we cannot afford
        anything that is not very simple calculation. We tried a power iteration approximation
        algorithm implemented in `_utils.fast_min_eigv_()`. But even such power iteration algorithm
        is slow as it requires calculating a pseudo-inverse first.
        """
        cdef int i
        cdef double min, abs
        min = fabs(self.var_left[0])
        for i in range(self.n_outputs):
            abs = fabs(self.var_left[i])
            if abs < min:
                min = abs
        # The `update()` method maintains that var_left[k] = J_left[k, k], where J_left is the
        # un-normalized jacobian of the left child. Thus we normalize by weight(left)
        return min / self.weighted_n_left

    cdef double min_eig_right(self) nogil:
        """ Calculate proxy for minimum eigenvalue of jacobian of right child
        (see min_eig_left for more details).
        """
        cdef int i
        cdef double min, abs
        min = fabs(self.var_right[0])
        for i in range(self.n_outputs):
            abs = fabs(self.var_right[i])
            if abs < min:
                min = abs
        return min / self.weighted_n_right


cdef void _fast_invJ(DOUBLE_t* J, DOUBLE_t* invJ, SIZE_t n, double clip) nogil:
    """Fast inversion of a 2x2 array."""
    cdef double det
    if n == 1:
        invJ[0] = 1.0 / J[0] if fabs(J[0]) >= clip else 1/clip     # Explicit inverse calculation
    elif n == 2:
        # Explicit inverse calculation
        det = J[0] * J[3] - J[1] * J[2]
        if fabs(det) < clip:
            det = clip
        invJ[0] = J[3] / det
        invJ[1] = - J[1] / det
        invJ[2] = - J[2] / det
        invJ[3] = J[0] / det

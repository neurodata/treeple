import numpy as np
import numpy.random as rng
import scipy.signal
from scipy.sparse import issparse
from sklearn.base import is_classifier
from sklearn.tree import _tree
from sklearn.utils import check_random_state

from ._split import BaseObliqueSplitter
from .conv import _apply_convolution
from .oblique_tree import ObliqueSplitter, ObliqueTree, ObliqueTreeClassifier

try:
    from skimage.filters import gabor_kernel
except Exception as e:
    raise ImportError("This function requires scikit-image.")


def _check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class Conv2DSplitter(ObliqueSplitter):
    """Convolutional splitter.

    A class used to represent a 2D convolutional splitter, where splits
    are done on a convolutional patch.

    Note: The convolution function is currently just the
    summation operator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    image_height : int, optional (default=None)
        MORF required parameter. Image height of each observation.
    image_width : int, optional (default=None)
        MORF required parameter. Width of each observation.
    patch_height_max : int, optional (default=max(2, floor(sqrt(image_height))))
        MORF parameter. Maximum image patch height to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_height)))``.
    patch_height_min : int, optional (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    patch_width_max : int, optional (default=max(2, floor(sqrt(image_width))))
        MORF parameter. Maximum image patch width to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_width)))``.
    patch_width_min : int (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    discontiguous_height : bool, optional (default=False)
        Whether or not the rows of the patch are taken discontiguously or not.
    discontiguous_width : bool, optional (default=False)
        Whether or not the columns of the patch are taken discontiguously or not.

    Methods
    -------
    sample_proj_mat
        Will compute projection matrix, which has columns as the vectorized
        convolutional patches.

    Notes
    -----
    This class assumes that data is vectorized in
    row-major (C-style), rather then column-major (Fotran-style) order.
    """

    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height: bool = False,
        discontiguous_width: bool = False,
    ):
        super(Conv2DSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
        )
        # set sample dimensions
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height_max = patch_height_max
        self.patch_width_max = patch_width_max
        self.patch_height_min = patch_height_min
        self.patch_width_min = patch_width_min
        self.axis_sample_dims = [
            (patch_height_min, patch_height_max),
            (patch_width_min, patch_width_max),
        ]
        self.structured_data_shape = [image_height, image_width]
        self.discontiguous_height = discontiguous_height
        self.disontiguous_width = discontiguous_width

    def _get_rand_patch_idx(self, rand_height, rand_width):
        """Generate a random patch on the original data to consider as feature combination.

        This function assumes that data samples were vectorized. Thus contiguous convolutional
        patches are defined based on the top left corner. If the convolutional patch
        is "discontiguous", then any random point can be chosen.

        TODO:
        - refactor to optimize for discontiguous and contiguous case
        - currently pretty slow because being constructed and called many times

        Parameters
        ----------
        rand_height : int
            A random height chosen between ``[1, image_height]``.
        rand_width : int
            A random width chosen between ``[1, image_width]``.

        Returns
        -------
        patch_idxs : np.ndarray
            The indices of the selected patch inside the vectorized
            structured data.
        """
        # XXX: results in edge effect on the RHS of the structured data...
        # compute the difference between the image dimension and current random
        # patch dimension
        delta_height = self.image_height - rand_height + 1
        delta_width = self.image_width - rand_width + 1

        # sample the top left pixel from available pixels now
        top_left_point = rng.randint(delta_width * delta_height)

        # convert the top left point to appropriate index in full image
        vectorized_start_idx = int(
            (top_left_point % delta_width)
            + (self.image_width * np.floor(top_left_point / delta_width))
        )

        # get the (x_1, x_2) coordinate in 2D array of sample
        multi_idx = self._compute_vectorized_index_in_data(vectorized_start_idx)

        if self.debug:
            print(vec_idx, multi_idx, rand_height, rand_width)

        # get random row and column indices
        if self.discontiguous_height:
            row_idx = np.random.choice(
                self.image_height, size=rand_height, replace=False
            )
        else:
            row_idx = np.arange(multi_idx[0], multi_idx[0] + rand_height)
        if self.disontiguous_width:
            col_idx = np.random.choice(self.image_width, size=rand_width, replace=False)
        else:
            col_idx = np.arange(multi_idx[1], multi_idx[1] + rand_width)

        # create index arrays in the 2D image
        structured_patch_idxs = np.ix_(
            row_idx,
            col_idx,
        )

        # get the patch vectorized indices
        patch_idxs = self._compute_index_in_vectorized_data(structured_patch_idxs)

        return patch_idxs

    def _compute_index_in_vectorized_data(self, idx):
        return np.ravel_multi_index(
            idx, dims=self.structured_data_shape, mode="raise", order="C"
        )

    def _compute_vectorized_index_in_data(self, vec_idx):
        return np.unravel_index(vec_idx, shape=self.structured_data_shape, order="C")

    def sample_proj_mat(self, sample_inds):
        """
        Get the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """
        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # generate random heights and widths of the patch. Note add 1 because numpy
        # needs is exclusive of the high end of interval
        rand_heights = rng.randint(
            self.patch_height_min, self.patch_height_max + 1, size=self.proj_dims
        )
        rand_widths = rng.randint(
            self.patch_width_min, self.patch_width_max + 1, size=self.proj_dims
        )

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.proj_dims):
            rand_height = rand_heights[idx]
            rand_width = rand_widths[idx]
            # get patch positions
            patch_idxs = self._get_rand_patch_idx(
                rand_height=rand_height, rand_width=rand_width
            )

            # get indices for this patch
            proj_mat[patch_idxs, idx] = 1

        # apply summation operation over the sampled patch
        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat


class GaborSplitter(Conv2DSplitter):
    """Splitter using Gabor kernel activations.

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    max_features : [type]
        [description]
    feature_combinations : [type]
        [description]
    random_state : [type]
        [description]
    image_height : [type], optional
        [description], by default None
    image_width : [type], optional
        [description], by default None
    patch_height_max : [type], optional
        [description], by default None
    patch_height_min : int, optional
        [description], by default 1
    patch_width_max : [type], optional
        [description], by default None
    patch_width_min : int, optional
        [description], by default 1
    discontiguous_height : bool, optional
        [description], by default False
    discontiguous_width : bool, optional
        [description], by default False
    frequency : [type], optional
        [description], by default None
    theta : [type], optional
        [description], by default None
    bandwidth : int, optional
        [description], by default 1
    sigma_x : [type], optional
        [description], by default None
    sigma_y : [type], optional
        [description], by default None
    n_stds : int, optional
        [description], by default 3
    offset : int, optional
        [description], by default 0

    Notes
    -----
    This class only uses convolution with ``'same'`` padding done to
    prevent a change in the size of the output image compared to the
    input image.

    This splitter relies on pytorch to do convolutions efficiently
    and scikit-image to instantiate the Gabor kernels.
    """

    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height: bool = False,
        discontiguous_width: bool = False,
        frequency=None,
        theta=None,
        bandwidth=1,
        sigma_x=None,
        sigma_y=None,
        n_stds=3,
        offset=0,
    ):
        super(GaborSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
            image_height=image_height,
            image_width=image_width,
            patch_height_max=patch_height_max,
            patch_height_min=patch_height_min,
            patch_width_max=patch_width_max,
            patch_width_min=patch_width_min,
            discontiguous_height=discontiguous_height,
            discontiguous_width=discontiguous_width,
        )
        # filter parameters
        self.frequency = frequency
        self.theta = theta
        self.bandwidth = bandwidth
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.n_stds = n_stds
        self.offset = offset

    # not used.
    def _convolutional_kernel_matrix(
        self, kernel, image_height, image_width, mode="full"
    ):
        """Manually doing convolution matrix.

        # sample mtry times different filters
        for idx in range(self.proj_dims):
            frequency = rng.rand()  # spatial frequency
            theta = rng.uniform() * 2 * np.pi  # orientation in radians
            bandwidth = rng.uniform() * 5  # bandwidth of the filter
            n_stds = rng.randint(1, 4)

            # get the random kernel
            kernel_params = {
                "frequency": frequency,
                "theta": theta,
                "bandwidth": bandwidth,
                "n_stds": n_stds,
            }
            kernel = gabor_kernel(**kernel_params)
            proj_kernel_params.append(kernel_params)

            # apply kernel as a full discrete linear convolution
            # over sub-sampled patch
            conv_kernel_mat = self._convolutional_kernel_matrix(
                kernel, image_height=patch_height, image_width=patch_width
            )
            convolved_X = self.X[:, patch_idxs] @ conv_kernel_mat.T

            proj_X[:, idx] = convolved_X.real.sum()

        Parameters
        ----------
        kernel : [type]
            [description]
        image_height : [type]
            [description]
        image_width : [type]
            [description]
        mode : str, optional
            [description], by default "full"

        Returns
        -------
        [type]
            [description]
        """
        # not used
        # reference: https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
        if mode == "same":
            pad_size = ((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2)
            image_height = image_height + pad_size[0]
            image_width = image_width + pad_size[1]

        # get output size of the data
        output_size = (
            image_height + kernel.shape[0] - 1,
            image_width + kernel.shape[1] - 1,
        )

        # zero-pad filter matrix
        pad_width = [
            (output_size[0] - kernel.shape[0], 0),
            (0, output_size[1] - kernel.shape[1]),
        ]
        kernel_padded = np.pad(kernel, pad_width=pad_width)

        # create the toeplitz matrix for each row of the filter
        toeplitz_list = []
        for i in range(kernel_padded.shape[0]):
            c = kernel_padded[
                i, :
            ]  # i th row of the F to define first column of toeplitz matrix

            # first row for the toeplitz function should be defined otherwise
            # the result is wrong
            r = np.hstack([c[0], np.zeros(int(image_width * image_height / 2) - 1)])

            # create the toeplitz matrix
            toeplitz_m = scipy.linalg.toeplitz(c, r)

            assert toeplitz_m.shape == (kernel_padded.shape[1], len(r))

            #     print(toeplitz_m.shape)
            toeplitz_list.append(toeplitz_m)

        # create block matrix
        zero_block = np.zeros(toeplitz_m.shape)
        block_seq = []
        for idx, block in enumerate(toeplitz_list):
            if idx == 0:
                block_seq.append([block, zero_block])
            else:
                block_seq.append([block, toeplitz_list[idx - 1]])
        doubly_block_mat = np.block(block_seq)
        return doubly_block_mat

    def _sample_kernel(self):
        """Sample a random Gabor kernel.

        Returns
        -------
        kernel : instance of skimage.filters.gabor_kernel
            A 2D Gabor kernel (K x K).
        kernel_params: dict
            A dictionary of keys and values of the corresponding
            2D Gabor ``kernel`` parameters.

        Raises
        ------
        ImportError
            if ``scikit-image`` is not installed.
        """
        frequency = rng.rand()  # spatial frequency
        theta = rng.uniform() * 2 * np.pi  # orientation in radians
        bandwidth = rng.uniform() * 5  # bandwidth of the filter
        n_stds = rng.randint(1, 4)

        # get the random kernel
        kernel_params = {
            "frequency": frequency,
            "theta": theta,
            "bandwidth": bandwidth,
            "n_stds": n_stds,
        }
        kernel = gabor_kernel(**kernel_params)
        return kernel, kernel_params

    def _apply_convolution(self, sample_X, kernel, image_height, image_width):
        """Apply convolution of a kernel to image data.

        Parameters
        ----------
        sample_X : np.ndarray (n_samples, n_dimensions)
            [description]
        kernel : [type]
            [description]
        image_height : int
            [description]
        image_width : int
            [description]

        Returns
        -------
        [type]
            [description]
        """
        output = _apply_convolution(
            sample_X, kernel, image_height=image_height, image_width=image_width
        )
        return output

    def sample_proj_mat(self, sample_inds, apply_conv_first=True):
        """
        Get the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated weighted projection matrix.

        Notes
        -----
        This will get the basis matrix based on the Gabor kernel and also
        the patch selection vector.
        """
        # store the kernel parameters of each "projection"
        proj_kernel_params = []

        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        if apply_conv_first:
            # sample kernel
            kernel, kernel_params = self._sample_kernel()

            # apply convolution
            output = self._apply_convolution(
                self.X[sample_inds, :],
                kernel=kernel,
                image_height=self.image_height,
                image_width=self.image_width,
            )

            # TODO: handle imaginary and real kernel convolution
            output = output[:, 0, ...].numpy()

            # reformat to vectorized shape
            assert output.ndim == 3
            output = output.reshape(len(sample_inds), self.n_features)

            # keep track of the kernel parameters
            proj_kernel_params.append(kernel_params)
        else:
            # generate random heights and widths of the patch. Note add 1 because numpy
            # needs is exclusive of the high end of interval
            rand_height = rng.randint(
                self.patch_height_min, self.patch_height_max + 1, size=None
            )
            rand_width = rng.randint(
                self.patch_width_min, self.patch_width_max + 1, size=None
            )

            # choose patch
            # get patch positions
            patch_idxs = self._get_rand_patch_idx(
                rand_height=rand_height, rand_width=rand_width
            )

            proj_mat[patch_idxs, :] = 1.0

        # sample mtry times different filters
        for idx in range(self.proj_dims):
            patch_weights = np.zeros(
                (self.n_features, self.proj_dims), dtype=np.float64
            )

            if apply_conv_first:
                # generate random heights and widths of the patch. Note add 1 because numpy
                # needs is exclusive of the high end of interval
                rand_height = rng.randint(
                    self.patch_height_min, self.patch_height_max + 1, size=None
                )
                rand_width = rng.randint(
                    self.patch_width_min, self.patch_width_max + 1, size=None
                )

                # choose patch
                # get patch positions
                patch_idxs = self._get_rand_patch_idx(
                    rand_height=rand_height, rand_width=rand_width
                )

                proj_mat[patch_idxs, idx] = 1.0
                # patch_weights[patch_idxs, idx] = output[:, patch_idxs]
            else:
                # sample kernel
                kernel, kernel_params = self._sample_kernel()

                # apply convolution
                output = self._apply_convolution(
                    self.X[sample_inds, :],
                    kernel=kernel,
                    image_height=rand_height,
                    image_width=rand_width,
                )

                # reformat to vectorized shape
                output = output.flatten()

                # keep track of the kernel parameters
                proj_kernel_params.append(kernel_params)

                # get the output weights
                patch_weights[:, patch_idxs] = output[:, patch_idxs]
                proj_mat[patch_idxs, idx] = patch_weights

        # apply projection matrix
        proj_X = output @ proj_mat

        return proj_X, proj_mat, proj_kernel_params


class SampleGraphSplitter(ObliqueSplitter):
    """Convolutional splitter.

    A class used to represent a 2D convolutional splitter, where splits
    are done on a convolutional patch.

    Note: The convolution function is currently just the
    summation operator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    image_height : int, optional (default=None)
        MORF required parameter. Image height of each observation.
    image_width : int, optional (default=None)
        MORF required parameter. Width of each observation.
    patch_height_max : int, optional (default=max(2, floor(sqrt(image_height))))
        MORF parameter. Maximum image patch height to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_height)))``.
    patch_height_min : int, optional (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    patch_width_max : int, optional (default=max(2, floor(sqrt(image_width))))
        MORF parameter. Maximum image patch width to randomly select from.
        If None, set to ``max(2, floor(sqrt(image_width)))``.
    patch_width_min : int (default=1)
        MORF parameter. Minimum image patch height to randomly select from.
    discontiguous_height : bool, optional (default=False)
        Whether or not the rows of the patch are taken discontiguously or not.
    discontiguous_width : bool, optional (default=False)
        Whether or not the columns of the patch are taken discontiguously or not.

    Methods
    -------
    sample_proj_mat
        Will compute projection matrix, which has columns as the vectorized
        convolutional patches.

    Notes
    -----
    This class assumes that data is vectorized in
    row-major (C-style), rather then column-major (Fotran-style) order.
    """

    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        sample_strategies: list,
        sample_dims: list,
        patch_dims: list = None,
    ):
        super(SampleGraphSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
        )

        if axis_sample_graphs is None and axis_data_dims is None:
            raise RuntimeError(
                "Either the sample graph must be instantiated, or "
                "the data dimensionality must be specified. Both are not right now."
            )

        # error check sampling graphs
        if axis_sample_graphs is not None:
            # perform error check on the passes in sample graphs and dimensions
            if len(axis_sample_graphs) != len(axis_sample_dims):
                raise ValueError(
                    f"The number of sample graphs \
                ({len(axis_sample_graphs)}) must match \
                the number of sample dimensions ({len(axis_sample_dims)}) in MORF."
                )
            if not all([x.ndim == 2 for x in axis_sample_graphs]):
                raise ValueError(
                    f"All axis sample graphs must be \
                                    2D matrices."
                )
            if not all([x.shape[0] == x.shape[1] for x in axis_sample_graphs]):
                raise ValueError(f"All axis sample graphs must be " "square matrices.")

            # XXX: could later generalize to remove this condition
            if not all([_check_symmetric(x) for x in axis_sample_graphs]):
                raise ValueError("All axis sample graphs must" "be symmetric.")

        # error check data dimensions
        if axis_data_dims is not None:
            # perform error check on the passes in sample graphs and dimensions
            if len(axis_data_dims) != len(axis_sample_dims):
                raise ValueError(
                    f"The number of data dimensions "
                    "({len(axis_data_dims)}) must match "
                    "the number of sample dimensions ({len(axis_sample_dims)}) in MORF."
                )

            if X.shape[1] != np.sum(axis_data_dims):
                raise ValueError(
                    f"The specified data dimensionality "
                    "({np.sum(axis_data_dims)}) does not match the dimensionality "
                    "of the data (i.e. # columns in X: {X.shape[1]})."
                )

        # set sample dimensions
        self.structured_data_shape = sample_dims
        self.sample_dims = sample_dims
        self.sample_strategies = sample_strategies

    def sample_proj_mat(self, sample_inds):
        """
        Get the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """
        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # generate random heights and widths of the patch. Note add 1 because numpy
        # needs is exclusive of the high end of interval
        rand_heights = rng.randint(
            self.patch_height_min, self.patch_height_max + 1, size=self.proj_dims
        )
        rand_widths = rng.randint(
            self.patch_width_min, self.patch_width_max + 1, size=self.proj_dims
        )

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.proj_dims):
            rand_height = rand_heights[idx]
            rand_width = rand_widths[idx]
            # get patch positions
            patch_idxs = self._get_rand_patch_idx(
                rand_height=rand_height, rand_width=rand_width
            )

            # get indices for this patch
            proj_mat[patch_idxs, idx] = 1

        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat

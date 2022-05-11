import numpy as np
import numpy.random as rng
from scipy.sparse import issparse
from sklearn.base import is_classifier
from sklearn.tree import _tree
from sklearn.utils import check_random_state
import numbers

from ._split import BaseObliqueSplitter
from .oblique_tree import ObliqueSplitter, ObliqueTreeClassifier, ObliqueTree

from .conv import _apply_convolution
from .morf_split import Conv2DSplitter, GaborSplitter

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE


class Conv2DObliqueTreeClassifier(ObliqueTreeClassifier):
    """Convolutional patch oblique tree classifier.

    Parameters
    ----------
    n_estimators : int, optional
        [description], by default 500
    max_depth : [type], optional
        [description], by default None
    min_samples_split : int, optional
        [description], by default 1
    min_samples_leaf : int, optional
        [description], by default 1
    min_impurity_decrease : int, optional
        [description], by default 0
    min_impurity_split : int, optional
        [description], by default 0
    feature_combinations : float, optional
        [description], by default 1.5
    max_features : int, optional
        [description], by default 1
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
    bootstrap : bool, optional
        [description], by default True
    n_jobs : [type], optional
        [description], by default None
    random_state : [type], optional
        [description], by default None
    warm_start : bool, optional
        [description], by default False
    verbose : int, optional
        [description], by default 0
    """

    def __init__(
        self,
        *,
        max_depth=np.inf,
        min_samples_split=1,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        min_impurity_split=0,
        feature_combinations=1.5,
        max_features=1,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height=False,
        discontiguous_width=False,
        bootstrap=True,
        random_state=None,
        warm_start=False,
        verbose=0,
    ):
        super(Conv2DObliqueTreeClassifier, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            feature_combinations=feature_combinations,
            max_features=max_features,
            random_state=random_state,
            bootstrap=bootstrap,
            warm_start=warm_start,
            verbose=verbose,
        )

        # s-rerf params
        self.discontiguous_height = discontiguous_height
        self.discontiguous_width = discontiguous_width
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height_max = patch_height_max
        self.patch_height_min = patch_height_min
        self.patch_width_max = patch_width_max
        self.patch_width_min = patch_width_min

    def _check_patch_params(self):
        # Check that image_height and image_width are divisors of
        # the num_features.  This is the most we can do to
        # prevent an invalid value being passed in.
        if (self.n_features_ % self.image_height) != 0:
            raise ValueError("Incorrect image_height given:")
        else:
            self.image_height_ = self.image_height
        if (self.n_features_ % self.image_width) != 0:
            raise ValueError("Incorrect image_width given:")
        else:
            self.image_width_ = self.image_width

        # If patch_height_{min, max} and patch_width_{min, max} are
        # not set by the user, set them to defaults.
        if self.patch_height_max is None:
            self.patch_height_max_ = max(2, np.floor(np.sqrt(self.image_height_)))
        else:
            self.patch_height_max_ = self.patch_height_max
        if self.patch_width_max is None:
            self.patch_width_max_ = max(2, np.floor(np.sqrt(self.image_width_)))
        else:
            self.patch_width_max_ = self.patch_width_max
        if 1 <= self.patch_height_min <= self.patch_height_max_:
            self.patch_height_min_ = self.patch_height_min
        else:
            raise ValueError("Incorrect patch_height_min")
        if 1 <= self.patch_width_min <= self.patch_width_max_:
            self.patch_width_min_ = self.patch_width_min
        else:
            raise ValueError("Incorrect patch_width_min")

    def _set_splitter(self, X, y):
        return Conv2DSplitter(
            X,
            y,
            self.max_features,
            self.feature_combinations,
            self.random_state,
            self.image_height_,
            self.image_width_,
            self.patch_height_max_,
            self.patch_height_min_,
            self.patch_width_max_,
            self.patch_width_min_,
            self.discontiguous_height,
            self.discontiguous_width,
        )

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        """
        Predict final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The training samples.
        y : array of shape [n_samples]
            Labels for the training samples.
        sample_weight : [type], optional
            [description], by default None
        check_input : bool, optional
            [description], by default True
        X_idx_sorted : [type], optional
            [description], by default None

        Returns
        -------
        ObliqueTreeClassifier
            The fit classifier.

        Raises
        ------
        ValueError
            [description]
        """
        # check random state - sklearn
        random_state = check_random_state(self.random_state)

        # check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        # check_y_params = dict(ensure_2d=False, dtype=None)
        # X, y = self._validate_data(
        #     X, y, validate_separately=(check_X_params, check_y_params)
        # )
        # if issparse(X):
        #     X.sort_indices()

        #     if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
        #         raise ValueError(
        #             "No support for np.int64 index based " "sparse matrices"
        #         )

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_

        # check patch parameters
        self._check_patch_params()

        # mark if tree is used for classification
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        # Check parameters
        self.max_depth = (2 ** 31) - 1 if self.max_depth is None else self.max_depth
        # self.max_leaf_nodes = (-1 if self.max_leaf_nodes is None
        #                   else self.max_leaf_nodes)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. Allowed string "
                    'values are "auto", "sqrt" or "log2".'
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 0
        self.max_features = max_features

        # note that this is done in scikit-learn, but results in a matrix
        # multiplication error because y is 2D
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        # create the splitter
        splitter = self._set_splitter(X, y)

        # create the Oblique tree
        self.tree = ObliqueTree(
            splitter,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_depth,
            self.min_impurity_split,
            self.min_impurity_decrease,
        )
        self.tree.build()
        return self


class GaborTree(ObliqueTree):
    """A Gabor tree.

    Parameters
    ----------
    splitter : [type]
        [description]
    min_samples_split : [type]
        [description]
    min_samples_leaf : [type]
        [description]
    max_depth : [type]
        [description]
    min_impurity_split : [type]
        [description]
    min_impurity_decrease : [type]
        [description]
    """

    def __init__(
        self,
        splitter,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        min_impurity_split,
        min_impurity_decrease,
    ):
        super(GaborTree, self).__init__(
            splitter,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            min_impurity_split,
            min_impurity_decrease,
        )

    def _transform_data(self, X, proj_vec, transform_params):
        kernel = gabor_kernel(**cur.transform_params)
        output = _apply_convolution(X[i], kernel, image_height, image_width)
        proj_X = output @ cur.proj_vec
        return proj_X


class GaborTreeClassifier(Conv2DObliqueTreeClassifier):
    """Gabor tree classifier.

    Parameters
    ----------
    max_depth : [type], optional
        [description], by default np.inf
    min_samples_split : int, optional
        [description], by default 1
    min_samples_leaf : int, optional
        [description], by default 1
    min_impurity_decrease : int, optional
        [description], by default 0
    min_impurity_split : int, optional
        [description], by default 0
    feature_combinations : float, optional
        [description], by default 1.5
    max_features : int, optional
        [description], by default 1
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
    bootstrap : bool, optional
        [description], by default True
    random_state : [type], optional
        [description], by default None
    warm_start : bool, optional
        [description], by default False
    verbose : int, optional
        [description], by default 0
    """

    def __init__(
        self,
        *,
        max_depth=np.inf,
        min_samples_split=1,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        min_impurity_split=0,
        feature_combinations=1.5,
        max_features=1,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height=False,
        discontiguous_width=False,
        bootstrap=True,
        random_state=None,
        warm_start=False,
        verbose=0,
    ):
        super(GaborTreeClassifier, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            feature_combinations=feature_combinations,
            max_features=max_features,
            image_height=image_height,
            image_width=image_width,
            patch_height_max=patch_width_max,
            patch_height_min=patch_width_min,
            patch_width_max=patch_width_max,
            patch_width_min=patch_width_min,
            discontiguous_height=discontiguous_height,
            discontiguous_width=discontiguous_width,
            random_state=random_state,
            bootstrap=bootstrap,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _tree_class(self):
        return GaborTree

    def _set_splitter(self, X, y):
        return GaborSplitter(
            X,
            y,
            self.max_features,
            self.feature_combinations,
            self.random_state,
            self.image_height_,
            self.image_width_,
            self.patch_height_max_,
            self.patch_height_min_,
            self.patch_width_max_,
            self.patch_width_min_,
            self.discontiguous_height,
            self.discontiguous_width,
        )

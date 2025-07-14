import numpy as np
from joblib import Parallel, delayed
from scipy import stats as ss
from sklearn.utils import shuffle
from statsmodels.stats.multitest import multipletests

from ..ensemble._supervised_forest import (
    ObliqueRandomForestClassifier,
    PatchObliqueRandomForestClassifier,
)


class NeuroExplainableOptimalFIT:
    """
    A feature important testing method for assessing feature importance in
    datasets using permutation testing with oblique random forest classifiers
    (MORF[1] or SPORF[2]). This method provides p-values for each feature to
    make it possible choosing the important features based on printed p-values.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, default=1
        Controls the verbosity when fitting and predicting.

    n_permutations : int, default=5000
        Number of permutations to use for the test.

    clf_type: {"SPORF", "MORF"}, default="SPORF"
        Type of the estimators in the FIT algorithm. See as reference.

    alpha: float, default=0.05
        The threshold for p-values of feature importance. When p-value<0.05,
        the feature can be recognized as important feature.

    *Attributes:
    ----------
    feature_importances_ : ndarray, shape=(n_features,)
        Importances of each feature.

    p_values_ : ndarray, shape=(n_features,)
        P-values for each features. Calculated by neofit algorithm.

    significant_features_ : ndarray, shape=(n_features,)
        Selected important features which have p-values below alpha(0.05).

    Reference:
    ----------
    [1] Adam Li, Haoyin Xu, et al. "Manifold Oblique Random Forests.",/
      IEEE Transactions on Neural Networks and Learning Systems, 2023.
    [2] Tomita, Tyler M., et al. "Sparse Projection Oblique Randomer Forests.",/
      Journal of Machine Learning Research, 21(104), 1-39, 2020.

    Example:
    --------
    >>> from treeple.stats import NeuroExplainableOptimalFIT
    >>> from sklearn.dataset import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=1000,
    ...                          n_informative=2, n_redundant=0,
    ...                          random_state=0, shuffle=False)
    >>> neofit = NeuroExplainableOptimalFIT(
    ...         n_estimators = 5000,
    ...         n_permutations = 100000,
    ...         clf_type = "SPORF",
    ...         alpha = 0.05,
    ...         verbose = 1)
    >>> p_values = neofit.feat_imp_test(X, y)
    >>> p_values, important_features, X_important = neofit.get_significant_features(X, y)
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        n_jobs=-1,
        random_state=None,
        verbose=1,
        n_permutations=5000,
        clf_type="SPORF",
        alpha=0.05,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.n_permutations = n_permutations
        self.clf_type = clf_type
        self.alpha = alpha

    def construct_orf(self, random_state=None):
        """
        Construct the estimator based on the type chose in the params.


        """
        if self.clf_type == "MORF":
            return PatchObliqueRandomForestClassifier(
                n_estimators=1,
                max_patch_dims=np.array((5, 2)),
                data_dims=np.array((28, 28)),
                n_jobs=1,
                max_features=self.max_features,
                bootstrap=False,
                verbose=self.verbose,
                oob_score=False,
                random_state=random_state,
            )
        elif self.clf_type == "SPORF":
            return ObliqueRandomForestClassifier(
                n_estimators=1,
                n_jobs=1,
                max_features=self.max_features,
                bootstrap=False,
                verbose=self.verbose,
                oob_score=False,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Classifier type {self.clf_type} not implemented yet.")

    def train(self, ii, X, y):
        """
        Train a classifier based on MORF or SPORF,
        and return the feature importance and OOB decisions.

        Parameters:
        -----------
        ii: int
            The random seed.
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values. In classification problems, it should be class labels.

        Returns:
        --------
        tuple
            (feature_importance, padded_oob_decisions)
            feature_importance: array-like of shape (n_features,)
                The feature importance values for all the input features in training set.
            padded_oob_decisions: array-like of shape (n_samples, n_classes)
                The OOB decisions.
        """
        rng = np.random.default_rng(ii if self.random_state is None else self.random_state + ii)
        bootstrap_idx = rng.choice(
            len(X), size=len(X), replace=True
        )  # make sure the bootstrap strategy is the consistent

        orf = self.construct_orf(random_state=ii)
        orf.fit(X[bootstrap_idx, :], y[bootstrap_idx])
        oob_idx = np.setdiff1d(np.arange(len(y)), bootstrap_idx)
        oob_decisions = orf.predict_proba(X[oob_idx, :])
        padded_oob_decisions = np.zeros((len(y), oob_decisions.shape[1]))
        padded_oob_decisions[oob_idx, :] = oob_decisions

        return orf.feature_importances_, padded_oob_decisions

    @staticmethod
    def compute_ranks(feature_importance):
        """
        Precompute ranks of features.
        The rank from top to the bottom will be from most important to the least important.

        Parameters:
        -----------
        feature_importance: array-like of shape (n_samples, n_features)
            The feature importance get from the forest.

        Returns:
        --------
        ranks: array-like of shape (n_samples, n_features)
            The ranks of the feature importance.
        """
        return np.apply_along_axis(
            lambda x: ss.rankdata(1 - x, method="max"), axis=1, arr=feature_importance
        )

    def statistics(self, ranks, idx):
        """
        Calculate the feature importance test statistic.
        Ensures correct rank ordering and functionality preservation.

        Parameters:
        -----------
        ranks: array-like of shape (n_samples, n_features)
            The ranks of the feature importance.
        idx: array-like of shape (2 * n_estimators,)
            The indices of the feature importance.

        Returns:
        --------
        stat: array-like of shape (n_features,)
            The feature importance test statistic.
        """
        stat = np.zeros(ranks.shape[1])

        for ii in range(self.n_estimators):
            r = ranks[idx[ii]]
            r_0 = ranks[idx[self.n_estimators + ii]]
            stat += (r_0 > r) * 1  # Boolean Comparison

        stat /= self.n_estimators
        return stat

    def perm_stat(self, ranks):
        """
        Helper function that calculates the null distribution.

        Parameters:
        -----------
        ranks: array-like of shape (n_samples, n_features)
            The ranks of the feature importance.

        Returns:
        --------
        stat: array-like of shape (n_features,)
            The feature importance test statistic.
        """
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(2 * self.n_estimators)
        return self.statistics(ranks, idx)

    def test(self, feature_importance):
        """
        Permutation test to compare real vs shuffled feature importance.

        Parameters:
        -----------
        feature_importance: array-like of shape (n_samples, n_features)
            The feature importance.
        n_permutations: int
            The number of permutations. Ref to Coleman et al. [3]

        Returns:
        --------
        tuple
            (stat, p_val): The test statistic and p-values.

        Reference:
        ----------
        [3] Coleman, T., et al. "Distributed, partial feature ranking using sparse oblique trees." ,/
        arXiv preprint arXiv:2310.19722 (2023).
        """
        # Precompute ranks once using the correct method
        ranks = self.compute_ranks(feature_importance)

        # Compute actual statistic
        stat = self.statistics(ranks, np.arange(2 * self.n_estimators))

        # Parallel computation of null distribution
        null_stat = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self.perm_stat)(ranks) for _ in range(self.n_permutations)
            )
        )

        # Compute p-values
        count = np.sum(null_stat >= stat, axis=0)
        p_val = (1 + count) / (1 + self.n_permutations)

        return stat, p_val

    def get_p(self, feat_imp_all, feat_imp_all_rand):
        """
        Calculate p-values with multiple testing correction.

        Parameters:
        -----------
        feat_imp_all: array-like of shape (n_estimators, n_features)
            Feature importance from original data.

        feat_imp_all_rand: array-like of shape (n_estimators, n_features)
            Feature importance from shuffled data.

        Returns:
        --------
        p_corrected: array-like of shape (n_features,)
            Corrected p-values.
        """
        _, p = self.test(np.concatenate((feat_imp_all, feat_imp_all_rand)))

        # Apply Bonferroni-Holm correction
        p_corrected = multipletests(p, method="holm")[1]
        return p_corrected

    def feat_imp_test(self, X, y):
        """
        Main method to test for significant features.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        p_corrected : array-like of shape (n_features,)
            Corrected p-values for each feature.
        """
        # Training on original data
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train)(ii, X, y) for ii in range(self.n_estimators)
        )
        feat_imp_all, _ = zip(*results)

        # Training on shuffled data
        y_shuffled = shuffle(y, random_state=0)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train)(ii, X, y_shuffled) for ii in range(self.n_estimators)
        )
        feat_imp_all_rand, _ = zip(*results)

        # Computing p-values
        p_corrected = self.get_p(np.array(feat_imp_all), np.array(feat_imp_all_rand))

        return p_corrected

    def get_significant_features(self, X, y):
        """
        Find significant features (p < alpha) and return their indices.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        significant_features : array-like
            Boolean array indicating significant features.
        p_values : array-like
            Corrected p-values for each feature.
        X_important : array-like
            Data with only significant features.
        """
        p_values = self.feat_imp_test(X, y)
        significant_features = p_values < self.alpha

        print(
            f"Found {np.sum(significant_features)} significant features out of {len(significant_features)}"
        )
        # print the top 10 features: name, index
        if X.shape[1] > 1:
            print(f"Significant features: {np.where(significant_features)[0][:10]}")
        else:
            print(f"Significant feature: {np.where(significant_features)[0][:10]}")

        if np.sum(significant_features) > 0:
            X_important = X[:, significant_features]
        else:
            X_important = X

        return p_values, significant_features, X_important

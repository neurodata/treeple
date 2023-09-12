import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..ensemble import HonestForestClassifier


def auc_calibrator(tree, X, y, test_size=0.2, permute_y=False):
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=test_size
    )

    # individual tree permutation of y labels
    if permute_y:
        y_train = np.random.permutation(y_train)

    tree.fit(X_train, y_train)
    y_pred = tree.predict_proba(X_test)[:, 1]

    # Fill test set posteriors & set rest NaN
    posterior = np.full(y.shape, np.nan)
    posterior[indices_test] = y_pred

    return posterior


def perm_stat(clf, x, z, y, random_state=None):
    if z is not None:
        permuted_Z = np.random.permutation(z)
        X_permutedZ = np.hstack((x, permuted_Z))
    else:
        X_permutedZ = np.random.permutation(x)

    perm_stat = clf.statistic(X_permutedZ, y)
    return perm_stat


def perm_half(clf, z, y, x_pos):
    permuted_Z = np.random.permutation(z)
    perm_stat, perm_pos = clf.statistic(permuted_Z, y, return_pos=True)
    null_pos = forest_pos(x_pos + perm_pos, y)
    null_stat = roc_auc_score(null_pos[:, 0], null_pos[:, 1], max_fpr=clf.limit)

    return null_stat


def pos_diff(observe_pos, perm_pos, limit):
    total_pos = np.random.shuffle(np.concatenate((observe_pos, perm_pos)))

    half_ind = len(total_pos) * 0.5
    half_pos = total_pos[:half_ind]
    end_pos = total_pos[half_ind:]

    half_pos_final = forest_pos(half_pos, y)
    half_stat = roc_auc_score(half_pos_final[:, 0], half_pos_final[:, 1], max_fpr=limit)

    end_pos_final = forest_pos(end_pos, y)
    end_stat = roc_auc_score(end_pos_final[:, 0], end_pos_final[:, 1], max_fpr=limit)

    return abs(half_stat - end_stat)


def forest_pos(posterior, y):
    # Average all posteriors
    posterior_final = np.nanmean(posterior, axis=0)

    # Ignore all NaN values (samples not tested)
    true_final = y.ravel()[~np.isnan(posterior_final)].reshape(-1, 1)
    posterior_final = posterior_final[~np.isnan(posterior_final)].reshape(-1, 1)

    return np.hstack((true_final, posterior_final))


class MIGHT:
    def __init__(
        self,
        n_estimators=500,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        limit=0.05,
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            tree_estimator=tree_estimator,
        )
        self.limit = limit

    def statistic(
        self,
        x,
        y,
        stat="AUC",
        workers=1,
        test_size=0.2,
        initial=True,
        return_pos=False,
        permute_y=False,
    ):
        # Initialize trees
        if initial:
            self.clf.fit(x[0:2], y.ravel()[0:2])

        # Compute posteriors with train test splits
        posterior = Parallel(n_jobs=workers)(
            delayed(auc_calibrator)(tree, x, y.ravel(), test_size, permute_y)
            for tree in (self.clf.estimators_)
        )

        posterior_final = forest_pos(posterior, y)

        if stat == "AUC":
            self.stat = roc_auc_score(
                posterior_final[:, 0], posterior_final[:, 1], max_fpr=self.limit
            )
        elif stat == "MI":
            H_YX = np.mean(entropy(posterior_final[:, 1], base=np.exp(1)))
            _, counts = np.unique(posterior_final[:, 0], return_counts=True)
            H_Y = entropy(counts, base=np.exp(1))
            self.stat = max(H_Y - H_YX, 0)

        if return_pos:
            return self.stat, posterior

        return self.stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        observe_stat = self.statistic(x, y)

        null_dist = np.array(
            Parallel(n_jobs=workers)([delayed(perm_stat)(self, x, None, y) for _ in range(reps)])
        )
        pval = (1 + (null_dist >= observe_stat).sum()) / (1 + reps)

        return observe_stat, null_dist, pval


class MIGHT_MV:
    def __init__(
        self,
        n_estimators=500,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        limit=0.05,
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            tree_estimator=tree_estimator,
        )
        self.limit = limit

    def statistic(
        self,
        x,
        y,
        stat="AUC",
        workers=1,
        test_size=0.2,
        initial=True,
        return_pos=False,
        permute_y=False,
    ):
        # Initialize trees
        if initial:
            self.clf.fit(x[0:2], y.ravel()[0:2])

        # Compute posteriors with train test splits
        posterior = Parallel(n_jobs=workers)(
            delayed(auc_calibrator)(tree, x, y.ravel(), test_size, permute_y)
            for tree in (self.clf.estimators_)
        )

        posterior_final = forest_pos(posterior, y)

        if stat == "AUC":
            self.stat = roc_auc_score(
                posterior_final[:, 0], posterior_final[:, 1], max_fpr=self.limit
            )
        elif stat == "MI":
            H_YX = np.mean(entropy(posterior_final[:, 1], base=np.exp(1), axis=1))
            _, counts = np.unique(posterior_final[:, 0], return_counts=True)
            H_Y = entropy(counts, base=np.exp(1))
            self.stat = max(H_Y - H_YX, 0)

        if return_pos:
            return self.stat, posterior

        return self.stat

    def test(self, x, z, y, reps=1000, workers=1, random_state=None):
        XZ = np.hstack((x, z))
        observe_stat = self.statistic(XZ, y)

        null_dist = np.array(
            Parallel(n_jobs=workers)([delayed(perm_stat)(self, x, z, y) for _ in range(reps)])
        )
        pval = (1 + (null_dist >= observe_stat).sum()) / (1 + reps)

        return observe_stat, null_dist, pval

    def test_twin(self, x, z, y, reps=1000, workers=1, random_state=None):
        x_stat, x_pos = self.statistic(x, y, return_pos=True)

        # TODO: determine whether we need the forest

        z_stat, z_pos = self.statistic(z, y, return_pos=True)

        observe_pos = forest_pos(x_pos + z_pos, y)
        observe_stat = roc_auc_score(observe_pos[:, 0], observe_pos[:, 1], max_fpr=self.limit)

        null_dist = np.array(
            Parallel(n_jobs=workers)([delayed(perm_half)(self, z, y, x_pos) for _ in range(reps)])
        )
        pval = (1 + (null_dist >= observe_stat).sum()) / (1 + reps)

        return observe_stat, null_dist, pval

    def test_diff(self, x, z, y, reps=1000, workers=1):
        XZ = np.hstack((x, z))
        observe_stat, observe_pos = self.statistic(XZ, y, return_pos=True)

        # Compute statistic for permuted sets
        permuted_Z = np.random.permutation(z)
        X_permutedZ = np.hstack((x, permuted_Z))
        perm_stat, perm_pos = self.statistic(X_permutedZ, y, return_pos=True)

        # Boostrap sample the posterior from the two forests
        null_stats = np.array(
            Parallel(n_jobs=workers)(
                [delayed(pos_diff)(observe_pos, perm_pos, limit=self.limit) for _ in range(reps)]
            )
        )

        stat = observe_stat - perm_stat

        pval = (1 + (null_stats >= stat).sum()) / (1 + reps)
        return stat, null_stats, pval

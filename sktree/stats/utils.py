from numpy.typing import ArrayLike
import numpy as np

from sklearn.metrics import mean_squared_error, roc_auc_score

METRIC_FUNCTIONS = {
    'mse': mean_squared_error,
    'auc': roc_auc_score
}

def pvalue(observe_stat: float, permuted_stat: ArrayLike) -> float:
    """Compute pvalue with Coleman method.

    Implements the pvalue calculation from Algorithm 1. See
    :footcite:`coleman2022scalable` for full details.

    Parameters
    ----------
    observe_stat : float
        The observed test statistic.
    permuted_stat : ArrayLike of shape (n_repeats,)
        The array of test statistics computed on permutations.

    Returns
    -------
    pval : float
        The pvalue.
    """
    n_repeats = len(permuted_stat)
    pval = (1 + (permuted_stat >= observe_stat).sum()) / (1 + n_repeats)
    return pval


def compute_null_distribution(X_test, y_test, forest, perm_forest, metric: str='mse', n_repeats: int=1000, seed: int=None):
    """Compute null distribution using Coleman method.

    The null distribution is comprised of two forests.

    Parameters
    ----------
    X_test : _type_
        _description_
    y_test : _type_
        _description_
    forest : _type_
        _description_
    perm_forest : _type_
        _description_
    metric : str, optional
        _description_, by default 'mse'
    n_repeats : int, optional
        _description_, by default 1000
    seed : int, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    rng = np.random.default_rng(seed)

    metric_func = METRIC_FUNCTIONS[metric]

    # sample two sets of equal number of trees from the combined forest
    y_pred_proba_normal = forest.predict_proba(X_test)
    y_pred_proba_perm = perm_forest.predict_proba(X_test)
    all_y_pred = np.concatenate((y_pred_proba_normal, y_pred_proba_perm), axis=0)

    n_samples = len(y_test)

    # pre-allocate memory for the index array
    index_arr = np.arange(n_samples * 2, dtype=int)

    metric_star = []
    metric_star_pi = []
    for idx in range(n_repeats):
        # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
        rng.shuffle(index_arr)
        first_half_index = index_arr[:n_samples]
        second_half_index = index_arr[n_samples:]

        # compute two instances of the metric from the sampled trees
        first_half_metric = metric_func(y_true=y_test, y_pred=all_y_pred[first_half_index])
        second_half_metric = metric_func(y_true=y_test, y_pred=all_y_pred[second_half_index])

        metric_star.append(first_half_metric)
        metric_star_pi.append(second_half_metric)
    
    return metric_star, metric_star_pi
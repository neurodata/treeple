import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import entropy


def approximate_clf_mutual_information_with_H_Y(
    means, covs, n=10000, class_probs=[0.5, 0.5], base=np.exp(1), seed=None
):
    """Approximate MI for multivariate Gaussian for a classification setting.

    Parameters
    ----------
    means : list of ArrayLike of shape (n_dims_,)
        A list of means to sample from for each class.
    covs : list of ArrayLike of shape (n_dims_, n_dims_)
        A list of covariances to sample from for each class.
    n : total number of samples
        The total number of simulation samples
    class_probs : list, optional
        List of class probabilities, by default [0.5, 0.5] for
        balanced binary classification.
    base : float, optional
        The bit base to use, by default np.exp(1) for natural logarithm.
    seed : int, optional
        Random seed for the multivariate normal, by default None.

    Returns
    -------
    I_XY : float
        Estimated mutual information.
    H_Y : float
        Estimated entropy of Y, the class labels.
    H_Y_on_X : float
        The conditional entropy of Y given X.
    """
    np.random.seed(seed)
    P_Y = class_probs

    # Generate samples
    pdf_class = []
    X = []
    for i in range(len(means)):
        pdf_class.append(multivariate_normal(means[i], covs[i], allow_singular=True))
        X.append(
            np.random.normal(means[i], covs[i], size=int(n * P_Y[i])).reshape(-1, 1)
        )

    X = np.vstack(X)

    # Calculate P(X) by law of total probability
    P_X_l = []
    P_X_on_Y = []
    for i in range(len(means)):
        P_X_on_Y.append(pdf_class[i].pdf(X))
        P_X_l.append(P_X_on_Y[-1] * P_Y[i])
    P_X = sum(P_X_l)

    # Calculate P(Y|X) by Bayes' theorem
    P_Y_on_X = []
    for i in range(len(means)):
        P_Y_on_X.append((P_X_on_Y[i] * P_Y[i] / P_X).reshape(-1, 1))

    P_Y_on_X = np.hstack(P_Y_on_X)
    P_Y_on_X = P_Y_on_X[~np.isnan(P_Y_on_X)].reshape(-1, 2)

    # Calculate the entropy of Y by class counts
    H_Y = entropy(P_Y, base=base)

    # Calculate the conditional entropy of Y on X
    H_Y_on_X = np.mean(entropy(P_Y_on_X, base=base, axis=1))

    MI = H_Y - H_Y_on_X
    return MI, H_Y, H_Y_on_X
  

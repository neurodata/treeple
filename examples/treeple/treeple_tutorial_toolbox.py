import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from scipy.stats import entropy


def Calculate_SA(y_true, y_pred_proba, max_fpr=0.02) -> float:
    # check the shape of true labels
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # find the positive class and calculate fpr and tpr
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    sa98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label="ROC Curve")

    spec = int((1 - max_fpr) * 100)
    plt.axvline(
        x=max_fpr,
        color="r",
        ymin=0,
        ymax=sa98,
        label="S@" + str(spec) + " = " + str(round(sa98, 2)),
        linestyle="--",
    )
    plt.axhline(y=sa98, xmin=0, xmax=max_fpr, color="r", linestyle="--")
    plt.legend()

    return sa98


def Calculate_MI(y_true, y_pred_proba):
    # calculate the conditional entropy
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))

    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    # calculate the entropy of labels
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def Calculate_pAUC(y_true, y_pred_proba, max_fpr=0.1) -> float:
    # check the shape of true labels
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # find the positive class and calculate fpr and tpr
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label="ROC Curve")
    # Calculate pAUC at the specific threshold
    pAUC = roc_auc_score(y_true, y_pred_proba[:, 1], max_fpr=max_fpr)

    pos = np.where(fpr == max_fpr)[0][-1]
    plt.fill_between(
        fpr[:pos],
        tpr[:pos],
        color="r",
        alpha=0.6,
        label="pAUC@90 = " + str(round(pAUC, 2)),
        linestyle="--",
    )
    plt.legend()
    return pAUC

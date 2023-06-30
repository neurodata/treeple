from sklearn.utils.validation import check_is_fitted


def check_is_forest(
    est,
    allow_tree=False,
    ensure_fitted: bool = True,
):
    if ensure_fitted:
        check_is_fitted(est)

    if not hasattr(est, "apply"):
        raise ValueError(f"estimator {est} must be a tree or forest")

    if not allow_tree and not hasattr(est, "estimator"):
        raise ValueError("estimator must be a forest, not a tree")

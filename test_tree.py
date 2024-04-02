from sklearn.utils.estimator_checks import check_fit_score_takes_y

from sktree.tree import MultiViewDecisionTreeClassifier

est = MultiViewDecisionTreeClassifier()
est.partial_fit
check_fit_score_takes_y("mvtree", MultiViewDecisionTreeClassifier())

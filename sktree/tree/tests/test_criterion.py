import pytest
import numpy as np
from sktree.tree.unsupervised._unsup_criterion import CriterionTester


def test_node_impurity_equality():
    # Create instances of FastBIC and FasterBIC with test data
    n_outputs = 1
    n_classes = np.array([2])

    X = np.array([0, 1, 2, 3, 5, 10, 20, 200]).reshape(-1).astype(np.float32)
    n_samples = X.shape[0]

    # Create a new array which will be used to store nonzero
    # samples from the feature of interest
    samples = np.arange(n_samples, dtype=np.intp)
    sample_weight = np.ones(n_samples, dtype=np.float64)
    weighted_n_samples = np.sum(sample_weight)
    
    crit_tester = CriterionTester()
    crit_tester.init(sample_weight, weighted_n_samples, samples, X)
    crit_tester.reset()
    for pos in range(1, n_samples):
        print("Doing pos: ", pos)
        crit_tester.update(pos)
        fast_bic_imp, faster_bic_imp = crit_tester.node_impurity()
        print(fast_bic_imp, faster_bic_imp)
        assert fast_bic_imp == faster_bic_imp

        (fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right) = crit_tester.children_impurity()
        print((fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right))
        assert fast_bic_left == faster_bic_left
        assert fast_bic_right == faster_bic_right

        fast_bic_imp, faster_bic_imp = crit_tester.proxy_impurity()
        print(fast_bic_imp, faster_bic_imp)
        assert fast_bic_imp == faster_bic_imp

        (fast_bic_weights, faster_bic_weights) = crit_tester.weighted_n_samples()
        assert all(weight == other_weight for weight, other_weight in zip(fast_bic_weights, faster_bic_weights))
        print(crit_tester.weighted_n_samples())

        fast_sum_right, faster_sum_right = crit_tester.sum_right()
        assert fast_sum_right == faster_sum_right

    crit_tester.reset()
    crit_tester.update(1)

    crit_tester.set_sample_pointers(3, 6)
    crit_tester.init_feature_vec()
    crit_tester.update(5)
    n_samples = 3

    X[0] = 100
    print("Doing pos: ", 5)
    fast_bic_imp, faster_bic_imp = crit_tester.node_impurity()
    print(fast_bic_imp, faster_bic_imp)
    assert fast_bic_imp == faster_bic_imp

    (fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right) = crit_tester.children_impurity()
    print((fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right))
    assert fast_bic_left == faster_bic_left
    assert fast_bic_right == faster_bic_right

    print(crit_tester.weighted_n_samples())
    (fast_bic_weights,faster_bic_weights) = crit_tester.weighted_n_samples()
    assert all(weight == other_weight for weight, other_weight in zip(fast_bic_weights, faster_bic_weights))

    fast_bic_imp, faster_bic_imp = crit_tester.proxy_impurity()
    print(fast_bic_imp, faster_bic_imp)
    assert fast_bic_imp == faster_bic_imp

    crit_tester.init_feature_vec()
    for pos in range(4, 6):
        print("Doing pos: ", pos)
        crit_tester.update(pos)
        fast_bic_imp, faster_bic_imp = crit_tester.node_impurity()
        print(fast_bic_imp, faster_bic_imp)
        assert fast_bic_imp == faster_bic_imp

        (fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right) = crit_tester.children_impurity()
        print((fast_bic_left, fast_bic_right), (faster_bic_left, faster_bic_right))
        assert fast_bic_left == faster_bic_left
        assert fast_bic_right == faster_bic_right

        print(crit_tester.weighted_n_samples())
        (fast_bic_weights,faster_bic_weights) = crit_tester.weighted_n_samples()
        assert all(weight == other_weight for weight, other_weight in zip(fast_bic_weights, faster_bic_weights))

        fast_bic_imp, faster_bic_imp = crit_tester.proxy_impurity()
        print(fast_bic_imp, faster_bic_imp)
        assert fast_bic_imp == faster_bic_imp

        fast_sum_right, faster_sum_right = crit_tester.sum_right()
        assert fast_sum_right == faster_sum_right
    # assert False
    # fast_bic = FastBIC(n_outputs, n_classes)
    # faster_bic = FasterBIC(n_outputs, n_classes)

    # fast_bic.init(
    #     sample_weight,
    #     weighted_n_samples,
    #     samples,
    #     feature_values,
    # )
    # # Call the cpdef node_impurity function for both instances
    # impurity_fast = fast_bic.node_impurity()
    # impurity_faster = faster_bic.node_impurity()

    # # Compare the results using pytest's built-in assert statement
    # assert pytest.approx(impurity_fast, rel=1e-5) == impurity_faster


# def test_children_impurity_equality():
#     # Create instances of FastBIC and FasterBIC with test data
#     fast_bic = FastBIC(...)
#     faster_bic = FasterBIC(...)

#     # Call the cpdef children_impurity function for both instances
#     left_impurity_fast, right_impurity_fast = fast_bic.children_impurity()
#     left_impurity_faster, right_impurity_faster = faster_bic.children_impurity()

#     # Compare the results using pytest's built-in assert statement
#     assert pytest.approx(left_impurity_fast, rel=1e-5) == left_impurity_faster
#     assert pytest.approx(right_impurity_fast, rel=1e-5) == right_impurity_faster

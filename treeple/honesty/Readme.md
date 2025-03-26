# Honesty

## Use
There is now a submodule called `_lib_experimental/sklearn_fork`, parallel to `_lib/sklearn_fork`. Build, install, and import it as an editable install using the same procedure you use with the main `neurodata` fork submodule. This of course means you must either pick one or the other at a time, or import them under different aliases.

For examples of `HonestRandomForestClassifier` use, see the test functions `test_honest_forest_separation` and `test_honest_forest_iris_criterion` in `_lib_experimental/sklearn_fork/sklearn/ensemble/tests/test_forest.py`.


## `scikit-learn`
### Architectural Changes
Adding honesty to `scikit-learn` in a sensible way required adding some architectural elements to the existing `scikit-learn` tree implementation. In particular, in order to make honesty a module composable into arbitrary types of trees, I added:
1. Added an injectable split rejection criteria pattern to `Splitter` so that arbitrary lists of split rejection criteria could be specified at runtime, and execute with no perceivable marginal overhead.
2. A lightweight event broker and handler framework to `TreeBuilder` so that interested parties can observe tree build status. In particular, this feature is used by the honesty module to grow a shadow tree which partitions the honest set using structure tree splits, and has veto power over structure tree candidate splits using the aforementioned injectable split rejection criteria.

There are more technically detailed comments in the files themselves.


### Summary of File Changes
- `ensemble/forest.py`
    - Added `HonestRandomForestClassifier`.
- `ensemble/tests/test_forest.py`
    - Added some unit tests for `HonestRandomForestClassifier`.
- `tree/_classes.py`
    - Refactored `BaseDecisionTree._fit` to separate all the data prep from the call to `BaseDecisionTree._build_tree`.
    - Added a data-bearing class `BuildTreeArgs` to carry all the post-`_prep_data` state information around.
    - Moved creation of `Criterion` from `BaseDecisionTree._build_tree` to an overridable factory method called from `BaseDecisionTree._fit`, so that `Criterion` is passed into `BaseDecisionTree._build_tree` as a parameter.
    - The idea is that committing to an implementation of `Criterion` should be hoisted as "high" as possible, but as currently implemented instantiation depends on class distribution analysis done in `BaseDecisionTree._prep_data`.
- `tree/_events.{pxd, pyx}`
    - Added for event broker/handler implementation.
- `tree/_honest_tree.py`
    - Honest classification tree implementation.
- `tree/_honesty.{pxd, pyx}`
    - Honesty module implementation.
- `tree/_partitioner.{pxd, pyx}`
    - Added to break `Partitioner` out of `Splitter` module for clearer reuse.
    - Also refactored the existing `{Dense, Sparse}Partitioner` fused type design to prevent the proliferation of concrete container classes required by that design.
- `tree/_sort.{pxd, pyx}`
    - Added to break out of `Splitter` module since these functions are used by both `Splitter` and `Partitioner`, and we want to avoid cyclic dependencies.
- `tree/_splitter.{pxd, pyx}`
    - Updated to introduce injectable split rejection criteria.
    - Refactored a bit to accommodate factory creation of extended `SplitRecord` types, as used by obliqueness in `treeple`.
- `tree/_test.{pxd, pyx}`
    - Added to implement some necessary cython test functionality for honesty.
- `tree/_tree.{pxd, pyx}`
    - Updated to introduce event firing to tree build process.
    - Refactored to break out a large block of duplicate code in `TreeBuilder.build` from the `neurodata` fork into its own method `TreeBuilder._build_body`, invoked from `TreeBuilder.build`.
- `tree/tests/test_tree.py`
    - Added some unit tests for honesty.

### Suggested Future Work
The `tree` package in `scikit-learn` is overdue for a good deal of refactoring. I would do it in multiple passes, at least the following:
- Eliminate all the introspection high up in the inheritance hierarchy. An obvious example is in `BaseDecisionTree._prep_data` (formerly in `BaseDecisionTree._fit`), where the function asks its containing class "am I a classifier?" This antipattern of switching behavior which requires foreknowledge of future structure occurs throughout the existing codebase, and makes it impossible to extend and reuse existing code without either modifying it or duplicating a great deal of it. In the case of `is_classifier(self)`, stop calling it from functions in `BaseDecisionTree`, and push that switched functionality down into actual classifiers where it belongs.
- Hoist commitments to particular implementations of interfaces. For instance, the aforementioned `Criterion` selection logic buried in the implementation of tree build. Parameterize these things, and defer commitment as "late" as possible, all the way back to the runtime interface presented to the user if possible.
- Be more judicious with inheritance, and more prolific with composition. Functionality inserted at some point in the inheritance hierarchy cannot easily be reused in sibling branches; if you want the functionality, you have to inherit, forcing your new code into a corner of the hierarchy where it doesn't necessarily belong. An obvious place to begin unwinding this in the current codebase is to merge upstream the defused `Partitioner` refactor that I did, and eliminate all the classes that were forced to exist by the fused `Partitioner`, deferring selection of concrete implementation to runtime and passing it in as a parameter value.
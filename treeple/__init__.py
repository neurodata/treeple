"""Scikit manifold oblique random forests."""

import logging
import os
import sys

__version__ = "0.9.0dev0"
logger = logging.getLogger(__name__)


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __treeple_SETUP__  # type: ignore
except NameError:
    __treeple_SETUP__ = False

if __treeple_SETUP__:
    sys.stderr.write("Running from treeple source directory.\n")
    sys.stderr.write("Partial import of treeple during the build process.\n")
    # We are not importing the rest of treeple during the build
    # process, as it may not be compiled yet
else:
    try:
        from . import _lib, tree, ensemble, experimental, stats
        from ._lib.sklearn.ensemble._forest import (
            RandomForestClassifier,
            RandomForestRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
        )
        from .neighbors import NearestNeighborsMetaEstimator
        from .ensemble import ExtendedIsolationForest, MultiViewRandomForestClassifier
        from .ensemble._unsupervised_forest import (
            UnsupervisedRandomForest,
            UnsupervisedObliqueRandomForest,
        )
        from .ensemble._supervised_forest import (
            ExtraObliqueRandomForestClassifier,
            ExtraObliqueRandomForestRegressor,
            ObliqueRandomForestClassifier,
            ObliqueRandomForestRegressor,
            PatchObliqueRandomForestClassifier,
            PatchObliqueRandomForestRegressor,
        )
        from .ensemble._honest_forest import HonestForestClassifier
    except ImportError as e:
        print(e.msg)
        msg = """Error importing treeple: you cannot import treeple while
        being in treeple source directory; please exit the treeple source
        tree first and relaunch your Python interpreter."""
        raise ImportError(msg) from e

    __all__ = [
        "_lib",
        "tree",
        "experimental",
        "ensemble",
        "stats",
        "ExtraObliqueRandomForestClassifier",
        "ExtraObliqueRandomForestRegressor",
        "NearestNeighborsMetaEstimator",
        "ObliqueRandomForestClassifier",
        "ObliqueRandomForestRegressor",
        "PatchObliqueRandomForestClassifier",
        "PatchObliqueRandomForestRegressor",
        "UnsupervisedRandomForest",
        "UnsupervisedObliqueRandomForest",
        "HonestForestClassifier",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "ExtendedIsolationForest",
        "MultiViewRandomForestClassifier",
    ]

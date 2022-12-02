"""Scikit manifold oblique random forests."""
import logging
import os
import sys

__version__ = '0.0.0dev0'
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
    __SKMORF_SETUP__
except NameError:
    __SKMORF_SETUP__ = False

if __SKMORF_SETUP__:
    sys.stderr.write("Partial import of skmorf during the build process.\n")
    # We are not importing the rest of scikit-morf during the build
    # process, as it may not be compiled yet
else:
    from . import __check_build
    from sklearn.utils._show_versions import show_versions

    __all__ = [
        "tree"
    ]

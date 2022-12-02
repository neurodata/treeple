import os
import shutil
import sys
import numpy as np
import builtins

import skmorf
from distutils.command.build_ext import build_ext
from distutils.core import Distribution
from distutils.core import Extension
from distutils.errors import DistutilsPlatformError

# This is a bit (!) hackish: we are setting a global variable so that the
# main sklearn __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by scikit-learn to
# recursively build the compiled extensions in sub-packages is based on the
# Python import machinery.
builtins.__SKMORF_SETUP__ = True

# See: https://numpy.org/doc/stable/reference/c-api/deprecations.html
# Defines that our code is clean against numpy version
DEFINE_MACRO_NUMPY_C_API = (
    "NPY_NO_DEPRECATED_API",
    "NPY_1_7_API_VERSION",
)

compile_args = [
    # "-march=native", 
    "-O3", 
    # "-msse", "-msse2", "-mfma", "-mfpmath=sse"
    ]
link_args = []
include_dirs = [np.get_include()]
libraries = []
if os.name == "posix":
    libraries.append("m")
languages = ['c++']

# C/C++/Cython Extensions
# If you are interested in defining C extensions, then take a look at
# poetry and the package pendulum. https://github.com/sdispater/pendulum
with_extensions = os.getenv("SKMORF_EXTENSIONS", None)
if with_extensions == "1" or with_extensions is None:
    with_extensions = True
if with_extensions == "0" or hasattr(sys, "pypy_version_info"):
    with_extensions = False

extensions = []
if with_extensions:
    extensions = [
        Extension(
            "*",
            ["skmorf/*.pyx"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            include_dirs=include_dirs,
            libraries=libraries,
        )
    ]

# Custom build_ext command to set OpenMP compile flags depending on os and
# compiler. Also makes it possible to set the parallelism level via
# and environment variable (useful for the wheel building CI).
# build_ext has to be imported after setuptools
from numpy.distutils.command.build_ext import build_ext  # noqa


class ExtBuilder(build_ext):
    # This class allows C extension building to fail.

    built_extensions = []

    def finalize_options(self):
        super().finalize_options()
        if self.parallel is None:
            # Do not override self.parallel if already defined by
            # command-line flag (--parallel or -j)

            parallel = os.environ.get("SKLEARN_BUILD_PARALLEL")
            if parallel:
                self.parallel = int(parallel)
        if self.parallel:
            print("setting parallel=%d " % self.parallel)

    def build_extensions(self):
        from sklearn._build_utils.openmp_helpers import get_openmp_flag

        # all extensions need some macros added
        # for example, the relative NUMPY C API
        # see above.
        for ext in self.extensions:
            ext.define_macros.append(DEFINE_MACRO_NUMPY_C_API)
        
        # we also define compile time extras, which will
        # allow each extension to be built with for example
        # openMP support. We look inside our repo for 
        # whether or not OPENMP is supported or not.
        if skmorf._OPENMP_SUPPORTED:
            openmp_flag = get_openmp_flag(self.compiler)

            for e in self.extensions:
                e.extra_compile_args += openmp_flag
                e.extra_link_args += openmp_flag

        build_ext.build_extensions(self)

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            print(
                "  Unable to build the C extensions, "
                "Scikit-morf will use the pure python code instead."
            )



def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    distribution = Distribution({"name": "scikit-morf", "ext_modules": extensions})
    distribution.package_dir = "skmorf"

    cmd = ExtBuilder(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        if not os.path.exists(output):
            continue

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)

    return setup_kwargs


if __name__ == "__main__":
    build({})
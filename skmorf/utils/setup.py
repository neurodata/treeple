import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):  # noqa
    config = Configuration("utils", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_extension(
        "_random",
        sources=["_random.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
        #  extra_compile_args=["-Xpreprocessor", "-fopenmp",],
        #  extra_link_args=["-Xpreprocessor", "-fopenmp"],
        language="c++",
    )
    config.add_extension('_openmp_helpers',
                         sources=['_openmp_helpers.pyx'],
                         libraries=libraries)
    config.add_data_files("_random.pxd")
    return config



if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())

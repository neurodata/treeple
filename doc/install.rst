:orphan:

Installation
============

Dependencies
------------

* ``numpy`` (>=1.23)
* ``scipy`` (>=1.5.0)
* ``scikit-learn`` (>=1.3)
* ``joblib`` (>=1.0.0)
* ``matplotlib`` (optional)

**treeple** supports Python >= 3.9.

Installing with ``pip``
-----------------------

**treeple** is available on `PyPI <https://pypi.org/project/treeple/>`_. Just run

.. code-block:: bash

    pip install treeple

Installing from source with Meson
---------------------------------

To install **treeple** from source, first clone the `repository <https://github.com/neurodata/treeple>`_:

.. code-block:: bash

    git clone https://github.com/neurodata/treeple.git
    cd treeple

    # ideally, you should always start within a virtual environment
    conda create -n sklearn-dev python=3.9
    conda activate sklearn-dev

Then run installation of build packages

.. code-block:: bash

    pip install -r build_requirements.txt
    pip install spin

    # use spin CLI to run Meson build locally
    ./spin build -j 2

    # you can now run tests
    ./spin test

via pip, you will be able to install in editable mode (pending Meson-Python support).

.. code-block:: bash

    pip install -e .

    # if editing Cython files
    pip install --verbose --no-build-isolation --editable .

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/neurodata/treeple/zipball/master

Conda (Recommended)
-------------------
First, create a virtual environment using Conda.

    conda create -n sklearn-dev python=3.9

# activate the virtual environment and install necessary packages to build from source

    conda activate sklearn-dev
    conda install -c conda-forge numpy scipy cython joblib threadpoolctl pytest compilers llvm-openmp

Next, `treeple` from source:

    pip install .[build]

    # if editing Cython files
    pip install --verbose --no-build-isolation --editable .

To install the package from github, clone the repository and then `cd` into the directory.:

    ./spin build

    # if you would like an editable install of treeple for dev purposes
    pip install --verbose --no-build-isolation --editable .

    pip install https://api.github.com/repos/neurodata/treeple/zipball/main


    pip install https://api.github.com/repos/neurodata/scikit-learn/zipball/obliquepr

Note that currently, we need to build the development version of scikit-learn with oblique trees within this `PR <https://github.com/scikit-learn/scikit-learn/pull/22754>`_.

Checkout this PR code, and build from source, using scikit-learn's build from source page instructions.

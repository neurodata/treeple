[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/neurodata/scikit-tree/tree/main.svg?style=svg)](https://circleci.com/gh/neurodata/scikit-tree/tree/main)
[![Main](https://github.com/neurodata/scikit-tree/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/neurodata/scikit-tree/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/neurodata/scikit-tree/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/neurodata/scikit-tree)

scikit-tree
===========

scikit-tree is a scikit-learn compatible API for building state-of-the-art decision trees. These include
unsupervised trees, oblique trees, uncertainty trees, quantile trees and causal trees.

History of oblique trees
========================
In 2001, Leo Breiman proposed two types of Random Forests. One was known as ``Forest-RI``, which is the axis-aligned traditional random forest. One was known as ``Forest-RC``, which is the random oblique linear combinations random forest. This leveraged random combinations of features to perform splits. MORF builds upon ``Forest-RC`` by proposing additional functions to combine features.

Installation
------------
Our installation will try to follow scikit-learn installation as close as possible, as we contain Cython code subclassed, or inspired by the scikit-learn tree submodule.

Building locally with Meson
---------------------------
Make sure you have the necessary packages installed

    # install general package dependencies
    poetry install --with style,test,docs

    # install build dependencies
    pip install meson ninja meson-python Cython

Run the following to build the local files

    # generate ninja make files
    meson build --prefix=$PWD/build

    # compile
    ninja -C build

    # install scikit-tree package
    meson install -C build

    # to check installation
    python -c "from sktree import tree"
    python -c "import sklearn; print(sklearn.__version__);"

Conda (Recommended)
-------------------
First, create a virtual environment using Conda.

    conda create -n sklearn-dev python=3.9

# activate the virtual environment and install necessary packages to build from source
    
    conda activate sklearn-dev
    conda install -c conda-forge numpy scipy cython joblib threadpoolctl pytest compilers llvm-openmp poetry


Next, `sktree` from source:


    pip install -e .

    # if editing Cython files
    pip install --verbose --no-build-isolation --editable .


To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # if you would like an editable install of dodiscover for dev purposes
    pip install -e .

    pip install https://api.github.com/repos/neurodata/scikit-tree/zipball/main


    pip install https://api.github.com/repos/neurodata/scikit-learn/zipball/obliquepr

Note that currently, we need to build the development version of scikit-learn with oblique trees within this [PR](https://github.com/scikit-learn/scikit-learn/pull/22754).

Checkout this PR code, and build from source, using scikit-learn's build from source page instructions.


Alpha Functionality
-------------------

We can impose a Gabor or wavelet filter bank. To do so, install ``skimage`` and ``pywavelets``.

    pip install scikit-image
    pip install PyWavelets


Using with Jupyter Notebook
---------------------------

To setup an ipykernel with jupyter notebook, then do:

    python -m ipykernel install --name sklearn --user 

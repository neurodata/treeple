[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/neurodata/scikit-tree/tree/main.svg?style=svg)](https://circleci.com/gh/neurodata/scikit-tree/tree/main)
[![Main](https://github.com/neurodata/scikit-tree/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/neurodata/scikit-tree/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/neurodata/scikit-tree/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/neurodata/scikit-tree)

scikit-tree
===========

scikit-tree is a scikit-learn compatible API for building state-of-the-art decision trees. These include unsupervised trees, oblique trees, uncertainty trees, quantile trees and causal trees.

We welcome contributions for modern tree-based algorithms. We use Cython to achieve fast C/C++ speeds, while abiding by a scikit-learn compatible (tested) API. Moreover, our Cython internals are easily extensible because they follow the internal Cython API of scikit-learn as well.

**Dependency on a fork of scikit-learn**
Due to the current state of scikit-learn's internal Cython code for trees, we have to instead leverage a maintained fork of scikit-learn at https://github.com/neurodata/scikit-learn, where specifically, the `fork` branch is used to build and install this repo. We keep that fork well-maintained and up-to-date with respect to the main sklearn repo. The only difference is the refactoring of the `tree/` submodule. This fork is released as a separate package named ``scikit-learn-tree``, which is used in code under the namespace ``sklearn_fork``. It is necessary to use this fork for anything related to:

- `RandomForest*`
- `ExtraTrees*`
- or any importable items from the `tree/` submodule, whether it is a Cython or Python object

Currently, scikit-tree depends on a refactored fork of the scikit-learn codebase at https://github.com/neurodata/scikit-learn/, which will be maintained to not diverge from the upstream scikit-learn. We call this fork an alias `scikit-learn-tree`. Within this fork though, we will maintain a refactoring of the `tree/` submodule that more easily allows 3rd party trees to take advantage of the Cython and Python APIs. You will need to download ``scikit-learn-tree`` from this fork following the installation instructions.

If you are developing for scikit-tree, we will always depend on the most up-to-date commit of `https://github.com/neurodata/scikit-learn/fork`. This branch is consistently maintained for changes upstream that occur in the scikit-learn tree submodule. This ensures that our fork maintains consistency and robustness due to bug fixes and improvements upstream. Thus if you are developing and contributing for scikit-tree, then you should clone the fork and pip install the latest commit of `https://github.com/neurodata/scikit-learn/fork`.

On the other hand, releases of scikit-tree will occur simultaneously with a tagged version of https://github.com/neurodata/scikit-learn/ (for example https://github.com/neurodata/scikit-learn/v1.1-refactoredtrees), which will then install a tagged version of the sklearn fork. This ensures that any releases of scikit-tree always work, but will not be necessarily forwards/backwards compatible with `https://github.com/neurodata/scikit-learn/fork`.

Documentation
=============

See here for the documentation for our dev version: https://docs.neurodata.io/scikit-tree/dev/index.html

Why oblique trees and why trees beyond those in scikit-learn?
=============================================================
In 2001, Leo Breiman proposed two types of Random Forests. One was known as ``Forest-RI``, which is the axis-aligned traditional random forest. One was known as ``Forest-RC``, which is the random oblique linear combinations random forest. This leveraged random combinations of features to perform splits. [MORF](1) builds upon ``Forest-RC`` by proposing additional functions to combine features. Other modern tree variants such as Canonical Correlation Forests (CCF), or unsupervised random forests are also important at solving real-world problems using robust decision tree models.

Installation
============
Our installation will try to follow scikit-learn installation as close as possible, as we contain Cython code subclassed, or inspired by the scikit-learn tree submodule.

AS OF NOW, scikit-tree is in development stage and the installation is still finicky due to the upstream scikit-learn's stalled refactoring PRs of the tree submodule. Once those
are merged, the installation will be simpler. The current recommended installation is done locally with meson.

Dependencies
------------

We minimally require:

    * Python (>=3.8)
    * numpy
    * scipy
    * scikit-learn
    * scikit-learn-tree (>= 1.2.3)

Building locally with Meson (RECOMMENDED)
-----------------------------------------
Make sure you have the necessary packages installed

    # install build dependencies
    pip install numpy scipy meson ninja meson-python Cython scikit-learn scikit-learn-tree

    # you may need these optional dependencies to build scikit-learn locally
    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

    # (caution only if you know what you're doing) or if you're a developer and need some latest changes on the fork:
    pip install scikit-learn@git+https://git@github.com/neurodata/scikit-learn.git@fork

We use the ``spin`` CLI to abstract away build details:

    # run the build using Meson/Ninja
    ./spin build
    
    # you can run the following command to see what other options there are
    ./spin --help
    ./spin build --help
    
    # For example, you might want to start from a clean build
    ./spin build --clean
    
    # or build in parallel for faster builds
    ./spin build -j 2

    # you will need to double check the build-install has the proper path 
    # this might be different from machine to machine
    export PYTHONPATH=${PWD}/build-install/usr/lib/python3.9/site-packages

    # run specific unit tests
    ./spin test -- sktree/tree/tests/test_tree.py

    # you can bring up the CLI menu
    ./spin --help

You can also do the same thing using Meson/Ninja itself. Run the following to build the local files:

    # generate ninja make files
    meson build --prefix=$PWD/build

    # compile
    ninja -C build

    # install scikit-tree package
    meson install -C build

    export PYTHONPATH=${PWD}/build/lib/python3.9/site-packages

    # to check installation, you need to be in a different directory
    cd docs;  
    python -c "from sktree import tree"
    python -c "import sklearn; print(sklearn.__version__);"

References
==========
[1]: [`Li, Adam, et al. "Manifold Oblique Random Forests: Towards Closing the Gap on Convolutional Deep Networks." arXiv preprint arXiv:1909.11799 (2019)`](https://arxiv.org/abs/1909.11799)
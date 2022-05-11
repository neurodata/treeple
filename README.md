<!-- [![Codecov](https://codecov.io/gh/adam2392/mne-hfo/branch/master/graph/badge.svg)](https://codecov.io/gh/adam2392/mne-hfo)
![.github/workflows/main.yml](https://github.com/adam2392/mne-hfo/workflows/.github/workflows/main.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/adam2392/mne-hfo.svg?style=svg)](https://circleci.com/gh/adam2392/mne-hfo)
![License](https://img.shields.io/pypi/l/mne-bids)
[![Code Maintainability](https://api.codeclimate.com/v1/badges/3afe97439ec5133ce267/maintainability)](https://codeclimate.com/github/adam2392/mne-hfo/maintainability)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) -->

Scikit-MORF
===========

Scikit-morf is a scikit-learn compatible API for building manifold oblique random forests (MORF). These class of random forests are built on combinations of features rather then individual features themselves.

In 2001, Leo Breiman proposed two types of Random Forests. One was known as ``Forest-RI``, which is the axis-aligned traditional random forest. One was known as ``Forest-RC``, which is the random oblique linear combinations random forest. This leveraged random combinations of features to perform splits. MORF builds upon ``Forest-RC`` by proposing additional functions to combine features.

.. target for :end-before: title-end-content

Installation
------------
Using conda instructions from sklearn:

    conda create -n sklearn-dev -c conda-forge python numpy scipy cython \
    joblib threadpoolctl pytest compilers llvm-openmp

    conda activate sklearn-dev
    
    # install files from Pipfile
    pip install pipenv 
    pipenv install --dev --skip-lock

    # or install via requirements.txt
    pip install -r requirements.txt

    # clean out any left-over build files
    make clean

    # build the package
    make build-dev

To install the necessary development packages, run:

    pip install -r test_requirements.txt

    # check code style
    make pep

then use Makefile recipe to build dev version. You'll need Cython installed.

    make build-dev

Alpha Functionality
-------------------

We can impose a Gabor or wavelet filter bank. To do so, install ``skimage`` and ``pywavelets``.

    pip install scikit-image
    pip install PyWavelets


Using with Jupyter Notebook
---------------------------

To setup an ipykernel with jupyter notebook, then do:

    python -m ipykernel install --name sklearn --user 

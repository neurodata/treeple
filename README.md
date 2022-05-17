[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/adam2392/scikit-morf/tree/main.svg?style=svg)](https://circleci.com/gh/adam2392/scikit-morf/tree/main)
[![Main](https://github.com/adam2392/scikit-morf/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/adam2392/scikit-morf/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/adam2392/scikit-morf/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/adam2392/scikit-morf)

Scikit-MORF
===========

Scikit-morf is a scikit-learn compatible API for building manifold oblique random forests (MORF). These class of random forests are built on combinations of features rather then individual features themselves.

History
=======
In 2001, Leo Breiman proposed two types of Random Forests. One was known as ``Forest-RI``, which is the axis-aligned traditional random forest. One was known as ``Forest-RC``, which is the random oblique linear combinations random forest. This leveraged random combinations of features to perform splits. MORF builds upon ``Forest-RC`` by proposing additional functions to combine features.

Installation
------------

Conda (Recommended)
-------------------
First, create a virtual environment using Conda.

```bash
conda create -n sklearn-dev python=3.8

# activate the virtual environment
conda activate sklearn-dev
```

Next, `skmorf` from source:

```
pip install -e .
```

Finally, install all requirements for development and docs

```bash
pip install -r requirements_testing.txt
pip install -r requirements_doc.txt
```

Note that currently, we need to build the development version of scikit-learn with oblique trees within this [PR](https://github.com/scikit-learn/scikit-learn/pull/22754).

Checkout this PR code, and build from source, using scikit-learn's build from source page instructions.


Some outdated instructions
--------------------------
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

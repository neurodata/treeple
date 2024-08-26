<!-- TOC -->

- [Requirements](#requirements)
- [Setting up your development environment](#setting-up-your-development-environment)
- [Building the project from source](#building-the-project-from-source)
    - [Summary: Building locally with Meson For developers](#summary-building-locally-with-meson-for-developers)
- [Development Tasks](#development-tasks)
- [Advanced Updating submodules](#advanced-updating-submodules)
- [Cython and C++](#cython-and-c)
- [Making a Release](#making-a-release)
    - [Releasing on PyPi for pip installs](#releasing-on-pypi-for-pip-installs)
    - [Releasing documentation](#releasing-documentation)

<!-- /TOC -->

# Requirements

- Python 3.9+
- numpy>=1.25.0
- scipy>=1.5.0
- scikit-learn>=1.5.0

For the other requirements, inspect the ``pyproject.toml`` file.

# Setting up your development environment

We recommend using miniconda, as python virtual environments may not setup properly compilers necessary for our compiled code. For detailed information on setting up and managing conda environments, see <https://conda.io/docs/test-drive.html>.

<!-- Setup a conda env -->

    conda create -n treeple
    conda activate treeple

**Make sure you specify a Python version if your system defaults to anything less than Python 3.9.**

**Any commands should ALWAYS be after you have activated your conda environment.**
Next, install necessary build dependencies. For more information, see <https://scikit-learn.org/stable/developers/advanced_installation.html>.

    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

Assuming these steps have worked properly and you have read and followed any necessary scikit-learn advanced installation instructions, you can then install dependencies for treeple.

If you are developing locally, you will need the build dependencies to compile the Cython / C++ code:

    pip install -r build_requirements.txt

Additionally, you need to install the latest build of scikit-learn:

    pip install --force -r build_sklearn_requirements.txt

Other requirements can be installed as such:

    pip install .
    pip install .[style]
    pip install .[test]
    pip install .[doc]

# Building the project from source

We leverage meson to build treeple from source. We utilize a CLI tool, called [spin](https://github.com/scientific-python/spin), which wraps certain meson commands to make building easier.

For example, the following command will build the project completely from scratch

    spin build --clean

If you have part of the build already done, you can run:

    spin build

The following command will test the project

    spin test

For other commands, see

    spin --help

Note at this stage, you will be unable to run Python commands directly. For example, ``pytest ./treeple`` will not work.

However, after installing and building the project from source using meson, you can leverage editable installs to make testing code changes much faster.

    spin install

This will now link the meson build to your Python runtime. Now if you run

    pytest ./treeple

the unit-tests should run.

Summary: Building locally with Meson (For developers)
-----------------------------------------------------
Make sure you have the necessary packages installed.

    # install build dependencies
    pip install -r build_requirements.txt

    # you may need these optional dependencies to build scikit-learn locally
    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

``YOUR_PYTHON_VERSION`` below should be any of the acceptable versions of Python for treeple. We use the ``spin`` CLI to abstract away build details:

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
    export PYTHONPATH=${PWD}/build-install/usr/lib/python<YOUR_PYTHON_VERSION>/site-packages

    # run specific unit tests
    ./spin test -- treeple/tree/tests/test_tree.py

    # you can bring up the CLI menu
    ./spin --help

You can also do the same thing using Meson/Ninja itself. Run the following to build the local files:

    # generate ninja make files
    meson build --prefix=$PWD/build

    # compile
    ninja -C build

    # install treeple package
    meson install -C build

    export PYTHONPATH=${PWD}/build/lib/python<YOUR_PYTHON_VERSION>/site-packages

    # to check installation, you need to be in a different directory
    cd docs;
    python -c "from treeple import tree"
    python -c "import sklearn; print(sklearn.__version__);"

After building locally, you can use editable installs (warning: this only registers Python changes locally)

    pip install --no-build-isolation --editable .

Or if you have spin v0.8+ installed, you can just run directly

    spin install

# Development Tasks

There are a series of top-level tasks available.

    make pre-commit

This leverage pre-commit to run a series of precommit checks.

# (Advanced) Updating submodules

treeple relies on a submodule of a forked-version of scikit-learn for certain Python and Cython code that extends the ``DecisionTree*`` models. Usually, if a developer is making changes, they should go over to the ``submodulev3`` branch on ``https://github.com/neurodata/scikit-learn`` and
submit a PR to make changes to the submodule.

This should **ALWAYS** be supported by some use-case in treeple. We want the minimal amount of code-change in our forked version of scikit-learn to make it very easy to merge in upstream changes, bug fixes and features for tree-based code.

Once a PR is submitted and merged, the developer can update the submodule here in treeple, so that we leverage the new commit. You **must** update the submodule commit ID and also commit this change, so that way the build leverages the new submodule commit ID.

    git submodule update --init --recursive --remote
    git add -A
    git commit -m "Update submodule" -s

Now, you can re-build the project using the latest submodule changes.

    spin build --clean

# Cython and C++

The general design of treeple follows that of the tree-models inside scikit-learn, where tree-based models are inherently Cythonized, or written with C++. Then the actual forest (e.g. RandomForest, or ExtraForest) is just a Python API wrapper that creates an ensemble of the trees.

In order to develop new tree models, generally Cython and C++ code will need to be written in order to optimize the tree building process, otherwise fitting a single forest model would take very long.

# Making a Release

treeple is in-line with scikit-learn and thus relies on each new version released there. Moreover, treeple relies on compiled code, so releases are a bit more complex than the typical Python package.

## Releasing on PyPi (for pip installs)

GH Actions will build wheels for each Python version and OS. Then the wheels needs to be uploaded to PyPi. The following steps outline the process:

1. Download wheels from GH Actions and put all wheels into a ``dist/`` folder in the root (local) of the project.

<https://github.com/neurodata/treeple/actions/workflows/build_wheels.yml> will have all the wheels for common OSes built for each Python version.

2. Upload wheels to test PyPi
This is to ensure that the wheels are built correctly and can be installed on a fresh environment. For more information, see <https://packaging.python.org/guides/using-testpypi/>. You will need to follow the instructions to create an account and get your API token for testpypi and pypi.

```
twine upload dist/* --repository testpypi
```

Verify that installations work as expected on your machine.

3. Upload wheels

```
twine upload dist/*
```

or if you have two-factor authentication enabled: <https://pypi.org/help/#apitoken>

    twine upload dist/* --repository treeple

4. Update version number on ``meson.build`` and ``pyproject.toml`` to the relevant version.

See https://github.com/neurodata/treeple/pull/160 as an example.

## Releasing documentation

1. Build the documentation locally

```
spin docs
```

2. Make a copy of the documentation in the ``docs/_build/html`` folder somewhere outside of the git folder.

3. Push the documentation to the ``gh-pages`` branch

```
git checkout gh-pages
```

Create a new folder for the new version, e.g. ``v0.8`` if you are releasing version 0.8.0.

Copy the contents of the locally build ``docs/_build/html`` folder to newly created folder at the root of the ``gh-pages`` branch.

4. Update the versions pointer file in main `doc/_static/versions.json` to point to the new version.

e.g. If we are releasing ``0.8.0``, then you will see:


Change the development version to the next version i.e. v0.9 and rename v0.8 version appropriately:

```
    {
        "name": "0.9 (devel)",
        "version": "dev",
        "url": "https://docs.neurodata.io/treeple/dev/"
    },
    {
        "name": "0.8",
        "version": "0.8",
        "url": "https://docs.neurodata.io/treeple/v0.8/"
    },
```


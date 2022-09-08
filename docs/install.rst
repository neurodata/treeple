:orphan:

Installation
============

Dependencies
------------

* ``numpy`` (>=1.23)
* ``scipy`` (>=1.6.0)
* ``scikit-learn`` (>=1.1)
* ``joblib`` (>=1.0.0)
* ``pandas`` (>=1.1)
* ``matplotlib`` (optional)

**scikit-morf** supports Python >= 3.8.

Installing with ``pip``
-----------------------

**scikit-morf** is available [on PyPI](https://pypi.org/project/scikit-morf/). Just run

.. code-block:: bash

    pip install scikit-morf

    # or if you use poetry which is recommended
    poetry add scikit-morf

## Installing from source

To install **scikit-morf** from source, first clone [the repository](https://github.com/adam2392/scikit-morf):

.. code-block:: bash

    git clone https://github.com/adam2392/scikit-morf.git
    cd scikit-morf

Then run installation via poetry (recommended)

.. code-block:: bash

    poetry install

or via pip

.. code-block:: bash

    pip install -e .

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/adam2392/mne-hfo/zipball/master

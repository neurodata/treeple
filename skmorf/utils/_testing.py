from contextlib import contextmanager
from distutils.version import LooseVersion
from functools import partial, wraps
import os
import inspect
from io import StringIO
from shutil import rmtree
import sys
import tempfile
import traceback
from unittest import SkipTest
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose



def nottest(f):
    """Mark a function as not a test (decorator)."""
    f.__test__ = False
    return f

@nottest
def run_tests_if_main():
    """Run tests in a given file if it is run as a script."""
    local_vars = inspect.currentframe().f_back.f_locals
    if local_vars.get('__name__', '') != '__main__':
        return
    import pytest
    code = pytest.main([local_vars['__file__'], '-v'])
    if code:
        raise AssertionError('pytest finished with errors (%d)' % (code,))

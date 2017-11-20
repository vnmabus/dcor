"""Utility functions"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections

import numba

import numpy as np


HypothesisTest = collections.namedtuple('HypothesisTest', ['p_value',
                                        'statistic'])


def _jit(function):
    """
    Compile a function using a jit compiler.

    The function is always compiled to check errors, but is only used outside
    tests, so that code coverage analysis can be performed in jitted functions.

    The tests set sys._called_from_test in conftest.py.

    """
    import sys

    compiled = numba.jit(function)

    if hasattr(sys, '_called_from_test'):
        return function
    else:  # pragma: no cover
        return compiled


def _check_kwargs_empty(kwargs):
    """Raise an apropiate exception if the kwargs dictionary is not empty."""
    if kwargs:
        raise TypeError("Unexpected keyword argument '{arg}'".format(
            arg=list(kwargs.keys())[0]))


def _sqrt(x):
    """
    Return square root of an ndarray.

    This sqrt function for ndarrays tries to use the exponentiation operator
    if the objects stored do not supply a sqrt method.

    """
    try:
        return np.sqrt(x)
    except AttributeError:
        exponent = 0.5

        try:
            exponent = np.take(x, 0).from_float(exponent)
        except AttributeError:
            pass

        return x ** exponent


def _transform_to_2d(t):
    """Convert vectors to column matrices, to always have a 2d shape."""
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t

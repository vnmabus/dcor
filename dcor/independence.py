"""
Functions for testing independence of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors are independent.
"""

from __future__ import absolute_import, division, print_function

from . import _dcor_internals
from . import _hypothesis
from ._utils import _random_state_init, _transform_to_2d
from ._dcor import u_distance_correlation_sqr
import numpy as np
import scipy.stats


def _distance_covariance_test_imp(x, y,
                                  _centering,
                                  _product,
                                  exponent=1,
                                  num_resamples=0,
                                  random_state=None
                                  ):
    """
    Real implementation of :func:`distance_covariance_test`.

    This function is used to make parameters ``num_resamples``, ``exponent``
    and ``random_state`` keyword-only in Python 2.

    """
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    _dcor_internals._check_same_n_elements(x, y)

    random_state = _random_state_init(random_state)

    # Compute U-centered matrices
    u_x = _dcor_internals._distance_matrix_generic(
        x,
        centering=_centering,
        exponent=exponent)
    u_y = _dcor_internals._distance_matrix_generic(
        y,
        centering=_centering,
        exponent=exponent)

    # Use the dcov statistic
    def statistic_function(distance_matrix):
        return u_x.shape[0] * _product(
            distance_matrix, u_y)

    return _hypothesis._permutation_test_with_sym_matrix(
        u_x,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state)


def distance_covariance_test(x, y, **kwargs):
    """
    distance_covariance_test(x, y, *, num_resamples=0, exponent=1, \
    random_state=None)

    Test of distance covariance independence.

    Compute the test of independence based on the distance
    covariance, for two random vectors.

    The test is a permutation test where the null hypothesis is that the two
    random vectors are independent.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.
    num_resamples: int
        Number of permutations resamples to take in the permutation test.
    random_state: {None, int, array_like, numpy.random.RandomState}
        Random state to generate the permutations.

    Returns
    -------
    HypothesisTest
        Results of the hypothesis test.

    See Also
    --------
    distance_covariance

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1, 0, 0, 1],
    ...               [0, 1, 1, 1],
    ...               [1, 1, 1, 1],
    ...               [1, 1, 0, 1]])
    >>> dcor.independence.distance_covariance_test(a, a)
    HypothesisTest(p_value=1.0, statistic=208.0)
    >>> dcor.independence.distance_covariance_test(a, b)
    ...                                      # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=11.75323056...)
    >>> dcor.independence.distance_covariance_test(b, b)
    HypothesisTest(p_value=1.0, statistic=1.3604610...)
    >>> dcor.independence.distance_covariance_test(a, b,
    ... num_resamples=5, random_state=0)
    HypothesisTest(p_value=0.5, statistic=11.7532305...)
    >>> dcor.independence.distance_covariance_test(a, b,
    ... num_resamples=5, random_state=13)
    HypothesisTest(p_value=0.3333333..., statistic=11.7532305...)
    >>> dcor.independence.distance_covariance_test(a, a,
    ... num_resamples=7, random_state=0)
    HypothesisTest(p_value=0.125, statistic=208.0)

    """
    return _distance_covariance_test_imp(
        x, y,
        _centering=_dcor_internals.double_centered,
        _product=_dcor_internals.mean_product,
        ** kwargs)


def _partial_distance_covariance_test_imp(x, y, z, num_resamples=0,
                                          random_state=None):
    """
    Real implementation of :func:`partial_distance_covariance_test`.

    This function is used to make parameters ``num_resamples``
    and ``random_state`` keyword-only in Python 2.

    """
    random_state = _random_state_init(random_state)

    # Compute U-centered matrices
    u_x = _dcor_internals._u_distance_matrix(x)
    u_y = _dcor_internals._u_distance_matrix(y)
    u_z = _dcor_internals._u_distance_matrix(z)

    # Compute projections
    proj = _dcor_internals.u_complementary_projection(u_z)

    p_xz = proj(u_x)
    p_yz = proj(u_y)

    # Use the pdcor statistic
    def statistic_function(distance_matrix):
        return u_x.shape[0] * _dcor_internals.u_product(
            distance_matrix, p_yz)

    return _hypothesis._permutation_test_with_sym_matrix(
        p_xz,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state)


def partial_distance_covariance_test(x, y, z, **kwargs):
    """
    partial_distance_covariance_test(x, y, z, num_resamples=0, exponent=1,
    random_state=None)

    Test of partial distance covariance independence.

    Compute the test of independence based on the partial distance
    covariance, for two random vectors conditioned on a third.

    The test is a permutation test where the null hypothesis is that the first
    two random vectors are independent given the third one.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    z: array_like
        Observed random vector. The columns correspond with the individual
        random variables while the rows are individual instances of the random
        vector.
    num_resamples: int
        Number of permutations resamples to take in the permutation test.
    random_state: {None, int, array_like, numpy.random.RandomState}
        Random state to generate the permutations.

    Returns
    -------
    HypothesisTest
        Results of the hypothesis test.

    See Also
    --------
    partial_distance_covariance

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1, 0, 0, 1],
    ...               [0, 1, 1, 1],
    ...               [1, 1, 1, 1],
    ...               [1, 1, 0, 1]])
    >>> c = np.array([[1000, 0, 0, 1000],
    ...               [0, 1000, 1000, 1000],
    ...               [1000, 1000, 1000, 1000],
    ...               [1000, 1000, 0, 1000]])
    >>> dcor.independence.partial_distance_covariance_test(a, a, b)
    ...                                       # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=142.6664416...)
    >>> dcor.independence.partial_distance_covariance_test(a, b, c)
    ...                                      # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=7.2690070...e-15)
    >>> dcor.independence.partial_distance_covariance_test(b, b, c)
    ...                                      # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=2.2533380...e-30)
    >>> dcor.independence.partial_distance_covariance_test(a, b, c,
    ... num_resamples=5, random_state=0)
    HypothesisTest(p_value=0.1666666..., statistic=7.2690070...e-15)
    >>> dcor.independence.partial_distance_covariance_test(a, b, c,
    ... num_resamples=5, random_state=13)
    HypothesisTest(p_value=0.1666666..., statistic=7.2690070...e-15)
    >>> dcor.independence.partial_distance_covariance_test(a, c, b,
    ... num_resamples=7, random_state=0)
    HypothesisTest(p_value=1.0, statistic=-7.5701764...e-12)

    """
    return _partial_distance_covariance_test_imp(x, y, z, **kwargs)


def distance_correlation_t_statistic(x, y):
    """
    Transformation of the bias corrected version of distance correlation used
    in :func:`distance_correlation_t_test`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        T statistic.

    See Also
    --------
    distance_correlation_t_test

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1, 0, 0, 1],
    ...               [0, 1, 1, 1],
    ...               [1, 1, 1, 1],
    ...               [1, 1, 0, 1]])
    >>> with np.errstate(divide='ignore'):
    ...     dcor.independence.distance_correlation_t_statistic(a, a)
    inf
    >>> dcor.independence.distance_correlation_t_statistic(a, b)
    ...                                      # doctest: +ELLIPSIS
    -0.4430164...
    >>> with np.errstate(divide='ignore'):
    ...     dcor.independence.distance_correlation_t_statistic(b, b)
    inf

    """
    bcdcor = u_distance_correlation_sqr(x, y)

    n = x.shape[0]
    v = n * (n-3) / 2

    return np.sqrt(v - 1) * bcdcor / np.sqrt(1 - bcdcor**2)


def distance_correlation_t_test(x, y):
    """
    Test of independence for high dimension based on convergence to a Student t
    distribution. The null hypothesis is that the two random vectors are
    independent.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    HypothesisTest
        Results of the hypothesis test.

    See Also
    --------
    distance_correlation_t_statistic

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1, 0, 0, 1],
    ...               [0, 1, 1, 1],
    ...               [1, 1, 1, 1],
    ...               [1, 1, 0, 1]])
    >>> with np.errstate(divide='ignore'):
    ...     dcor.independence.distance_correlation_t_test(a, a)
    ...                                      # doctest: +ELLIPSIS
    HypothesisTest(p_value=0.0, statistic=inf)
    >>> dcor.independence.distance_correlation_t_test(a, b)
    ...                                      # doctest: +ELLIPSIS
    HypothesisTest(p_value=0.6327451..., statistic=-0.4430164...)
    >>> with np.errstate(divide='ignore'):
    ...     dcor.independence.distance_correlation_t_test(b, b)
    ...                                      # doctest: +ELLIPSIS
    HypothesisTest(p_value=0.0, statistic=inf)

    """
    t_test = distance_correlation_t_statistic(x, y)

    n = x.shape[0]
    v = n * (n-3) / 2
    df = v - 1

    p_value = 1 - scipy.stats.t.cdf(t_test, df=df)

    return _hypothesis.HypothesisTest(p_value=p_value, statistic=t_test)

"""
Functions for testing independence of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors are independent.
"""

from __future__ import absolute_import, division, print_function

import numpy as _np

from . import _dcor_internals
from . import _utils
from ._utils import _random_state_init, _check_kwargs_empty


def partial_distance_covariance_test(x, y, z, **kwargs):
    """
    partial_distance_covariance_test(x, y, z, num_resamples=0, exponent=1,
    random_state=None)

    Test of partial distance covariance independence.

    Compute the test of independence based on the partial distance
    covariance, for two random vectors conditioned on a third.

    The test is a permutation test where the null hypothesis is that all
    random vectors have the same distribution.

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
    # pylint:disable=too-many-locals
    random_state = _random_state_init(kwargs.pop("random_state", None))

    # B
    num_resamples = kwargs.pop("num_resamples", 0)

    _check_kwargs_empty(kwargs)

    # Compute U-centered matrices
    u_x = _dcor_internals._u_distance_matrix(x)
    u_y = _dcor_internals._u_distance_matrix(y)
    u_z = _dcor_internals._u_distance_matrix(z)

    # Compute projections
    proj = _dcor_internals.u_complementary_projection(u_z)

    p_xz = proj(u_x)
    p_yz = proj(u_y)

    num_dimensions = u_x.shape[0]

    # epsilon_n
    observed_pdcov = num_dimensions * _dcor_internals.u_product(p_xz, p_yz)

    # epsilon^(b)_n
    bootstrap_pdcov = _np.ones(num_resamples, dtype=observed_pdcov.dtype)

    for bootstrap in range(num_resamples):
        permuted_index = random_state.permutation(num_dimensions)

        permuted_p_xz = p_xz[_np.ix_(permuted_index, permuted_index)]

        pdcov = num_dimensions * _dcor_internals.u_product(permuted_p_xz, p_yz)

        bootstrap_pdcov[bootstrap] = pdcov

    extreme_results = bootstrap_pdcov > observed_pdcov
    p_value = (_np.sum(extreme_results) + 1) / (num_resamples + 1)

    return _utils.HypothesisTest(
        p_value=p_value,
        statistic=observed_pdcov
    )

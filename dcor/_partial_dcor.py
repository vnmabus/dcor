"""Functions for computing partial distance covariance and correlation"""

import numpy as np

from ._dcor_internals import (_u_distance_matrix, u_complementary_projection,
                              u_product)
from ._utils import _sqrt


def partial_distance_covariance(x, y, z):
    """
    Partial distance covariance estimator.

    Compute the estimator for the partial distance covariance of the
    random vectors corresponding to :math:`x` and :math:`y` with respect
    to the random variable corresponding to :math:`z`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    z: array_like
        Random vector with respect to which the partial distance covariance
        is computed. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the partial distance covariance.

    See Also
    --------
    partial_distance_correlation

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> c = np.array([[1, 3, 4],
    ...               [5, 7, 8],
    ...               [9, 11, 15],
    ...               [13, 15, 16]])
    >>> dcor.partial_distance_covariance(a, a, c) # doctest: +ELLIPSIS
    0.0024298...
    >>> dcor.partial_distance_covariance(a, b, c)
    0.0347030...
    >>> dcor.partial_distance_covariance(b, b, c)
    0.4956241...

    """
    a = _u_distance_matrix(x)
    b = _u_distance_matrix(y)
    c = _u_distance_matrix(z)

    proj = u_complementary_projection(c)

    return u_product(proj(a), proj(b))


def partial_distance_correlation(x, y, z):  # pylint:disable=too-many-locals
    """
    Partial distance correlation estimator.

    Compute the estimator for the partial distance correlation of the
    random vectors corresponding to :math:`x` and :math:`y` with respect
    to the random variable corresponding to :math:`z`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    z: array_like
        Random vector with respect to which the partial distance correlation
        is computed. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the partial distance correlation.

    See Also
    --------
    partial_distance_covariance

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1], [1], [2], [2], [3]])
    >>> b = np.array([[1], [2], [1], [2], [1]])
    >>> c = np.array([[1], [2], [2], [1], [2]])
    >>> dcor.partial_distance_correlation(a, a, c)
    1.0
    >>> dcor.partial_distance_correlation(a, b, c) # doctest: +ELLIPSIS
    -0.5...
    >>> dcor.partial_distance_correlation(b, b, c)
    1.0
    >>> dcor.partial_distance_correlation(a, c, c)
    0.0

    """
    a = _u_distance_matrix(x)
    b = _u_distance_matrix(y)
    c = _u_distance_matrix(z)

    aa = u_product(a, a)
    bb = u_product(b, b)
    cc = u_product(c, c)
    ab = u_product(a, b)
    ac = u_product(a, c)
    bc = u_product(b, c)

    denom_sqr = aa * bb
    r_xy = ab / _sqrt(denom_sqr) if denom_sqr != 0 else denom_sqr
    r_xy = np.clip(r_xy, -1, 1)

    denom_sqr = aa * cc
    r_xz = ac / _sqrt(denom_sqr) if denom_sqr != 0 else denom_sqr
    r_xz = np.clip(r_xz, -1, 1)

    denom_sqr = bb * cc
    r_yz = bc / _sqrt(denom_sqr) if denom_sqr != 0 else denom_sqr
    r_yz = np.clip(r_yz, -1, 1)

    denom = _sqrt(1 - r_xz ** 2) * _sqrt(1 - r_yz ** 2)

    return (r_xy - r_xz * r_yz) / denom if denom != 0 else denom

"""
Internal functions for distance covariance and correlation.

The functions in this module are used for performing computations related with
distance covariance and correlation.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import warnings

import numpy as np

from . import distances
from ._utils import _transform_to_2d


def _check_valid_dcov_exponent(exponent):
    if not 0 < exponent < 2:
        warning_msg = ('Distance covariance is not guaranteed to '
                       'characterize independence if the exponent value is '
                       'not in the range (0, 2). The exponent passed '
                       'is {exponent}.'.format(exponent=exponent))

        warnings.warn(warning_msg)


def _check_same_n_elements(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same number of examples. The '
                         'number of samples of x is {x_shape} while the '
                         'number of samples of y is {y_shape}.'.format(
                             x_shape=x.shape[0],
                             y_shape=y.shape[0]))


def _float_copy_to_out(out, origin):
    """
    Copy origin to out and return it.

    If ``out`` is None, a new copy (casted to floating point) is used. If
    ``out`` and ``origin`` are the same, we simply return it. Otherwise we
    copy the values.

    """
    if out is None:
        out = origin / 1  # The division forces cast to a floating point type
    elif out is not origin:
        np.copyto(out, origin)
    return out


def _double_centered_imp(a, out=None):
    """
    Real implementation of :func:`double_centered`.

    This function is used to make parameter ``out`` keyword-only in
    Python 2.

    """
    out = _float_copy_to_out(out, a)

    dim = np.size(a, 0)

    mu = np.sum(a) / (dim * dim)
    sum_cols = np.sum(a, 0, keepdims=True)
    sum_rows = np.sum(a, 1, keepdims=True)
    mu_cols = sum_cols / dim
    mu_rows = sum_rows / dim

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= mu_rows
    out -= mu_cols
    out += mu

    return out


def double_centered(a, **kwargs):
    r"""
    double_centered(a, *, out=None)

    Return a copy of the matrix :math:`a` which is double centered.

    A matrix is double centered if both the sum of its columns and the sum of
    its rows are 0.

    In order to do that, for every element its row and column averages are
    subtracted, and the total average is added.

    Thus, if the element in the i-th row and j-th column of the original
    matrix :math:`a` is :math:`a_{i,j}`, then the new element will be

    .. math::

        \tilde{a}_{i, j} = a_{i,j} - \frac{1}{N} \sum_{l=1}^N a_{il} -
        \frac{1}{N}\sum_{k=1}^N a_{kj} + \frac{1}{N^2}\sum_{k=1}^N a_{kj}.

    Parameters
    ----------
    a : (N, N) array_like
        Original matrix.
    out: None or array_like
        If not None, specifies where to return the resulting array. This
        array should allow non integer numbers.

    Returns
    -------
    (N, N) ndarray
        Double centered matrix.

    See Also
    --------
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2], [3, 4]])
    >>> dcor.double_centered(a)
    array([[0., 0.],
           [0., 0.]])
    >>> b = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    >>> dcor.double_centered(b)
    array([[ 0.44444444, -0.22222222, -0.22222222],
           [-0.22222222,  0.11111111,  0.11111111],
           [-0.22222222,  0.11111111,  0.11111111]])
    >>> c = np.array([[1., 2., 3.], [2., 4., 5.], [3., 5., 6.]])
    >>> dcor.double_centered(c, out=c)
    array([[ 0.44444444, -0.22222222, -0.22222222],
           [-0.22222222,  0.11111111,  0.11111111],
           [-0.22222222,  0.11111111,  0.11111111]])
    >>> c
    array([[ 0.44444444, -0.22222222, -0.22222222],
           [-0.22222222,  0.11111111,  0.11111111],
           [-0.22222222,  0.11111111,  0.11111111]])

    """
    return _double_centered_imp(a, **kwargs)


def _u_centered_imp(a, out=None):
    """
    Real implementation of :func:`u_centered`.

    This function is used to make parameter ``out`` keyword-only in
    Python 2.

    """
    out = _float_copy_to_out(out, a)

    dim = np.size(a, 0)

    u_mu = np.sum(a) / ((dim - 1) * (dim - 2))
    sum_cols = np.sum(a, 0, keepdims=True)
    sum_rows = np.sum(a, 1, keepdims=True)
    u_mu_cols = np.ones((dim, 1)).dot(sum_cols / (dim - 2))
    u_mu_rows = (sum_rows / (dim - 2)).dot(np.ones((1, dim)))

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= u_mu_rows
    out -= u_mu_cols
    out += u_mu

    # The diagonal is zero
    out[np.eye(dim, dtype=bool)] = 0

    return out


def u_centered(a, **kwargs):
    r"""
    u_centered(a, *, out=None)

    Return a copy of the matrix :math:`a` which is :math:`U`-centered.

    If the element of the i-th row and j-th column of the original
    matrix :math:`a` is :math:`a_{i,j}`, then the new element will be

    .. math::

        \tilde{a}_{i, j} =
        \begin{cases}
        a_{i,j} - \frac{1}{n-2} \sum_{l=1}^n a_{il} -
        \frac{1}{n-2} \sum_{k=1}^n a_{kj} +
        \frac{1}{(n-1)(n-2)}\sum_{k=1}^n a_{kj},
        &\text{if } i \neq j, \\
        0,
        &\text{if } i = j.
        \end{cases}

    Parameters
    ----------
    a : (N, N) array_like
        Original matrix.
    out: None or array_like
        If not None, specifies where to return the resulting array. This
        array should allow non integer numbers.

    Returns
    -------
    (N, N) ndarray
        :math:`U`-centered matrix.

    See Also
    --------
    double_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    >>> dcor.u_centered(a)
    array([[ 0. ,  0.5, -1.5],
           [ 0.5,  0. , -4.5],
           [-1.5, -4.5,  0. ]])
    >>> b = np.array([[1., 2., 3.], [2., 4., 5.], [3., 5., 6.]])
    >>> dcor.u_centered(b, out=b)
    array([[ 0. ,  0.5, -1.5],
           [ 0.5,  0. , -4.5],
           [-1.5, -4.5,  0. ]])
    >>> b
    array([[ 0. ,  0.5, -1.5],
           [ 0.5,  0. , -4.5],
           [-1.5, -4.5,  0. ]])

    Note that when the matrix is 1x1 or 2x2, the formula performs
    a division by 0

    >>> import warnings
    >>> b = np.array([[1, 2], [3, 4]])
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     dcor.u_centered(b)
    array([[ 0., nan],
           [nan,  0.]])

    """
    return _u_centered_imp(a, **kwargs)


def mean_product(a, b):
    r"""
    Average of the elements for an element-wise product of two matrices.

    If the matrices are square it is

    .. math::
        \frac{1}{n^2} \sum_{i,j=1}^n a_{i, j} b_{i, j}.

    Parameters
    ----------
    a: array_like
        First input array to be multiplied.
    b: array_like
        Second input array to be multiplied.

    Returns
    -------
    numpy scalar
        Average of the product.

    See Also
    --------
    u_product

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 4], [1, 2, 4], [1, 2, 4]])
    >>> b = np.array([[1, .5, .25], [1, .5, .25], [1, .5, .25]])
    >>> dcor.mean_product(a, b)
    1.0
    >>> dcor.mean_product(a, a)
    7.0

    If the matrices involved are not square, but have the same dimensions,
    the average of the product is still well defined

    >>> c = np.array([[1, 2], [1, 2], [1, 2]])
    >>> dcor.mean_product(c, c)
    2.5

    """
    return np.mean(a * b)


def u_product(a, b):
    r"""
    Inner product in the Hilbert space of :math:`U`-centered distance matrices.

    This inner product is defined as

    .. math::
        \frac{1}{n(n-3)} \sum_{i,j=1}^n a_{i, j} b_{i, j}

    Parameters
    ----------
    a: array_like
        First input array to be multiplied.
    b: array_like
        Second input array to be multiplied.

    Returns
    -------
    numpy scalar
        Inner product.

    See Also
    --------
    mean_product

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> dcor.u_product(u_a, u_a)
    6.0
    >>> dcor.u_product(u_a, u_b)
    -8.0

    Note that the formula is well defined as long as the matrices involved
    are square and have the same dimensions, even if they are not in the
    Hilbert space of :math:`U`-centered distance matrices

    >>> dcor.u_product(a, a)
    132.0

    Also the formula produces a division by 0 for 3x3 matrices

    >>> import warnings
    >>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     dcor.u_product(b, b)
    inf

    """
    n = np.size(a, 0)

    return np.sum(a * b) / (n * (n - 3))


def u_projection(a):
    r"""
    Return the orthogonal projection function over :math:`a`.

    The function returned computes the orthogonal projection over
    :math:`a` in the Hilbert space of :math:`U`-centered distance
    matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A`
    is defined as

    .. math::
        \text{proj}_A(B) = \begin{cases}
        \frac{\langle A, B \rangle}{\langle A, A \rangle} A,
        & \text{if} \langle A, A \rangle \neq 0, \\
        0, & \text{if} \langle A, A \rangle = 0.
        \end{cases}

    where :math:`\langle {}\cdot{}, {}\cdot{} \rangle` is the scalar
    product in the Hilbert space of :math:`U`-centered distance
    matrices, given by the function :py:func:`u_product`.

    Parameters
    ----------
    a: array_like
        :math:`U`-centered distance matrix.

    Returns
    -------
    callable
        Function that receives a :math:`U`-centered distance matrix and
        computes its orthogonal projection over :math:`a`.

    See Also
    --------
    u_complementary_projection
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> proj_a = dcor.u_projection(u_a)
    >>> proj_a(u_a)
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> proj_a(u_b)
    array([[-0.        ,  2.66666667, -1.33333333, -1.33333333],
           [ 2.66666667, -0.        , -1.33333333, -1.33333333],
           [-1.33333333, -1.33333333, -0.        ,  2.66666667],
           [-1.33333333, -1.33333333,  2.66666667, -0.        ]])

    The function gives the correct result if
    :math:`\\langle A, A \\rangle = 0`.

    >>> proj_null = dcor.u_projection(np.zeros((4, 4)))
    >>> proj_null(u_a)
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

    """
    c = a
    denominator = u_product(c, c)

    docstring = """
    Orthogonal projection over a :math:`U`-centered distance matrix.

    This function was returned by :code:`u_projection`. The complete
    usage information is in the documentation of :code:`u_projection`.

    See Also
    --------
    u_projection
    """

    if denominator == 0:

        def projection(a):  # noqa
            return np.zeros_like(c)

    else:

        def projection(a):  # noqa
            return u_product(a, c) / denominator * c

    projection.__doc__ = docstring
    return projection


def u_complementary_projection(a):
    r"""
    Return the orthogonal projection function over :math:`a^{\perp}`.

    The function returned computes the orthogonal projection over
    :math:`a^{\perp}` (the complementary projection over a)
    in the Hilbert space of :math:`U`-centered distance matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A^{\perp}`
    is defined as

    .. math::
        \text{proj}_{A^{\perp}}(B) = B - \text{proj}_A(B)

    Parameters
    ----------
    a: array_like
        :math:`U`-centered distance matrix.

    Returns
    -------
    callable
        Function that receives a :math:`U`-centered distance matrices
        and computes its orthogonal projection over :math:`a^{\perp}`.

    See Also
    --------
    u_projection
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> proj_a = dcor.u_complementary_projection(u_a)
    >>> proj_a(u_a)
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> proj_a(u_b)
    array([[ 0.0000000e+00, -4.4408921e-16,  4.0000000e+00, -4.0000000e+00],
           [-4.4408921e-16,  0.0000000e+00, -4.0000000e+00,  4.0000000e+00],
           [ 4.0000000e+00, -4.0000000e+00,  0.0000000e+00, -4.4408921e-16],
           [-4.0000000e+00,  4.0000000e+00, -4.4408921e-16,  0.0000000e+00]])
    >>> proj_null = dcor.u_complementary_projection(np.zeros((4, 4)))
    >>> proj_null(u_a)
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])

    """
    proj = u_projection(a)

    def projection(a):
        """
        Orthogonal projection over the complementary space.

        This function was returned by :code:`u_complementary_projection`.
        The complete usage information is in the documentation of
        :code:`u_complementary_projection`.

        See Also
        --------
        u_complementary_projection

        """
        return a - proj(a)

    return projection


def _distance_matrix_generic(x, centering, exponent=1):
    """Compute a centered distance matrix given a matrix."""
    _check_valid_dcov_exponent(exponent)

    x = _transform_to_2d(x)

    # Calculate distance matrices
    a = distances.pairwise_distances(x, exponent=exponent)

    # Double centering
    a = centering(a, out=a)

    return a


def _distance_matrix(x, exponent=1):
    """Compute the double centered distance matrix given a matrix."""
    return _distance_matrix_generic(x, centering=double_centered,
                                    exponent=exponent)


def _u_distance_matrix(x, exponent=1):
    """Compute the :math:`U`-centered distance matrices given a matrix."""
    return _distance_matrix_generic(x, centering=u_centered,
                                    exponent=exponent)


def _mat_sqrt_inv(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Eliminate negative values
    np.clip(eigenvalues, a_min=0, a_max=None, out=eigenvalues)

    eigenvalues[eigenvalues == 0] = np.inf
    eigenvalues_sqrt_inv = 1 / np.sqrt(eigenvalues)

    return eigenvectors.dot(np.diag(eigenvalues_sqrt_inv)).dot(eigenvectors.T)


def _af_inv_scaled(x):
    """Scale a random vector for using the affinely invariant measures"""
    x = _transform_to_2d(x)

    cov_matrix = np.atleast_2d(np.cov(x, rowvar=False))

    cov_matrix_power = _mat_sqrt_inv(cov_matrix)

    return x.dot(cov_matrix_power)

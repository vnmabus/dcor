"""
Internal functions for distance covariance and correlation.

The functions in this module are used for performing computations related with
distance covariance and correlation.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, TypeVar

from . import distances
from ._utils import ArrayType, _transform_to_2d, get_namespace

T = TypeVar("T", bound=ArrayType)

if TYPE_CHECKING:
    try:
        from typing import Protocol
    except ImportError:
        from typing_extensions import Protocol
else:
    Protocol = object


class Centering(Protocol):
    """Callback protocol for centering method."""

    def __call__(self, __a: T, *, out: T | None) -> T:
        ...


class MatrixCentered(Protocol):
    """Callback protocol for centering method."""

    def __call__(self, __a: T, *, exponent: float) -> T:
        ...


def _check_valid_dcov_exponent(exponent: float) -> None:
    if not 0 < exponent < 2:
        warning_msg = (
            f'Distance covariance is not guaranteed to '
            f'characterize independence if the exponent value is '
            f'not in the range (0, 2). The exponent passed '
            f'is {exponent}.'
        )

        warnings.warn(warning_msg)


def _check_same_n_elements(x: T, y: T) -> None:
    xp = get_namespace(x, y)

    x = xp.asarray(x)
    y = xp.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f'x and y must have the same number of examples. The '
            f'number of samples of x is {x.shape[0]} while the '
            f'number of samples of y is {y.shape[0]}.'
        )


def _float_copy_to_out(out: T | None, origin: T) -> T:
    """
    Copy origin to out and return it.

    If ``out`` is None, a new copy (casted to floating point) is used. If
    ``out`` and ``origin`` are the same, we simply return it. Otherwise we
    copy the values.

    """
    if out is None:
        return origin / 1  # The division forces cast to a floating point type

    if out is not origin:
        out[...] = origin[...]
    return out


def double_centered(a: T, *, out: T | None = None) -> T:
    r"""
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

    Args:
        a: Original square matrix.
        out: If not None, specifies where to return the resulting array. This
            array should allow non integer numbers.

    Returns:
        Double centered matrix.

    See Also:
        u_centered

    Examples:
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
    out = _float_copy_to_out(out, a)

    xp = get_namespace(a)

    mu = xp.mean(a)
    mu_cols = xp.mean(a, axis=0, keepdims=True)
    mu_rows = xp.mean(a, axis=1, keepdims=True)

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= mu_rows
    out -= mu_cols
    out += mu

    return out


def u_centered(a: T, *, out: T | None = None) -> T:
    r"""
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

    Args:
        a: Original square matrix.
        out: If not None, specifies where to return the resulting array. This
            array should allow non integer numbers.

    Returns:
        :math:`U`-centered matrix.

    See Also:
        double_centered

    Examples:
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
    out = _float_copy_to_out(out, a)

    dim = a.shape[0]

    xp = get_namespace(a)

    u_mu = xp.sum(a) / ((dim - 1) * (dim - 2))
    sum_cols = xp.sum(a, axis=0, keepdims=True)
    sum_rows = xp.sum(a, axis=1, keepdims=True)
    u_mu_cols = sum_cols / (dim - 2)
    u_mu_rows = sum_rows / (dim - 2)

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= u_mu_rows
    out -= u_mu_cols
    out += u_mu

    # The diagonal is zero
    out[xp.eye(dim, dtype=xp.bool)] = 0

    return out


def mean_product(a: T, b: T) -> T:
    r"""
    Average of the elements for an element-wise product of two matrices.

    If the matrices are square it is

    .. math::
        \frac{1}{n^2} \sum_{i,j=1}^n a_{i, j} b_{i, j}.

    Args:
        a: First input array to be multiplied.
        b: Second input array to be multiplied.

    Returns:
        Average of the product.

    See Also:
        u_product

    Examples:
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
    xp = get_namespace(a, b)
    return xp.mean(a * b)


def u_product(a: T, b: T) -> T:
    r"""
    Inner product in the Hilbert space of :math:`U`-centered distance matrices.

    This inner product is defined as

    .. math::
        \frac{1}{n(n-3)} \sum_{i,j=1}^n a_{i, j} b_{i, j}

    Args:
        a: First input array to be multiplied.
        b: Second input array to be multiplied.

    Returns:
        Inner product.

    See Also:
        mean_product

    Examples:
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
    n = a.shape[0]

    xp = get_namespace(a, b)

    return xp.sum(a * b) / (n * (n - 3))


def u_projection(a: T) -> Callable[[T], T]:
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

    Args:
        a: :math:`U`-centered distance matrix.

    Returns:
        Function that receives a :math:`U`-centered distance matrix and
        computes its orthogonal projection over :math:`a`.

    See Also:
        u_complementary_projection
        u_centered

    Examples:
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

    See Also:
        u_projection
    """

    xp = get_namespace(a)

    if denominator == 0:

        def projection(a: T) -> T:  # noqa
            return xp.zeros_like(c)

    else:

        def projection(a: T) -> T:  # noqa
            return u_product(a, c) / denominator * c

    projection.__doc__ = docstring
    return projection


def u_complementary_projection(a: T) -> Callable[[T], T]:
    r"""
    Return the orthogonal projection function over :math:`a^{\perp}`.

    The function returned computes the orthogonal projection over
    :math:`a^{\perp}` (the complementary projection over a)
    in the Hilbert space of :math:`U`-centered distance matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A^{\perp}`
    is defined as

    .. math::
        \text{proj}_{A^{\perp}}(B) = B - \text{proj}_A(B)

    Args:
        a: :math:`U`-centered distance matrix.

    Returns:
        Function that receives a :math:`U`-centered distance matrices
        and computes its orthogonal projection over :math:`a^{\perp}`.

    See Also:
        u_projection
        u_centered

    Examples:
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

    def projection(a: T) -> T:
        """
        Orthogonal projection over the complementary space.

        This function was returned by :code:`u_complementary_projection`.
        The complete usage information is in the documentation of
        :code:`u_complementary_projection`.

        See Also:
            u_complementary_projection

        """
        return a - proj(a)

    return projection


def _distance_matrix_generic(
    x: T,
    centering: Centering,
    exponent: float = 1,
) -> T:
    """Compute a centered distance matrix given a matrix."""
    _check_valid_dcov_exponent(exponent)

    x, = _transform_to_2d(x)

    # Calculate distance matrices
    a = distances.pairwise_distances(x, exponent=exponent)

    # Double centering
    a = centering(a, out=a)

    return a


def _distance_matrix(x: T, *, exponent: float = 1) -> T:
    """Compute the double centered distance matrix given a matrix."""
    return _distance_matrix_generic(
        x,
        centering=double_centered,
        exponent=exponent,
    )


def _u_distance_matrix(x: T, *, exponent: float = 1) -> T:
    """Compute the :math:`U`-centered distance matrices given a matrix."""
    return _distance_matrix_generic(
        x,
        centering=u_centered,
        exponent=exponent,
    )


def _mat_sqrt_inv(matrix: T) -> T:
    xp = get_namespace(matrix)

    eigenvalues, eigenvectors = xp.linalg.eigh(matrix)

    # Eliminate negative values
    eigenvalues[eigenvalues <= 0] = xp.inf
    eigenvalues_sqrt_inv = 1 / xp.sqrt(eigenvalues)

    return eigenvectors * eigenvalues_sqrt_inv @ eigenvectors.T


def _cov(x: T) -> T:
    """Equivalent to np.cov(x, rowvar=False)."""
    x, = _transform_to_2d(x)

    xp = get_namespace(x)

    mean = xp.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    return (x_centered.T @ x_centered) / (x.shape[0] - 1)


def _af_inv_scaled(x: T) -> T:
    """Scale a random vector for using the affinely invariant measures."""
    x, = _transform_to_2d(x)

    cov_matrix = _cov(x)

    cov_matrix_power = _mat_sqrt_inv(cov_matrix)

    return (x @ cov_matrix_power) / (x.shape[0] - 1)

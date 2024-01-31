"""
Internal functions for distance covariance and correlation.

The functions in this module are used for performing computations related with
distance covariance and correlation.
"""
from __future__ import annotations

import warnings
from typing import Callable, Literal, Protocol, Tuple, TypeVar, overload

from . import distances
from ._utils import ArrayType, CompileMode, _transform_to_2d, array_namespace

Array = TypeVar("Array", bound=ArrayType)


class Centering(Protocol):
    """Callback protocol for centering method."""

    def __call__(self, __a: Array, *, out: Array | None) -> Array:
        ...


class MatrixCentered(Protocol):
    """Callback protocol for centering method."""

    def __call__(self, __a: Array, *, exponent: float) -> Array:
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


def _check_same_n_elements(x: Array, y: Array) -> None:
    xp = array_namespace(x, y)

    x = xp.asarray(x)
    y = xp.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f'x and y must have the same number of examples. The '
            f'number of samples of x is {x.shape[0]} while the '
            f'number of samples of y is {y.shape[0]}.'
        )


def _float_copy_to_out(out: Array | None, origin: Array) -> Array:
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


def _symmetric_matrix_sums(a: Array) -> Tuple[Array, Array]:
    """Compute row, column and total sums of a symmetric matrix."""
    # Currently there is no portable way to check the order (Fortran/C)
    # across different array libraries.
    # Thus, we assume data is C-contiguous and then the faster array is 1.
    fast_axis = 1

    xp = array_namespace(a)

    axis_sum = xp.sum(a, axis=fast_axis)
    total_sum = xp.sum(axis_sum)

    return axis_sum, total_sum


@overload
def _dcov_terms_naive(
    x: Array,
    y: Array,
    *,
    exponent: float,
    compile_mode: CompileMode = CompileMode.AUTO,
    return_var_terms: Literal[False] = False,
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    None,
    None,
]:
    pass


@overload
def _dcov_terms_naive(
    x: Array,
    y: Array,
    *,
    exponent: float,
    compile_mode: CompileMode = CompileMode.AUTO,
    return_var_terms: Literal[True],
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    pass


def _dcov_terms_naive(
    x: Array,
    y: Array,
    *,
    exponent: float,
    compile_mode: CompileMode = CompileMode.AUTO,
    return_var_terms: bool = False,
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array | None,
    Array | None,
]:
    """Return terms used in dcov."""
    if compile_mode not in {CompileMode.AUTO, CompileMode.NO_COMPILE}:
        raise NotImplementedError(
            f"Compile mode {compile_mode} not implemented.",
        )

    xp = array_namespace(x, y)
    a, b = _compute_distances(
        x,
        y,
        exponent=exponent,
    )

    a_vec = xp.reshape(a, -1)
    b_vec = xp.reshape(b, -1)

    mean_prod = a_vec @ b_vec
    a_sums = _symmetric_matrix_sums(a)
    b_sums = _symmetric_matrix_sums(b)

    mean_prod_a = None
    mean_prod_b = None

    if return_var_terms:
        mean_prod_a = a_vec @ a_vec
        mean_prod_b = b_vec @ b_vec

    return mean_prod, *a_sums, *b_sums, mean_prod_a, mean_prod_b


def _dcov_from_terms(
    mean_prod: Array,
    a_axis_sum: Array,
    a_total_sum: Array,
    b_axis_sum: Array,
    b_total_sum: Array,
    n_samples: int,
    bias_corrected: bool = False,
) -> Array:
    """Compute distance covariance WITHOUT centering first."""
    first_term = mean_prod / n_samples
    second_term = a_axis_sum / n_samples @ b_axis_sum
    third_term = a_total_sum / n_samples * b_total_sum

    if bias_corrected:
        first_term /= (n_samples - 3)
        second_term /= (n_samples - 2) * (n_samples - 3)
        third_term /= (n_samples - 1) * (n_samples - 2) * (n_samples - 3)
    else:
        first_term /= n_samples
        second_term /= n_samples**2
        third_term /= n_samples**3

    return first_term - 2 * second_term + third_term


def double_centered(a: Array, *, out: Array | None = None) -> Array:
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
        a: Original symmetric square matrix.
        out: If not None, specifies where to return the resulting array. This
            array should allow non integer numbers.

    Returns:
        Double centered matrix.

    See Also:
        u_centered

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2], [2, 4]])
        >>> dcor.double_centered(a)
        array([[ 0.25, -0.25],
               [-0.25,  0.25]])
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

    dim = a.shape[0]
    axis_sum, total_sum = _symmetric_matrix_sums(a)

    total_mean = total_sum / dim**2
    axis_mean = axis_sum / dim

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= axis_mean[None, :]
    out -= axis_mean[:, None]
    out += total_mean

    return out


def u_centered(a: Array, *, out: Array | None = None) -> Array:
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
        a: Original symmetric square matrix.
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

    axis_sum, total_sum = _symmetric_matrix_sums(a)

    total_u_mean = total_sum / ((dim - 1) * (dim - 2))
    axis_u_mean = axis_sum / (dim - 2)

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= axis_u_mean[None, :]
    out -= axis_u_mean[:, None]
    out += total_u_mean

    # The diagonal is zero
    xp = array_namespace(a)
    out[xp.eye(dim, dtype=xp.bool)] = 0

    return out


def mean_product(a: Array, b: Array) -> Array:
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
    xp = array_namespace(a, b)
    return xp.mean(a * b)


def u_product(a: Array, b: Array) -> Array:
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

    xp = array_namespace(a, b)

    return xp.sum(a * b) / (n * (n - 3))


def u_projection(a: Array) -> Callable[[Array], Array]:
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

    xp = array_namespace(a)

    if denominator == 0:

        def projection(a: T) -> T:  # noqa
            return xp.zeros_like(c)

    else:

        def projection(a: T) -> T:  # noqa
            return u_product(a, c) / denominator * c

    projection.__doc__ = docstring
    return projection


def u_complementary_projection(a: Array) -> Callable[[Array], Array]:
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

    def projection(a: Array) -> Array:
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


def _compute_distances(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
) -> Tuple[Array, Array]:
    """Compute a centered distance matrix given a matrix."""
    _check_valid_dcov_exponent(exponent)

    x, y = _transform_to_2d(x, y)

    # Calculate distance matrices
    a = distances.pairwise_distances(x, exponent=exponent)
    b = distances.pairwise_distances(y, exponent=exponent)

    return a, b


def _distance_matrix_generic(
    x: Array,
    centering: Centering,
    exponent: float = 1,
) -> Array:
    """Compute a centered distance matrix given a matrix."""
    _check_valid_dcov_exponent(exponent)

    x, = _transform_to_2d(x)

    # Calculate distance matrices
    a = distances.pairwise_distances(x, exponent=exponent)

    # Double centering
    a = centering(a, out=a)

    return a


def _u_distance_matrix(x: Array, *, exponent: float = 1) -> Array:
    """Compute the :math:`U`-centered distance matrices given a matrix."""
    return _distance_matrix_generic(
        x,
        centering=u_centered,
        exponent=exponent,
    )


def _mat_sqrt_inv(matrix: Array) -> Array:
    xp = array_namespace(matrix)

    eigenvalues, eigenvectors = xp.linalg.eigh(matrix)

    # Eliminate negative values
    eigenvalues[eigenvalues <= 0] = xp.inf
    eigenvalues_sqrt_inv = 1 / xp.sqrt(eigenvalues)

    return eigenvectors * eigenvalues_sqrt_inv @ eigenvectors.T


def _cov(x: Array) -> Array:
    """Equivalent to np.cov(x, rowvar=False)."""
    x, = _transform_to_2d(x)

    xp = array_namespace(x)

    mean = xp.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    return (x_centered.T @ x_centered) / (x.shape[0] - 1)


def _af_inv_scaled(x: Array) -> Array:
    """Scale a random vector for using the affinely invariant measures."""
    x, = _transform_to_2d(x)

    cov_matrix = _cov(x)

    cov_matrix_power = _mat_sqrt_inv(cov_matrix)

    return (x @ cov_matrix_power) / (x.shape[0] - 1)

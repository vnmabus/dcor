"""Energy distance functions"""

import warnings

import numpy as np

from . import distances
from ._utils import _transform_to_2d


def _check_valid_energy_exponent(exponent):
    if not 0 < exponent < 2:
        warning_msg = ('The energy distance is not guaranteed to be '
                       'a valid metric if the exponent value is '
                       'not in the range (0, 2). The exponent passed '
                       'is {exponent}.'.format(exponent=exponent))

        warnings.warn(warning_msg)


def _energy_distance_from_distance_matrices(
        distance_xx, distance_yy, distance_xy, average=None, stat_type='v'):
    """
    Compute energy distance with precalculated distance matrices.

    Parameters
    ----------
    average: Callable[[ArrayLike], float]
        A function that will be used to calculate an average of distances.
        This defaults to np.mean.
    stat_type: Literal['u', 'v']
        If 'u', calculate energy distance using
        Hoeffding's unbiased U-statistics. Otherwise, use von Mises's biased
        V-statistics
    """
    if average is None:
        average = np.mean

    if stat_type == 'u':
        return (
                2 * average(distance_xy) -
                average(np.triu(distance_xx)) -
                average(np.triu(distance_yy))
        )
    else:
        return (
            2 * average(distance_xy) -
            average(distance_xx) -
            average(distance_yy)
        )


def energy_distance(x, y, *, average=None, exponent=1, stat_type='v'):
    """
    Estimator for energy distance.

    Computes the estimator for the energy distance of the
    random vectors corresponding to :math:`x` and :math:`y`.
    Both random vectors must have the same number of components.

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
    average: Callable[[ArrayLike], float]
        A function that will be used to calculate an average of distances.
        This defaults to np.mean.
    stat_type: Literal['u', 'v']
        If 'u', calculate energy distance using
        Hoeffding's unbiased U-statistics. Otherwise, use von Mises's biased
        V-statistics

    Returns
    -------
    numpy scalar
        Value of the estimator of the energy distance.

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
    ...               [1, 1, 1, 1]])
    >>> dcor.energy_distance(a, a)
    0.0
    >>> dcor.energy_distance(a, b) # doctest: +ELLIPSIS
    20.5780594...
    >>> dcor.energy_distance(b, b)
    0.0

    A different exponent for the Euclidean distance in the range
    :math:`(0, 2)` can be used:

    >>> dcor.energy_distance(a, a, exponent=1.5)
    0.0
    >>> dcor.energy_distance(a, b, exponent=1.5)
    ... # doctest: +ELLIPSIS
    99.7863955...
    >>> dcor.energy_distance(b, b, exponent=1.5)
    0.0

    """
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    _check_valid_energy_exponent(exponent)

    distance_xx = distances.pairwise_distances(x, exponent=exponent)
    distance_yy = distances.pairwise_distances(y, exponent=exponent)
    distance_xy = distances.pairwise_distances(x, y, exponent=exponent)

    return _energy_distance_from_distance_matrices(
        distance_xx=distance_xx,
        distance_yy=distance_yy,
        distance_xy=distance_xy,
        average=average,
        stat_type=stat_type
    )

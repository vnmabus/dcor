"""Energy distance functions"""

import warnings

import numpy as np

from enum import Enum, auto

from . import distances
from ._utils import _transform_to_2d


class EstimationStatistic(Enum):
    """
    A type of estimation statistic used for calculating energy distance.
    """
    @classmethod
    def from_string(cls, item):
        """
        Allows EstimationStatistic.from_string('u'),
        EstimationStatistic.from_string('V'),
        EstimationStatistic.from_string('V_STATISTIC'),
        EstimationStatistic.from_string('u_statistic') etc
        """
        upper = item.upper()
        if upper == 'U':
            return cls.U_STATISTIC
        elif upper == 'V':
            return cls.V_STATISTIC
        else:
            return cls[upper]

    U_STATISTIC = auto()
    """
    Hoeffding's unbiased U-statistics
    (does not include the distance from each point to itself)
    """

    V_STATISTIC = auto()
    """
    von Mises's biased V-statistics
    (does include the distance from each point to itself)
    """


def _check_valid_energy_exponent(exponent):
    if not 0 < exponent < 2:
        warning_msg = ('The energy distance is not guaranteed to be '
                       'a valid metric if the exponent value is '
                       'not in the range (0, 2). The exponent passed '
                       'is {exponent}.'.format(exponent=exponent))

        warnings.warn(warning_msg)


def _energy_distance_from_distance_matrices(
        distance_xx, distance_yy, distance_xy, average=None,
        estimation_stat=EstimationStatistic.V_STATISTIC):
    """
    Compute energy distance with precalculated distance matrices.

    Parameters
    ----------
    average: Callable[[ArrayLike], float]
        A function that will be used to calculate an average of distances.
        This defaults to np.mean.
    estimation_stat: Union[str, EstimationStatistic]
        If EstimationStatistic.U_STATISTIC, calculate energy distance using
        Hoeffding's unbiased U-statistics. Otherwise, use von Mises's biased
        V-statistics.
        If this is provided as a string, it will first be converted to
        an EstimationStatistic enum instance.
    """
    if isinstance(estimation_stat, str):
        estimation_stat = EstimationStatistic.from_string(estimation_stat)

    if average is None:
        average = np.mean

    if estimation_stat == EstimationStatistic.U_STATISTIC:
        # If using u-statistics, we exclude the central diagonal of 0s for the
        # within-sample distances
        distance_xx = distance_xx[np.triu_indices_from(distance_xx, k=1)]
        distance_yy = distance_yy[np.triu_indices_from(distance_yy, k=1)]

    return (
        2 * average(distance_xy) -
        average(distance_xx) -
        average(distance_yy)
    )


def energy_distance(x, y, *, average=None, exponent=1,
                    estimation_stat=EstimationStatistic.V_STATISTIC):
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
    estimation_stat: Union[str, EstimationStatistic]
        If EstimationStatistic.U_STATISTIC, calculate energy distance using
        Hoeffding's unbiased U-statistics. Otherwise, use von Mises's biased
        V-statistics.
        If this is provided as a string, it will first be converted to
        an EstimationStatistic enum instance.

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
        estimation_stat=estimation_stat
    )

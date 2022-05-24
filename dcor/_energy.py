"""Energy distance functions."""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import Callable, TypeVar, Union

from . import distances
from ._utils import ArrayType, _transform_to_2d, get_namespace

T = TypeVar("T", bound=ArrayType)


class EstimationStatistic(Enum):
    """A type of estimation statistic used for calculating energy distance."""

    @classmethod
    def from_string(cls, string: str) -> EstimationStatistic:
        """
        Parse the estimation statistic from a string.

        The string is converted to upercase first. Valid values are:
            - ``"U_STATISTIC"`` or ``"U"``: for the unbiased version.
            - ``"V_STATISTIC"`` or ``"V"``: for the biased version.

        Examples:
            >>> from dcor import EstimationStatistic
            >>>
            >>> EstimationStatistic.from_string('u')
            <EstimationStatistic.U_STATISTIC: 1>
            >>> EstimationStatistic.from_string('V')
            <EstimationStatistic.V_STATISTIC: 2>
            >>> EstimationStatistic.from_string('V_STATISTIC')
            <EstimationStatistic.V_STATISTIC: 2>
            >>> EstimationStatistic.from_string('u_statistic')
            <EstimationStatistic.U_STATISTIC: 1>

        """
        upper = string.upper()
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


EstimationStatisticLike = Union[EstimationStatistic, str]


def _check_valid_energy_exponent(exponent: float) -> None:
    if not 0 < exponent < 2:
        warning_msg = (
            f'The energy distance is not guaranteed to be '
            f'a valid metric if the exponent value is '
            f'not in the range (0, 2). The exponent passed '
            f'is {exponent}.'
        )

        warnings.warn(warning_msg)


def _get_flat_upper_matrix(x: T, k: int) -> T:
    """Get flat upper matrix from diagonal k."""
    xp = get_namespace(x)
    x_mask = xp.triu(xp.ones_like(x, dtype=xp.bool), k=k)
    x_mask_flat = xp.reshape(x_mask, -1)
    x_flat = xp.reshape(x, -1)

    return x_flat[x_mask_flat]


def _energy_distance_from_distance_matrices(
    distance_xx: T,
    distance_yy: T,
    distance_xy: T,
    average: Callable[[T], T] | None = None,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
) -> T:
    """
    Compute energy distance with precalculated distance matrices.

    Args:
        distance_xx: Pairwise distances of X.
        distance_yy: Pairwise distances of Y.
        distance_xy: Pairwise distances between X and Y.
        average: A function that will be used to calculate an average of
            distances. This defaults to the mean.
        estimation_stat: If EstimationStatistic.U_STATISTIC, calculate energy
            distance using Hoeffding's unbiased U-statistics. Otherwise, use
            von Mises's biased V-statistics.
            If this is provided as a string, it will first be converted to
            an EstimationStatistic enum instance.

    """
    xp = get_namespace(distance_xx, distance_yy, distance_xy)

    if isinstance(estimation_stat, str):
        estimation_stat = EstimationStatistic.from_string(estimation_stat)

    if average is None:
        average = xp.mean

    if estimation_stat == EstimationStatistic.U_STATISTIC:
        # If using u-statistics, we exclude the central diagonal of 0s for the
        distance_xx = _get_flat_upper_matrix(distance_xx, k=1)
        distance_yy = _get_flat_upper_matrix(distance_yy, k=1)

    return (
        2 * average(distance_xy)
        - average(distance_xx)
        - average(distance_yy)
    )


def energy_distance(
    x: T,
    y: T,
    *,
    average: Callable[[T], T] | None = None,
    exponent: float = 1,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
) -> T:
    """
    Estimator for energy distance.

    Computes the estimator for the energy distance of the
    random vectors corresponding to :math:`x` and :math:`y`.
    Both random vectors must have the same number of components.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`.
        average: A function that will be used to calculate an average of
            distances. This defaults to the mean.
        estimation_stat: Union[str, EstimationStatistic]
            If EstimationStatistic.U_STATISTIC, calculate energy distance using
            Hoeffding's unbiased U-statistics. Otherwise, use von Mises's
            biased V-statistics.
            If this is provided as a string, it will first be converted to
            an EstimationStatistic enum instance.

    Returns:
        Value of the estimator of the energy distance.

    Examples:
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
    x, y = _transform_to_2d(x, y)

    _check_valid_energy_exponent(exponent)

    distance_xx = distances.pairwise_distances(x, exponent=exponent)
    distance_yy = distances.pairwise_distances(y, exponent=exponent)
    distance_xy = distances.pairwise_distances(x, y, exponent=exponent)

    return _energy_distance_from_distance_matrices(
        distance_xx=distance_xx,
        distance_yy=distance_yy,
        distance_xy=distance_xy,
        average=average,
        estimation_stat=estimation_stat,
    )

"""
Distance correlation and related e-statistics.

This package provide a Python implementation of
distance correlation and other e-statistics, such as
energy distance.
"""

import errno as _errno
import os as _os
import pathlib as _pathlib

from . import distances  # noqa
from . import homogeneity  # noqa
from . import independence  # noqa
from ._dcor import (  # noqa
    CompileMode,
    DistanceCovarianceMethod,
    distance_correlation,
    distance_correlation_af_inv,
    distance_correlation_af_inv_sqr,
    distance_correlation_sqr,
    distance_covariance,
    distance_covariance_sqr,
    distance_stats,
    distance_stats_sqr,
    u_distance_correlation_sqr,
    u_distance_covariance_sqr,
    u_distance_stats_sqr,
)
from ._dcor_internals import (  # noqa
    double_centered,
    mean_product,
    u_centered,
    u_complementary_projection,
    u_product,
    u_projection,
)
from ._energy import EstimationStatistic, energy_distance  # noqa
from ._partial_dcor import partial_distance_covariance  # noqa
from ._partial_dcor import partial_distance_correlation
from ._rowwise import RowwiseMode, rowwise

try:
    __version__ = (
        _pathlib.Path(_os.path.dirname(__file__)) / 'VERSION'
    ).read_text().strip()
except FileNotFoundError:
    __version__ = "0.0"

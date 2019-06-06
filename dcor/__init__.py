"""
Distance correlation and related e-statistics.

This package provide a Python implementation of
distance correlation and other e-statistics, such as
energy distance.
"""

import errno as _errno
import os as _os

from . import distances  # noqa
from . import homogeneity  # noqa
from . import independence  # noqa
from ._dcor import (distance_covariance_sqr, distance_covariance,  # noqa
                   distance_correlation_sqr, distance_correlation,
                   distance_stats_sqr, distance_stats,
                   u_distance_covariance_sqr,
                   u_distance_correlation_sqr,
                   u_distance_stats_sqr,
                   distance_correlation_af_inv_sqr,
                   distance_correlation_af_inv)
from ._dcor_internals import (double_centered, u_centered,  # noqa
                              mean_product, u_product,
                              u_projection,
                              u_complementary_projection)
from ._energy import energy_distance  # noqa
from ._pairwise import pairwise
from ._partial_dcor import (partial_distance_covariance,  # noqa
                            partial_distance_correlation)

try:
    with open(_os.path.join(_os.path.dirname(__file__),
                            '..', 'VERSION'), 'r') as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"

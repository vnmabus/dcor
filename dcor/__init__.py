import errno as _errno
import os as _os

from .dcor import (double_centered, u_centered,
                   average_product, u_product,
                   distance_covariance_sqr, distance_covariance,
                   distance_correlation_sqr, distance_correlation,
                   distance_stats_sqr, distance_stats,
                   u_distance_covariance_sqr,
                   u_distance_correlation_sqr,
                   u_distance_stats_sqr)


try:
    with open(_os.path.join(_os.path.dirname(__file__),
                            '..', 'VERSION'), 'r') as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"

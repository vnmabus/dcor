import os
from .dcor import (double_centered, u_centered,
                   average_product, u_product,
                   distance_covariance_sqr, distance_covariance,
                   distance_correlation_sqr, distance_correlation,
                   distance_stats_sqr, distance_stats,
                   u_distance_covariance_sqr,
                   u_distance_correlation_sqr,
                   u_distance_stats_sqr)

with open(os.path.join(os.path.dirname(__file__),
                       '..', 'VERSION'), 'r') as version_file:
    __version__ = version_file.read().strip()

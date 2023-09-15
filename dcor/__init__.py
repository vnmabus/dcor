"""
Distance correlation and related e-statistics.

This package provide a Python implementation of
distance correlation and other e-statistics, such as
energy distance.
"""

from . import distances, homogeneity, independence
from ._dcor import (
    DistanceCovarianceMethod as DistanceCovarianceMethod,
    Stats as Stats,
    distance_correlation as distance_correlation,
    distance_correlation_af_inv as distance_correlation_af_inv,
    distance_correlation_af_inv_sqr as distance_correlation_af_inv_sqr,
    distance_correlation_sqr as distance_correlation_sqr,
    distance_covariance as distance_covariance,
    distance_covariance_sqr as distance_covariance_sqr,
    distance_stats as distance_stats,
    distance_stats_sqr as distance_stats_sqr,
    u_distance_correlation_sqr as u_distance_correlation_sqr,
    u_distance_covariance_sqr as u_distance_covariance_sqr,
    u_distance_stats_sqr as u_distance_stats_sqr,
)
from ._dcor_internals import (
    double_centered as double_centered,
    mean_product as mean_product,
    u_centered as u_centered,
    u_complementary_projection as u_complementary_projection,
    u_product as u_product,
    u_projection as u_projection,
)
from ._energy import (
    EstimationStatistic as EstimationStatistic,
    energy_distance as energy_distance,
)
from ._hypothesis import HypothesisTest as HypothesisTest
from ._partial_dcor import (
    partial_distance_correlation,
    partial_distance_covariance as partial_distance_covariance,
)
from ._rowwise import RowwiseMode as RowwiseMode, rowwise as rowwise
from ._utils import CompileMode as CompileMode

__version__ = "0.6"

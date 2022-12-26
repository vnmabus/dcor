API List
========

List of functions
-----------------
A complete list of all functions provided by dcor.

Biased estimators for distance covariance and distance correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These functions compute the usual (biased) estimators for the distance
covariance and distance correlation and their squares.

.. autosummary::
   :toctree: functions
   
   dcor.distance_covariance
   dcor.distance_covariance_sqr
   dcor.distance_correlation
   dcor.distance_correlation_sqr
   dcor.distance_stats
   dcor.distance_stats_sqr
   
Unbiased and bias-corrected estimators for distance covariance and distance correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These functions compute the unbiased estimators for the square of the distance
covariance and the bias corrected estimator for the square of the distance correlation. 
As these estimators are signed, no functions are provided for taking the square root.

.. autosummary::
   :toctree: functions
   
   dcor.u_distance_covariance_sqr
   dcor.u_distance_correlation_sqr
   dcor.u_distance_stats_sqr
   
Affinely invariant distance correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These functions compute the estimators for the affinely invariant distance correlation,
a variant of distance correlation that is invariant by invertible affine transformations
of the input parameters.

.. autosummary::
   :toctree: functions
   
   dcor.distance_correlation_af_inv_sqr
   dcor.distance_correlation_af_inv

Partial distance covariance and partial distance correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These functions compute the estimators for the partial distance
covariance and partial distance correlation.

.. autosummary::
   :toctree: functions
   
   dcor.partial_distance_covariance
   dcor.partial_distance_correlation
   
Energy distance
^^^^^^^^^^^^^^^
The following function is an estimator for the energy distance between
two random vectors.

.. autosummary::
   :toctree: functions
   
   dcor.energy_distance
   
Homogeneity test
^^^^^^^^^^^^^^^^
The following functions are used to test if random vectors have the same
distribution.

.. autosummary::
   :toctree: functions
   
   dcor.homogeneity.energy_test_statistic
   dcor.homogeneity.energy_test
   
Independence test
^^^^^^^^^^^^^^^^^
The following functions are used to test if two random vectors are independent.

.. autosummary::
   :toctree: functions
   
   dcor.independence.distance_covariance_test
   dcor.independence.distance_correlation_t_statistic
   dcor.independence.distance_correlation_t_test

Internal computations
^^^^^^^^^^^^^^^^^^^^^
These functions are used for computing the estimators of the squared
distance covariance, and are also provided by this package.

.. autosummary::
   :toctree: functions
   
   dcor.double_centered
   dcor.u_centered
   dcor.mean_product
   dcor.u_product
   dcor.u_projection
   dcor.u_complementary_projection

Compute distance matrices
^^^^^^^^^^^^^^^^^^^^^^^^^
These functions are used for computing distance matrices.

.. autosummary::
   :toctree: functions
   
   dcor.distances.pairwise_distances
   
Compute the same measure between different random variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function is used to compute a measure such as distance
covariance/correlation between different random variables. It will
use an optimized implementation (possibly parallelized in multicore
machines) when it is available.

.. autosummary::
   :toctree: functions
   
   dcor.rowwise
   
List of classes
---------------
A complete list of all classes provided by dcor.

.. autosummary::
   :toctree: functions
   
   dcor.CompileMode
   dcor.DistanceCovarianceMethod
   dcor.EstimationStatistic
   dcor.HypothesisTest
   dcor.RowwiseMode
   dcor.Stats
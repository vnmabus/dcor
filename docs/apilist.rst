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

Partial distance covariance and partial distance correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These functions compute the estimators for the partial distance
covariance and partial distance correlation.

.. autosummary::
   :toctree: functions
   
   dcor.partial_distance_covariance
   dcor.partial_distance_correlation

Internal computations
^^^^^^^^^^^^^^^^^^^^^
These functions are used for computing the estimators of the squared
distance covariance, and are also provided by this package.

.. autosummary::
   :toctree: functions
   
   dcor.double_centered
   dcor.u_centered
   dcor.average_product
   dcor.u_product
   dcor.u_projection
   dcor.u_complementary_projection

.. dcor documentation master file, created by
   sphinx-quickstart on Thu Sep 14 14:53:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dcor version |version|
======================

|build-status|

Distance covariance and distance correlation are
dependency measures between random vectors introduced in :cite:`a-distance_correlation`.

This package provide functions for calculating several statistics
related with distance covariance and distance correlation, including
biased and unbiased estimators of both dependency measures.

References
----------
.. bibliography:: refs.bib
   :labelprefix: A
   :keyprefix: a-

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   theory
   apilist
   dcor

dcor is developed `on Github <http://github.com/vnmabus/dcor>`_. Please
report `issues <https://github.com/vnmabus/dcor/issues>`_ there as well.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |build-status| image:: https://api.travis-ci.org/vnmabus/dcor.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/vnmabus/dcor

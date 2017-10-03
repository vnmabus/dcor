dcor
====

|build-status| |docs|

dcor: distance correlation and distance covariance in Python.

Distance covariance and distance correlation are
dependency measures between random vectors introduced in [SRB07]_.

This package offers functions for calculating several statistics
related with distance covariance and distance correlation, including
biased and unbiased estimators of both dependency measures.

Installation
============

dcor is on PyPi and can be installed using :code:`pip`:

.. code::

   pip install dcor
   
It is also available for :code:`conda`:

.. code::

   conda install -c vnmabus dcor

Requirements
------------

dcor is available in Python 3.5 or above and in Python 2.7, in all operating systems.

Documentation
=============
The documentation can be found in https://dcor.readthedocs.io/en/latest/?badge=latest

References
==========

.. [SRB07] Gábor J. Székely, Maria L. Rizzo, and Nail K. Bakirov. Measuring and
           testing dependence by correlation of distances. The Annals of
           Statistics, 35(6):2769–2794, 12 2007.
           doi:10.1214/009053607000000505.

.. |build-status| image:: https://api.travis-ci.org/vnmabus/dcor.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/vnmabus/dcor

.. |docs| image:: https://readthedocs.org/projects/dcor/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://dcor.readthedocs.io/en/latest/?badge=latest
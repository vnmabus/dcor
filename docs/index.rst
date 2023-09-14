.. dcor documentation master file, created by
   sphinx-quickstart on Thu Sep 14 14:53:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dcor version |version|
======================

|tests| |docs| |coverage| |pypi| |conda| |zenodo|

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
   :hidden:
   :maxdepth: 4
   :caption: Contents:

   installation
   theory
   auto_examples/index
   apilist
   energycomparison
   Release Notes <https://github.com/vnmabus/dcor/releases>
   citing
   development
   contributors

dcor is developed `on Github <http://github.com/vnmabus/dcor>`_. Please
report `issues <https://github.com/vnmabus/dcor/issues>`_ there as well.

.. |tests| image:: https://github.com/vnmabus/dcor/actions/workflows/main.yml/badge.svg
    :alt: Tests
    :scale: 100%
    :target: https://github.com/vnmabus/dcor/actions/workflows/main.yml

.. |docs| image:: https://readthedocs.org/projects/dcor/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://dcor.readthedocs.io/en/latest/?badge=latest
    
.. |coverage| image:: http://codecov.io/github/vnmabus/dcor/coverage.svg?branch=develop
    :alt: Coverage Status
    :scale: 100%
    :target: https://codecov.io/gh/vnmabus/dcor/branch/develop
    
.. |pypi| image:: https://badge.fury.io/py/dcor.svg
    :alt: Pypi version
    :scale: 100%
    :target: https://pypi.python.org/pypi/dcor/
    
.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/dcor
    :alt: Available in Conda
    :scale: 100%
    :target: https://anaconda.org/conda-forge/dcor
    
.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3468124.svg
    :alt: Zenodo DOI
    :scale: 100%
    :target: https://doi.org/10.5281/zenodo.3468124

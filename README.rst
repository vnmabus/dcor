dcor
====

|tests| |docs| |coverage| |repostatus| |versions| |pypi| |conda| |zenodo|

dcor: distance correlation and energy statistics in Python.

E-statistics are functions of distances between statistical observations
in metric spaces.

Distance covariance and distance correlation are
dependency measures between random vectors introduced in [SRB07]_ with
a simple E-statistic estimator.

This package offers functions for calculating several E-statistics
such as:

- Estimator of the energy distance [SR13]_.
- Biased and unbiased estimators of distance covariance and
  distance correlation [SRB07]_.
- Estimators of the partial distance covariance and partial
  distance covariance [SR14]_.

It also provides tests based on these E-statistics:

- Test of homogeneity based on the energy distance.
- Test of independence based on distance covariance.

Installation
============

dcor is on PyPi and can be installed using :code:`pip`:

.. code::

   pip install dcor
   
It is also available for :code:`conda` using the :code:`conda-forge` channel:

.. code::

   conda install -c conda-forge dcor
   
Previous versions of the package were in the :code:`vnmabus` channel. This
channel will not be updated with new releases, and users are recommended to
use the :code:`conda-forge` channel.

Requirements
------------

dcor is available in Python 3.8 or above in all operating systems.
The package dcor depends on the following libraries:

- numpy
- numba >= 0.51
- scipy
- joblib

Citing dcor
===========

Please, if you find this software useful in your work, reference it citing the following paper:

.. code-block::
  
  @article{ramos-carreno+torrecilla_2023_dcor,
    author = {Ramos-Carreño, Carlos and Torrecilla, José L.},
    doi = {10.1016/j.softx.2023.101326},
    journal = {SoftwareX},
    month = {2},
    title = {{dcor: Distance correlation and energy statistics in Python}},
    url = {https://www.sciencedirect.com/science/article/pii/S2352711023000225},
    volume = {22},
    year = {2023},
  }

You can additionally cite the software repository itself using:

.. code-block::

  @misc{ramos-carreno_2022_dcor,
    author = {Ramos-Carreño, Carlos},
    doi = {10.5281/zenodo.3468124},
    month = {3},
    title = {dcor: distance correlation and energy statistics in Python},
    url = {https://github.com/vnmabus/dcor},
    year = {2022}
  }

If you want to reference a particular version for reproducibility, check the version-specific DOIs available in Zenodo.

Documentation
=============
The documentation can be found in https://dcor.readthedocs.io/en/latest/?badge=latest

References
==========

.. [SR13] Gábor J. Székely and Maria L. Rizzo. Energy statistics: a class of
           statistics based on distances. Journal of Statistical Planning and
           Inference, 143(8):1249 – 1272, 2013.
           URL:
           http://www.sciencedirect.com/science/article/pii/S0378375813000633,
           doi:10.1016/j.jspi.2013.03.018.
.. [SR14]  Gábor J. Székely and Maria L. Rizzo. Partial distance correlation
           with methods for dissimilarities. The Annals of Statistics,
           42(6):2382–2412, 12 2014.
           doi:10.1214/14-AOS1255.
.. [SRB07] Gábor J. Székely, Maria L. Rizzo, and Nail K. Bakirov. Measuring and
           testing dependence by correlation of distances. The Annals of
           Statistics, 35(6):2769–2794, 12 2007.
           doi:10.1214/009053607000000505.

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
    
.. |repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
   
.. |versions| image:: https://img.shields.io/pypi/pyversions/dcor
   :alt: PyPI - Python Version
   :scale: 100%
    
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
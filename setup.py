# encoding: utf-8

"""dcor: distance correlation and distance covariance in Python.

Distance covariance and distance correlation are
dependency measures between random vectors introduced in [SRB07]_.

This package offers functions for calculating several statistics
related with distance covariance and distance correlation, including
biased and unbiased estimators of both dependency measures.

.. rubric:: References

.. [SRB07] Gábor J. Székely, Maria L. Rizzo, and Nail K. Bakirov. Measuring and
           testing dependence by correlation of distances. The Annals of
           Statistics, 35(6):2769–2794, 12 2007.
           doi:10.1214/009053607000000505.
"""

import os
import sys

from setuptools import setup, find_packages


needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

DOCLINES = (__doc__ or '').split("\n")

with open(os.path.join(os.path.dirname(__file__),
                       'VERSION'), 'r') as version_file:
    version = version_file.read().strip()

setup(name='dcor',
      version=version,
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      url='https://dcor.readthedocs.io',
      author='Carlos Ramos Carreño',
      author_email='vnmabus@gmail.com',
      include_package_data=True,
      platforms=['any'],
      license='MIT',
      packages=find_packages(),
      python_requires='>=2.7, >=3.5, <4',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=['numpy',
                        'numba',
                        'scipy',
                        'setuptools'],
      setup_requires=pytest_runner,
      tests_require=['pytest'],
      test_suite='dcor.tests',
      zip_safe=False)

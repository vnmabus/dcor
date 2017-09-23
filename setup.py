# encoding: utf-8

'''
Created on 13 sept. 2017

@author: Carlos Ramos Carreño
'''

import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__),
                       'VERSION'), 'r') as version_file:
    version = version_file.read().strip()

setup(name='dcor',
      version=version,
      description='Distance correlation and distance covariance in Python',
      url='https://github.com/vnmabus/dcor',
      author='Carlos Ramos Carreño',
      author_email='vnmabus@gmail.com',
      include_package_data=True,
      license='MIT',
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      test_suite='dcor.tests',
      zip_safe=False)

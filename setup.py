# encoding: utf-8

'''
Created on 13 sept. 2017

@author: Carlos Ramos Carreño
'''

from setuptools import setup

setup(name='dcor',
      version='0.1',
      description='Distance correlation and distance covariance in Python',
      url='https://github.com/vnmabus/dcor',
      author='Carlos Ramos Carreño',
      author_email='vnmabus@gmail.com',
      license='MIT',
      packages=['dcor'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      test_suite='dcor.tests',
      zip_safe=False)

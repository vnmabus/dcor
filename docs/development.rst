Developer information
=====================

This page contains the necessary information to contribute to this package.

Contributing guide
------------------

The guide for contributors is in the repository:
https://github.com/vnmabus/dcor/blob/develop/CONTRIBUTING.md.

Building the documentation
--------------------------

In order to build the documentation, please first install the necessary
packages listed in ``readthedocs-requirements.txt``:

.. code-block:: bash

	pip install -r readthedocs-requirements.txt

The documentation and its configuration is stored in the ``docs`` subfolder.
In order to build the documentation using Sphinx, execute the following
commands in that folder:

.. code-block:: bash

	make clean
	make html
	
Code of conduct
---------------

The code of conduct for contributors is in the repository:
https://github.com/vnmabus/dcor/blob/develop/CODE_OF_CONDUCT.md
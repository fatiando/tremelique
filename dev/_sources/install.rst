.. _install:

Installing
==========

There are different ways to install Tremelique:

.. tab-set::

    .. tab-item:: pip

        Using the `pip <https://pypi.org/project/pip/>`__ package manager:

        .. code:: bash

            python -m pip install tremelique

    .. tab-item:: conda/mamba

        Using the `conda package manager <https://conda.io/>`__ (or ``mamba``)
        that comes with the Anaconda/Miniconda distribution:

        .. code:: bash

            conda install tremelique --channel conda-forge

    .. tab-item:: Development version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/fatiando/tremelique

.. note::

    The commands above should be executed in a terminal. On Windows, use the
    ``cmd.exe`` or the "Anaconda Prompt"/"Miniforge Prompt" app if you're using
    Anaconda/Miniforge.

.. tip::

   We recommend using the
   `Miniforge distribution <https://conda-forge.org/download/>`__
   to ensure that you have the ``conda`` package manager available.
   Installing Miniforge does not require administrative rights to your computer
   and doesn't interfere with any other Python installations in your system.
   It's also much smaller than the Anaconda distribution and is less likely to
   break when installing new software.

Which Python?
-------------

You'll need **Python 3.9 or greater**.
See :ref:`python-versions` if you require support for older versions.


Dependencies
------------

The required dependencies should be installed automatically when you install
Bordado using ``conda`` or ``pip``. Optional dependencies have to be
installed manually.

.. note::

    See :ref:`dependency-versions` for the our policy of oldest supported
    versions of each dependency.

Required:

* `numpy <http://www.numpy.org/>`__
* `numba <https://numba.pydata.org/>`__
* `matplotlib <https://matplotlib.org/>`__

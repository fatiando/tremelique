# Copyright (c) 2025 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
These are the functions and classes that make up the Tremelique API.
"""

from ._acoustic import Acoustic
from ._version import __version__
from ._wavelets import RickerWavelet

# Append a leading "v" to the generated version by setuptools_scm
__version__ = f"v{__version__}"

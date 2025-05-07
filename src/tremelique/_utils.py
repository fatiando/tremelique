# Copyright (c) 2025 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Misc utilities that don't fit anywhere else.
"""

import base64
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numba
import numpy as np
from IPython.display import Video


def anim_to_html(anim, fps=6, dpi=30):
    """
    Convert a matplotlib animation to a video embedded in an HTML <video> tag.

    Uses avconv (default) or ffmpeg. Both need to be installed on the system
    for this to work.

    Returns an IPython.display.HTML object for embedding in the notebook.

    Adapted from `the yt project docs
    <http://yt-project.org/doc/cookbook/embedded_webm_animation.html>`__.
    """
    plt.close(anim._fig)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temporary_file = pathlib.Path(tmp.name)
    try:
        anim.save(
            temporary_file,
            fps=fps,
            dpi=dpi,
            writer="ffmpeg",
            extra_args=["-vcodec", "libx264"],
        )
        with temporary_file.open(mode="rb") as videofile:
            video = videofile.read()
    finally:
        # Remove the file always, even if an exception occurred.
        temporary_file.unlink()
    return Video(
        data=base64.b64encode(video).decode(),
        embed=True,
        mimetype="video/mp4",
        width=800,
        html_attributes="controls",
    )


@numba.jit(nopython=True)
def apply_damping(array, nx, nz, pad, decay):
    """
    Apply a decay factor to the values of the array in the padding region.

    The decay/damping is meant to slowly kill reflections off of the left,
    right, and bottom of the domain. The decay is exponential and should use
    a decay factor that is small to avoid causing unwanted reflections.
    """
    # Damping on the left
    for i in range(nz):
        for j in range(pad):
            array[i, j] *= np.exp(-((decay * (pad - j)) ** 2))
    # Damping on the right
    for i in range(nz):
        for j in range(nx - pad, nx):
            array[i, j] *= np.exp(-((decay * (j - nx + pad)) ** 2))
    # Damping on the bottom
    for i in range(nz - pad, nz):
        for j in range(nx):
            array[i, j] *= np.exp(-((decay * (i - nz + pad)) ** 2))


@numba.jit(nopython=True)
def xz2ps(ux, uz, p, s, nx, nz, dx, dz):
    """
    Convert ux and uz to P and S waves.
    """
    tmpx = dx * 12.0
    tmpz = dz * 12.0
    for i in range(2, nz - 2):
        for j in range(2, nx - 2):
            p[i, j] = (
                -uz[i + 2, j] + 8 * uz[i + 1, j] - 8 * uz[i - 1, j] + uz[i - 2, j]
            ) / tmpz + (
                -ux[i, j + 2] + 8 * ux[i, j + 1] - 8 * ux[i, j - 1] + ux[i, j - 2]
            ) / tmpx
            s[i, j] = (
                -ux[i + 2, j] + 8 * ux[i + 1, j] - 8 * ux[i - 1, j] + ux[i - 2, j]
            ) / tmpz - (
                -uz[i, j + 2] + 8 * uz[i, j + 1] - 8 * uz[i, j - 1] + uz[i, j - 2]
            ) / tmpx
    # Fill in the borders with the same values
    for i in range(nz):
        p[i, nx - 2] = p[i, nx - 3]
        p[i, nx - 1] = p[i, nx - 2]
        p[i, 1] = p[i, 2]
        p[i, 0] = p[i, 1]
        s[i, nx - 2] = s[i, nx - 3]
        s[i, nx - 1] = s[i, nx - 2]
        s[i, 1] = s[i, 2]
        s[i, 0] = s[i, 1]
    for j in range(nx):
        p[nz - 2, j] = p[nz - 3, j]
        p[nz - 1, j] = p[nz - 2, j]
        p[1, j] = p[2, j]
        p[0, j] = p[1, j]
        s[nz - 2, j] = s[nz - 3, j]
        s[nz - 1, j] = s[nz - 2, j]
        s[1, j] = s[2, j]
        s[0, j] = s[1, j]


def lame_lamb(pvel, svel, dens):
    r"""
    Calculate the Lame parameter :math:`\lambda`.

    It's related to P and S wave velocities (:math:`\alpha` and :math:`\beta`)
    and the density (:math:`\rho`).

    .. math::

        \lambda = \alpha^2 \rho - 2\beta^2 \rho

    Parameters
    ----------
    * pvel : float or array
        The P wave velocity.
    * svel : float or array
        The S wave velocity.
    * dens : float or array
        The density.

    Returns
    -------
    * lambda : float or array
        The Lame parameter.

    Examples
    --------
    Scalars can be used:

    >>> print(lame_lamb(2000, 1000, 2700))
    5400000000

    Or numpy arrays:

    >>> import numpy as np
    >>> pv = np.array([2000., 3000.])
    >>> sv = np.array([1000., 1700.])
    >>> dens = np.array([2700., 3100.])
    >>> print(lame_lamb(pv, sv, dens))
    [5.400e+09 9.982e+09]
    """
    lamb = dens * pvel**2 - 2 * dens * svel**2
    return lamb


def lame_mu(svel, dens):
    r"""
    Calculate the Lame parameter :math:`\mu`.

    It's related to S wave velocity (:math:`\beta`) and the density
    (:math:`\rho`).

    .. math::

        \mu = \beta^2 \rho

    Parameters
    ----------
    * svel : float or array
        The S wave velocity.
    * dens : float or array
        The density.

    Returns
    -------
    * mu : float or array
        The Lame parameter.

    Examples
    --------
    We can use scalars:

    >>> print(lame_mu(1000, 2700))
    2700000000

    Or numpy arrays:

    >>> import numpy as np
    >>> sv = np.array([1000., 1700.])
    >>> dens = np.array([2700., 3100.])
    >>> print(lame_mu(sv, dens))
    [2.700e+09 8.959e+09]
    """
    mu = dens * svel**2
    return mu

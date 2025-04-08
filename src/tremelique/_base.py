# Copyright (c) 2025 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Base class for 2D finite-difference simulations.
"""

import abc
import base64
import os
import tempfile
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import rich.progress
from IPython.core.pylabtools import print_figure
from IPython.display import Image, Video
from ipywidgets import widgets
from matplotlib import pyplot as plt


def anim_to_html(anim, fps=6, dpi=30):
    """
    Convert a matplotlib animation object to a video embedded in an HTML
    <video> tag.

    Uses avconv (default) or ffmpeg.

    Returns an IPython.display.HTML object for embedding in the notebook.

    Adapted from `the yt project docs
    <http://yt-project.org/doc/cookbook/embedded_webm_animation.html>`__.
    """
    plt.close(anim._fig)
    with NamedTemporaryFile(suffix=".mp4") as f:
        anim.save(
            f.name, fps=fps, dpi=dpi, writer="ffmpeg", extra_args=["-vcodec", "libx264"]
        )
        with open(f.name, "rb") as f:
            video = f.read()
    return Video(
        data=base64.b64encode(video).decode(),
        embed=True,
        mimetype="video/mp4",
        width=800,
        html_attributes="controls",
    )


class BaseSimulation(abc.ABC):
    """
    Base class for 2D simulations.

    Implements the ``run`` method and delegates actual timestepping to the
    abstract ``_timestep`` method.

    Handles creating an HDF5 cache file, plotting snapshots of the simulation,
    printing a progress bar to stderr, and creating an IPython widget to
    explore the snapshots.

    Overloads ``__getitem__``. Indexing the simulation object is like
    indexing the HDF5 cache file. This way you can treat the simulation
    object as a numpy array.

    Attributes:

    * simsize: int
        number of iterations that has been run
    * cachefile: str
        The hdf5 cachefile file path where the simulation is stored
    * shape: tuple
        2D panel numpy.shape without padding

    """

    def __init__(
        self, cachefile, spacing, shape, dt=None, padding=50, taper=0.007, verbose=True
    ):
        if np.size(spacing) == 1:  # equal space increment in x and z
            self.dx, self.dz = spacing, spacing
        else:
            self.dx, self.dz = spacing
        self.shape = shape  # 2D panel shape without padding
        self.verbose = verbose
        self.sources = []
        # simsize stores the total size of this simulation
        # after some or many runs
        self.simsize = 0  # simulation number of interations already ran
        # it is the `run` iteration time step indexer
        self.it = -1  # iteration time step index (where we are)
        # `it` and `simsize` together allows indefinite simulation runs
        if cachefile is None:
            cachefile = self._create_tmp_cache()
        self.cachefile = cachefile
        self.padding = padding  # padding region size
        self.taper = taper
        self.dt = dt

    def _create_tmp_cache(self):
        """
        Creates the temporary file used
        to store data in hdf5 format

        Returns:

        * _create_tmp_cache: str
            returns the name of the file created

        """
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".h5",
            prefix="{}-".format(self.__class__.__name__),
            dir=os.path.curdir,
            delete=False,
        )
        fname = tmpfile.name
        tmpfile.close()
        return fname

    @abc.abstractmethod
    def from_cache(fname, verbose=True):
        pass

    @abc.abstractmethod
    def _init_panels(self):
        pass

    @abc.abstractmethod
    def _init_cache(self, npanels, chunks=None, compression="lzf", shuffle=True):
        pass

    @abc.abstractmethod
    def _expand_cache(self, npanels):
        pass

    @abc.abstractmethod
    def _cache_panels(self, npanels, tp1, iteration, simul_size):
        pass

    def _get_cache(self, mode="r"):
        """
        Get the cache file as h5py file object

        Parameters:

        * mode: str
            'r' or 'w'
            for reading or writing

        Returns:

        * cache : h5py file object

        """
        return h5py.File(self.cachefile, mode)

    def __getitem__(self, index):
        """
        Get an iteration of the panels object from the hdf5 cache file.
        """
        pass

    @abc.abstractmethod
    def _plot_snapshot(self, frame, **kwargs):
        pass

    def snapshot(self, frame, embed=False, raw=False, ax=None, **kwargs):
        """
        Returns an image (snapshot) of the 2D wavefield simulation
        at a desired iteration number (frame) or just plot it.

        Parameters:

        * frame : int
            The time step iteration number
        * embed : bool
            True to plot it inline
        * raw : bool
            True for raw byte image
        * ax : None or matplotlib Axes
            If not None, will assume this is a matplotlib Axes
            and make the plot on it.

        Returns:

        * image:
            raw byte image if raw=True
            jpeg picture if embed=True
            or None

        """
        if ax is None:
            fig = plt.figure(facecolor="white")
            ax = plt.subplot(111)
        if frame < 0:
            title = self.simsize + frame
        else:
            title = frame
        # plt.sca(ax)
        ax = plt.gca()
        fig = ax.get_figure()
        plt.title("Time frame {:d}".format(title))
        self._plot_snapshot(frame, **kwargs)
        nz, nx = self.shape
        mx, mz = nx * self.dx, nz * self.dz
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mz)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.invert_yaxis()
        # Check the aspect ratio of the plot and adjust figure size to match
        aspect = min(self.shape) / max(self.shape)
        try:
            aspect /= ax.get_aspect()
        except TypeError:
            pass
        if nx > nz:
            width = 10
            height = width * aspect * 0.8
        else:
            height = 8
            width = height * aspect * 1.5
        fig.set_size_inches(width, height)
        plt.tight_layout()
        if raw or embed:
            png = print_figure(fig, dpi=70)
            plt.close(fig)
        if raw:
            return png
        elif embed:
            return Image(png)

    def _repr_png_(self):
        """
        Display one time frame of this simulation
        """
        return self.snapshot(-1, raw=True)

    def explore(self, every=1, **kwargs):
        """
        Interactive visualization of simulation results.

        Allows to move back and forth on simulation frames
        using the IPython widgets feature.

        .. warning::

            Only works when running in an IPython notebook.

        """
        plotargs = kwargs

        def plot(Frame):
            image = Image(self.snapshot(Frame, raw=True, **plotargs))
            return image

        slider = widgets.IntSlider(
            min=0, max=self.it, step=every, value=self.it, description="Frame"
        )
        widget = widgets.interactive(plot, Frame=slider)
        return widget

    @abc.abstractmethod
    def _timestep(self, panels, tm1, t, tp1, iteration):
        pass

    def run(self, iterations):
        """
        Run this simulation given the number of iterations.

        * iterations: int
            number of time step iterations to run
        """
        # Calls the following abstract methods:
        # `_init_cache`, `_expand_cache`, `_init_panels`
        # and `_cache_panels` and  `_time_step`
        nz, nx = self.shape
        dz, dx = self.dz, self.dx
        u = self._init_panels()  # panels must be created first

        # Initialize the cache on the first run
        if self.simsize == 0:
            self._init_cache(iterations)
        else:  # increase cache size by iterations
            self._expand_cache(iterations)

        iterator = range(iterations)
        if self.verbose:
            iterator = rich.progress.track(iterator)

        for iteration in iterator:
            t, tm1 = iteration % 2, (iteration + 1) % 2
            tp1 = tm1
            self.it += 1
            self._timestep(u, tm1, t, tp1, self.it)
            self.simsize += 1
            #  won't this make it slower than it should? I/O
            self._cache_panels(u, tp1, self.it, self.simsize)

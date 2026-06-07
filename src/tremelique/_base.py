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
import atexit
import contextlib
import tempfile
import warnings
from pathlib import Path

import h5netcdf
import matplotlib.pyplot as plt
import numpy as np
import rich.progress
from IPython.core.pylabtools import print_figure
from IPython.display import Image
from ipywidgets import widgets


class BaseSimulation(abc.ABC):
    """
    Base class for 2D finite-difference seismic wave simulations.

    Implements the ``run`` method and delegates actual time stepping to the
    abstract ``_timestep`` method.

    Handles creating an HDF5 cache file, plotting snapshots of the simulation,
    printing a progress bar to stderr, and creating an IPython widget to
    explore the snapshots.

    Overloads ``__getitem__``. Indexing the simulation object is like
    indexing the HDF5 cache file. This way you can treat the simulation
    object as a numpy array.

    Attributes
    ----------
    simsize: int
        Number of iterations that has been run.
    cachefile: str
        The hdf5 cachefile file path where the simulation is stored.
    shape: tuple
        2D wavefield shape without padding as (nz, nx).

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
            self._temp_cache = True
        else:
            self._temp_cache = False
        self.cachefile = cachefile
        self.padding = padding  # padding region size
        self.taper = taper
        self.dt = dt

    @atexit.register
    def _delete_tmp_cache(self):
        """
        Clear the temporary cache file when the object is destroyed.
        """
        if self._temp_cache and Path(self.cachefile).exists():
            try:
                Path(self.cachefile).unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                msg_temp_file = (
                    f"Error! Failed to delete temporary file {self.cachefile}: {e}."
                )
                warnings.warn(msg_temp_file, RuntimeWarning, stacklevel=2)

    def _create_tmp_cache(self):
        """
        Create the temporary file used to store data in h5netcdf (.h5) format.

        Returns
        -------
        path : str
            The absolute path of the temporary file created.

        """
        directory = Path.cwd()

        if self is not None:
            if self is not None:
                prefix = self.__class__.__name__ + "-"
            else:
                prefix = "Simulation-"

        tmp_args = {
            "suffix": ".h5",
            "prefix": prefix,
            "dir": directory,
            "delete": False,
        }
        with tempfile.NamedTemporaryFile(**tmp_args) as tmpfile:
            path = tmpfile.name
        return path

    @staticmethod
    @abc.abstractmethod
    def from_cache(fname, verbose=True):
        pass

    @abc.abstractmethod
    def _init_wavefield(self):
        pass

    @abc.abstractmethod
    def _init_cache(self, num_steps):
        pass

    @abc.abstractmethod
    def _expand_cache(self, num_steps):
        pass

    @abc.abstractmethod
    def _cache_wavefield(self, u, tp1, iteration, simul_size):
        pass

    def _get_cache(self, mode="r"):
        """
        Get the cache file as h5netcdf file object.

        Parameters
        ----------
        mode: str
            'r' or 'w' for reading or writing.

        Returns
        -------
        cache : :class:`h5py.File`
            The open HDF5 file object for the cache.
        """
        if not Path(self.cachefile).resolve().parent.exists():
            Path(self.cachefile).resolve().parent.mkdir(parents=True, exist_ok=True)

        file = h5netcdf.File(self.cachefile, mode)
        try:
            yield file
        finally:
            with contextlib.suppress(Exception):
                file.close()

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        Get an iteration of the pressure field object from the hdf5 cache file.
        """

    @abc.abstractmethod
    def _plot_snapshot(self, frame, **kwargs):
        """
        Plot a given frame as an image.
        """

    def snapshot(self, frame, embed=False, raw=False, ax=None, **kwargs):
        """
        Create an image of the 2D wavefield simulation at given frame.

        The image can be returned as a raw PNG or embedded into an
        :class:`IPython.display.Image`.

        Parameters
        ----------
        frame : int
            The time step iteration number.
        embed : bool
            True to plot it inline.
        raw : bool
            True for raw byte image.
        ax : None or matplotlib Axes
            If not None, will assume this is a matplotlib Axes and make the
            plot on it.

        Returns
        -------
        image : bytes-array or :class:`IPython.display.Image` or None
            Raw byte image if ``raw=True`` PNG picture if ``embed=True`` or
            None.
        """
        if ax is None:
            fig = plt.figure(facecolor="white")
            ax = plt.subplot(111)

        title_frame = self.simsize + frame if frame < 0 else frame

        dt = getattr(self, "dt", 0) or 0
        t = title_frame * dt

        ax = plt.gca()
        fig = ax.get_figure()

        ax.set_title(f"Iteration: {title_frame} | t = {t:.3f} s")

        self._plot_snapshot(frame, **kwargs)

        nz, nx = self.shape
        dx = getattr(self, "dx", 1.0)
        dz = getattr(self, "dz", 1.0)

        max_x, max_z = nx * dx, nz * dz

        ax.set_xlim(0, max_x)
        ax.set_ylim(-max_z, 0)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")

        aspect = min(self.shape) / max(self.shape)

        with contextlib.suppress(TypeError):
            aspect /= ax.get_aspect()

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
        if embed:
            return Image(png)
        return None

    def _repr_png_(self):
        """
        Display one time frame of this simulation.
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
    def _timestep(self, u, tm1, t, tp1, iteration):
        """
        Run the simulation forward one step in time.
        """

    def run(self, iterations):
        """
        Run this simulation given the number of iterations.

        Parameters
        ----------
        iterations : int
            Number of time step iterations to run.
        """
        # Calls the following abstract methods: `_init_cache`, `_expand_cache`,
        # `_init_wavefield` and `_cache_wavefield` and  `_time_step`. All of them
        # must be implemented in the child classes.

        u = self._init_wavefield()  # pressure must be created first

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
            self._cache_wavefield(u, tp1, self.it, self.simsize)

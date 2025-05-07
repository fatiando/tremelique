# Copyright (c) 2025 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Class for acoustic wave simulation.
"""

import pickle

import h5py
import numba
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy import sqrt

from ._base import BaseSimulation
from ._utils import anim_to_html, apply_damping


class Acoustic(BaseSimulation):
    """
    Simulate the propagation of acoustic waves in 2D.

    Solves the pressure wave equation using the equivalent staggered grid
    finite-difference method of [DiBartolo2012]_. Uses a damping scheme to
    suppress reflections from the left, right, and bottom boundaries. The top
    boundary has a free-surface boundary condition.
    """

    def __init__(
        self,
        velocity,
        density,
        spacing,
        cachefile=None,
        dt=None,
        padding=50,
        taper=0.005,
        verbose=True,
    ):
        super().__init__(
            cachefile, spacing, velocity.shape, dt, padding, taper, verbose
        )
        self.density = density
        self.velocity = velocity
        if self.dt is None:
            self.dt = self.maxdt()

    def __getitem__(self, index):
        """
        Get an iteration of the panels object from the hdf5 cache file.

        Parameters
        ----------
        * index: index or slice
            Index for slicing hdf5 data set.

        Returns
        -------
        * panels : array
            Numpy array with the 2D panels at the given index.

        """
        with self._get_cache() as f:
            data = f["panels"][index]
        return data

    @staticmethod
    def from_cache(fname, verbose=True):
        """
        Create a simulation object from a pre-existing HDF5 file.

        Parameters
        ----------
        * fname: str
            HDF5 file path containing a previous simulation stored
        * verbose: bool
            Progress status shown or not

        Returns
        -------
        * simulation : :class:`tremelique.Acoustic`
            The simulation class instance.
        """
        with h5py.File(fname, "r") as f:
            vel = f["velocity"]
            dens = f["density"]
            panels = f["panels"]
            dx = panels.attrs["dx"]
            dz = panels.attrs["dz"]
            dt = panels.attrs["dt"]
            padding = panels.attrs["padding"]
            taper = panels.attrs["taper"]
            sim = Acoustic(
                vel[:],
                dens[:],
                (dx, dz),
                dt=dt,
                padding=padding,
                taper=taper,
                cachefile=fname,
            )
            sim.simsize = panels.attrs["simsize"]
            sim.it = panels.attrs["iteration"]
            sim.sources = pickle.loads(f["sources"].value.tostring())
        sim.set_verbose(verbose)
        return sim

    def _init_cache(self, npanels, chunks=None, compression="lzf", shuffle=True):
        """
        Initialize the hdf5 cache file with this simulation parameters.

        Parameters
        ----------
        * npanels: int
            number of 2D panels needed for this simulation run
        *  chunks : HDF5 data set option
            (Tuple) Chunk shape, or True to enable auto-chunking.
        * compression: HDF5 data set option
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        * shuffle: (bool) HDF5 data set option
            (T/F) Enable shuffle filter.
        """
        nz, nx = self.shape
        if chunks is None:
            chunks = (1, nz // 10, nx // 10)
        with self._get_cache(mode="w") as f:  # create HDF5 data sets
            nz, nx = self.shape
            dset = f.create_dataset(
                "panels",
                (npanels, nz, nx),
                maxshape=(None, nz, nx),
                chunks=chunks,
                compression=compression,
                shuffle=shuffle,
                dtype=np.float32,
            )
            dset.attrs["shape"] = self.shape
            dset.attrs["simsize"] = self.simsize
            dset.attrs["iteration"] = self.it
            dset.attrs["dx"] = self.dx
            dset.attrs["dz"] = self.dz
            dset.attrs["dt"] = self.dt
            dset.attrs["padding"] = self.padding
            dset.attrs["taper"] = self.taper
            f.create_dataset("velocity", data=self.velocity)
            f.create_dataset("density", data=self.density)
            f.create_dataset("sources", data=np.void(pickle.dumps(self.sources)))

    def _expand_cache(self, npanels):
        """
        Expand the hdf5 cache for more iterations.

        Parameters
        ----------
        *  npanels: int
            number of 2D panels needed for this simulation run
        """
        with self._get_cache(mode="a") as f:
            cache = f["panels"]
            cache.resize(self.simsize + npanels, axis=0)

    def _cache_panels(self, u, tp1, iteration, simul_size):
        """
        Save the last calculated panels to the hdf5 cache.

        Parameters
        ----------
        * panels : tuple or variable
            tuple or variable containing all 2D panels needed for
            this simulation
        * tp1 : int
            panel time index
        * iteration:
            iteration number
        * simul_size:
            number of iterations that has been run
        """
        # Save the panel to disk
        with self._get_cache(mode="a") as f:
            cache = f["panels"]
            cache[simul_size - 1] = u[tp1]
            cache.attrs["simsize"] = simul_size
            # I need to update the attribute with this iteration number
            # so that simulation runs properly after reloaded from file
            cache.attrs["iteration"] = iteration

    def _init_panels(self):
        """
        Start the simulation panels used for finite difference solution.

        Keep consistency of simulations if loaded from file.

        Returns
        -------
        * return:
            panels object

        """
        # If this is the first run, start with zeros, else, get the last two
        # panels from the cache so that the simulation can be resumed
        if self.simsize == 0:
            nz, nx = self.shape
            u = np.zeros((2, nz, nx), dtype=np.float32)
        else:
            with self._get_cache() as f:
                cache = f["panels"]
                u = cache[self.simsize - 2 : self.simsize][::-1]
        return u

    def add_point_source(self, position, wavelet):
        """
        Add a point source of energy to this simulation.

        Parameters
        ----------
        * position : tuple
            The (x, z) coordinates of the source
        * source : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an
            example source)
        """
        self.sources.append([position, wavelet])

    def _timestep(self, u, tm1, t, tp1, iteration):
        """
        Take a step in the finite-difference simulation.
        """
        nz, nx = self.shape
        # Take the step
        timestep_esg(
            u[tp1],
            u[t],
            u[tm1],
            3,
            nx - 3,
            3,
            nz - 3,
            self.dt,
            self.dx,
            self.dz,
            self.velocity,
            self.density,
        )
        # Apply boundary conditions and damping
        apply_damping(u[t], nx, nz, self.padding, self.taper)
        nonreflexive_bc(
            u[tp1], u[t], nx, nz, self.dt, self.dx, self.dz, self.velocity, self.density
        )
        apply_damping(u[tp1], nx, nz, self.padding, self.taper)
        # Update the wave sources
        for pos, src in self.sources:
            i, j = pos
            scale = -self.density[i, j] * (self.velocity[i, j] * self.dt) ** 2
            u[tp1, i, j] += scale * src(iteration * self.dt)

    def _plot_snapshot(self, frame, **kwargs):
        """
        Plot a given frame as an image.
        """
        with h5py.File(self.cachefile) as f:
            data = f["panels"][frame]
        scale = kwargs.pop("cutoff", np.abs(data).max())
        nz, nx = self.shape
        dx, dz = nx * self.dx, nz * self.dz
        if "extent" not in kwargs:
            kwargs["extent"] = [0, dx, dz, 0]
        if "cmap" not in kwargs:
            kwargs["cmap"] = plt.cm.seismic
        plt.imshow(data, vmin=-scale, vmax=scale, **kwargs)
        plt.colorbar(pad=0, aspect=30).set_label("Pressure")

    def animate(
        self,
        every=1,
        cutoff=None,
        ax=None,
        cmap=plt.cm.seismic,
        embed=False,
        fps=10,
        dpi=70,
        **kwargs,
    ):
        """
        Create a 2D animation from the current state of the simulation.
        """
        if ax is None:
            plt.figure(facecolor="white")
            ax = plt.subplot(111)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        fig = ax.get_figure()
        nz, nx = self.shape
        # Check the aspect ratio of the plot and adjust figure size to match
        aspect = min(self.shape) / max(self.shape)
        if nx > nz:
            width = 10
            height = width * aspect * 0.8
        else:
            height = 10
            width = height * aspect * 1.5
        fig.set_size_inches(width, height)
        # Separate the arguments for imshow
        imshow_args = {"cmap": cmap}
        if cutoff is not None:
            imshow_args["vmin"] = -cutoff
            imshow_args["vmax"] = cutoff
        wavefield = ax.imshow(np.zeros(self.shape), **imshow_args)
        fig.colorbar(wavefield, pad=0, aspect=30).set_label("Pressure")
        ax.set_title("iteration: 0")
        frames = self.simsize // every

        def plot(i):
            ax.set_title(f"iteration: {i * every:d}")
            u = self[i * every]
            wavefield.set_array(u)
            return wavefield

        anim = animation.FuncAnimation(fig, plot, frames=frames, **kwargs)
        if embed:
            return anim_to_html(anim, fps=fps, dpi=dpi)
        plt.show()
        return anim

    def maxdt(self):
        """
        Calculate the maximum allowed time interval that is safe to use.
        """
        nz, nx = self.shape
        x1, x2, z1, z2 = [0, nx * self.dx, 0, nz * self.dz]
        spacing = min([(x2 - x1) / (nx - 1), (z2 - z1) / (nz - 1)])
        # Be conservative and use 0.6x the recommended value
        return 0.6 * 0.606 * spacing / self.velocity.max()


@numba.jit(nopython=True)
def timestep_esg(u_tp1, u_t, u_tm1, x1, x2, z1, z2, dt, dx, dz, vel, dens):
    """
    Perform a single time step in the finite-difference solution.

    Updates the pressure (u) from the pressure at the previous 2 times using
    the equivalent staggered grid method [DiBartolo2012]_.
    """
    dt2 = dt**2
    dx2 = dx**2
    dz2 = dz**2
    for i in range(z1, z2):
        for j in range(x1, x2):
            zderiv = (1.125 / dz2) * (
                0.5
                * (1 / dens[i + 1, j] + 1 / dens[i, j])
                * (
                    1.125 * (u_t[i + 1, j] - u_t[i, j])
                    - (u_t[i + 2, j] - u_t[i - 1, j]) / 24.0
                )
                - 0.5
                * (1 / dens[i, j] + 1 / dens[i - 1, j])
                * (
                    1.125 * (u_t[i, j] - u_t[i - 1, j])
                    - (u_t[i + 1, j] - u_t[i - 2, j]) / 24.0
                )
            ) - (1 / (24 * dz2)) * (
                0.5
                * (1 / dens[i + 2, j] + 1 / dens[i + 1, j])
                * (
                    1.125 * (u_t[i + 2, j] - u_t[i + 1, j])
                    - (u_t[i + 3, j] - u_t[i, j]) / 24.0
                )
                - 0.5
                * (1 / dens[i - 1, j] + 1 / dens[i - 2, j])
                * (
                    1.125 * (u_t[i - 1, j] - u_t[i - 2, j])
                    - (u_t[i, j] - u_t[i - 3, j]) / 24.0
                )
            )
            xderiv = (1.125 / dx2) * (
                0.5
                * (1 / dens[i, j + 1] + 1 / dens[i, j])
                * (
                    1.125 * (u_t[i, j + 1] - u_t[i, j])
                    - (u_t[i, j + 2] - u_t[i, j - 1]) / 24.0
                )
                - 0.5
                * (1 / dens[i, j] + 1 / dens[i, j - 1])
                * (
                    1.125 * (u_t[i, j] - u_t[i, j - 1])
                    - (u_t[i, j + 1] - u_t[i, j - 2]) / 24.0
                )
            ) - (1.0 / (24.0 * dx2)) * (
                0.5
                * (1 / dens[i, j + 2] + 1 / dens[i, j + 1])
                * (
                    1.125 * (u_t[i, j + 2] - u_t[i, j + 1])
                    - (u_t[i, j + 3] - u_t[i, j]) / 24.0
                )
                - 0.5
                * (1 / dens[i, j - 1] + 1 / dens[i, j - 2])
                * (
                    1.125 * (u_t[i, j - 1] - u_t[i, j - 2])
                    - (u_t[i, j] - u_t[i, j - 3]) / 24.0
                )
            )
            u_tp1[i, j] = (
                2 * u_t[i, j]
                - u_tm1[i, j]
                + dt2 * dens[i, j] * vel[i, j] ** 2 * (xderiv + zderiv)
            )


@numba.jit(nopython=True)
def nonreflexive_bc(u_tp1, u_t, nx, nz, dt, dx, dz, mu, dens):
    """
    Apply nonreflexive boundary contitions to elastic SH waves.
    """
    # Left
    for i in range(nz):
        for j in range(3):
            u_tp1[i, j] = (
                u_t[i, j]
                + dt * sqrt(mu[i, j] / dens[i, j]) * (u_t[i, j + 1] - u_t[i, j]) / dx
            )
    # Right
    for i in range(nz):
        for j in range(nx - 3, nx):
            u_tp1[i, j] = (
                u_t[i, j]
                - dt * sqrt(mu[i, j] / dens[i, j]) * (u_t[i, j] - u_t[i, j - 1]) / dx
            )
    # Bottom
    for i in range(nz - 3, nz):
        for j in range(nx):
            u_tp1[i, j] = (
                u_t[i, j]
                - dt * sqrt(mu[i, j] / dens[i, j]) * (u_t[i, j] - u_t[i - 1, j]) / dz
            )
    # Top
    for j in range(nx):
        u_tp1[2, j] = u_tp1[3, j]
        u_tp1[1, j] = u_tp1[2, j]
        u_tp1[0, j] = u_tp1[1, j]


def scalar_maxdt(area, shape, maxvel):
    r"""
    Calculate the maximum time step that can be used safely.

    This is derived for a simulation with 4th order space and 1st order time.

    References
    ----------
    Alford R.M., Kelly K.R., Boore D.M. (1974) Accuracy of finite-difference
    modeling of the acoustic wave equation Geophysics, 39 (6), P. 834-842

    Chen, Jing-Bo (2011) A stability formula for Lax-Wendroff methods
    with fourth-order in time and general-order in space for
    the scalar wave equation Geophysics, v. 76, p. T37-T42

    Convergence

    .. math::

         \Delta t \leq \frac{2 \Delta s}{ V \sqrt{\sum_{a=-N}^{N} (|w_a^1| +
         |w_a^2|)}}
         = \frac{ \Delta s \sqrt{3}}{ V_{max} \sqrt{8}}

    Where :math:`w_a` are the centered differences weights.

    Parameters
    ----------
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
        at zmin, the deepest at zmax.
    * shape : (nz, nx)
        The number of nodes in the finite difference grid
    * maxvel : float
        The maximum velocity in the medium

    Returns
    -------
    * maxdt : float
        The maximum time step
    """
    x1, x2, z1, z2 = area
    nz, nx = shape
    spacing = min([(x2 - x1) / (nx - 1), (z2 - z1) / (nz - 1)])
    factor = np.sqrt(3.0 / 8.0)
    factor -= factor / 100.0  # 1% smaller to guarantee criteria
    # the closer to stability criteria the better the convergence
    return factor * spacing / maxvel

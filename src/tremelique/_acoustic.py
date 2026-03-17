# Copyright (c) 2025 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Class for acoustic wave simulation.
"""

import base64
import os
import pickle

import h5netcdf
import numba
import numpy as np
import xarray as xr
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
        model,
        cachefile=None,
        dt=None,
        padding=50,
        taper=0.005,
        verbose=True,
    ):
        self.model = model
        self.velocity = model["velocity"].values
        self.density = model["density"].values

        dx = model.attrs.get("dx", 1.0)  # se dx não existir dx = 1.0
        dz = model.attrs.get("dz", 1.0)  # se dz não existir dz = 1.0
        spacing = (dx, dz)

        super().__init__(
            cachefile, spacing, self.velocity.shape, dt, padding, taper, verbose
        )

        if self.dt is None:
            self.dt = self.maxdt()

    @property
    def wavefield(self):
        return xr.open_dataset(
            self.cachefile, engine="h5netcdf", phony_dims="sort"
        ).panels

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
        return self.wavefield["panels"].isel(time=index).values

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

        ds = xr.open_dataset(fname, engine="h5netcdf")

        velocity = ds["velocity"].values.astype(np.float32)
        density = ds["density"].values.astype(np.float32)

        dx = ds.attrs["dx"]
        dz = ds.attrs["dz"]
        dt = ds.attrs["dt"]
        padding = ds.attrs["padding"]
        taper = ds.attrs["taper"]

        z = ds["z"].values
        x = ds["x"].values

        model = xr.Dataset(
            data_vars=dict(
                velocity=(("z", "x"), velocity), density=(("z", "x"), density)
            ),
            coords=dict(z=z, x=x),
            attrs=dict(dx=dx, dz=dz, spacing=(dx, dz), shape=velocity.shape),
        )

        sim = Acoustic(
            model=model,
            cachefile=fname,
            dt=dt,
            padding=padding,
            taper=taper,
            verbose=verbose,
        )

        sim.simsize = int(ds.attrs["simsize"])
        sim.it = int(ds.attrs["iteration"])

        if "sources" in ds:
            sim.sources = pickle.loads(base64.b64decode(ds["sources"].values.item()))
        else:
            sim.sources = []

        ds.close()
        return sim

    def _init_cache(self, npanels):
        """
        Initialize the h5netcdf cache file with this simulation parameters.

        Parameters
        ----------
        * npanels: int
            number of 2D panels needed for this simulation run
        """

        if os.path.exists(self.cachefile):
            try:
                os.remove(self.cachefile)
            except PermissionError:
                raise RuntimeError(
                    "Cache file is already open"
                    "Do not acess sim wavefield before sim.run()"
                )

        nz, nx = self.shape

        # coordenadas fisicas
        time = np.arange(npanels, dtype=np.int32) * self.dt
        z = self.model.z.values.astype(np.float32)
        x = self.model.x.values.astype(np.float32)

        # criação do Dataset completo

        ds = xr.Dataset(
            data_vars=dict(
                panels=(
                    ("time", "z", "x"),
                    np.zeros((npanels, nz, nx), dtype=np.float32),
                ),
                velocity=(("z", "x"), self.velocity.astype(np.float32)),
                density=(("z", "x"), self.density.astype(np.float32)),
                sources=(
                    (),
                    base64.b64encode(pickle.dumps(self.sources)).decode("ascii"),
                ),
            ),
            coords=dict(time=time, z=z, x=x),
            attrs=dict(
                dx=self.dx,
                dz=self.dz,
                dt=self.dt,
                padding=self.padding,
                taper=self.taper,
                simsize=0,
                iteration=self.it,
                shape=self.shape,
            ),
        )
        # unidades físicas (CF-like)
        ds["time"].attrs["units"] = "s"
        ds["x"].attrs["units"] = "m"
        ds["x"].attrs["long_name"] = "horizontal"
        ds["z"].attrs["units"] = "m"
        ds["z"].attrs["long_name"] = "vertical"
        ds["panels"].attrs["units"] = "Pa"
        ds["panels"].attrs["long_name"] = "pressure"

        ds.to_netcdf(
            self.cachefile, engine="h5netcdf", mode="w", unlimited_dims=["time"]
        )

    def _expand_cache(self, npanels):
        """
        Expand the hdf5 cache for more iterations.

        Parameters
        ----------
        *  npanels: int
            number of 2D panels needed for this simulation run
        """
        # with self._get_cache(mode="a") as f:
        #    cache = f["panels"]
        #    cache.resize(self.simsize + npanels, axis=0)
        with h5netcdf.File(self.cachefile, mode="a") as f:
            current_size = f.dimensions["time"].size  # tamanho atual da dimensão tempo
            new_size = current_size + npanels  # novo tamanho

            f.resize_dimension("time", new_size)  # expande a dimensão de tempo

            dt = f.attrs.get("dt", self.dt)
            new_times = np.arange(current_size, new_size, dtype=np.int32) * dt
            f.variables["time"][current_size:new_size] = new_times

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
        with h5netcdf.File(self.cachefile, mode="a") as f:
            f.variables["panels"][simul_size - 1, :, :] = u[tp1]
            f.attrs["simsize"] = simul_size
            f.attrs["iteration"] = iteration

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
            ds = xr.open_dataset(self.cachefile, engine="h5netcdf")
            u = (
                ds["panels"]
                .isel(time=slice(self.simsize - 2, self.simsize))
                .values[::-1]
            )
        return u

    def add_point_source(self, position, wavelet):
        """
        Add a point source of energy to this simulation.

        Parameters
        ----------
        * position : tuple
            The (x, z) coordinates physical
        * source : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an
            example source)
        """
        x, z = position
        
        x0, z0 = float(self.model.x.values[0]), float(self.model.z.values[0])

        index_x = int(round(abs((x-x0)/self.dx)))
        index_z = int(round(abs((z-z0)/self.dz)))

        nz, nx = self.shape

        if not (0 <= index_x < nx):
            raise ValueError(
                f"A coordenada X da fonte ({x}m) está fora do limite do model"
            )
        if not (0 <= index_z < nz):
            raise ValueError(
                f"A coordenada z da fonte ({z}m) está fora do limite do model"
            )

        self.sources.append(
            ((index_z, index_x), wavelet)
        )

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

        ds = xr.open_dataset(self.cachefile, engine="h5netcdf", phony_dims="sort")
        if frame < 0:
            frame = ds["panels"].shape[0] + frame

        data = ds["panels"].isel(time=frame).values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)

        if "cutoff" in kwargs:
            scale = kwargs.pop("cutoff")
        else:
            scale = float(np.nanmax(np.abs(data)))
            if scale == 0:
                scale = 1.0

        ds.close()

        nz, nx = self.shape
        width = nx * self.dx
        height = nz * self.dz

        if "extent" not in kwargs:
            kwargs["extent"] = [0, width, -height, 0]  # [xmin, xmax, zmax, zmin]

        if "cmap" not in kwargs:
            kwargs["cmap"] = "seismic"

        ax = plt.gca()
        im = ax.imshow(data, vmin=-scale, vmax=scale, **kwargs)

        if not ax.images or len(ax.images) == 1:
            plt.colorbar(im, ax=ax, pad=0.02, aspect=30).set_label("Pressure (Pa)")

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
        Create a 2D animation equivalent to animate, but using physical axes (meters) and pressure in Pa.
        """
        if ax is None:
            plt.figure(facecolor="white")
            ax = plt.subplot(111)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")

        fig = ax.get_figure()
        nz, nx = self.shape
        dx, dz = self.dx, self.dz

        aspect = min(self.shape) / max(self.shape)
        if nx > nz:
            width = 10
            height = width * aspect * 0.8
        else:
            height = 10
            width = height * aspect * 1.5

        fig.set_size_inches(width, height)

        imshow_args = {"cmap": cmap}
        if cutoff is not None:
            imshow_args["vmin"] = -cutoff
            imshow_args["vmax"] = cutoff

        extent = [0, nx * dx, -(nz * dz), 0]

        wavefield = ax.imshow(
            np.zeros(self.shape, dtype=np.float32), extent=extent, **imshow_args
        )

        fig.colorbar(wavefield, pad=0.02, aspect=30).set_label("Pressure (Pa)")
        frames = self.simsize // every
        ds = xr.open_dataset(self.cachefile, engine="h5netcdf", phony_dims="sort")

        def plot(i):
            it = i * every
            ax.set_title(f"t = {it * self.dt:.3f} s")
            u = ds["panels"][it, :, :].values.astype(np.float32)
            wavefield.set_array(u)
            return wavefield

        anim = animation.FuncAnimation(fig, plot, frames=frames, **kwargs)

        if embed:
            video = anim_to_html(anim, fps=fps, dpi=dpi)
            ds.close()
            return video

        plt.show()
        ds.close()
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

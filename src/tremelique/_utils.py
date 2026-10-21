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

import bordado as bd
import matplotlib.pyplot as plt
import numba
import numpy as np
import xarray as xr
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


def homogeneous_model(region, spacing, properties=None, **kwargs):
    """
    Create a homogeneous model using Xarray.

    Parameters
    ----------
    region : tuple
        Physical boundaries of the model as (xmin, xmax, zmin, zmax).
        The z-axis is positive downwards.
    spacing : float or tuple
        Grid spacing (dx, dz).
    properties : dict, optional
        Dictionary containing physical properties and their values.
        For example: ``{"velocity": 1500, "density": 1000}``.
    **kwargs : dict
        Additional physical properties passed as keyword arguments.

    Returns
    -------
    model : xarray.Dataset
        Dataset containing the coordinates 'z' and 'x' with the specified
        properties.
    """
    if properties is None:
        properties = {}

    properties = {**properties, **kwargs}

    if isinstance(spacing, (int, float)):
        dx = dz = spacing
    else:
        dx, dz = spacing

    coords = bd.grid_coordinates(region=region, spacing=spacing, adjust="region")

    z = coords[1][:, 0]  # Get the first column (depth along z)
    x = coords[0][0, :]  # Get the first row (position along x))
    shape = coords[0].shape

    model_data = {}

    for name, value in properties.items():
        data_array = np.full(shape, value, dtype="float32")
        model_data[name] = xr.DataArray(
            data=data_array, coords={"z": z, "x": x}, dims=("z", "x"), name=name
        )

    model = xr.Dataset(model_data)

    model = model.assign_coords(z=-model.z)

    model["x"].attrs["units"] = "m"
    model["x"].attrs["long_name"] = "horizontal"
    model["z"].attrs["units"] = "m"
    model["z"].attrs["long_name"] = "vertical"
    model["velocity"].attrs["units"] = "m/s"
    model["velocity"].attrs["long_name"] = "wave velocity"
    model["density"].attrs["units"] = "kg/m³"
    model["density"].attrs["long_name"] = "density"

    model.attrs = {
        "dx": dx,
        "dz": dz,
        "spacing": spacing,
        "region": region,
        "shape": shape,
        "n_layers": "1",
        "model_type": "homogeneous_model",
    }

    return model


def add_layer(model, z, z_bottom=None, properties=None, **kwargs):
    """
    Add a horizontal layer to an existing Xarray model.

    Parameters
    ----------
    model : xarray.Dataset
        Stratified model with dimensions ('z', 'x').
    z : float
        Depth where the layer begins.
    z_bottom : float, optional
        Depth where the layer ends. If None, the layer extends to the
        bottom of the model.
    properties : dict, optional
        Dictionary containing physical properties of the layer.
    **kwargs : dict
        Additional physical properties passed as keyword arguments
        (e.g., velocity=2500, density=1500).

    Returns
    -------
    model : xarray.Dataset
        The updated stratified model.
    """
    if properties is None:
        properties = {}
    properties = {**properties, **kwargs}

    if z_bottom is None:
        dz = model.attrs.get("dz", 0)
        z_bottom = model.z.values[-1] + dz

    z_coords = model.z.values
    mask = (z_coords >= z) & (z_coords < z_bottom)
    z_layer = z_coords[mask]

    for prop, value in properties.items():
        model[prop].loc[{"z": z_layer}] = value

    return model


def add_layers(model, layers):
    """
    Add multiple horizontal layers to an existing Xarray model.

    Parameters
    ----------
    model : xarray.Dataset
        Stratified model with dimensions ('z', 'x').
    layers : list of dict
        A list of dictionaries where each dictionary defines a layer.
        Must contain 'z' (start depth) and optionally 'z_bottom' (end depth)
        along with any physical properties.
        Example:
        [
            {"z": 0, "z_bottom": 300, "velocity": 3000},
            {"z": 300, "z_bottom": 1000, "velocity": 4200},
            {"z": 1000, "velocity": 5000}
        ]

    Returns
    -------
    model : xarray.Dataset
        The updated stratified model.
    """
    for layer in layers:
        z = layer.pop("z")
        z_bottom = layer.pop("z_bottom", None)
        add_layer(model, z=z, z_bottom=z_bottom, **layer)

    return model


def layered_model(region, spacing, layers):
    """
    Create a stratified model in a simplified way.

    Parameters
    ----------
    region : tuple
        Physical boundaries of the model as (xmin, xmax, zmin, zmax).
    spacing : float or tuple
        Grid spacing (dx, dz) or a single float for both.
    layers : list of dict
        Definition of each layer's properties.
        Example:
        {
            "z": depth of the layer top (in meters, not index),
            "velocity": value,
            "density": value
        }
        The first layer must start at z = 0.

    Returns
    -------
    model : xarray.Dataset
        Stratified model with dimensions ('z', 'x').
    """
    # Grid spacing
    if isinstance(spacing, (int, float)):
        dx = dz = spacing
    else:
        dx, dz = spacing

    # Physical coordinates
    coords = bd.grid_coordinates(region=region, spacing=spacing, adjust="region")
    z = coords[1][:, 0]
    x = coords[0][0, :]
    shape = coords[0].shape

    # Sort the list of layers by depth (z) from top to bottom
    layers = sorted(layers, key=lambda layer: layer["z"])

    # Physical properties
    properties_names = []
    for key in layers[0]:
        if key != "z":
            properties_names.append(key)

    model_data = {}

    for name in properties_names:
        value = layers[0][name]
        data_array = np.full(shape, value, dtype="float32")
        model_data[name] = xr.DataArray(
            data=data_array, coords={"z": z, "x": x}, dims=("z", "x"), name=name
        )

    # Apply deeper layers
    for i in range(
        1, len(layers)
    ):  # Layer 0 is used to create the base model (np.full)
        z_top = layers[i]["z"]  # Depth where the layer starts

        # Define where the layer ends
        z_bottom = (
            layers[i + 1]["z"] if i < len(layers) - 1 else z[-1] + dz
        )  # if there is a layer below

        mask = (z >= z_top) & (z < z_bottom)  # depth mask

        for name in properties_names:
            model_data[name].loc[{"z": z[mask]}] = layers[i][name]

    model = xr.Dataset(model_data)

    model = model.assign_coords(z=-model.z)

    model["x"].attrs["units"] = "m"
    model["x"].attrs["long_name"] = "horizontal"
    model["z"].attrs["units"] = "m"
    model["z"].attrs["long_name"] = "vertical"
    model["velocity"].attrs["units"] = "m/s"
    model["velocity"].attrs["long_name"] = "wave velocity"
    model["density"].attrs["units"] = "kg/m^3"
    model["density"].attrs["long_name"] = "density"

    model.attrs = {
        "dx": dx,
        "dz": dz,
        "spacing": spacing,
        "region": region,
        "shape": shape,
        "n_layers": len(layers),
        "model_type": "layered_model",
    }

    return model


def add_vertical_fault(model, x_fault, throw):
    """
    Apply a vertical fault to a stratified model.

    Parameters
    ----------
    model : xarray.Dataset
        The original model (e.g., generated by layered_model).
    x_fault : float
        The horizontal position (in meters) where the fault occurs.
    throw : float
        The vertical throw (in meters). Positive values represent
        a normal fault (hanging wall moves down), while negative values
        represent a reverse fault.

    Returns
    -------
    model: xarray.Dataset
        The updated model with the applied fault.
    """
    model = model.copy(deep=True)

    dz = model.attrs.get("dz", 1.0)

    # Operates on indices
    shift_cells = int(np.round(throw / dz))

    faulted_velocity = model["velocity"].values.copy()
    faulted_density = model["density"].values.copy()

    x_coords = model.x.values

    # Create a list of column indices to the right of the fault
    right_of_fault = np.where(x_coords >= x_fault)[0]

    for col_idx in right_of_fault:
        faulted_velocity[:, col_idx] = np.roll(
            faulted_velocity[:, col_idx], shift_cells
        )
        faulted_density[:, col_idx] = np.roll(faulted_density[:, col_idx], shift_cells)

        # If greater than 0, normal fault
        if shift_cells > 0:
            # Overwrite edge artifacts shifted to the top with the topmost valid layer
            faulted_velocity[:shift_cells, col_idx] = faulted_velocity[
                shift_cells, col_idx
            ]
            faulted_density[:shift_cells, col_idx] = faulted_density[
                shift_cells, col_idx
            ]

        # If less than 0, it is a reverse fault
        elif shift_cells < 0:
            # Overwrite edge artifacts shifted to the bottom with the deepest valid layer
            faulted_velocity[shift_cells:, col_idx] = faulted_velocity[
                shift_cells - 1, col_idx
            ]
            faulted_density[shift_cells:, col_idx] = faulted_density[
                shift_cells - 1, col_idx
            ]

    model["velocity"].values = faulted_velocity
    model["density"].values = faulted_density

    if "faults" not in model.attrs:
        model.attrs["faults"] = []
    model.attrs["faults"].append({"type": "vertical", "x": x_fault})

    return model


def add_horizontal_fault(model, z_fault, shift):
    """
    Apply a horizontal fault (detachment) to a stratified model.

    Parameters
    ----------
    model : xarray.Dataset
        The original model (e.g., generated by layered_model).
    z_fault : float
        The depth (in meters) of the fault plane. The block of rock
        above this depth will be displaced.
    shift : float
        The horizontal shift (in meters). Positive values move the block
        to the right, while negative values move it to the left.

    Returns
    -------
    model: xarray.Dataset
        The updated model with the applied fault.
    """
    model = model.copy(deep=True)

    dx = model.attrs.get("dx", 1.0)

    # Operates on indices
    shift_cells = int(np.round(shift / dx))

    faulted_velocity = model["velocity"].values.copy()
    faulted_density = model["density"].values.copy()

    z_coords = model.z.values

    # Using np.abs() because the z coordinates in the model are negative
    above_fault = np.where(np.abs(z_coords) <= z_fault)[0]

    for row_idx in above_fault:
        faulted_velocity[row_idx, :] = np.roll(
            faulted_velocity[row_idx, :], shift_cells
        )
        faulted_density[row_idx, :] = np.roll(faulted_density[row_idx, :], shift_cells)

        # If greater than 0, block moved right
        if shift_cells > 0:
            # Overwrite edge artifacts shifted to the left with the leftmost valid column
            faulted_velocity[row_idx, :shift_cells] = faulted_velocity[
                row_idx, shift_cells
            ]
            faulted_density[row_idx, :shift_cells] = faulted_density[
                row_idx, shift_cells
            ]

        # If less than 0, block moved left
        elif shift_cells < 0:
            # Overwrite edge artifacts shifted to the right with the rightmost valid column
            faulted_velocity[row_idx, shift_cells:] = faulted_velocity[
                row_idx, shift_cells - 1
            ]
            faulted_density[row_idx, shift_cells:] = faulted_density[
                row_idx, shift_cells - 1
            ]

    model["velocity"].values = faulted_velocity
    model["density"].values = faulted_density

    if "faults" not in model.attrs:
        model.attrs["faults"] = []
    model.attrs["faults"].append({"type": "horizontal", "z": z_fault})

    return model


def add_dipping_fault(model, position, dip_degrees, shift):
    """
    Apply a dipping normal fault to a stratified model.

    Parameters
    ----------
    model : xarray.Dataset
        The original model (e.g., generated by layered_model).
    x_fault : float
        The horizontal position (in meters) where the fault reaches the surface.
    dip_degrees : float
        The dip angle of the fault in degrees.
    throw : float
        The vertical throw (in meters). Positive values move the hanging wall down.

    Returns
    -------
    model: xarray.Dataset
        The updated model with the applied fault.
    """
    model = model.copy(deep=True)
    dx = model.attrs.get("dx", 1.0)
    dz = model.attrs.get("dz", 1.0)

    x_coords = model.x.values
    z_coords = model.z.values
    nz, nx = model["velocity"].shape

    if isinstance(position, (list, tuple)):
        if len(position) == 2:
            if dip_degrees >= 0:
                xmin, zmax = position
                xmax = x_coords[-1]
            else:
                xmax, zmax = position
                xmin = x_coords[0]
            zmin = 0.0
        elif len(position) == 4:
            xmin, xmax, zmin, zmax = position
        # Erro fora de região
    else:
        if dip_degrees >= 0:
            xmin = float(position)
            xmax = x_coords[-1]
        else:
            xmin = x_coords[0]
            xmax = float(position)
        zmin = 0.0
        zmax = None

    original_faulted_velocity = model["velocity"].values
    original_faulted_density = model["density"].values
    faulted_velocity = original_faulted_velocity.copy()
    faulted_density = original_faulted_density.copy()

    dip_rad = np.radians(abs(dip_degrees))

    shift_z = int(np.round(shift / dz))  # Operates on indices
    direction = 1 if dip_degrees >= 0 else -1
    shift_x = int(np.round((shift / np.tan(dip_rad)) / dx)) * direction

    zmin_idx = np.argmin(np.abs(z_coords - (-zmin)))
    zmax_idx = np.argmin(np.abs(z_coords - (-zmax))) if zmax is not None else nz

    for col_idx, x in enumerate(x_coords):
        # Assuming the fault dips to the right
        if xmin <= x <= xmax:
            # Calculate depth of the fault plane for this column
            # Using negative values to match the model's z coordinates
            if dip_degrees >= 0:
                z_fault = -zmin - ((x - xmin) * np.tan(dip_rad))
            else:
                z_fault = -zmin - ((xmax - x) * np.tan(dip_rad))

            # Find the nearest Z index for the fault plane
            z_idx = np.argmin(np.abs(z_coords - z_fault))
            limit_idx = min(z_idx, zmax_idx)

            source_col = np.clip(col_idx - shift_x, 0, nx - 1)

            for r in range(zmin_idx, limit_idx):
                source_row = np.clip(r - shift_z, 0, nz - 1)
                faulted_velocity[r, col_idx] = original_faulted_velocity[
                    source_row, source_col
                ]
                faulted_density[r, col_idx] = original_faulted_density[
                    source_row, source_col
                ]

    model["velocity"].values = faulted_velocity
    model["density"].values = faulted_density

    if "faults" not in model.attrs:
        model.attrs["faults"] = []
    model.attrs["faults"].append(
        {
            "type": "dipping",
            "xmin": xmin,
            "xmax": xmax,
            "zmin": zmin,
            "zmax": zmax,
            "dip": dip_degrees,
        }
    )

    return model

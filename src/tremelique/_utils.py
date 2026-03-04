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
import xarray as xr
import bordado as bd



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

def create_homogeneous_model(shape, properties):
    """
    Cria um modelo homenegeo com NumPy
    """
    model_arrays = {}
    for prop_name, prop_value in properties.items():
        model_arrays[prop_name] = np.full(shape, prop_value, dtype=np.float32)
    return model_arrays


def homogeneous_model_xarray(region, spacing, properties=None, **kwargs):
    """
    Cria um modelo homegeneo com Xarray
    region: delimitação fisica (xmin, xmax, zmin, zmax) -> z é positivo quando para baixo
    shape: numero de pontos no grid
    properties: dicionario com as propriedades físicas e seus valores
    ex properties: {"velocity":1500, "density":1000}
    
    Returns: 
    model:: xarray_Dataset (dataset com coordenadas de 'z' e 'x')
    """

    if properties is None:
        properties = {}

    properties = {**properties, **kwargs}

    if isinstance(spacing,(int, float)):
        dx = dz = spacing
    else:
        dx, dz = spacing

    xmin, xmax, zmin, zmax = region

    #cria as coord fisicas
    coords = bd.grid_coordinates(region=region, spacing=spacing, adjust="region")
    
    z = coords[1][:,0] # pega a primeira coluna (profundidade ao longo de z)
    x = coords[0][0,:] # pega a primeira linha (posição ao longo de x)
    shape = coords[0].shape
    nz, nx = shape

    #cria as propriedades
    model_data = {}
    
    for name, value in properties.items():
        data_array = np.full(shape, value, dtype="float32")
        model_data[name] = xr.DataArray(
            data = data_array,
            coords = {"z": z, "x": x},
            dims = ("z","x"),
            name = name
        )
    
    model = xr.Dataset(model_data)

    model.attrs = { 
                    "dx": dx,
                    "dz": dz,
                    "region": region, #[xmin, xmax, zmin, zmax]
                    "shape": shape, #(dx, dz) -----> Conversar com o Leo pra ver oq é melhor (dz, dx) ou (dx, dz)
                    "spacing": spacing, #(nz, nx)
                    "n_layers": 1,
                }
    return model

    
def add_layer_xarray(model, z_top, z_bottom, properties):
    """
    Adiciona (ou sobrescreve) uma camada em um modelo xarray existente.

    Parameters
    ----------

    model: xr.Dataset
        modelo estratificado com dims ('z', 'x')
    z_top: float
        profundidade do topo da camada
    z_bottom: float
        profundidade da base da camada
    properties: dict
        propriedades fisicas das camada
        ex: {"velocity": 2500, "density": 1500}
    
    Returns
    -------
    xr.Dataset
    """
    z = model.z.values
    mask = (z>=z_top) & (z<z_bottom)
    
    for prop, value in properties.items():
        model[prop].loc[dict(z=z[mask])] = value
    
    return model


def add_layer(model, z, z_bottom=None, properties=None, **kwargs):
    """
    Adiciona uma camada horizontal em um modelo xarray
    
    Parameters
    ----------
    model: xr.Dataset
        Modelo com dims ('z', 'x')
    z: float
        Profundidade onde a camada começa
    z_bottom: float
        Profundidade onde a camada termina (se None vai até o final do modelo)
    properties: dict, optional
        Propriedades físicas da camada
    **kwargs:
        Maneira mais "simples" de definir propriedades
        Ex: velocity=2500, density=1500 
    
    Returns
    -------
    xr.Dataset
        Modelo estratificado ('z', 'x')
    """
    if properties is None:
        properties={}
    properties = {**properties, **kwargs}

    if z_bottom is None:
        dz = model.attrs.get("dz",0)
        z_bottom = model.z.values[-1] + dz
        
    z_coords = model.z.values
    mask = (z_coords>=z) & (z_coords<z_bottom)
    z_layer = z_coords[mask]

    for prop, value in properties.items():
        model[prop].loc[dict(z=z_layer)] = value

    return model


def add_layers(model, layers):
    """
    Adiciona múltiplas camadas a um modelo xarray existente com base no add_layer
    
    Parameters
    ----    
    layers: list or dict
    [
        {"z": 0, "z_bottom":300, "velocity":3000}
        {"z": 300, "z_bottom":1000, "velocity":4200}
        {"z": 1000, "velocity":5000}
    ]

    Returns
    -------
    xr.Dataset
        Modelo estratificado ('z', 'x')
    """
    for layer in layers:
        z = layer.pop("z")
        z_bottom = layer.pop("z_bottom", None)
        add_layer(model, z=z, z_bottom=z_bottom, **layer)
    
    return model

def layered_model(region, spacing, layers):
    """
    Cria um modelo estratificado de uma maneira mais simples para o usuario

    Parameters
    ----------
    region: tuple
        (xmin, xmax, zmin, zmax)
    spacing: float or tuple
        (dx, dz)
    layers: list of dict
        Definição de como a camada deve ser:
        {
            "z": profundidade do topo da camada (em metros, não em indice),
            "velocity": valor,
            "density": valor
        }
        A primeira camada deve começar em z = 0

    Returns
    -------
    xr.Dataset
        Modelo estratificado ('z', 'x')
    """
    
    #spacing
    if isinstance(spacing,(int, float)):
        dx = dz = spacing
    else:
        dx, dz = spacing

    #coordenadas fisicas
    coords = bd.grid_coordinates(region=region, spacing=spacing, adjust="region")
    z = coords[1][:,0]
    x = coords[0][0,:]
    shape = coords[0].shape
    nz, nx = shape

    #ordena a lista de camadas por profundidade em z do topo ao fundo
    layers = sorted(layers, key=lambda l: l["z"])

    #propriedades físicas
    properties = layers[0]["properties"]
    properties_names = properties.keys()

    #modelo base
    model_data = {}

    for name, value in properties.items():
        data_array = np.full(shape, value, dtype="float32")
        model_data[name] = xr.DataArray(
            data=data_array,
            coords={"z": z, "x": x},
            dims=("z", "x"),
            name=name
            )
    
    #aplicação de camadas mais a baixo
    for i in range(1, len(layers)): #camada 0 é usada pra criar o modelo base (np.full)
        z_top = layers[i]["z"] #profundidade onde a camada começa
        
        #vai definir onde a camada termina
        if i<len(layers) - 1: 
            z_bottom = layers[i+1]["z"] #se existe camada abaixo
        else:
            z_bottom = z[-1] + dz #se não existe camada abaixo
        
        mask = (z>=z_top) & (z<z_bottom) #filtro

        for name in properties_names: 
            #value = layers[i]["properties"][name]
            #model_data[name].loc[dict(z=z[mask])] = value
            model_data[name].loc[dict(z=z[mask])] = layers[i]["properties"][name] #junção das linhas anteriores
    
    model = xr.Dataset(model_data)

    model.attrs = {
        "dx":dx,
        "dz":dz,
        "spacing":spacing,
        "region":region,
        "shape":shape,
        "n_layers":len(layers),
        "model_type":"layered_model"
    }
    
    return model

def model(region, spacing, layers=None, properties=None, **kwargs):
    """
    Create homogeneous or layered xarray

    Parameters
    ----------
    region: tuple
        (xmin, xmax, zmin, zmax)
    spacing: float or tuple
        (dx, dz)
    layers: list or dict, optional
        layers definiton:
            {"z": depth, "velocity": value, "density": value}
    properties: dict, optional
        for homogeneus model
    **kwargs:
        alternative way to define homogeneous properties

    Returns
    -------
    model: xr.Dataset
    """

    if isinstance(spacing, (int, float)):
        dx = dz = spacing
    else:
        dx, dz = spacing

    grid = bd.grid_coordinates(region=region, spacing=spacing, adjust="region")
    z = grid[1][:,0]
    x = grid[0][0,:]
    nz, nx = grid[0].shape #shape vai pegar o formato ("esqueleto") de grid, vai guardar o numero de linhas e numero de colunas (y, x) -> (z, x) 
    shape = (nz, nx)

    if layers is None:
        #homogeneous model

        if properties is None:
            properties = {}
        
        properties = {**properties, **kwargs}
        if len(properties) == 0:
            raise ValueError("Defina as propriedades físicas (velocity, densitty) do modelo homogêneo!")
        
        model_data = {}

        for name, value in properties.items():
            data_array = np.full(shape, value, dtype="float32")
            model_data[name] = xr.DataArray(
                data=data_array,
                coords={"z":z, "x":x},
                dims=("z","x"),
                name=name
                )
        model_type = "homogeneous"
        n_layers = 1
    
    else:
        #layered model
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError("'layers' não pode ser uma lista vazia")
        
        layers = sorted(layers, key=lambda l: l["z"]) #vai ordenar as camadas por z em ordem crescente
        
        if layers[0]["z"] != z[0]:
            raise ValueError("A primeira camada deve começar no topo do modelo, z=0")
        
        properties_names = []

        for key in layers[0].keys():
            if key != "z":
                properties_names.append(key)

        model_data = {}

        for name in properties_names:
            data_array = np.full(shape, layers[0][name], dtype="float32")
            model_data[name] = xr.DataArray(
                data=data_array,
                coords={"z":z, "x":x},
                dims=("z","x"),
                name=name
            )
        
        for i in range(1, len(layers)):
            z_top = layers[i]["z"]

            if i < len(layers) - 1:
                z_bottom = layers[i+1]["z"]
            else:
                z_bottom = z[-1] + dz

            mask = (z >= z_top) & (z < z_bottom)

            for name in properties_names:
                model_data[name].loc[dict(z=z[mask])] = layers[i][name]
        
        model_type = "layered"
        n_layers = len(layers)
    
    model = xr.Dataset(model_data)
    model.attrs = {
        "dx":dx,
        "dz":dz,
        "spacing":spacing,
        "region":region,
        "shape":shape,
        "model_type":model_type,
        "n_layers":n_layers
    }

    return model
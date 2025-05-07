.. _overview:

Overview
========

.. jupyter-execute::

    import tremelique as tr
    import numpy as np

    shape = (300, 400)
    spacing = 5
    extent = [0, shape[1]*spacing, shape[0]*spacing, 0]
    velocity = np.zeros(shape, dtype='float32') + 1500  # m/s
    density = np.zeros(shape, dtype='float32') + 1000  # kg/m³

    pwave = tr.Acoustic(velocity, density, spacing=spacing)
    pwave.add_point_source((0, shape[1]//2), tr.RickerWavelet(1, 60))
    pwave.run(300)
    pwave.animate(every=10, embed=True, dpi=50, cutoff=0.5)


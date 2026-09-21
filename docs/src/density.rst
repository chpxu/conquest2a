Densities
=========

The classes under the `density` module handle the processing and plotting of charge and band densities. Note that there is no explicit dependence on any of the classes used in C2a, and therefore could be used standalone. However, it only supports *orthorhombic cells*.

The ``density`` class supports the following features:

1. Loading arbitrary amount of cube data
2. Ability to freely perform sequential file operations. For example, if you pass in files `["a", "b", "c"]` and "+*" then the class will contain the resulting data as ``(a + b) * c``, i.e. it applies operations sequentially and left-to-right like VESTA.
3. Slicing through any :math:`hkl`-plane and translating it along the plane normal with any ``offset``

Two extra classes are also contained in this module: ``chden`` and ``bandden``. After searching for files, each class exposes a :class:`density` instance under the ``.density`` attribute.
- ``chden`` (same name but functionally different from versions before 0.4.0) searches for ``{stub}_up.cube`` and ``{stub}_dn.cube`` (for spin-polarised) or ``{stub}.cube`` for spin-unpolarised.
- ``bandden`` searches for all ``Band*.cube`` files. It supports filtering by multiple bands, multiple :math:`k`-points and spin (``S1`` or ``S2`` as CONQUEST only supports collinear spin)

Finally, the module has a plotting class ``plot_density``. This class handles all the steps and functions to process a density, and then allows you to plot said density and optionally save it to a file. This class also exposes its final ``Figure``, ``Axes`` and ``imshow`` after calling ``run()``.


Example Usage
=============
This example creates a final figure from two different band density instances. Each band density instance exposes its Axes so that we can use them on the external subplot.

.. code-block:: python

  import matplotlib.pyplot as plt
  import numpy as np
  from conquest2a.density import density, bandden, plot_densities

  # Create figure instance
  fig, (ax_val, ax_cond) = plt.subplots(1, 2, figsize=(8, 5))

  path_to_dir = r"path/to/directory/with/band/densities" + "/"
  den1 = bandden(directory=path_to_dir,  # str
                  hkl=[0,0,1],           # The slicing plane, Miller indices
                  offset=0.0,            # How much to translate the plane by [Bohr]
                  bands=[87,88],         # Bands to search for. Note since spin and kpts is None, it will search for all spins and files which are summed over kpts
                  operations="+++",      # This example uses 2 bands, so since kpts=None, I get 4 files (87_S1, 87_S2, 88_S1, 88_S2). I then tell the class to add up the data across all the files.
                  units="ang",           # The volumetric data from CONQUEST has Bohr units. This converts everything to angstroms
                  spin=None,             # Spin filter. None is all spins
                  kpts=None              # kpts filter, can also be a list[int] and used in conjunction with bands.
                )
  plot1  = plot_densities(den1,
                          show_atoms=True, # Plots markers of the atom locations projected onto the slicing plane, default False
                          extension: str = "png", # Filename extension, default to "png" and only used when no filename is supplied later
                          cbar_label: str | None = None, # Colourbar label
                          )
  # fig, ax, imshow
  fig, ax, im1 = plot1.run(
    "test.png", # File name
    ax=ax_val,  # Axes instance. Default None (it will create its own)
    atom_number=5, # Atom number to set as the origin of the plot
    thickness=1.5, # Perpendicular distance from the plane (Bohr) to look for atoms, default 1.0
    vmax=1.10, # Density values above this will be clipped. Depends on units. Default is maximum of the data
    xlabel=r"$[0y0](\mathrm{\AA})$", ylabel=r"$[x00] (\mathrm{\AA})$",  # Labels, can be None (hides) or True (default generated)
    xlim=(-2.5,2.5), ylim=(-2.5,2.5), # limits
    window_repeat=1, # How many repeats of the initial densiy to extend by
    atom_repeat=1, # How many repeats of atom searches
    grid_points=500, # Grid points for interpolating density. I recommend at least 500 for a clear image. Default is 1000
    save=False, # Whether to save an image automatically, default True
    show_colorbar=False # Whether to have a colourbar, default True. False is useful if you're plotting on an external Axes and wish to control subplots yourself.
    )




  den2 = bandden(directory=path_to_dir, hkl=[-2.42693,-3.6017,-1], offset=-2.21346, operations="+", bands=[89], units="ang")

  blah2 = plot_densities(den2, show_atoms=True)
  _,_,im2 = blah2.run("test_cond.png",ax=ax_cond, save=False, thickness=1, shift=(-2.39,4.2), xlabel=r"$[-\frac{3}{4},\frac{1}{2},-\frac{1}{5}][\mathrm{\AA}]$", ylabel=r"$[-1,0,4][\mathrm{\AA}]$", show_ticks=True, show_colorbar=False, atom_symbols=["Mn", "O"], vmax="1.1", window_repeat=2, atom_repeat=1, grid_points=500, xlim=(-3,3),ylim=(-3, 3))


API
===

.. automodule:: conquest2a.density
  :members:
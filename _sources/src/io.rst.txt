File I/O
========

Coordinate file conversion to and from the CONQUEST coordinate file is handled by the `io.write_coords` and `io.read_coords`

Example usage
+++++++++++++

First, define :class:`conquest_species`, e.g.

.. code-block:: python

  from conquest2a.conquest import conquest_species, conquest_coordinates

  cs = conquest_species({1:"K", 2: "Cu", 3: "Cu", 4: "F"})

Then, assuming you have a completely filled out :class:`conquest_coordinates` instance, you can just write immediately:

.. code-block:: python

  from conquest2a.io import write_coords
  cq = write_coords("dest_file.cell", cs, format="cell")

Check the API below. Writing supports ``cell``, ``vasp`` (POSCAR), ``xsf`` (including additional vector quantities), ``ext/xyz`` as well as writing out a CONQUEST coordinates file (default extension is ``.dat``).

If you instead want to _read_ in a coordinate file,

.. code-block:: python

  from conquest2a.io import read_coords
  cq = read_coords("read_file.cell", cs, format="cell")
  coords = cq.coords # Now you can use this to do whatever

This will now expose a :class:`conquest_coordinates` instance for you to use. Currently supports reading CONQUEST files, ``cell`` and POSCAR ``.vasp`` files.

API
+++

.. automodule:: conquest2a.io
  :members:
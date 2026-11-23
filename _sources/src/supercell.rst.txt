
Transforms and supercells
=========================

C2a can handle transforming unit cells like in VESTA, with the important feature of being able to preserve species numbers. In CONQUEST, elements with different spins are given their own species, e.g. species 1 could be Mn (up) and species 2 could be Mn (down). Thus, creating and transforming cells with spin patterns is now trivial.

In VESTA, you can give atoms vectors. Transforming a unit cell preserves symmetry and thus the arrows (by choosing "equivalent sites"). However, once the symmetry is removed, the vectors are not maintained (understandably so). Since VESTA has no knowledge of "species" when outputting unit cells, it has no notion that a set of atoms are meant to have particular spins, and that spins are mapped to species. For small cells this isn't an issue, you could set atoms to different elements temporarily and manually edit the coordinates file afterwards, but for large cells this is unfeasible and a time waste, which is where this module aims to fill the gap as CONQUEST is designed for large-scale simulations.


Example Usage
+++++++++++++

In this section, we demonstrate use of the :class:`transform_unit_cell` class. The following snippet is modified from the unit tests of this module.


The following code block will transform an 80-atom supercell with a spin pattern onto the symmetry-equivalent 40-atom unit cell.

.. code-block:: python

  from conquest2a.conquest import conquest_species, conquest_coordinates
  from conquest2a.cell.transform import transform_unit_cell
  from conquest2a.io import read_coords, write_coords # To read and write coordinate files

  run_dir = "path/to/directory"
  input_cell = f"{data_dir}/unitcell.dat" # Cell to apply transform to
  final_cell_file = f"{data_dir}/expected_transformed_cell.dat" # File to write new unit cell to

  # Always define your species
  BMO = conquest_species({1: "O", 2: "Bi", 3: "Mn", 4: "Mn"})
  # Transformation matrix, same convention as VESTA
  P = np.array([[1 / 2, 0, 1 / 2], [0, 1, 0], [-1 / 2, 0, 1 / 2]])

  # conquest_coordinates instance of input file
  coords = read_coords(input_cell, species, "cq").coords
  transformer = transform_unit_cell(coords)
  # Apply transform
  result = transformer.transform(P)
  # Access new conquest_coordinates instance of your transformed cell, including correct species!
  new_cell = result.transformed_cell_coords

  # Write new_cell into CONQUEST coordinates format
  write_coords(final_cell_file, new_cell, format='cq')

To instead make a supercell, use the :class:`supercell` class or just use the :class:`transform` with a supercell matrix. The former is demonstrated below:

.. code-block:: python
  from conquest2a.conquest import conquest_species, conquest_coordinates
  from conquest2a.cell.transform import supercell
  from conquest2a.io import read_coords, write_coords # To read and write coordinate files

  BMO = conquest_species({1: "O", 2: "Bi", 3: "Mn", 4: "Mn"})

  # conquest_coordinates instance of input file
  coords = read_coords(input_cell, species, "cq").coords
  sc = supercell(repeats_x=1, repeats_y=1, repeats_z=1, orig_coords=coords)
  sc_coords = sc.supercell_coords # New conquest_coordinates instance you can do whatever with

API
+++

.. automodule:: conquest2a.cell.supercell
  :members:

.. automodule:: conquest2a.cell.transform
  :members:
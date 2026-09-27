General Classes
===============

The classes under the `conquest` module are used across the library, and are often required to be initialised and passed as arguments to other classes.

Example usage
+++++++++++++

The most important class in this library is :class:`conquest_coordinates`. It is a full Python class which holds information about a CONQUEST coordinates file. It requires a :class:`conquest_species` to initialise it, which maps CONQUEST species numbers to real elements.

Suppose you have a spin-polarised KCuF\ :sub:`3` system with a spin up and down Cu atoms. You might do:

.. code-block:: python

  from conquest2a.conquest import conquest_species, conquest_coordinates

  kcuf3 = conquest_species({1: "K", 2: "F", 3: "Cu", 4: "Cu"})
  coords = conquest_coordinates("coords.dat", kcuf3)

Now the variable ``coords`` contains:
- ``coords.atoms``: list of :class:`Atom`, in the order of their appearance in ``coords.dat``
- ``coords.element_map``: a ``dict[str, list[Atom]]`` where each key is an element label whose value is all the atoms with that element label
- ``coords.lattice_vectors``: a NumPy array of lattice vectors. Does not factor in units (you should check if you need angstroms or bohr units. CONQUEST assumes bohr by default).

There is also the :class:`atom_charge` class, which uses the ``AtomCharge.dat`` file to calculate spin ``up - dn`` for each atom and then assigns it to the ``Atom.spins`` array.

.. code-block:: python

  from conquest2a.conquest import conquest_species, conquest_coordinates, atom_charge

  kcuf3 = conquest_species({1: "K", 2: "F", 3: "Cu", 4: "Cu"})
  coords = conquest_coordinates("coords.dat", kcuf3)
  atom_charge("AtomCharge.dat", coords) # Mutates atom.spin for atom in coords.atoms

The other classes are utility classes for other types of CONQUEST pre/post-processing. ``block_processor`` is a generic class to inherit from to parse files which separate blocks of data by the ``&`` character - isolated on its own line - as is for spin-polarised CONQUEST output like pDOS and band structures. ``processor_base`` is another generic class to inherit from for any generic data processing from files.

API
+++

.. automodule:: conquest2a.conquest
  :members:
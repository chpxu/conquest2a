PDOS
====
The processing of (p)DOS files is entirely handled by the single class :class:`pdos`. It will analyse and store DOS information in dicts.

The :class:`pdos` class accepts 2 parameters: a path to a directory containing (p)DOS files and a parameter ``mode`` which is either ``t``, ``l`` or ``lm``. if the total DOS is requested from CONQUEST, then a ``DOS.dat`` will be placed in the directory. Its contents are simply the energy, DOS and the local density states (columns) sorted in blocks of ascending spin (Spin 1, then 2). Alternatively, pDOS can be calculated for each atom and can be angular momentum resolved (:math:`l`) or further resolved into each quantised momentum :math:`lm`. In this case, for any choice of atom (or all atoms), ``AtomNNNNNNN_[l|lm].dat`` files are output to the running directory. The columns are then the energy, then the total DOS at that energy, then columns of pDOS values sorted in ascending order of :math:`l,m`` going towards the right. The files are also placed in blocks for each spin.

See `this file <https://github.com/chpxu/conquest2a/blob/main/tests/data/Atom00000002DOS_lm.dat>__`for an example :math:`lm`-resolved pDOS and `this other file <https://github.com/chpxu/conquest2a/blob/main/tests/data/Atom00000001DOS_l.dat>__` for an example :math:`l`-resolved pDOS file.


Usage
+++++
The following code block details how to use each mode.

.. code-block:: python

  from conquest2a.pdos import pdos
  # TDOS
  dos_proc = pdos("my/directory", mode="t")
  dos_proc.get_pdos()
  energies = dos_proc.energy_values[1] # energy
  dos = dos_proc.pdos_dict["tdos"]     # total DOS
  spin_up_dos, spin_dn_dos = dos[0], dos[1] # unpack DOS of each spin
  ldos = dos_proc.pdos_dict["ldos"]    # same behaviour for ldos key

  # l-resolved pDOS
  pdos_lproc = pdos("my/directory", mode="l")
  pdos_lproc.get_pdos(atom_number=1)            # Will look in my/directory, for l-resolved pDOS files with the atom number.
  pdos_l = pdos_lproc.pdos_dict                  # Now contains pDOS information sorted by angular momentum "0", "1", "2"
  s_pdos = pdos["0"]                          # s-orbital pdos
  p_pdos = pdos["1"]                          # p-orbital pdos
  d_pdos = pdos["2"]                          # d-orbital pdos
  # lm-resolved pDOS
  pdos_lmproc = pdos("my/directory", mode="lm")

  pdos_lmproc.get_pdos(atom_number=1)            # Will look in my/directory, for lm-resolved pDOS files with the atom number.
  pdos_lm = pdos_lmproc.pdos_dict                  # Now contains pDOS information sorted by angular momentum "0,0", "1,-1", "1,0", "1,1", "2,-2", "2,-1" ...

For ``mode="lm"`` and ``l``, each key corresponds to a Python list of NumPy arrays, sorted in order of spin. Therefore, the first element of ``s_pdos`` in the code block above has 2 elements: ``[np.array(...), np.array(...)]`` and the first element (index ``0``) corresponds to Spin 1, the second element (index ``1``) corresponds to Spin 2. This is the same for ``pdos_lm["2,-1"] = [np.array(...), np.array(...)]``.

The code will abort if any files cannot be found, including ``DOS.dat`` and if it cannot find a pDOS file associated with the supplied atom number.

API
+++

.. automodule:: conquest2a.pdos
  :members:
# 0.4.0

## Features
- New density API, `conquest2a.density`
    - generic processing is handled by the `density` class inside the `density` module. It now handles a generic set of parameters to slice and extract the relevant data from supplied files. It now supports combining an arbitrary amount of files in a specified way. For example, if you pass in files `["a", "b", "c"]` and "+*" then the class will contain the resulting data as `(a + b) * c`, i.e. it applies operations sequentially and left-to-right like VESTA.
    - Subclasses `chden` and `bandden` handle file searching for charge density output or band density output.These classes then attach a `density` instance to themselves with the filtered files
    - `bandden` supports filtering by band number, $k$-point and spin
    - `chden` supports filtering for spin-(un)polarised calculations, as well as searching for stubs instead.
    - The generic density plotter ``plot_density`` also supports shifting the origin of the plot, either arbitrarily or by inputting an atom number.
        - ``plot_density`` also supports exposing its `ax` and `im` instances, to allow further customisation or to plot in an external figure.
        - It also has much more customisation, allowing custom keywords for `ax.text`, setting `xlim/ylim`, filtering atom labels, satisfying PBC conditions for the slicing planes allowing for repeats, custom labels, themes etc
- New IO API, `conquest2a.io`
    - File writing and reading is now handled entirely by the new `io.write_coords` and `io.read_coords`, no need to go class hunting
    - Improved object-oriented API: define `conquest_species`, create `write/read_coords` which will automaticaly write file for you
    - Automatic file extension detection if you don't supply a format
- New `cell` module, `conquest2a.cell`
    - VESTA-like unit cell transformations are handled with the new `cell.transform` module. There is only one class: `transform_unit_cell`
    - Define a transformation matrix $P$ and call it as `transform_unit_cell.transform(P)` which will return a new `conquest_coordinates` instance you can use
    - The `Atom` class now has a new member: `symmetry_number`. When doing unit cell transforms, it is set to the original atom number which created it. This may be useful for identifying a bulk set of atoms to manipulate.
- Refactored DOS API
    - Everything now happens inside the new `pdos` class. It accepts the same parameters as before: path to a directory and what mode
    - Now, you just need to instantiate the class, e.g. `pdos_instance = pdos("thisdir", "lm)`. Then you need to just call `pdos_instance.get_pdos()`. If you supply a positive integer, it will automatically search for $l$ or $lm$-resolved pDOS files in the directory, otherwise it will look for `DOS.dat`.
    - Resulting DOS/pDOS data now accessible via `pdos_dict` and `energy_values` attributes. For DOS, the keys `pdos_dict` are `tdos` and `ldos`, whilst for pDOS they keys are still `l` or `l,m` and they are still sorted by ascending angular momentum
    - The fundamental searching and processing is identical to v0.3.0

## Other
- Tests introduced for `conquest` module
- Tests introduced for new `density` module on real data
- Tests introduced for new `io` module
- Original `supercell` module now moved under `cell.supercell`

## Bug fixes
- Unit conversion errors (everyone's favourite)
- Fix newline bug in writing CONQUEST cooordinate files
- Make resetting pDOS more robust
- Fix atom labels in density being transposed the wrong way
<!--  -->
# 0.3.0

## Features
- Introduced basic plotting into `pdos` module
- Introduce Sphinx documentation at https://conquest-to-vasp.readthedocs.io/en/latest/
- Interopability with ASE
    - `conquest2a.conquest.Atom` now has a `to_ase()` method which returns an ASE `Atom` object
- Band structure plotting

## Fixes

- Fix calculation of grid vectors in charge density plot
- Fix incorrect placement of atom labels in charge density plot
- Convert `chden` to use Bohr units completely

## Other
- BREAKING: the `conquest_input` class has been renamed to `conquest_species` to better reflect its purpose. The argument `conquest_input` will now take in this new class instead.
- BREAKING: drop Python 3.10
- fix: `black` excluding `conquest2a` directory
- `hkl` now accepts a basic tuple of ints instead of a NumPy array
- Add more type hints throughout
- Add `@override` where appropriate
- Update Nix deps to 26.05
<!--  -->
# 0.2.0

## Features
- Refactored project structure
- Created `_types.py` and `constants.py`.
- Split file writer classes into `src/writers.py`
- Classes dealing with processing the Conquest coordinates file is in `src/conquest.py`
- Classes making Conquest coordinates-compliant supercells are in `src/supercell.py` (NEW)
<!-- - Classes generating a Conquest_input file with sane defaults is in `src/generate_run.py` (NEW)-->
- `xsf` file-format with spin (NEW)
    - Modify the `Atom` class with a `spin` attribute that can optionally be filed
    - Create a `atom_charge` class to process `AtomCharge.dat`
    - create an `xsf_writer` class in `src/writers.py` to write this data, including the spin
- pDOS processing (NEW!)
    - Supports reading pDOS files for $l,m$ and $l$ decomposed situations
- Band structure processing (`band.py`)
- $k$-nearest-neighbour searching via periodic KDTrees (`algo/kdtree.py`).
- Extracting forces and stresses from static output files (`read/quantities.py`)


## Misc
- All modules now are integrated with NumPy.
- `Atoms` class now has attributes `spin` and `forces`, both of which are NumPy arrays

## Development

- Added flake8, pylint, mypy and black to the repo
- Added some initial tests in `tests/` with pytest

# 0.1.0 - Initial release

- Script for converting coordinate files to and from VASP and (ext)XYZ format.

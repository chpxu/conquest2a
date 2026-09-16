from collections import defaultdict
import sys
import re
from pathlib import Path
from io import TextIOWrapper
from typing import IO, Any, Literal

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
import numpy as np
from conquest2a.conquest import Atom, conquest_species, conquest_coordinates, processor_base
from conquest2a.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR
from conquest2a._types import REAL_ARRAY


class write_coords(processor_base):
    """
    Class to convert data held in a :class:`conquest_coordinates` instance into various coordinate file formats: xsf, cell, vasp, xyz and extxyz.


    :param dest: Path to output file.
    :type dest: str
    :param cq_coords: :class:`conquest_coordinates` instance with data
    :type cq_coords: conquest_coordinates
    :param format: Type of coordinate file to write
    :type format: Literal["cq","vasp","cell","xyz","extxyz", "xsf"]
    :param precision: Number of decimal places, defaults to 10
    :type precision: int, optional
    :param mode: File IO mode, defaults to "w"
    :type mode: str, optional
    :param encoding: File encoding, defaults to "utf-8"
    :type encoding: str, optional
    :param convert_to: Whether to convert lattice parameters and cartesian coordinates to specified units, defaults to None (data is preserved as-is)
    :type convert_to: Literal["bohr", "ang"] | None, optional
    :param comment_line: What unit factor to target, defaults to ``None`` (content written as-is)
    :type comment_line: str, optional
    :param time: Animation time (extxyz only), defaults to 0.0
    :type time: float, optional
    :param write_spin: Whether to write out ``atom.spins`` (XSF only, mutually exclusive with ``write_force``), defaults to False
    :type write_spin: bool, optional
    :param write_force: Whether to write out ``atom.force`` (XSF only, mutually exclusive with ``write_spin``), defaults to False
    :type write_force: bool, optional
    :raises ValueError: If write_force and write_spin are True, error out.
    """

    def __init__(
        self,
        dest: str,
        cq_coords: conquest_coordinates,
        format: Literal["cq", "vasp", "cell", "xyz", "extxyz", "xsf"],
        precision: int = 10,
        mode: str = "w",
        encoding: str = "utf-8",
        convert_to: Literal["bohr", "ang"] | None = None,
        comment_line: str = "",
        time: float = 0.0,
        write_spin: bool = False,
        write_force: bool = False,
    ) -> None:
        self._FORMATS: dict[str, str] = {
            "cell": "write_cell",
            "cq": "write_conquest",
            "vasp": "write_poscar",
            "xyz": "write_xyz",
            "extxyz": "write_extxyz",
        }
        self.mode: str = mode
        self.data: conquest_coordinates = cq_coords
        self.format: str = format
        self.dest_path: str = dest.strip()
        self.encoding: str = encoding
        self.precision: int = precision
        self.convert_units: str | None = convert_to
        self.time: float = time
        self.write_spin: bool = write_spin
        self.write_force: bool = write_force
        if self.write_force and self.write_spin:
            raise ValueError(
                "Cannot write both atom forces and atom spins to same coordinate file."
            )
        super().__init__(path=dest)
        self.comment_line: str = comment_line
        self.file: IO[Any]

        self.open_file()
        self.write()
        self.close_file(self.file)

    @override
    def open_file(self) -> None:
        self.file = open(self.dest_path, mode=self.mode, encoding=self.encoding)

    def close_file(self, file: TextIOWrapper | IO[Any]) -> None:
        file.close()

    def _write_2d_array_with_precision(self, arr: REAL_ARRAY) -> str:
        prec = self.precision
        return "\n".join(" ".join(f"%0.{prec}f" % x for x in y) for y in arr)

    def _write_1d_array_with_precision(self, arr: REAL_ARRAY) -> str:
        prec = self.precision
        return "".join(" ".join(f"%0.{prec}f" % x for x in arr))

    def _lattice_vect_units_converter(self, arr: REAL_ARRAY) -> REAL_ARRAY:
        if self.convert_units == "bohr":
            return arr * ANGSTROM_TO_BOHR
        elif self.convert_units == "ang":
            return arr * BOHR_TO_ANGSTROM
        return arr

    def create_atoms_str(self) -> tuple[str, str]:
        """Creates a line for an atom, for the XYZ file.

        :return: Components of the line to write: the element and the numbers
        :rtype: tuple[str, str]
        """
        num_ele: dict[str, int] = self.data.number_of_elements()
        ele_string: str = " ".join(list(num_ele.keys()))
        num_string: str = " ".join(str(x) for x in num_ele.values())
        return ele_string, num_string

    def write_conquest(self) -> None:
        """Method to write a CONQUEST coordinates file given a :class:`conquest_coordinates` instance."""
        self.file.write(
            self._write_2d_array_with_precision(
                self._lattice_vect_units_converter(self.data.lattice_vectors)
            )
        )
        self.file.write("\n")
        self.file.write(self.data.natoms)
        self.file.write("\n")
        for atom in self.data.atoms:
            move_str: str = " ".join(x for x in atom.can_move)
            atom_str: str = self._write_1d_array_with_precision(atom.coords)
            self.file.write(f"{atom_str} {atom.species} {move_str}")
            self.file.write("\n")

    def write_vasp(self) -> None:
        """Method to write a basic VASP POSCAR file given a :class:`~conquest.conquest_coordinates` instance.

        :param dest: Path to write the new coordinates file.
        :type dest: ``str``
        :param data: :class:`conquest.conquest_coordinates` instance to write.
        :type data: ``conquest_coordinates``
        :param encoding: File encoding, defaults to "utf-8"
        :type encoding: ``str``, optional
        :param is_angstrom: Whether the data in ``conquest_coordinates`` is already in angstroms instead of Bohrs, defaults to ``False``.
        :type is_angstrom: ``bool``, optional
        """
        ele_string, num_string = self.create_atoms_str()
        self.file.write(f"{ele_string}\n")
        self.file.write("1.0\n")
        self.file.write(
            self._write_2d_array_with_precision(
                self._lattice_vect_units_converter(self.data.lattice_vectors)
            )
        )
        self.file.write(f"{ele_string}\n")
        self.file.write(f"{num_string}\n")
        self.file.write("Direct\n")
        for atoms in self.data.element_map:
            for atom in self.data.element_map[atoms]:
                self.file.write(self._write_1d_array_with_precision(atom.coords))
                self.file.write("\n")

    def write_cell(self) -> None:
        self.file.write("%block LATTICE_CART\n")
        self.file.write(f"Bohr\n")
        self.file.write(self._write_2d_array_with_precision(self.data.lattice_vectors))
        self.file.write("%endblock LATTICE_CART\n")

        self.file.write("%block POSITIONS_FRAC\n")
        for atom in self.data.atoms:
            line = f"{atom.label} {atom.coords[0]:.{self.precision}f} {atom.coords[1]:.{self.precision}f} {atom.coords[2]:.{self.precision}f}"
            if atom.spins[2] != 0.0:
                line += f" SPIN {atom.spins[2]:.{self.precision}f}"
            self.file.write(line + "\n")
        self.file.write("%endblock POSITIONS_FRAC\n")

    def write_xyz(self) -> None:
        """Method to write a ``.xyz`` for a basic XYZ file given a :class:`~conquest.conquest_coordinates` instance."""
        self.file.write(f"{self.data.natoms}")
        self.file.write(f"{self.comment_line}\n")
        for atoms in self.data.element_map:
            for atom in self.data.element_map[atoms]:
                self.file.write(
                    rf"{atoms} {self._write_1d_array_with_precision(self._lattice_vect_units_converter(atom.cart_coords))}"
                )
                self.file.write("\n")

    def write_extxyz(self) -> None:
        """Method to write a ``.extxyz`` for a basic XYZ file given a :class:`~conquest.conquest_coordinates` instance.

        The main advantage of `.extxyz` is the ability to specify columns and the time, which is very useful for animations. I recommend just using CONQUEST's ability to output ``.extxyz`` files at different timesteps however.
        """

        def extxyz_comment_line() -> str:
            """Creates the ``.extxyz`` comment line: specifies columns, formats and time.

            :return: The file's comment line.
            :rtype: ``str``
            """
            lattice: list[float] = []
            for single_vector in self.data.lattice_vectors:
                lattice.append(single_vector[0])
                lattice.append(single_vector[1])
                lattice.append(single_vector[2])
            property_str = "Properties=species:S:1:pos:R:3"
            time_str: str = f"Time={str(self.time)}"
            lat: str = " ".join(str(x) for x in lattice)
            return f'Lattice="{lat}" {property_str} {time_str}'

        self.comment_line = extxyz_comment_line()
        self.write_xyz()

    def write_xsf(self) -> None:
        """Method to write an XSF file optionally with a vector attached to each atom"""
        self.file.write("CRYSTAL\n")
        self.file.write("PRIMVEC\n")
        self._write_2d_array_with_precision(
            self._lattice_vect_units_converter(self.data.lattice_vectors)
        )
        self.file.write("PRIMCOORD\n")
        natom_line: str = f'{" ".join(self.data.natoms.split())} 1\n'
        self.file.write(natom_line)
        for element, atoms in self.data.element_map.items():
            for atom in atoms:
                pos_string: str = self._write_1d_array_with_precision(
                    self._lattice_vect_units_converter(atom.cart_coords)
                )
                extra: REAL_ARRAY = np.array([0.0, 0.0, 0.0])
                if self.write_spin:
                    extra = atom.spins
                elif self.write_force:
                    extra = atom.forces
                extra2: str = self._write_1d_array_with_precision(extra)
                self.file.write(f" {element} {pos_string} {extra2}\n")

    def write(self) -> None:
        """Wrapper function to write file based off ``self.format``.
        :raises ValueError: If ``self.format`` is unsupported.
        """
        try:
            method_name = self._FORMATS[self.format]
        except KeyError:
            supported = ", ".join(sorted(self._FORMATS))
            raise ValueError(
                f"Unsupported format '{self.format}'; supported formats are: {supported}"
            ) from None

        reader = getattr(self, method_name)
        reader()


# Reading
class read_coords(processor_base):
    """Class to read external coordinates data and store them in a :class:`conquest_coordinates` instance.

    :param path: Path to file to read
    :type path: str
    :param species: :class:`conquest_species` instance
    :type species: conquest_species
    :param format: File format, defaults to ``None`` in which case it will infer from the file extension of ``path``
    :type format: Literal["vasp", "cell"] | None, optional
    :param encoding: File encoding, defaults to "utf-8"
    :type encoding: str, optional
    :param cq_units: Units to use in :class:`conquest_coordinates`, defaults to "bohr". `vasp` is assumed to be read entirely in angstroms.
    :type cq_units: Literal["bohr", "ang"], optional
    """

    def __init__(
        self,
        path: str,
        species: conquest_species,
        format: Literal["vasp", "cell", "cq"] | None = None,
        encoding: str = "utf-8",
        cq_units: Literal["bohr", "ang"] = "bohr",
    ) -> None:
        self._FORMATS: dict[str, str] = {
            "cell": "read_cell",
            "vasp": "read_poscar",
            "cq": "read_conquest",
        }
        self.path: str = path
        self.species: conquest_species = species
        self.format: Literal["vasp", "cell", "cq"] = (
            format if format is not None else self._get_format_from_path()
        )
        self.file: IO[Any]
        self.encoding: str = encoding
        self.cq_units: Literal["bohr", "ang"] = cq_units
        self._atom_counter: int = 0
        self.fix_ions: bool = False
        super().__init__(self.path)
        self.resolve_path(self.path)
        self.open_file()
        # This will store the data of the read file
        self.coords: conquest_coordinates = conquest_coordinates(self.species)
        self.read()
        self.coords.get_cartesian_positions()
        self.coords.assign_atom_labels()
        self.coords.index_to_atom_map()
        self.close_file(self.file)

    @override
    def open_file(self) -> None:
        self.file = open(self.path, mode="r", encoding=self.encoding)

    def close_file(self, file: TextIOWrapper | IO[Any]) -> None:
        file.close()

    def _get_format_from_path(self) -> Literal["vasp", "cell", "cq"]:
        extension: str = Path(self.path).suffix
        if extension == ".cell":
            return "cell"
        if extension == ".vasp":
            return "vasp"
        raise ValueError("Class read_coords currently only supports conquest, cell and poscar")

    def _resolve_species_names(self, counts: list[int], elements: list[str] | None) -> list[str]:
        if elements is not None:
            return [element.strip() for element in elements]
        labels: list[str] = []
        seen: set[str] = set()
        for _, label in sorted(self.species.species_dict.items()):
            if label not in seen:
                labels.append(label)
                seen.add(label)
        if len(labels) != len(counts):
            raise RuntimeError("Number of element types and number of counts are mismatched")
        return labels

    def _resolve_species_id(self, label: str) -> int:
        if label not in self.species.allowed_element_labels:
            raise ValueError(f"'{label}' is not a valid element.")
        matches = sorted(idx for idx, lbl in self.species.species_dict.items() if lbl == label)
        if not matches:
            raise ValueError(f"Element '{label}' is not present in the supplied species map.")
        return matches[0]

    def _make_triclinic_lattice(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> REAL_ARRAY:
        r"""Given lattice parameters, make a P1 triclinic formulation:

        1. :math:`v_1 = (a, 0, 0)`,
        2. :math:`v_2 = (b*\cos(\gamma), b*\sin(\gamma), 0)`
        3. :math:`v_3 = (c*\cos(\beta),  \c*(\cos(\alpha)-\cos(\beta)\cos(\gamma))/\sin(\gamma),
            c*\sqrt( 1 + 2*\cos(\alpha)\cos(\beta)\cos(\gamma)
                        - \cos(\alpha)^2-\cos(\beta)^2-\cos(\gamma)^2 )/\sin(\gamma) )`
        where :math:`alpha` is the angle between axis :math:`b` and :math:`c`
                :math:`beta` is the angle between axis :math:`a` and :math:`c`
                :math:`gamma` is the angle between axis :math:`a` and :math:`b`

        :param a: First lattice parameter
        :type a: float
        :param b: Second lattice parameter
        :type b: float
        :param c: Third lattice parameter
        :type c: float
        :param alpha: angle between axis :math:`b` and :math:`c`
        :type alpha: float (radians)
        :param beta: angle between axis :math:`a` and :math:`c`
        :type beta: float (radians)
        :param gamma: angle between axis :math:`a` and :math:`b`
        :type gamma: float (radians)
        :return: New lattice vectors in triclinic form
        :rtype: c2at.REAL_ARRAY
        """

        a_vec = np.array([a, 0.0, 0.0])
        b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
        cz2 = np.cos(alpha) ** 2 + np.cos(beta) ** 2 + np.cos(gamma) ** 2
        cz = (c / np.sin(gamma)) * np.sqrt(
            1.0 + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) - cz2
        )
        c_vec = np.array(
            [
                c * np.cos(beta),
                (c / np.sin(gamma)) * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)),
                cz,
            ]
        )
        return np.vstack((a_vec, b_vec, c_vec))

    def _read_1d_array_string(self, arr: str, count: int = 3) -> REAL_ARRAY:
        final_vect: REAL_ARRAY = np.array(arr.split(), dtype=float)[:count]
        return final_vect

    def _read_percent_blocks(self) -> dict[str, list[str]]:
        """Process coordinate files that are separated with %block/%endblock

        :raises ValueError: If there are multiple blocks with the same name (case-insensitive, so upper and lower case are considered the same)
        :return: A dictionary keyed by the block name and then the lines
        :rtype: dict[str, list[str]]
        """
        blocks: defaultdict[str, list[list[str]]] = defaultdict(list)
        current_block: str | None = None
        current_lines: list[str] = []
        block_start_re = re.compile(r"^\s*%block\s+(\S+)", re.IGNORECASE)
        block_end_re = re.compile(r"^\s*%endblock\b(?:\s+(\S+))?", re.IGNORECASE)
        for line in self.file:
            start_block = block_start_re.match(line)
            end_block = block_end_re.match(line)

            if start_block:
                # Beginning of block means new data, so reset, and store block name
                current_block = (start_block.group(1)).lower()
                current_lines = []
            elif end_block:
                # End of block means we now store the data
                if current_block is not None:
                    blocks[current_block].append(current_lines)
                current_block = None
            elif current_block is not None:
                # Inside a block, so store the line.
                # Do not do a full trim as entries are whitespace-separated
                stripped = line.rstrip("\n")
                if stripped != "":
                    current_lines.append(stripped)
        result: dict[str, list[str]] = {}
        for name, data in blocks.items():
            if len(data) > 1:
                raise ValueError(f"File {self.path} had multiple blocks with {name}")
            result[name] = data[0]
        return result

    def _read_lattice_vectors(self, lines: list[str]) -> REAL_ARRAY:
        """Given a list of strings, convert them to NumPy arrays of lattice vectors

        Developer note: this should be called _after_ the units have been determined.

        This function will set the :class:`conquest_coordinates.lattice_vectors` directly.

        Expect something like: ``["0.0 0.5 0.3", "0.1 0.5 0.3", "-0.9 0.5 0.3"]``
        Becomes:

        ..
            np.array([
                [0.0, 0.5, 0.3],
                [0.1, 0.5, 0.3],
                [-0.9, 0.5, 0.3]
            ])

        :param lines: List of string to convert to lattice vector arrays
        :type lines: list[str]
        :raises RuntimeError: If a string to convert to lattice vector does not have 3 space-separated entries
        :raises RuntimeError: If there are not exactly 3 lattice vectors at the end of the conversion
        :return: 2D array of lattice vectors. Each row is a lattice vector
        :rtype: REAL_ARRAY
        """
        if len(lines) != 3:
            raise ValueError("Number of lattice vectors was not 3")
        vectors: list[REAL_ARRAY] = []
        for vect_str in lines:
            vector = self._read_1d_array_string(vect_str)
            if len(vector) != 3:
                raise RuntimeError("Expected three position components.")
            vectors.append(vector)

        lattice_vect: REAL_ARRAY = np.vstack(vectors)

        if len(lattice_vect) != 3:
            raise RuntimeError("Must contain exactly three lattice vectors.")
        return lattice_vect

    def read_conquest(self) -> None:
        """This method reads a CONQUEST coordinate file.

        CONQUEST coords file split into 3 main chunks:
            * first 3 lines are lattice vectors
            * fourth line is the total number of atoms in the unit cell
            * the following lines describe each atom and look like
                <double> <double> <double> <int> <char> <char> <char>
        """

        conquest_lattice_data_str: list[str] = [next(self.file).strip() for _ in range(3)]

        self.coords.lattice_vectors = self._read_lattice_vectors(conquest_lattice_data_str)
        self.coords.natoms = next(self.file)
        atom_data: list[str] = self.file.readlines()
        atom_data_stripped: list[str] = [atom for atom in atom_data if atom.strip()]
        atom_number = 1
        for atom in atom_data_stripped:
            split_atom_data: list[str] = atom.strip().split()
            self.coords.atoms.append(
                Atom(
                    species=int(split_atom_data[3]),
                    can_move=split_atom_data[4:],
                    coords=np.array(split_atom_data[:3]).astype(float),
                    number=atom_number,
                )
            )
            atom_number += 1

    def read_cell(self) -> None:
        """Parse a CASTEP cell file completely and fill out a ``conquest_coordinate`` instance

        :raises KeyError: If neither lattice_cart nor lattice_abc are in the cell file
        :raises KeyError: If neither position_cart nor position_abs are in the cell file
        """

        def _parse_atom(line: str, cart: bool = False) -> Atom:
            """
            Parse an atom positions line of the form:

            ::

                Element label xpos ypos zpos
                Element label xpos ypos zpos SPIN spinval

            A line of the form ``Element label xpos ypos zpos SPIN sx sy sz`` will raise an error as non-collinear spin is unsupported in CONQUEST.

            :param line: the atom line with or without `SPIN` flag
            :type line: ``str ``
            :return: ``Atom`` instance of the atom
            :rtype: ``Atom``
            :raises ValueError: Exits if the line does not have the correct format or atom is wrong
            :raises ValueError: Exits if non-collinear spin is detected
            """
            parts: list[str] = line.split()
            if len(parts) < 4:
                raise ValueError(f"Atom line is formatted incorrectly: {line!r}")

            label: str = parts[0]
            if label not in self.coords.conquest_input.allowed_element_labels:
                raise ValueError("Element label was not a valid element")

            position = np.array([float(x) for x in parts[1:4]])
            # No spin by default
            spin: REAL_ARRAY = np.array([0.0, 0.0, 0.0])
            rest: list[str] = parts[4:]
            if len(rest) > 0:
                if rest[0].upper() != "SPIN":
                    raise ValueError(f"Unexpected trailing content in atom line: {line!r}")
                spin_values: list[str] = rest[1:]
                if len(spin_values) == 1:
                    spin = np.array([0.0, 0.0, float(spin_values[0])])
                elif len(spin_values) == 3:
                    raise ValueError(f"Non-collinear spin is not supported: {line!r}")
                else:
                    raise ValueError(
                        f"SPIN flag must be followed by 1 (collinear) or 3 (non-collinear) "
                        + f"values, got {len(spin_values)}: {line!r}"
                    )
            move_line = ["F", "F", "F"] if self.fix_ions else ["T", "T", "T"]
            spin_species: list[int] = self.coords.conquest_input.element_to_species_dict[label]
            atom_line: Atom = Atom(
                species=spin_species[-1] if spin[2] < 0.0 else spin_species[0],
                number=self._atom_counter,
                coords=(
                    position if not cart else position @ np.linalg.inv(self.coords.lattice_vectors)
                ),
                label=label,
                can_move=move_line,
                spins=spin,
                cart_coords=position if cart else position @ self.coords.lattice_vectors,
            )
            self._atom_counter += 1
            return atom_line

        # Return back to read_cell
        block_data: dict[str, list[str]] = self._read_percent_blocks()
        flags: dict[str, str] = defaultdict(str)
        self.file.seek(0)
        in_block = False
        for raw_line in self.file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%"):
                # %block ... opens a region, %endblock (with or without a name) closes it
                in_block = line.lower().startswith(r"%block")
                continue
            if in_block or line.startswith("#"):
                continue
            upper = line.upper()
            parts = upper.split()
            if parts[0].isalpha():
                flags[parts[0]] = parts[1:]
                continue
            self.fix_ions = True if flags[parts[0]] == "T" else False
        # in CELL file, can supply LATTICE_ABS or LATTICE_CART, which have different information
        blocks = block_data.keys()
        if "lattice_cart" in blocks and "lattice_abc" in blocks:
            raise ValueError(
                "Only one of LATTICE_CART and LATTICE_ABC may occur in a cell definition file."
            )
        if "lattice_cart" not in blocks and "lattice_abc" not in blocks:
            raise KeyError(f"Neither lattice_cart nor lattice_abc was found in {self.path}.")
        if "lattice_cart" in blocks:
            units = (block_data["lattice_cart"][0]).lower()
            lattice_vect = self._read_lattice_vectors(block_data["lattice_cart"][1:])  # avoid units
            if units == "ang" and "bohr" == self.cq_units:
                lattice_vect *= ANGSTROM_TO_BOHR
            if units == "bohr" and "ang" == self.cq_units:
                lattice_vect *= BOHR_TO_ANGSTROM
            self.coords.lattice_vectors = lattice_vect
        if "lattice_abc" in blocks:
            params: list[str] = block_data["lattice_abc"]
            units = params[0]
            a_s, b_s, c_s = params[1].split()

            alpha_s, beta_s, gamma_s = params[2].split()
            a, b, c, alpha, beta, gamma = (
                float(a_s),
                float(b_s),
                float(c_s),
                float(alpha_s),
                float(beta_s),
                float(gamma_s),
            )
            if units == "ang" and "bohr" == self.cq_units:
                a, b, c = a * ANGSTROM_TO_BOHR, b * ANGSTROM_TO_BOHR, c * ANGSTROM_TO_BOHR
            if units == "bohr" and "ang" == self.cq_units:
                a, b, c = a * BOHR_TO_ANGSTROM, b * BOHR_TO_ANGSTROM, c * BOHR_TO_ANGSTROM
            alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
            self.coords.lattice_vectors = self._make_triclinic_lattice(a, b, c, alpha, beta, gamma)

        # Atom position and spin
        if "positions_frac" in blocks and "positions_abs" in blocks:
            raise ValueError(
                "Only one of POSITIONS_FRAC and POSITION_ABS may occur in a cell definition file."
            )
        if "positions_frac" not in blocks and "positions_abs" not in blocks:
            raise KeyError(f"Neither POSITIONS_FRAC nor POSITION_ABS was found in in {self.path}.")
        if "positions_frac" in blocks:

            for atomline in block_data["positions_frac"]:
                self.coords.atoms.append(_parse_atom(atomline))
        # positions_abs has first line determining units
        if "positions_abs" in blocks:
            params = block_data["positions_abs"]
            atoms = params[1:]
            for atomline in atoms:
                self.coords.atoms.append(_parse_atom(atomline, cart=True))
        self.coords.natoms = str(self._atom_counter)

    def read_poscar(self) -> None:
        """Method which reads a POSCAR file extracting only what is needed to make a CONQUEST coordinates instance

        :raises ValueError: If the number of scaling factors is incorrect
        :raises NotImplementedError: Negative scale factors
        """
        selective_dynamics: bool = False
        direct: bool = True
        lines: list[str] = []
        with open(self.abs_input_path) as f:
            lines = [line.strip() for line in f]
        # Scan POSCAR file for selective_dynamics and Direct/Cartesian first
        for line in lines:
            selective_dynamics = True if "selective" in line.lower() else False
            direct = False if "cartesian" in line.lower() else True

        # Skip comment line
        line_idx = 1

        # Scale always follows comment line
        scale = lines[line_idx].split()
        if len(scale) != 3 and len(scale) != 1:
            raise ValueError("There were not 1 or 3 scale factors")
        # Normally scale factor is 1.0 so default is identity matrix
        if any(float(s) < 0 for s in scale):
            raise NotImplementedError("Negative scaling factors in POSCAR are unsupported")
        scale_matrix = np.eye(3)
        if len(scale) == 3:
            scale_matrix = np.diag([float(scale[0]), float(scale[1]), float(scale[2])])
        elif len(scale) == 1:
            scale_matrix = np.diag([float(scale[0]), float(scale[0]), float(scale[0])])
        # Now ion species and counts. We do not assume POTCAR available so require ion species and counts
        line_idx += 1
        # Lattice vectors
        latt_str = [lines[line_idx], lines[line_idx + 1], lines[line_idx + 2]]
        self.coords.lattice_vectors = self._read_lattice_vectors(latt_str) @ scale_matrix
        line_idx += 3
        # The default is ang for POSCAR anyways
        self.coords.lattice_vectors *= ANGSTROM_TO_BOHR
        # Ion species and numbers
        elements = lines[line_idx].split()  # store elements
        line_idx += 1
        counts = [int(t) for t in lines[line_idx].split()]  # store element counts
        line_idx += 1

        labels = self._resolve_species_names(counts, elements)

        if selective_dynamics:
            line_idx += 1

        line_idx += 1

        inv_lattice = np.linalg.inv(self.coords.lattice_vectors)

        for label, count in zip(labels, counts):
            species_id = self._resolve_species_id(label)
            for _ in range(count):
                parts: list[str] = lines[line_idx].split()
                line_idx += 1
                coord_vals = np.array([float(v) for v in parts[:3]])
                # Need to handle Direct vs Cartesian
                if direct:
                    pos = coord_vals % 1.0
                else:
                    # Return to fractional coordinates, taking into account scaling and ang units
                    pos = ANGSTROM_TO_BOHR * (coord_vals @ scale_matrix @ inv_lattice)
                can_move = parts[3:6] if selective_dynamics and len(parts) >= 6 else ["T", "T", "T"]
                atom = Atom(
                    species=species_id,
                    number=self._atom_counter,
                    coords=pos,
                    label=label,
                    can_move=can_move,
                )
                self.coords.atoms.append(atom)
                self._atom_counter += 1
        self.coords.natoms = str(self._atom_counter)

    def read(self) -> None:
        """Wrapper function to read file based off ``self.format``.
        :raises ValueError: If ``self.format`` is unsupported.
        """
        try:
            method_name = self._FORMATS[self.format]
            reader = getattr(self, method_name)
            reader()
        except KeyError:
            supported = ", ".join(sorted(self._FORMATS))
            raise ValueError(
                f"Unsupported format '{self.format}'; supported formats are: {supported}"
            ) from None


def main() -> None:
    kcuf3spec = conquest_species({1: "K", 2: "Cu", 3: "Cu", 4: "F"})
    print(kcuf3spec)
    cq = read_coords("tests/data/files/kcuf3.cell", species=kcuf3spec, format="cell")
    print(cq.coords.lattice_vectors)
    write_coords("tests/data/files/kcuf3.dat", cq.coords, format="cq", convert_to="bohr")


if __name__ == "__main__":
    main()

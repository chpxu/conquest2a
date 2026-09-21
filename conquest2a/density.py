import sys

if sys.version_info >= (3, 9):
    from typing import overload
    from collections.abc import Mapping, MutableMapping
else:
    from typing_extensions import MutableMapping, Mapping, overload

from copy import deepcopy
from collections.abc import Sequence
from typing import Any, Literal
from pathlib import Path
import os
from os.path import abspath, basename
import re
import numpy as np
from ase.atoms import Atoms
from ase.io.cube import read_cube_data
from ase.units import Bohr
from scipy.ndimage import map_coordinates
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import matplotlib.pyplot as plt
from conquest2a._types import INT_ARRAY, REAL_ARRAY
from conquest2a.conquest import processor_base

# Some default element colours but can be overridden
_ELEMENT_COLOURS: dict[str, str] = {
    # Alkali metals
    "Li": "#cc80ff",
    "Na": "#ab5cf2",
    "K": "#8f40d4",
    "Rb": "#702eb0",
    "Cs": "#57178f",
    "Fr": "#420066",
    # Others
    "La": "#70d4ff",
    # Alkaline earth metals
    "Be": "#c2ff00",
    "Mg": "#8aff00",
    "Ca": "#3dff00",
    "Sr": "#00ff00",
    "Ba": "#00c900",
    "Ra": "#007d00",
    # 3d transition metals
    "Sc": "#e6e6e6",
    "Ti": "#bfc2c7",
    "V": "#a6a6ab",
    "Cr": "#8a99c7",
    "Mn": "#9c7ac7",
    "Fe": "#e06633",
    "Co": "#f090a0",
    "Ni": "#50d050",
    "Cu": "#c88033",
    "Zn": "#7d80b0",
    # 4d transition metals
    "Y": "#94ffff",
    "Zr": "#94e0e0",
    "Nb": "#73c2c9",
    "Mo": "#54b5b5",
    "Tc": "#3b9e9e",
    "Ru": "#248f8f",
    "Rh": "#0a7d8c",
    "Pd": "#006985",
    "Ag": "#c0c0c0",
    "Cd": "#ffd98f",
    # 5d transition metals
    "Hf": "#4dc2ff",
    "Ta": "#4da6ff",
    "W": "#2194d6",
    "Re": "#267dab",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#d0d0e0",
    "Au": "#ffd123",
    "Hg": "#b8b8d0",
    # Post-transition metals
    "Al": "#bfa6a6",
    "Ga": "#c28f8f",
    "In": "#a67573",
    "Sn": "#668080",
    "Tl": "#a6544d",
    "Pb": "#575961",
    "Bi": "#9e4fb5",
    # Metalloids
    "B": "#ffb5b5",
    "Si": "#f0c8a0",
    "Ge": "#668f8f",
    "As": "#bd80e3",
    "Sb": "#9e63b5",
    "Te": "#d47a00",
    "Po": "#ab5c00",
    # Halogens
    "F": "#90e050",
    "Cl": "#1ff01f",
    "Br": "#a62929",
    "I": "#940094",
    # Reactive non-metals
    "H": "#ffffff",
    "C": "#404040",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "P": "#ff8000",
    "S": "#ffff30",
    "Se": "#ffa100",
    # Noble gases
    "He": "#d9ffff",
    "Ne": "#b3e3f5",
    "Ar": "#80d1e3",
    "Kr": "#5cb8d1",
    "Xe": "#429eb0",
    "Rn": "#428296",
}


class density(processor_base):
    """Process volumetric density data

    CONQUEST can output any number of ``.cube`` files, e.g.:
        - ``chden_up.cube``/``chden_dn.cube`` for a spin polarised calculation
        - a single cube file, e.g. ``chden.cube`` for unpolarised calculations
        - Band densities

    This class uses ASE's ``.cube`` file processor to get atoms and data.
    It is therefore suitable for generic ``.cube`` file processing.

    An arbitrary number of ``.cube`` files can be supplied via ``paths``. If more than
    one path is given, ``operations`` is a string describing how each subsequent
    file is combined with the running total, applied left to right, one character
    per file after the first, much like VESTA:

    - ``"+"`` add
    - ``"-"`` subtract
    - ``"x"`` multiply
    - ``"/"`` divide

    Example: for ``paths=[a, b, c, d]`` and ``operations="+/-"``, the resulting data is
    ``((a + b) / c) - d``.

    :param hkl: The :math:`hkl` slice of the crystal to plot charge densities in. This will accept floats if you really want a specific plane.
    :type hkl: :ref:`REAL ARRAY <types>`
    :param offset: The :math:`hkl` direction defines a family of planes. Use ``offset`` to select which one (i.e. wherein the unit cell).
    :type offset: ``float``
    :param paths: Path(s) to one or more charge density (``.cube``) files. At least
        one path must be provided.
    :type paths: ``Sequence[str]``
    :param operations: How to combine each file in ``paths`` with the running total of
        those before it, as a string with one of ``"+"``, ``"-"``, ``"x"``, ``"/"`` per
        character. Must have exactly ``len(paths) - 1`` characters. Not required (and
        ignored) when only one path is given, defaults to ``None``.
    :type operations: ``str | None``, optional
    :param units: The working units. The density data from CONQUEST is in spatial units of Bohr, and is the default.
    :type offset: ``Literal["ang", "bohr"]``
    :raises ValueError: If all Miller indices are 0: cannot slice through the origin.
    :raises ValueError: If no paths are provided.
    :raises ValueError: If the length of ``operations`` does not match ``len(paths) - 1``.
    :raises ValueError: If ``operations`` contains a character other than ``"+-x/"``.
    :raises ValueError: If cube files being combined have mismatched grid shapes.
    :raises ZeroDivisionError: If a ``"/"`` operation would divide by a zero-valued voxel.
    """

    def __init__(
        self,
        hkl: tuple[float, float, float],
        offset: float,
        paths: Sequence[str],
        operations: str | None = None,
        units: Literal["ang", "bohr"] = "bohr",
    ) -> None:
        self._VALID_OPERATION_CHARS: str = "+-x/"
        if hkl[0] == 0 and hkl[1] == 0 and hkl[2] == 0:
            raise ValueError("Miller indices (h k l) cannot all be zero.")
        if len(paths) == 0:
            raise ValueError("At least one cube file path must be provided.")

        operations = operations if operations is not None else ""
        if len(paths) > 1 and len(operations) != len(paths) - 1:
            raise ValueError(
                f"Expected {len(paths) - 1} operation character(s) to combine "
                + f"{len(paths)} cube files, got {len(operations)!r}."
            )
        for op in operations:
            if op not in self._VALID_OPERATION_CHARS:
                raise ValueError(
                    f"Unknown operation {op!r}; expected one of {list(self._VALID_OPERATION_CHARS)}."
                )

        self.hkl: REAL_ARRAY = np.array(hkl)
        self.offset: float = offset
        self.paths: list[str] = list(paths)
        self.operations: str = operations
        self.units: Literal["ang", "bohr"] = units
        super().__init__(path=self.paths[0])
        self._length_scale: float = 1.0 if units == "bohr" else Bohr
        self._density_scale: float = 1.0 if units == "bohr" else 1.0 / (Bohr**3)

        #: The (data, atoms) pair loaded from each cube file in ``paths``, in order.
        self.densities: list[tuple[REAL_ARRAY, Atoms]] = [self.load_cube(p) for p in self.paths]
        self.atoms: Atoms = self.densities[0][1]
        self.cell: REAL_ARRAY = np.array(self.atoms.get_cell()) / Bohr  # ASE cell information

        data: REAL_ARRAY = np.array(self.densities[0][0], copy=True)
        for (other_data, _), op in zip(self.densities[1:], self.operations):
            other_data = np.asarray(other_data)
            if other_data.shape != data.shape:
                raise ValueError(
                    "Cannot combine cube files with mismatched grid shapes: "
                    + f"{data.shape} vs {other_data.shape}."
                )
            if op == "/" and np.any(other_data == 0):
                raise ZeroDivisionError("Division by a cube file that contains zero-valued voxels.")
            if op == "+":
                data = np.add(data, other_data)
            elif op == "-":
                data = np.subtract(data, other_data)
            elif op == "x":
                data = np.multiply(data, other_data)
            else:  # op == "/"
                data = np.divide(data, other_data)
        self.data: REAL_ARRAY = data

    @overload
    def _to_bohr_length(self, value: float) -> float: ...
    @overload
    def _to_bohr_length(self, value: REAL_ARRAY) -> REAL_ARRAY: ...
    def _to_bohr_length(self, value: REAL_ARRAY | float) -> REAL_ARRAY | float:
        """Convert a length in the configured units back to Bohr for internal use."""
        return value / self._length_scale

    @overload
    def _to_output_length(self, value: float) -> float: ...
    @overload
    def _to_output_length(self, value: REAL_ARRAY) -> REAL_ARRAY: ...
    def _to_output_length(self, value: REAL_ARRAY | float) -> REAL_ARRAY | float:
        """Convert an internal Bohr length to the configured output units."""
        return value * self._length_scale

    def load_cube(self, filename: str) -> tuple[Any, Atoms]:
        self.resolve_path(filename=filename)
        with open(filename, "r", encoding="utf-8") as fh:
            cube: tuple[REAL_ARRAY, Atoms] = read_cube_data(fh)
        fh.close()
        density_data = cube[0]
        atoms_data = cube[1]
        return density_data, atoms_data

    def inplane_basis(
        self,
    ) -> tuple[REAL_ARRAY, REAL_ARRAY, REAL_ARRAY]:
        """Return two orthonormal Cartesian vectors :math:`(v_1, v_2)` spanning the :math:`[hkl]` plane
        and the unit plane-normal :math:`\\hat{n}`.

        1. The plane normal in Cartesian space is  :math:`n = ha^* + kb^* + lc^*`
        where :math:`a^*, b^*, c^*` are the reciprocal lattice vectors (rows of inv(cell)^T).
        2. Two fractional-space null vectors of :math:`[hkl]` are found via SVD.
        3. Those are converted to Cartesian, then Gram-Schmidt orthonormalised so the axes are perpendicular in real-space.

        """
        cell: REAL_ARRAY = self.cell
        recip: REAL_ARRAY = (np.linalg.inv(cell).T).astype(np.float64)
        n_cart = self.hkl[0] * recip[0] + self.hkl[1] * recip[1] + self.hkl[2] * recip[2]
        n_hat: REAL_ARRAY = n_cart / np.linalg.norm(n_cart)
        _, _, Vt = np.linalg.svd(self.hkl.reshape(1, 3))
        u1_frac, u2_frac = Vt[1], Vt[2]
        u1 = u1_frac @ cell
        u2 = u2_frac @ cell

        v1 = u1 / np.linalg.norm(u1)
        v2 = u2 - np.dot(u2, v1) * v1
        v2 /= np.linalg.norm(v2)

        return v1, v2, n_hat

    def plane_origin(self) -> REAL_ARRAY:
        """Return a Cartesian point lying on the plane  :math:`ha + kb + lc =` ``offset``.

        ``offset`` is a dimensionless fractional intercept (0-1 spans one
        interplanar period). Pick the simplest fractional coordinate
        satisfying the plane equation.

        :returns: Origin of the slice.
        :rtype: :ref:`REAL ARRAY <types>`
        """
        n: REAL_ARRAY = self.hkl / (self.hkl @ self.hkl)  # normal direction in fractional space
        centre: REAL_ARRAY = np.array([0.5, 0.5, 0.5])
        frac: REAL_ARRAY = centre - (centre @ self.hkl) * n + self.offset * n
        return frac @ self.cell

    def inplane_range(self, v1: REAL_ARRAY, v2: REAL_ARRAY) -> tuple[float, float]:
        length_1: float = sum(abs(self.cell[i] @ v1) for i in range(3))
        length_2: float = sum(abs(self.cell[i] @ v2) for i in range(3))
        return length_1, length_2

    def extract_slice(
        self,
        n_points: int = 1000,
        interp_order: int = 5,
        shift: tuple[float, float] = (0.0, 0.0),
        window_repeat: float | tuple[float, float] = 1.0,
        rotation_deg: float = 0.0,
    ) -> tuple[Any, REAL_ARRAY, REAL_ARRAY, REAL_ARRAY, REAL_ARRAY, REAL_ARRAY]:
        """Sample the charge density on the :math:`[hkl]` plane at fractional ``offset``.

        For each point on a 2D :math:`(t_1, t_2)` grid centred on the plane origin:

        1. Compute its Cartesian position:
            :math:`p = \\text{origin} + t_1 v_1 + t_2 v_2`
        2. Convert to fractional coordinates: ``s = p @ inv(cell)``
        3. Wrap into the unit cell to enforce periodic boundaries.
        4. Map fractional -> voxel index and interpolate via ``scipy.ndimage.map_coordinates``.

        :param n_points: Number of points to sample on the slice, defaults to 1000
        :type n_points: ``int``, optional
        :param interp_order: The polynomial degree for interpolation, defaults to 5
        :type interp_order: ``int``, optional
        :param shift: Vector to shift the origin on the plane by
        :type shift: ``tuple[float, float]``, optional
        :returns: A tuple containing:

            - **density** (:ref:`REAL ARRAY <types>`) -- Charge density in :math:`e/a_0^3`, shape ``(n_points, n_points)``
            - **v1** (:ref:`REAL ARRAY <types>`) -- First unit vector spanning the :math:`[hkl]` plane
            - **v2** (:ref:`REAL ARRAY <types>`) -- Second unit vector spanning the :math:`[hkl]` plane
            - **t1** (:ref:`REAL ARRAY <types>`) -- Scalar grid along :math:`v_1`
            - **t2** (:ref:`REAL ARRAY <types>`) -- Scalar grid along :math:`v_2`
            - **origin** (``float``) -- Cartesian slice origin in Bohr
        """
        if isinstance(window_repeat, (int, float)):
            window_repeat = (float(window_repeat), float(window_repeat))
        shift_bohr = (self._to_bohr_length(shift[0]), self._to_bohr_length(shift[1]))

        v1, v2, _ = self.inplane_basis()
        origin = self.plane_origin() + shift_bohr[0] * v1 + shift_bohr[1] * v2
        if abs(rotation_deg) > 1e-3:
            v1, v2 = self._rotate_inplane_vectors(v1, v2, rotation_deg)
        length_1, length_2 = self.inplane_range(v1, v2)
        length_1 *= window_repeat[0]
        length_2 *= window_repeat[1]
        t1 = np.linspace(-length_1, length_1, n_points)
        t2 = np.linspace(-length_2, length_2, n_points)
        grid_1, grid_2 = np.meshgrid(t1, t2, indexing="ij")
        pts_cart = origin + grid_1[..., None] * v1 + grid_2[..., None] * v2
        pts_frac = (pts_cart @ np.linalg.inv(self.cell)) % 1.0
        vox = pts_frac * self.data.shape
        data_padded = np.pad(self.data, 2 * interp_order + 1, mode="wrap")

        coords = vox.reshape(-1, 3).T + 2 * interp_order + 1
        density: REAL_ARRAY = map_coordinates(
            data_padded, coords, order=interp_order, mode="nearest"
        ).reshape(n_points, n_points)

        density = density * self._density_scale
        t1_out: REAL_ARRAY = self._to_output_length(t1)
        t2_out: REAL_ARRAY = self._to_output_length(t2)
        origin_out: REAL_ARRAY = self._to_output_length(origin)

        return density, v1, v2, t1_out, t2_out, origin_out

    @staticmethod
    def _rotate_inplane_vectors(
        v1: REAL_ARRAY, v2: REAL_ARRAY, angle_deg: float | None = None
    ) -> tuple[REAL_ARRAY, REAL_ARRAY]:
        if angle_deg is None or (abs(angle_deg) < 1e-3):
            return v1, v2
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        v1_rot = c * v1 + s * v2
        v2_rot = -s * v1 + c * v2
        return v1_rot, v2_rot

    def shift_onto_atom(self, atom_number: int) -> tuple[float, float]:
        """Produce the vector shift needed to set an atom to be at the origin of the plane.

        :param atom_number: The atom number to shift to
        :type atom_number: int
        :return: 2D vector to shift by
        :rtype: tuple[float, float]
        """
        v1, v2, _ = self.inplane_basis()
        origin = self.plane_origin()
        pos = self.atoms.get_positions()[atom_number] / Bohr
        disp = pos - origin
        disp_frac = disp @ np.linalg.inv(self.cell)
        disp_frac -= np.around(disp_frac)
        disp2 = disp_frac @ self.cell
        return self._to_output_length(disp2 @ v1), self._to_output_length(disp2 @ v2)

    def project_atoms(
        self,
        v1: REAL_ARRAY,
        v2: REAL_ARRAY,
        n_hat: REAL_ARRAY,
        origin: REAL_ARRAY,
        thickness: float = 1.0,
        indices: Sequence[int] | None = None,
        repeat: int | tuple[int, int, int] = 1,
    ) -> tuple[REAL_ARRAY, REAL_ARRAY, list[str]]:
        """Project atoms within ``thickness`` of the slice plane onto :math:`(v_1, v_2)` axes.

        :param v1: First unit vector spanning the :math:`[hkl]` plane
        :type v1: :ref:`REAL ARRAY <types>`
        :param v2: Second unit vector spanning the :math:`[hkl]` plane
        :type v2: :ref:`REAL ARRAY <types>`
        :param n_hat: Unit vector defining the slice
        :type n_hat: :ref:`REAL ARRAY <types>`
        :param origin: The origin of the slice
        :type origin: :ref:`REAL ARRAY <types>`
        :param thickness: The Cartesian distance perpendicular to the plane to consider atoms as lying on the slice, defaults to 1.0 Bohr.
        :type thickness: ``float``, optional
        :param indices: The atom indices to allow to be shown, default None (show all that are found)
        :type indices: ``Sequence[int] | None``, optional
        :returns: A tuple containing:

            - **t1_proj** (:ref:`REAL ARRAY <types>`) - Positions of label along :math:`v_1`
            - **t2_proj** (:ref:`REAL ARRAY <types>`) - Positions of label along :math:`v_2`
            - **syms** (``list[str]``) - List of atom labels
        """
        if isinstance(repeat, int):
            repeat = (repeat, repeat, repeat)
        n1, n2, n3 = repeat
        origin_bohr = self._to_bohr_length(origin)
        thickness_bohr = self._to_bohr_length(thickness)

        positions = self.atoms.get_positions() / Bohr
        symbols: Sequence[str] = self.atoms.get_chemical_symbols()
        considered = range(len(symbols)) if indices is None else list(indices)
        inv_cell: REAL_ARRAY = np.linalg.inv(self.cell).astype(float)
        base_frac = positions @ inv_cell
        t1_all: list[float] = []
        t2_all: list[float] = []
        syms_all: list[str] = []

        for i in range(-n1, n1 + 1):
            for j in range(-n2, n2 + 1):
                for k in range(-n3, n3 + 1):
                    cart = (base_frac + np.array([i, j, k])) @ self.cell
                    disp = cart - origin_bohr
                    dist_norm = disp @ n_hat
                    within_plane = np.abs(dist_norm) < thickness_bohr
                    for idx in considered:
                        if not within_plane[idx]:
                            continue
                        t1_all.append(float(disp[idx] @ v1))
                        t2_all.append(float(disp[idx] @ v2))
                        syms_all.append(symbols[idx])
        t1_out: REAL_ARRAY = self._to_output_length(np.array(t1_all))
        t2_out: REAL_ARRAY = self._to_output_length(np.array(t2_all))

        return np.array(t1_out), np.array(t2_out), syms_all


class chden(processor_base):
    """Class which looks for charge density files in a directory.

    :param directory: Path to directory containing band density files
    :type directory: ``str``
    :param hkl: The :math:`hkl` slice of the crystal to plot charge densities in.
    :type hkl: :ref:`INT ARRAY <types>`
    :param offset: The :math:`hkl` direction defines a family of planes. Use ``offset`` to select which one (i.e. wherein the unit cell).
    :type offset: ``float``
    :param operations: How to combine each file in ``paths`` with the running total of
        those before it, as a string with one of ``"+"``, ``"-"``, ``"x"``, ``"/"`` per
        character. Must have exactly ``len(paths) - 1`` characters. Not required (and
        ignored) when only one path is given, defaults to ``None``.
    :type operations: ``str | None``, optional
    :param charge_stub: the prefix of charge density files in CONQUEST. defaults to ``chden``.
    :type charge_stub: ``str``, optional
    :param spin: Whether to search for spin-polarised (True) or unpolarised charge density files. Default ``False``.
    :type spin: ``bool``
    :param units: The working units. The density data from CONQUEST is in spatial units of Bohr, and is the default.
    :type units: ``Literal["ang", "bohr"]``
    """

    def __init__(
        self,
        directory: str,
        hkl: tuple[int, int, int],
        offset: float,
        operations: str,
        charge_stub: str = "chden",
        spin: bool = False,
        units: Literal["ang", "bohr"] = "bohr",
    ) -> None:
        self.all_dens_files: list[str] = []
        self.filtered_dens_files: list[str] = []
        self.directory: str = directory
        self.charge_stub: str = charge_stub
        self._charge_regex: re.Pattern[str] = re.compile(rf"{charge_stub}\.cube")
        self._charge_spin_regex: re.Pattern[str] = re.compile(rf"{charge_stub}_(up|dn)\.cube")
        self.spin: int | None = spin
        self.units: Literal["ang", "bohr"] = units
        super().__init__(path=directory)
        self.locate_chden_files()
        self._filter_chden_files()
        self.density: density = density(
            hkl, offset, paths=self.filtered_dens_files, operations=operations, units=units
        )

    def locate_chden_files(self) -> list[str]:
        """Gets the paths to all charge density files.

        For spin-unpolarised calculation, CONQUEST will output just ``chden.cube``.
        For spin-polarised calculation, CONQUEST will output ``chden_up.cube`` and ``chden_dn.cube``.

        This output may change if the user has set ``Process.ChargeStub`` in ``Conquest_input`` which defaults to ``chden``. Since reading ``Conquest_input`` directly is unsupported, just assume the user will pass in any stub.

        :raises FileNotFoundError: If ``self.directory`` does not exist.
        :return: List of all paths to band density files.
        :rtype: ``list[str]``
        """
        abs_path: Path = Path(abspath(self.directory))
        if not abs_path.is_dir():
            raise FileNotFoundError(f'Directory specified: "{abs_path}", does not exist.')

        all_file_list: list[str] = []
        file_list: list[str] = []
        if len(self.all_dens_files) > 0:
            # Reset files array, i.e. if changing directory
            self.all_dens_files = []
        for _, _, files in os.walk(abs_path, topdown=True):
            file_list = files
            break
        for filename in file_list:
            if re.match(self._charge_regex, filename) or re.match(
                self._charge_spin_regex, filename
            ):
                all_file_list.append(filename)

        for file in all_file_list:
            full_path = f"{abs_path}/{file}"
            self.resolve_path(filename=full_path)
            self.all_dens_files.append(full_path)
        return self.all_dens_files

    def _filter_chden_files(self) -> list[str]:
        """Method to choose the spin-unpolarised .cube file or the spin-polarised files.

        :return: List of paths to selected charge density files.
        :rtype: ``list[str]``
        """
        pattern: re.Pattern[str]
        if self.spin:
            pattern = self._charge_spin_regex
        else:
            pattern = self._charge_regex

        self.filtered_dens_files = [path for path in self.all_dens_files if pattern.search(path)]
        return self.filtered_dens_files


class bandden(processor_base):
    """Class which looks for band density files in a directory.

    :param directory: Path to directory containing band density files
    :type directory: ``str``
    :param hkl: The :math:`hkl` slice of the crystal to plot charge densities in.
    :type hkl: :ref:`INT ARRAY <types>`
    :param offset: The :math:`hkl` direction defines a family of planes. Use ``offset`` to select which one (i.e. wherein the unit cell).
    :type offset: ``float``
    :param operations: How to combine each file in ``paths`` with the running total of
        those before it, as a string with one of ``"+"``, ``"-"``, ``"x"``, ``"/"`` per
        character. Must have exactly ``len(paths) - 1`` characters. Not required (and
        ignored) when only one path is given, defaults to ``None``.
    :type operations: ``str | None``, optional
    :param charge_stub: the prefix of charge density files in CONQUEST. defaults to ``chden``.
    :type operations: ``str``, optional
    :param bands: The specific bands to filter for
    :type bands: ``list[int]``
    :param spin: Whether to search for spin-up (``1``), spin-down (``2``) or any spin band density files (``None``). Default ``None``.
    :type spin: ``int | None``
    :param kpts: The specific :math:`k`-points to filter for. If ``None`` (default), just looks for band density files which were produced from summing over :math:`k`-points.
    :type kpts: ``list[int]``
    :param units: The working units. The density data from CONQUEST is in spatial units of Bohr, and is the default.
    :type units: ``Literal["ang", "bohr"]``
    """

    def __init__(
        self,
        directory: str,
        hkl: tuple[int, int, int],
        offset: float,
        operations: str,
        bands: list[int],
        spin: int | None = None,
        kpts: list[int] | None = None,
        units: Literal["ang", "bohr"] = "bohr",
    ) -> None:
        self.all_dens_files: list[str] = []
        self.filtered_dens_files: list[str] = []
        self.bands: list[int] = []
        self._bykpt_regex: re.Pattern[str] = re.compile(
            r"Band([0-9]{6})den_kp([0-9]{3})S(\d)\.cube"
        )
        self._sumkpt_regex: re.Pattern[str] = re.compile(r"Band([0-9]{6})den_totS(\d)\.cube")
        self.units: str = units
        self.directory: str = directory
        super().__init__(path=directory)
        self.locate_banddens_files()
        self._filter_banddens_files(bands, spin=spin, kpt=kpts)
        self.density: density = density(
            hkl, offset, paths=self.filtered_dens_files, operations=operations, units=units
        )

    def locate_banddens_files(self) -> list[str]:
        """Gets the paths to all band density files, sorted by band number, :math:`k` point then spin.

        :raises FileNotFoundError: If ``self.directory`` does not exist.
        :return: List of all paths to band density files.
        :rtype: ``list[str]``
        """
        abs_path: Path = Path(abspath(self.directory))
        if not abs_path.is_dir():
            raise FileNotFoundError(f'Directory specified: "{abs_path}", does not exist.')

        all_file_list: list[str] = []
        file_list: list[str] = []
        if len(self.all_dens_files) > 0:
            # Reset files array, i.e. if changing directory
            self.all_dens_files = []
            self.bands = []
        for _, _, files in os.walk(abs_path, topdown=True):
            file_list = files
            break
        for filename in file_list:
            if re.match(self._bykpt_regex, filename) or re.match(self._sumkpt_regex, filename):
                all_file_list.append(filename)
                match: re.Match[str] | None = re.search(r"Band([0-9]{6})", filename)
                if match:
                    self.bands.append(int(match.group(1)))

        self.bands = sorted(set(self.bands))
        all_file_list = sorted(all_file_list)
        for file in all_file_list:
            full_path = f"{abs_path}/{file}"
            self.resolve_path(filename=full_path)
            self.all_dens_files.append(full_path)
        return self.all_dens_files

    def _filter_banddens_files(
        self, band: list[int], spin: int | None = None, kpt: list[int] | None = None
    ) -> list[str]:
        """Method to filter band densities by band number, and optionally spin and k-point.

        :param bands: The specific bands to filter for
        :type bands: ``list[int]``
        :param spin: Whether to search for spin-up (``1``), spin-down (``2``) or any spin band density files (``None``). Default ``None``.
        :type spin: ``int | None``
        :param kpts: The specific :math:`k`-points to filter for. If ``None`` (default), just looks for band density files which were produced from summing over :math:`k`-points.
        :type kpts: ``list[int]``
        :raises ValueError: If ``band`` index supplied is not found.
        :raises ValueError: If ``spin`` index supplied is less than 1.
        :raises ValueError: If ``kpt`` index supplied is less than 1.
        :return: List of paths to band density files matching the given filters.
        :rtype: ``list[str]``
        """
        if spin is not None and spin < 1:
            raise ValueError("Spin index cannot be less than 1.")
        if kpt is not None and any(k < 1 for k in kpt):
            raise ValueError("k-point index cannot be less than 1.")
        if any(band not in self.bands for band in self.bands):
            raise ValueError(f"Selected band(s) {band} is not in the directory")

        band_set: set[int] = set(band)
        kpt_set: set[int] | None = None if kpt is None else set(kpt)
        # Filter by kpt or kpt sum
        pattern: re.Pattern[str] = self._bykpt_regex if kpt is not None else self._sumkpt_regex
        compiled: re.Pattern[str] = re.compile(pattern)

        category_matches: list[tuple[str, re.Match[str]]] = []
        for filepath in self.all_dens_files:
            match: re.Match[str] | None = compiled.search(basename(filepath))
            if match is None:
                continue
            if kpt_set is not None and int(match.group(2)) not in kpt_set:
                continue
            category_matches.append((filepath, match))

        # band number
        band_matches: list[tuple[str, re.Match[str]]] = [
            (filepath, match)
            for filepath, match in category_matches
            if int(match.group(1)) in band_set
        ]

        # If spin is none, just return the remaining filtered files
        if spin is None:
            self.filtered_dens_files = [filepath for filepath, _ in band_matches]
            return self.filtered_dens_files

        # Otherwise filter for specific files with matching spin from remaining files
        spin_group: int = 3 if kpt is not None else 2
        self.filtered_dens_files = [
            filepath for filepath, match in band_matches if int(match.group(spin_group)) == spin
        ]
        return self.filtered_dens_files


class plot_density:
    """Helper class to slice, analyse, and plot slices of volumetric density data.

    Takes a :class:`density` instance directly, or with
    :class:`chden` or :class:`bandden` that exposes a ``self.density`` attribute

    :param source: The density data to plot.
    :type source: :class:`density` | :class:`chden` | :class:`bandden`
    :param show_atoms: Whether to show atom labels on the plot, defaults to False.
    :type show_atoms: ``bool``, optional
    :param extension: File extension used for auto-generated filenames (leading
        "." optional), defaults to "png".
    :type extension: ``str``, optional
    :param cbar_label: Label for the colour bar. If not given, a sensible default
        is chosen based on the type of ``source`` - charge density and band
        density are conventionally expressed in different units.
    :type cbar_label: ``str | None``, optional
    """

    _DEFAULT_CBAR_LABELS: dict[type, dict[str, str]] = {
        chden: {
            "bohr": r"$\rho$  [$e\,a_0^{-3}$]",
            "ang": r"$\rho$  [$e\,\mathrm{\AA}^{-3}$]",
        },
        bandden: {
            "bohr": r"$\rho$ [Bohr$^{-3}$]",
            "ang": r"$\rho$ [$\mathrm{\AA}^{-3}$]",
        },
        density: {
            "bohr": r"$\rho$  [arb. units]",
            "ang": r"$\rho$  [arb. units]",
        },
    }

    def __init__(
        self,
        source: density | chden | bandden,
        show_atoms: bool = False,
        extension: str = "png",
        cbar_label: str | None = None,
    ) -> None:
        self.density: density = source.density if isinstance(source, (chden, bandden)) else source
        self.show_atoms: bool = show_atoms
        self.extension: str = extension.lstrip(".")
        self.cbar_label: str = (
            cbar_label
            or self._DEFAULT_CBAR_LABELS.get(type(source), self._DEFAULT_CBAR_LABELS[density])[
                self.density.units
            ]
        )

    def _length_unit_suffix(self) -> str:
        return r"$\,[a_0]$" if self.density.units == "bohr" else r"$\,[\mathrm{\AA}]$"

    def _miller_str(self, idx: int) -> str:
        if idx >= 0:
            return str(idx)
        return rf"$\overline{{{abs(idx)}}}$"

    def _vec_str(self, v: REAL_ARRAY) -> str:
        parts: list[str] = [self._miller_str(x) for x in v]
        return f"Position along [{','.join(parts)}]"

    def _default_filename(self) -> str:
        hkl = self.density.hkl
        filename = f"{hkl[0]}{hkl[1]}{hkl[2]}_{self.density.offset:.3f}"
        if self.density.operations:
            filename += f"_{self.density.operations}"
        return f"{filename}.{self.extension}"

    def _construct_atom_legend(
        self,
        ax: Any,
        syms: Sequence[str],
        atom_size: float,
        atom_bgcolor: Mapping[str, Mapping[str, Any]] | None = None,
        atom_edgecolor: Mapping[str, Mapping[str, Any]] | None = None,
        linestyle_map: Mapping[str, str] | None = None,
        default_linestyle: str = "solid",
        legend_linewidth: float = 1.0,
    ) -> list[Any]:
        legend_handles = []
        lin_map = linestyle_map or {}
        for sym in syms:
            color = (
                _ELEMENT_COLOURS.get(sym, "#00000000")
                if atom_bgcolor is None
                else atom_bgcolor[sym]
            )
            edgecolor = (
                _ELEMENT_COLOURS.get(sym, "#000000")
                if atom_edgecolor is None
                else atom_edgecolor[sym]
            )
            ls = lin_map.get(sym, default_linestyle)

            proxy = ax.scatter(
                [],
                [],
                s=atom_size,
                color=color,
                edgecolors=edgecolor,
                linestyle=ls,
                linewidths=legend_linewidth,
                label=sym,
            )
            legend_handles.append(proxy)

        return legend_handles

    def _plot_atoms(
        self,
        ax: Any,
        atom_data: tuple[REAL_ARRAY, REAL_ARRAY, list[str]],
        atom_symbols: Sequence[str] | None,
        label_atoms: bool = True,
        atom_size: float = 160,
        atom_fontsize: float = 5,
        atom_fontcolor: Mapping[str, str] | None = None,
        atom_bgcolor: Mapping[str, str] | None = None,
        atom_edgecolor: Mapping[str, str] | None = None,
        atom_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        def _atom_text(
            ax: Any,
            pos_x: REAL_ARRAY,
            pos_y: REAL_ARRAY,
            sym: str,
            fontsize: float = 5.0,
            color: str = "black",
            ha: str = "center",
            va: str = "center",
            **kwargs: Any,
        ) -> None:
            ax.text(
                pos_x,
                pos_y,
                sym,
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=color,
                zorder=6,
                fontweight="bold",
                clip_on=True,
                **kwargs,
            )

        t1s, t2s, syms = atom_data
        # Swap t1, t2 as coordinates are inverted from imshow
        for t1a, t2a, sym in zip(t2s, t1s, syms):
            # t1a, t2a = t2a, t1a
            if atom_symbols is not None and sym not in atom_symbols:
                continue
            bgcolor = "#000000"

            edgecolor: str | None = "#000000"
            if atom_bgcolor is None:
                bgcolor = _ELEMENT_COLOURS.get(sym, "#00000000")
            elif sym in atom_bgcolor.keys():
                bgcolor = atom_bgcolor[sym]

            if atom_edgecolor is None:
                edgecolor = None
            elif sym in atom_edgecolor.keys():
                edgecolor = atom_edgecolor[sym]

            color = bgcolor

            scatter_args: Mapping[str, Any] = {
                "s": atom_size,
                "color": color,
                "edgecolors": edgecolor,
                "zorder": 5,
                "clip_on": True,
            }
            per_atom_kwargs = (atom_kwargs or {}).get(sym, {})
            merged_args: Mapping[str, Any] = {**scatter_args, **per_atom_kwargs}
            ax.scatter(t1a, t2a, **merged_args)

            if label_atoms:
                textcolour = "black"
                if atom_fontcolor is not None and sym in atom_fontcolor.keys():
                    textcolour = atom_fontcolor[sym]
                _atom_text(
                    ax=ax,
                    pos_x=t1a,
                    pos_y=t2a,
                    sym=sym,
                    fontsize=atom_fontsize,
                    color=textcolour,
                    **kwargs,
                )

    def plot_slice(
        self,
        density_grid: REAL_ARRAY,
        t1: REAL_ARRAY,
        t2: REAL_ARRAY,
        v1: REAL_ARRAY,
        v2: REAL_ARRAY,
        atom_data: (
            tuple[REAL_ARRAY, REAL_ARRAY, list[str]] | None
        ) = None,  # (t1s, t2s, symbols) tuple or None
        log_scale: bool = False,
        vmin: float | None = 0.0,
        vmax: float | None = None,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        normalise: bool = False,
        owns_figure: bool = False,
        xlabel: str | bool | None = None,
        ylabel: str | bool | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        tick_kwargs: Mapping[str, Any] | None = None,
        cmap: str = "Blues",
        cbar_kwargs: Mapping[str, Any] | None = None,
        atom_symbols: Sequence[str] | None = None,
        label_atoms: bool = True,
        atom_size: float = 50,
        atom_fontsize: float = 8,
        atom_fontcolor: Mapping[str, str] | None = None,
        atom_bgcolor: Mapping[str, str] | None = None,
        atom_edgecolor: Mapping[str, str] | None = None,
        atom_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        **imshow_kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        """Create the plot and save it to disk.

        :param density_grid: Sliced density data, shape ``(n_points, n_points)``.
        :type density_grid: :ref:`REAL ARRAY <types>`
        :param t1: Scalar grid along :math:`v_1`
        :type t1: :ref:`REAL ARRAY <types>`
        :param t2: Scalar grid along :math:`v_2`
        :type t2:  :ref:`REAL ARRAY <types>`
        :param v1: First unit vector spanning the :math:`[hkl]` plane
        :type v1: :ref:`REAL ARRAY <types>`
        :param v2: Second unit vector spanning the :math:`[hkl]` plane
        :type v2: :ref:`REAL ARRAY <types>`
        :param atom_data: Data for atom labels, see ``density.project_atoms``, defaults to None
        :type atom_data: ``tuple[REAL_ARRAY, REAL_ARRAY, list[str]] | None``, optional
        :param log_scale: Whether to plot the density on a base-10 logarithmic scale -
            useful for revealing details without colour clipping, defaults to False
        :type log_scale: ``bool``, optional

        For extra params, please see ``plot_density.run()`` below.
        :returns: Tuple containing figure, axes and imshow instances created
        :rtype: ``tuple[Any, Any, Any]``
        """
        dx = float(np.mean(np.diff(t1)))
        dy = float(np.mean(np.diff(t2)))
        # Account for imshow centering
        extent = (
            t1[0] - dx / 2,
            t1[-1] + dx / 2,
            t2[0] - dy / 2,
            t2[-1] + dy / 2,
        )

        l1 = (xlim[1] - xlim[0]) if xlim is not None else (t1[-1] - t1[0])
        l2 = (ylim[1] - ylim[0]) if ylim is not None else (t2[-1] - t2[0])
        if normalise:
            density_grid = (density_grid - np.min(density_grid)) / (
                np.max(density_grid) - np.min(density_grid)
            )

        # If this data is meant to be part of some larger figure then we should reference external figure
        # Otherwise assume user wants a standalone plot and make and save own figure instance
        if not owns_figure and ax is not None:
            fig = ax.figure
        else:
            if figsize is None:
                fig_w = 5.0
                figsize = (fig_w, fig_w * (l2 / l1) + 0.5)
            fig, ax = plt.subplots(figsize=figsize)

        imshow_args: MutableMapping[str, Any] = {
            "origin": "lower",
            "extent": extent,
            "cmap": cmap,
            "aspect": "equal",
            "interpolation": "lanczos",
            "norm": "linear",
        }
        # vmax, vmin and Norm interact separately
        if log_scale:
            imshow_args["norm"] = colors.LogNorm()
        else:
            imshow_args["vmin"] = 0.0 if vmin is None else vmin
            imshow_args["vmax"] = float(np.max(density_grid)) if vmax is None else vmax
        imshow_args.update(imshow_kwargs)

        im = ax.imshow(density_grid, **imshow_args)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Axes ticks
        default_tick_args: Mapping[str, Any] = {"direction": "out", "which": "both"}
        if show_ticks:
            ax.tick_params(**tick_kwargs if tick_kwargs is not None else default_tick_args)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar customisation if enabled
        if show_colorbar:
            divider = mal(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax, fraction=0.04, pad=0.1, **(cbar_kwargs or {}))
            cbar.set_label(
                (r"$\log_{10}$ " + self.cbar_label) if log_scale else self.cbar_label,
                fontsize=10,
            )

        # Atom markers customisation
        if atom_data is not None:
            self._plot_atoms(
                ax,
                atom_data,
                atom_symbols=atom_symbols,
                label_atoms=label_atoms,
                atom_size=atom_size,
                atom_fontsize=atom_fontsize,
                atom_fontcolor=atom_fontcolor,
                atom_bgcolor=atom_bgcolor,
                atom_edgecolor=atom_edgecolor,
                atom_kwargs=atom_kwargs,
            )

        # Labels and titles
        default_x = self._vec_str(v1) + self._length_unit_suffix()
        default_y = self._vec_str(v2) + self._length_unit_suffix()
        if xlabel is not False:
            ax.set_xlabel(default_x if xlabel is None or xlabel is True else xlabel, fontsize=10)
        if ylabel is not False:
            ax.set_ylabel(default_y if ylabel is None or ylabel is True else ylabel, fontsize=10)
        if title is not None:
            ax.set_title(title)

        return fig, ax, im

    def run(
        self,
        filename: str | None = None,
        log_scale: bool = False,
        atom_number: int | None = None,
        shift: tuple[float, float] = (0.0, 0.0),
        thickness: float = 1,
        normalise: bool = False,
        rotation: float = 0.0,
        vmin: float | None = 0.0,
        vmax: float | None = None,
        figsize: tuple[float, float] | None = None,
        savefig_kwargs: Mapping[str, Any] | None = None,
        ax: Any | None = None,
        save: bool = True,
        xlabel: str | bool | None = None,
        ylabel: str | bool | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        cmap: str = "Blues",
        cbar_kwargs: Mapping[str, Any] | None = None,
        atom_symbols: Sequence[str] | None = None,
        atom_indices: Sequence[int] | None = None,
        label_atoms: bool = True,
        atom_fontcolor: Mapping[str, str] | None = None,
        atom_bgcolor: Mapping[str, str] | None = None,
        atom_edgecolor: Mapping[str, str] | None = None,
        atom_size: float = 160,
        atom_fontsize: float = 8,
        atom_kwargs: Mapping[str, Any] | None = None,
        grid_points: int = 500,
        window_repeat: float | tuple[float, float] = 1.0,
        atom_repeat: int | tuple[int, int, int] = 1,
        **imshow_kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        """Runs the full sequence of steps:

        1. Extracts a slice from the underlying :class:`density`
        2. Optionally projects nearby atoms onto the slice
        3. Plots and saves the result

        :param filename: Filename to save as. Will save with a useful name if not provided, defaults to None
        :type filename: ``str | None``, optional
        :param thickness: The Cartesian distance perpendicular to the plane to consider atoms as lying on the slice, defaults to 1 Bohr. Only useful if ``show_atoms=True``
        :type thickness: ``float``, optional
        :param normalise: Whether to normalise densities to the interval [0,1]. Useful if attempting to plot densities with different sums. Defaults to ``false``.
        :type normalise: ``bool``, optional
        :param rotation: The angle to rotate the in-plane basis by in degrees. Defaults to 0.0
        :type rotation: ``float``, optional (degrees)
        :param log_scale: Whether to plot the density on a base-10 logarithmic scale, defaults to False
        :type log_scale: ``bool``, optional
        :param atom_number: Atom number to center the density plot on
        :type atom_number: ``int``, optional
        :param shift: Vector to shift the origin on the plane by. Note the shift is :math:`(\\Delta v_1, \\Delta v_2)`.
        :type shift: ``tuple[float, float]``, optional
        :param vmin: Minimum value to set the colour scale at, defaults to 0.0
        :type vmin: ``float | None``, optional
        :param vmax: Maximum value to set the colour scale at, defaults to None
        :type vmax: ``float | None``, optional
        :param figsize: Override the auto-computed figure size, defaults to None
        :type figsize: ``tuple[float, float] | None``, optional
        :param savefig_kwargs: Extra keyword arguments forwarded to :func:`matplotlib.pyplot.savefig`, defaults to None
        :type savefig_kwargs: ``Mapping[str, Any] | None``, optional
        :param imshow_kwargs: Any further keyword arguments (e.g. ``cmap``,
            ``interpolation``) are forwarded to :func:`matplotlib.pyplot.imshow`.
        :param ax: the Axes instance to plot into. Default ``None`` (create its own)
        :type ax: matplotlib.pyplot.Axes | None
        :param save: Whether to save a copy of the density automatically, defaults to True
        :type save: bool
        :param xlabel: Plot's :math:`x`-axis label. If None, it is hidden. If ``True`` then uses an autogenerated one.
        :type xlabel: str | bool | None = None
        :param ylabel: Plot's :math:`y`-axis label. If None, it is hidden. If ``True`` then uses an autogenerated one.
        :type ylabel: str | bool | None = None
        :param xlim: x-axis limits. Defaults to (min, max) of data range
        :type xlim: tuple[float, float] | None = None
        :param ylim:  y-axis limits. Defaults to (min, max) of data range
        :type ylim: tuple[float, float] | None = None
        :param title: Axes plot title. None (default) produces no title.
        :type title: str | None = None
        :param show_colorbar: Whether to produce a colourbar alongside the density plot, defaults to True
        :type show_colorbar: bool
        :param show_ticks: Whether to show ticks on the density plots, defaults to True
        :type show_ticks: bool
        :param cmap: Colour theme to use. Defaults to matplotlib's "Blues". For publication, we recommend any theme with contrast, e.g. pure white background with a single colour gradient. Saves ink and easy to see!
        :type cmap: str, optional
        :param cbar_kwargs: Extra colourbar arguments to pass to matplotlib
        :type cbar_kwargs: Mapping[str, Any] | None, optional
        :param atom_symbols: Atom labels to filter out. Default is None, which filters nothing
        :type atom_symbols: Sequence[str] | None, optional
        :param atom_indices: Atoms to filter out. Default is None, which filters nothing
        :type atom_indices: Sequence[str] | None, optional
        :param label_atoms: Whether atom markers should show the element label, defaults to True (why else would you want them?)
        :param atom_fontcolor: Text colour of the atom labels
        :type atom_fontcolor: dict[str, str] | None, optional
        :param atom_bgcolor: Background colour of the marker corresponding to a specific element. Defaults to a specific colour dict (see source).
        :type atom_bgcolor: dict[str, str] | None, optional
        :param atom_edgecolor: Colour of the marker edge corresponding to a specific element. Defaults to white.
        :type atom_edgecolor: dict[str, str] | None, optional
        :param label_atoms: Whether atom markers should show the element label, defaults to True (why else woudl you want them?)
        :type label_atoms: bool, optional
        :param atom_size: Size of atom marker
        :type atom_size: float = 160, optional
        :param atom_fontsize: Fontsize of atom label. Defaults to 8pt.
        :type atom_fontsize: float = 8, optional
        :param grid_points: Number of grid points to use for interpolation. Defaults to 500. Very expensive beyond 1000.
        :type grid_points: int = 500, optional
        :param window_repeat: How many repeats to look for.
        :type window_repeat: float | tuple[float, float] = 1.0
        :param atom_repeat: How many atom repetitions to search for
        :type atom_repeat: int | tuple[int, int, int] = 1

        :returns: Tuple containing figure, axes and imshow instances created
        :rtype: ``tuple[Any, Any, Any]``
        """
        if atom_number is not None and atom_number < 0:
            raise ValueError("Cannot have a negative atom number")
        shift_vect = self.density.shift_onto_atom(atom_number) if atom_number is not None else shift
        density_grid, v1, v2, t1, t2, origin = self.density.extract_slice(
            n_points=grid_points,
            shift=shift_vect,
            window_repeat=window_repeat,
            rotation_deg=rotation,
        )
        atom_data: tuple[REAL_ARRAY, REAL_ARRAY, list[str]] | None = None
        if self.show_atoms:
            _, _, n_hat = self.density.inplane_basis()
            atom_data = self.density.project_atoms(
                v1, v2, n_hat, origin, thickness=thickness, indices=atom_indices, repeat=atom_repeat
            )
            print(f"  Atoms within {thickness} (Bohr) of plane: {len(atom_data[0])}")

        owns_figure: bool = ax is None
        fig, ax, im = self.plot_slice(
            density_grid,
            t1,
            t2,
            v1,
            v2,
            atom_data,
            log_scale=log_scale,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            ax=ax,
            normalise=normalise,
            owns_figure=owns_figure,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            title=title,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
            cmap=cmap,
            cbar_kwargs=cbar_kwargs,
            atom_symbols=atom_symbols,
            label_atoms=label_atoms,
            atom_size=atom_size,
            atom_fontsize=atom_fontsize,
            atom_fontcolor=atom_fontcolor,
            atom_bgcolor=atom_bgcolor,
            atom_edgecolor=atom_edgecolor,
            atom_kwargs=atom_kwargs,
            **imshow_kwargs,
        )
        if save:
            obj = plt if owns_figure else fig
            if owns_figure:
                fig.tight_layout()
            output = filename or self._default_filename()
            obj.savefig(output, **(savefig_kwargs or {}))
            print(f"Saved: {output}")
            if owns_figure:
                plt.close(fig)
        return fig, ax, im

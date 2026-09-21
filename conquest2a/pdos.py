from collections import defaultdict
import os
import sys
import re
from os.path import abspath
from pathlib import Path
from re import Match
from typing import Any, Literal

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
import numpy as np
import matplotlib.pyplot as plt
import conquest2a._types as c2at
from conquest2a.conquest import block_processor


class pdos(block_processor):
    """Initialise generic (p)DOS processor class.

    CONQUEST can produce a ``DOS.dat`` containing the total DOS and the local DOS, which ``lm="t"`` will process. To process :math:`l` and :math:`lm`-resolved PDOS files, set ``lm="l"`` and ``lm="lm"`` respectively.

    :param conquest_rundir: String or Path to the directory containing the (P)DOS files generated from CONQUEST's ``PostProcess`` tool.
    :type conquest_rundir: ``string | Path``
    :param lm: Determines the file-processing mode. Defaults to ``"t"``.

    :type lm: ``Literal["lm", "l", "t"]``, optional
    """

    def __init__(self, conquest_rundir: str | Path, lm: Literal["lm", "l", "t"] = "t") -> None:
        self.lm: Literal["lm", "l", "t"] = lm
        self.blocks: list[c2at.REAL_ARRAY] = []
        self.filename_regex: str = rf"Atom[0-9]{{7}}DOS\_{self.lm}\.dat"
        self.all_pdos_files: list[str] = []
        self.pdos_atoms: list[int] = []
        self.conquest_rundir: str | Path = conquest_rundir
        self.energy_values: dict[int, c2at.REAL_ARRAY] = {}
        self.current_block: list[Any] = []
        super().__init__()
        self.fermi_level: float = 0.0
        self.is_shifted_to_fermi: bool = True
        self.num_spins: int = 0
        self.pdos_dict: dict[str, list[c2at.REAL_ARRAY]] = defaultdict(list)
        self.resolve_path()
        self.locate_pdos_files()

    @override
    def process_headers(self, line: str) -> None:
        """Processes lines in PDOS output files starting with #

        :param line: line
        :type line: ``str``
        """
        if "# Spin" in line:
            self.num_spins += 1
        if "# Original" in line:
            result = re.findall(self.re_float, line)
            self.fermi_level = float(result[0])
        if not line.startswith("# DOS shifted"):
            self.is_shifted_to_fermi = False

    @override
    def process_block(self, line: str) -> None:
        if line == "&":
            if self.current_block:
                self.blocks.append(np.array(self.current_block, dtype=np.float64))
                self.current_block = []
        else:
            self.current_block.append(np.array(line.split()).astype(np.float64))

    def resolve_path(self) -> Path:
        """Checks whether the directory containing files exists.

        :raises FileNotFoundError: Raises error if directory does not exist
        :return: Path to the directory
        :rtype: ``Path``
        """
        abs_run_path: Path = Path(abspath(self.conquest_rundir))
        if abs_run_path.exists():
            return abs_run_path
        raise FileNotFoundError(f'Conquest directory specified: "{abs_run_path}", does not exist.')

    def locate_pdos_files(self) -> list[str]:
        """Gets the paths to all PDOS files of the right type and stores it in a list.

        :return: List of all paths to PDOS files given the mode.
        :rtype: ``list[str]``
        """
        if len(self.all_pdos_files) > 0:
            # Reset pdos files array, i.e. if changing directory
            self.all_pdos_files = []
        if self.lm == "t":
            self.all_pdos_files = ["DOS.dat"]
            return self.all_pdos_files
        abs_path: Path = self.resolve_path()
        file_list: list[str] = []
        for _, _, files in os.walk(abs_path, topdown=True):
            file_list = files
            break
        pdos_file_list: list[str] = []
        # The search is done because a directory can contain both lm, l resolved pDOS files
        for filename in file_list:
            match: Match[str] | None = re.search(r"([0-9]{7})", filename)
            if match:
                self.pdos_atoms.append(int(match.group(1)))
            res: Match[str] | None = re.match(self.filename_regex, filename)
            if res:
                pdos_file_list.append(filename)
        pdos_file_list = sorted(pdos_file_list)
        for file in pdos_file_list:
            self.all_pdos_files.append(f"{abspath(self.conquest_rundir)}/{file}")
        return self.all_pdos_files

    def read_dos_file(self) -> None:
        """Reads in a DOS.dat."""
        self.energy_values = {}
        for filename in self.all_pdos_files:
            self.read_file(filename)

    def read_pdos_file(self, atom: int) -> None:
        """Reads in the PDOS data of a file corresponding to an atom.

        CONQUEST outputs pdos filenames of the form ``AtomNNNNNNNDOS_lm.dat`` or ``AtomNNNNNNNDOS_l.dat``
        where NNNNNNN is a zero-padded atom number (as ordered in the coordinates file). This method is modified by the children :class:`pdos_l_processor` and :class:`pdos_lm_processor` for their purposes.

        :param atom: The atom to find and read in the PDOS for.
        :type atom: ``int``
        :raises ValueError: If the atom chosen does not have a PDOS file associated to it.
        """
        if atom not in self.pdos_atoms:
            raise ValueError("Chosen atom for pdos was not in the atom list")
        id: str = f"{atom:07d}"
        for filename in self.all_pdos_files:
            match: Match[str] | None = re.search(rf"Atom{id}DOS_{self.lm}\.dat", filename)
            if match:
                self.read_file(filename)
                return

    def _tdos(self) -> None:
        energy: c2at.REAL_ARRAY = self.blocks[0][:, 0]
        self.energy_values[0] = energy
        tdos_up: c2at.REAL_ARRAY = self.blocks[0][:, 1]
        tdos_dn: c2at.REAL_ARRAY = self.blocks[1][:, 1]
        ldos_up: c2at.REAL_ARRAY = self.blocks[0][:, 2]
        ldos_dn: c2at.REAL_ARRAY = self.blocks[1][:, 2]
        self.pdos_dict["tdos"] = [tdos_up, tdos_dn]
        self.pdos_dict["ldos"] = [ldos_up, ldos_dn]

    def _l_pdos(self) -> None:
        l_dict: dict[str, list[c2at.REAL_ARRAY]] = defaultdict(list)
        for idx, block in enumerate(self.blocks):
            energy: c2at.REAL_ARRAY = block[:, 0]
            self.energy_values[idx + 1] = energy
            pdos_values: c2at.REAL_ARRAY = block[:, 2:]
            num_l = pdos_values.shape[1]
            for l in range(num_l):
                l_dict[str(l)].append(pdos_values[:, l])
        self.pdos_dict = l_dict

    def _lm_pdos(self) -> None:
        lm_dict: dict[str, list[c2at.REAL_ARRAY]] = defaultdict(list)
        for idx, block in enumerate(self.blocks):
            energy = block[:, 0]
            self.energy_values[idx + 1] = energy
            pdos_values = block[:, 2:]
            num_lm = pdos_values.shape[1]

            l: int = 0
            m_count: int = 0
            m: int = 0
            for i in range(num_lm):
                if m_count >= (2 * l + 1):
                    l += 1
                    m_count = 0
                m = -l + m_count
                lm_key: str = f"{l},{m}"
                lm_dict[lm_key].append(pdos_values[:, i])
                m_count += 1
        self.pdos_dict = lm_dict

    def _clear_pdos(self) -> None:
        self.pdos_dict = {}

    def get_pdos(self, atom_number: int = 0) -> None:
        """Reads and stores the columns of a (p)DOS file in a dictionary sorted by ascending order of angular momentum :math:`l` and ascending order of :math:`m`."""
        self._clear_pdos()
        _pdos_funcs: dict[str, str] = {"lm": "_lm_pdos", "l": "_l_pdos", "t": "_tdos"}

        try:
            method_name = _pdos_funcs[self.lm]
        except KeyError:
            supported = ", ".join(sorted(_pdos_funcs))
            raise ValueError(
                f"Unsupported DOS mode '{self.lm}'; supported formats are: {supported}"
            ) from None

        reader = getattr(self, method_name)
        if atom_number > 0 and self.lm != "t":
            self.read_pdos_file(atom_number)
            reader()
        else:
            self.read_dos_file()
            reader()

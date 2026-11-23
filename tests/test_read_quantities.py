"""
Unit tests for read_static_output (conquest2a).

Generated with Claude, modified by chpxu
"""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conquest2a.conquest import conquest_species, conquest_coordinates
from conquest2a.io import read_coords
from conquest2a.read.quantities import read_static_output

EXAMPLE_OUTPUT = "data/output_static_example.txt"
EXAMPLE_COORDS = "data/output_example_coords.dat"

# Values used in the synthetic file
FREE = -2243.058015346185584
HF = -2243.058008992479245
DFT = -2243.058008992488794
STRESS = (-1.65706507, -2.31810410, -1.43000276)


def make_coords(natoms: int) -> SimpleNamespace:
    return SimpleNamespace(
        natoms=natoms,
        re_float=r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?",
        re_index=r"\d+",
        atoms=[SimpleNamespace(number=i + 1, forces=None) for i in range(natoms)],
    )


def expected_forces(natoms: int) -> np.ndarray:
    i = np.arange(1, natoms + 1)[:, None]
    return i * np.array([1e-3, -2e-3, 3e-3])


def build_output(
    natoms: int = 6,
    tail_lines: int = 24,
    include_free: bool = True,
    include_stress: bool = True,
    include_force_block: bool = True,
    n_force_blocks: int = 1,
) -> str:
    out = ["    CONQUEST", "    header junk"] * 200  # long preamble
    forces = expected_forces(natoms)

    for block in range(n_force_blocks):
        scale = 1.0 if block == n_force_blocks - 1 else 99.0  # earlier blocks are 'wrong'
        out += ["      | Number of electrons (u/d)=  1.0  2.0"]
        out += ["      |* band energy as 2Tr[K.H] =   -424.5 Ha"]
        if include_free:
            out += [f"      | Free Energy as E-TS      =  {FREE:.15f} Ha"]
        out += [
            f"      |* Harris-Foulkes energy   =  {HF:.15f} Ha",
            f"      |* DFT total energy        =  {DFT:.15f} Ha",
            "      |  estimated accuracy      =   0.000000000009550 Ha",
            "",
        ]
        if include_force_block:
            out += [
                "    force: Forces on atoms (Ha/a0)",
                "    force:  Atom   X              Y              Z",
            ]
            for n, (fx, fy, fz) in enumerate(forces * scale, start=1):
                out.append(f"    force: {n:5d} {fx:15.10f}{fy:15.10f}{fz:15.10f}")
            out += ["", "    force: Maximum force :        0.01 (Ha/a0) on atom 1 in x direction"]
            if include_stress:
                out += [
                    "    force: Force Residual:        0.01 Ha/a0",
                    "    force: Total stress:   "
                    f"{STRESS[0]:15.8f}{STRESS[1]:15.8f}{STRESS[2]:15.8f} GPa",
                    "    force: Average pressure:      1.80172398 GPa",
                ]
            out += ["", "    force: Finished force"]

    # Variable-length trailer (bibliography, LibXC references, timings...)
    out += [f"    trailer line {i}" for i in range(tail_lines)]
    return "\n".join(out) + "\n"


def run(tmp_path, natoms=6, **kwargs):
    f = tmp_path / "Conquest_out"
    f.write_text(build_output(natoms=natoms, **kwargs), encoding="utf-8")
    coords = make_coords(natoms)
    return read_static_output(str(f), coords), coords


# --------------------------------------------------------------------------
# Energies
# --------------------------------------------------------------------------
def test_energies_parsed(tmp_path):
    r, _ = run(tmp_path)
    assert r.free_energy == pytest.approx(FREE, abs=1e-9)
    assert r.harris_foulkes_energy == pytest.approx(HF, abs=1e-9)
    assert r.dft_energy == pytest.approx(DFT, abs=1e-9)


@pytest.mark.parametrize("tail_lines", [0, 5, 24, 25, 60, 500])
@pytest.mark.parametrize("natoms", [1, 6, 40])
def test_energies_found_for_any_trailer_length(tmp_path, natoms, tail_lines):
    """Regression: a fixed-size tail buffer missed the free energy line when
    the trailer (bibliography etc.) was long, or the system was large."""
    r, _ = run(tmp_path, natoms=natoms, tail_lines=tail_lines)
    assert r.free_energy == pytest.approx(FREE, abs=1e-9)
    assert r.harris_foulkes_energy == pytest.approx(HF, abs=1e-9)
    assert r.dft_energy == pytest.approx(DFT, abs=1e-9)


def test_missing_free_energy_is_inf_and_reported(tmp_path, capsys):
    r, _ = run(tmp_path, include_free=False)
    assert r.free_energy == np.inf
    assert "free energy" in capsys.readouterr().out.lower()
    # other energies are unaffected
    assert r.dft_energy == pytest.approx(DFT, abs=1e-9)


# --------------------------------------------------------------------------
# Forces
# --------------------------------------------------------------------------
def test_forces_assigned_to_correct_atoms(tmp_path):
    natoms = 6
    _, coords = run(tmp_path, natoms=natoms)
    exp = expected_forces(natoms)
    for atom, e in zip(coords.atoms, exp):
        np.testing.assert_allclose(atom.forces, e, atol=1e-9)


def test_forces_have_three_components(tmp_path):
    """Guards against re_float also capturing the atom index."""
    _, coords = run(tmp_path)
    for atom in coords.atoms:
        assert np.shape(atom.forces) == (3,)


def test_last_force_block_is_used(tmp_path):
    """For relaxations with several force blocks, the final one must win."""
    natoms = 4
    _, coords = run(tmp_path, natoms=natoms, n_force_blocks=3)
    for atom, e in zip(coords.atoms, expected_forces(natoms)):
        np.testing.assert_allclose(atom.forces, e, atol=1e-9)


def test_missing_force_block_raises(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, include_force_block=False)


def test_max_force_atom(tmp_path, capsys):
    natoms = 6
    r, _ = run(tmp_path, natoms=natoms)
    atom = r.get_max_force_atom()
    assert atom.number == natoms  # forces grow linearly with atom number
    out = capsys.readouterr().out
    assert f"Atom {natoms}" in out
    assert "z direction" in out  # z component (3e-3 * n) is the largest


# --------------------------------------------------------------------------
# Stresses
# --------------------------------------------------------------------------
def test_stresses_parsed(tmp_path):
    r, _ = run(tmp_path)
    np.testing.assert_allclose(r.stresses, STRESS, atol=1e-8)


def test_missing_stress_leaves_empty_array(tmp_path, capsys):
    r, _ = run(tmp_path, include_stress=False)
    assert r.stresses.size == 0
    assert "stress" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------
# Real output file
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not Path(EXAMPLE_OUTPUT).exists(), reason="data/output_static_example.txt not present"
)
@pytest.mark.skipif(
    not Path(EXAMPLE_COORDS).exists(), reason="data/output_example_coords.dat not present"
)
def test_real_output_file():
    spec = conquest_species({1: "Bi", 2: "O", 3: "Mn", 4: "Mn", 5: "Cu"})
    coords = read_coords(EXAMPLE_COORDS, spec, format="cq").coords
    r = read_static_output(str(REAL_OUTPUT), coords)

    assert r.free_energy == pytest.approx(-2243.058015346185584, abs=1e-9)
    assert r.harris_foulkes_energy == pytest.approx(-2243.058008992479245, abs=1e-9)
    assert r.dft_energy == pytest.approx(-2243.058008992488794, abs=1e-9)
    np.testing.assert_allclose(r.stresses, [-1.65706507, -2.31810410, -1.43000276], atol=1e-8)
    np.testing.assert_allclose(
        coords.atoms[0].forces, [0.0000000112, -0.0000001750, 0.0000001097], atol=1e-12
    )
    np.testing.assert_allclose(
        coords.atoms[12].forces, [0.0165184541, -0.0007769460, -0.0042728004], atol=1e-12
    )
    # "Maximum force: 0.01651845 on atom 13 in x direction"
    assert r.get_max_force_atom().number == 13

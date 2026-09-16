from conquest2a.conquest import *
from conquest2a.io import read_coords
from conquest2a.cell.supercell import *
import numpy as np
import pytest

test_species = conquest_species({1: "Bi", 2: "Mn", 3: "O"})
temp = read_coords("tests/data/test.dat", test_species, format="cq")
test_coords_proc: conquest_coordinates = temp.coords


def test_species_dict_stored() -> None:
    assert test_species.species_dict == {1: "Bi", 2: "Mn", 3: "O"}


def test_species_unique_elements() -> None:
    assert set(test_species.unique_elements) == {"Bi", "Mn", "O"}


def test_species_rejects_fake_element() -> None:
    with pytest.raises(ValueError):
        conquest_species({1: "Bi", 2: "NotAnElement"})


def test_species_allows_duplicate_labels() -> None:
    # Duplicated labels (e.g. spin up/down for the same element) are permitted.
    dup = conquest_species({1: "Mn", 2: "Mn"})
    assert dup.unique_elements == ["Mn"]


def test_dict_contains_only_real_elements_helper() -> None:
    assert test_species.dict_contains_only_real_elements() is True


# conquest_coordinates_processor


def test_natoms_and_atom_count_match() -> None:
    assert int(test_coords_proc.natoms) == 20
    assert len(test_coords_proc.atoms) == 20


def test_lattice_vectors_parsed() -> None:
    lattice = test_coords_proc.lattice_vectors
    assert lattice.shape == (3, 3)
    assert lattice[0][0] == pytest.approx(5.928)
    assert lattice[1][1] == pytest.approx(7.44)
    assert lattice[2][2] == pytest.approx(5.372)


def test_atom_labels_assigned() -> None:
    labels = {atom.label for atom in test_coords_proc.atoms}
    assert labels == {"Bi", "Mn", "O"}


def test_number_of_elements() -> None:
    counts = test_coords_proc.number_of_elements()
    assert counts == {"Bi": 4, "Mn": 4, "O": 12}


def test_element_map_contents_are_atoms_of_correct_label() -> None:
    element_map = test_coords_proc.element_map
    for label, atoms in element_map.items():
        assert all(atom.label == label for atom in atoms)


def test_cartesian_positions_shape_and_values() -> None:
    cart = test_coords_proc.get_cartesian_positions()
    assert cart.shape == (20, 3)
    first_atom = test_coords_proc.atoms[0]
    # Cartesian coords = frac coords @ lattice_vectors.T
    expected = first_atom.coords @ test_coords_proc.lattice_vectors.T
    assert np.allclose(first_atom.cart_coords, expected)


def test_atom_numbers_are_sequential() -> None:
    numbers = [atom.number for atom in test_coords_proc.atoms]
    assert numbers == list(range(1, 21))


# Atom class


def test_atom_str_contains_key_fields() -> None:
    atom = test_coords_proc.atoms[0]
    rendered = str(atom)
    assert f"Atom {atom.number} ({atom.label})" in rendered
    assert "Species Index" in rendered
    assert "Frac. Coords" in rendered
    assert "Cart. Coords" in rendered


def test_atom_to_ase_conversion() -> None:
    atom = test_coords_proc.atoms[0]
    ase_atom = atom.to_ase()
    assert ase_atom.symbol == atom.label
    assert np.allclose(ase_atom.position, atom.coords)


def test_atom_default_forces_and_spins() -> None:
    atom = test_coords_proc.atoms[5]
    assert np.array_equal(atom.forces, np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(atom.spins, np.array([0.0, 0.0, 0.0]))


# atom_charge


def test_atom_charge_assigns_spin_difference() -> None:
    charge_proc = atom_charge(
        "tests/data/test_original_AtomCharge.dat",
        test_coords_proc,
    )
    # spin = up - down, stored in the z-component
    for atom, charge_row in zip(test_coords_proc.atoms, charge_proc.conquest_charge_data):
        assert atom.spins[2] == pytest.approx(charge_row[1] - charge_row[2])
        assert atom.spins[0] == 0.0
        assert atom.spins[1] == 0.0


def test_atom_charge_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        atom_charge("tests/data/no_such_charge_file.dat", test_coords_proc)

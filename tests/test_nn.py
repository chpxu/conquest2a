import numpy as np
from conquest2a.conquest import *
from conquest2a.io import read_coords
from conquest2a.algo.nn import nearest_neighbours
import pytest

conquest_map = conquest_species({1: "O", 2: "Bi", 3: "Mn", 4: "Mn", 5: "Mn", 6: "Mn"})
path = "./tests/data/test_output_input_coords.in"
coordsproc = read_coords(path, conquest_map, format="cq").coords
nn = nearest_neighbours(
    coordsproc, coordsproc.atoms[8]
)  # Corresponds to B site Mn -> should return O atoms as nn


nn_self = nn.get_result(1)


def test_return_self() -> None:
    # Search always returns the atom itself as the first element!
    assert isinstance(nn_self, list)
    assert isinstance(nn_self[0], tuple)
    assert isinstance(nn_self[0][0], float)
    assert isinstance(nn_self[0][1], Atom)
    assert nn_self[0][1].label == "Mn"
    assert nn_self[0][1].label == coordsproc.atoms[8].label
    assert nn_self[0][1].number == coordsproc.atoms[8].number
    assert nn_self[0][1].number == 9
    assert nn_self[0][0] == 0.0  # Bohr


def test_nearest_neighbour() -> None:
    nn1 = nn.get_result(2)
    assert isinstance(nn1, list)
    # Check result is well-formed with expected types
    for result in nn1:
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert isinstance(result[1], Atom)
    # Search always returns the atom itself as the first element!
    assert nn1[0][1].label == "Mn"
    assert nn1[0][1].label == coordsproc.atoms[8].label
    assert nn1[0][1].number == coordsproc.atoms[8].number
    assert nn1[0][1].number == 9
    assert nn1[0][0] == 0.0  # Bohr

    # Now check the result by specifically probing for unique identifiable properties of the atom
    nn_atom = nn1[1][1]
    assert nn1[1][0] == 3.5664980205413435  # Bohr
    assert np.round(nn_atom.coords[0], decimals=8) == 0.00291016
    assert np.round(nn_atom.coords[1], decimals=8) == 0.79609625
    assert np.round(nn_atom.coords[2], decimals=8) == 0.35303287
    assert nn_atom.label == "O"
    assert nn_atom.number == 29


def test_nearest_neighbour2() -> None:
    nn2 = nn.get_result(3)
    # Search always returns the atom itself as the first element!
    for result in nn2:
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert isinstance(result[1], Atom)
    assert nn2[0][0] == 0.0  # Bohr
    # When searching for more neighbours, the results must be in the same order
    nn_atom = nn2[1][1]
    assert nn2[1][0] == 3.5664980205413435  # Bohr
    assert np.round(nn_atom.coords[0], decimals=8) == 0.00291016
    assert np.round(nn_atom.coords[1], decimals=8) == 0.79609625
    assert np.round(nn_atom.coords[2], decimals=8) == 0.35303287
    assert nn_atom.label == "O"
    assert nn_atom.number == 29
    # Now check the next nearest neighbo
    nn_atom2 = nn2[2][1]
    assert nn2[2][0] == 3.6250800070122247  # Bohr
    assert np.round(nn_atom2.coords[0], decimals=8) == 0.53486027
    assert np.round(nn_atom2.coords[1], decimals=8) == 0.64773048
    assert np.round(nn_atom2.coords[2], decimals=8) == 0.22858473
    assert nn_atom2.label == "O"
    assert nn_atom2.number == 26

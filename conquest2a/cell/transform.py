import copy
import itertools
import numpy as np
import conquest2a._types as c2at
from conquest2a.conquest import Atom, conquest_coordinates


class transform_unit_cell:
    """
    Class to handle unit cell transformations like VESTA.

    In VESTA, vectors on atoms are maintained if you have symmetry. If you remove the symmetry,
    the vectors are not kept. This class aims to reproduce the transformations of VESTA, but since
    atom labels, species numbers, spin and force arrays are not touched, this should preserve
    the spin and species labels, so a transformed unit cell can be immediately dumped with a writer.

    :param cq_coordinates: ``conquest_coordinates`` instance  of the cell to be transformed
    :type cq_coordinates: conquest_coordinates
    :param tol:  numerical tolerance used when wrapping/deduplicating atoms, defaults to 1e-4
    :type tol: float, optional
    """

    def __init__(self, cq_coordinates: conquest_coordinates, tol: float = 1e-4) -> None:
        self.coords: conquest_coordinates = cq_coordinates
        self.tol: float = tol
        self.transformed_cell_coords: conquest_coordinates = conquest_coordinates(
            self.coords.conquest_input
        )

    def _print_lattice_parameters(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> None:
        print(" === LATTICE PARAMETERS ===")
        print(
            f"a: {a}; b: {b}; c: {c}; alpha: {np.degrees(alpha)}, beta: {np.degrees(beta)}, gamma: {np.degrees(gamma)}"
        )

    def _get_lattice_parameters(
        self, coordinates_instance: conquest_coordinates
    ) -> tuple[float, float, float, float, float, float]:
        r"""
        This class extracts the lattice parameters :math:`a, b, c, \cos(\alpha), \cos(\beta), \cos(\gamma)`
        as is shown in VESTA, via the triclinic convention:
        1. :math:`v_1 = (a, 0, 0)`,
        2. :math:`v_2 = (b*\cos(gamma), b*\sin(gamma), 0)`
        3. :math:`v_3 = (c*\cos(beta),  \c*(\cos(alpha)-\cos(beta)\cos(gamma))/\sin(gamma),
            c*\sqrt( 1 + 2*\cos(alpha)\cos(beta)\cos(gamma)
                        - \cos(alpha)^2-\cos(beta)^2-\cos(gamma)^2 )/\sin(gamma) )`
        where :math:`alpha` is the angle between axis :math:`b` and :math:`c`
                :math:`beta` is the angle between axis :math:`a` and :math:`c`
                :math:`gamma` is the angle between axis :math:`a` and :math:`b`
        """

        # a,b,c
        lengths = np.linalg.norm(coordinates_instance.lattice_vectors, axis=1)
        a, b, c = lengths[0], lengths[1], lengths[2]
        alpha = np.acos(
            np.dot(coordinates_instance.lattice_vectors[1], coordinates_instance.lattice_vectors[2])
            / (b * c)
        )
        beta = np.acos(
            np.dot(coordinates_instance.lattice_vectors[0], coordinates_instance.lattice_vectors[2])
            / (a * c)
        )
        gamma = np.acos(
            np.dot(coordinates_instance.lattice_vectors[0], coordinates_instance.lattice_vectors[1])
            / (b * a)
        )

        return a, b, c, alpha, beta, gamma

    def _make_triclinic_lattice(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> c2at.REAL_ARRAY:
        r"""VESTA outputs new unit cell in a P1 triclinic formulation:

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

    def transform(self, P: c2at.REAL_ARRAY, dedupe: bool = True) -> conquest_coordinates:
        r"""
        Method to transform the unit cell using VESTA convention:

        The new basis vectors :math:`\mathrm{a}^{\prime}, \mathrm{b}^{\prime}, \mathrm{c}^{\prime}` are related to the basis vectors :math:`\mathrm{a}, \mathrm{b}, \mathrm{c}` by

        .. math::

            \begin{aligned}
            \left(\boldsymbol{a}^{\prime}, \boldsymbol{b}^{\prime}, \boldsymbol{c}^{\prime}\right)= & (\boldsymbol{a}, \boldsymbol{b}, \boldsymbol{c}) \boldsymbol{P} \\
            = & (\boldsymbol{a}, \boldsymbol{b}, \boldsymbol{c})\left(\begin{array}{lll}
            P_{11} & P_{12} & P_{13} \\
            P_{21} & P_{22} & P_{23} \\
            P_{31} & P_{32} & P_{33}
            \end{array}\right) \\
            = & \left(P_{11} \boldsymbol{a}+P_{21} \boldsymbol{b}+P_{31} \boldsymbol{c},\right. \\
            & P_{12} \boldsymbol{a}+P_{22} \boldsymbol{b}+P_{32} \boldsymbol{c}, \\
            & \left.P_{13} \boldsymbol{a}+P_{23} \boldsymbol{b}+P_{33} \boldsymbol{c}\right)
            \end{aligned}


        Note, the original :class:`Atom` in the original unit cell are deep copied to the new coordinates instance.
        The only property that is modified in the new Atoms is the fractional coordinate, and then cartesian coordinates
        This means atom numbers stay the same instead of being regenerated.

        :param P: Unit cell transformation matrix
        :type P: c2at.REAL_ARRAY
        :param dedupe: whether to remove duplicate atoms and wrap atoms into new unit cell lengths, defaults to `True`
        :type dedupe: bool, optional
        :return: New ``conquest_coordinates`` instance with new ``Atom`` s and lattice vectors
        :rtype: conquest_coordinates
        """

        if abs(np.linalg.det(P)) < 1e-8:
            raise ValueError("Rotation matrix is singular")
        if np.linalg.det(P) < 0.0:
            raise UserWarning("Rotation matrix will create a unit cell with opposite handedness!")

        # VESTA uses column convention, we use row convention
        transformed_lat_vect = (self.coords.lattice_vectors.T @ P).T
        inv_P = np.linalg.inv(P)

        corners = np.array([P @ np.array(v) for v in itertools.product([0, 1], repeat=3)])
        low = np.floor(corners.min(axis=0)).astype(int) - 1
        high = np.ceil(corners.max(axis=0)).astype(int) + 1

        offsets = np.array(
            list(
                itertools.product(
                    range(low[0], high[0] + 1),
                    range(low[1], high[1] + 1),
                    range(low[2], high[2] + 1),
                )
            )
        )

        new_atoms: list[Atom] = []
        should_be_added = set()
        for atom in self.coords.atoms:
            old_pos = atom.coords
            for n in offsets:
                img = old_pos + n
                new_pos = inv_P @ img
                is_inside = np.all(new_pos > -self.tol) and np.all(new_pos < 1.0 - self.tol)
                if not is_inside:
                    continue

                wrapped_pos = np.mod(new_pos, 1.0)
                if dedupe:
                    key = (
                        getattr(atom, "species", None),
                        tuple(np.round(wrapped_pos / self.tol).astype(int)),
                    )
                    if key in should_be_added:
                        continue
                    should_be_added.add(key)
                new_atom: Atom = copy.deepcopy(atom)
                new_atom.coords = wrapped_pos
                new_atom.symmetry_number = atom.number
                new_atoms.append(new_atom)

        # initialise new coordinates instance
        new_cq_coord = conquest_coordinates(self.coords.conquest_input)
        new_cq_coord.atoms = new_atoms
        new_cq_coord.natoms = str(len(new_atoms))
        new_cq_coord.lattice_vectors = transformed_lat_vect
        new_params: tuple[float, float, float, float, float, float] = self._get_lattice_parameters(
            new_cq_coord
        )
        new_cq_coord.lattice_vectors = self._make_triclinic_lattice(*new_params)
        new_cq_coord.index_to_atom_map()
        self.transformed_cell_coords = new_cq_coord
        self._print_lattice_parameters(*new_params)
        return new_cq_coord

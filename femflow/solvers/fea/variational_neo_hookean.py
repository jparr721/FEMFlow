from typing import Callable

import numpy as np

from .boundary_conditions import BoundaryConditions


class VariationalNeoHookean(object):
    def __init__(self, v: np.ndarray, f: np.ndarray, t: np.ndarray, E: float, nu: float):
        # Geometry
        self.v = v
        self.v0 = v
        self.f = f
        self.t = t

        self.q = np.array([])
        self.qdot = np.array([])

        self.E = E
        self.nu = nu
        self.lambda_ = (
            0.5 * (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        )
        self.mu = 0.5 * self.E / (2.0 * (1.0 + self.nu))

        self.k_selected = 1e5

    def simulate(self, q: np.ndarray, qdot: np.ndarray, dt: float, t: float):
        pass

    def assemble_forces(self) -> np.ndarray:
        """Assemble the forces of the fem system.

        Args:
            v0 (np.ndarray): v0 The reference space tetrahedra volumes

        Returns:
            np.ndarray: The per-forces
        """
        f = np.zeros(self.q.size)

        for i, v in enumerate(self.t):
            d_v = self._d_v_linear_tetrahedral_d_q(v, self.v0[i])
            index = self.t[i, 0]
            f[index : index + 3] += -d_v[0:3]

            index = self.t[i, 1]
            f[index : index + 3] += -d_v[3:6]

            index = self.t[i, 2]
            f[index : index + 3] += -d_v[6:9]

            index = self.t[i, 3]
            f[index : index + 3] += -d_v[9:12]

        return f

    def tetrahedral_linear_shape_functions(
        self, element: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Evaluates the linear shape functions for the tetrahedral element at a given
        reference space position

        Args:
            element (np.ndarray): element The tetrahedral element coordinates
            x (np.ndarray): x The position in the reference space

        Returns:
            np.ndarray: The 4x1 value of the basis function, phi
        """
        x0, x1, x2, x3 = self.v[element]
        t = np.zeros((3, 3))
        t[:, 0] = x1 - x0
        t[:, 1] = x2 - x0
        t[:, 2] = x3 - x0
        phi = np.zeros(4)
        phi[1:] = np.linalg.solve(t, x - x0)
        phi[0] = 1 - phi[1] - phi[2] - phi[3]
        return phi

    def d_tetrahedral_linear_shape_functions(self, element: np.ndarray) -> np.ndarray:
        """The derivative of the linear shape function for this element.

        Args:
            element (np.ndarray): element The tetrahedral element
            x (np.ndarray): x The reference space position vector

        Returns:
            np.ndarray: Derivative of the linear shape function for this element.
        """
        x0, x1, x2, x3 = self.v[element]
        t = np.zeros((3, 3))
        t[:, 0] = x1 - x0
        t[:, 1] = x2 - x0
        t[:, 2] = x3 - x0
        t_inv = np.linalg.inv(t)
        neg_one = -np.ones(3)
        d_phi = np.zeros((4, 3))
        d_phi[0] = neg_one.T * t_inv
        d_phi[1:4, 0:3] = t_inv
        return d_phi

    def _quadrature(self, volume: float, element: np.ndarray) -> np.ndarray:
        """Computes the quadrature as the weighted sum over the domain (tetrahedral).
        For the strain energy density integral function.

        Args:
            volume (float): volume The volume of the tetrahedral
            element (np.ndarray): element The vertex indices of the tetrahedral
            integrand (Callable): integrand The function to integrate

        Returns:
            np.ndarray The potential energy of the tetrahedral
        """
        t = np.zeros((4, 3))
        for i in range(4):
            index = 3 * element[i]
            t[i] = self.q[index : index + 3].T
        centroid = self._centroid(t)

        return volume * self._neo_hookean_linear_tetrahedral(element, centroid)

    def _centroid(self, t: np.ndarray) -> np.ndarray:
        n0 = np.linalg.norm(t[0]) ** 2

        A = np.zeros((3, 3))
        b = np.zeros(3)
        for k in range(3):
            A[k] = t[k + 1] - t[0]
            b[k] = (np.linalg.norm(t[k + 1]) ** 2) - n0

        return 0.5 * np.linalg.solve(A, b)

    def _neo_hookean_linear_tetrahedral(
        self, element: np.ndarray, tet_pt: np.ndarray
    ) -> np.ndarray:
        """Computes the gradient of the potential energy for a single tet.

        Args:
            element (np.ndarray): element The vertex indices of this tet.
            tet_pt (np.ndarray) A point within the tetrahedral.

        Returns:
            np.ndarray The gradient of the potential enery for the tet.
        """
        # Not tetrahedra, this is kinetic energy
        t = np.zeros((3, 4))
        for i in range(4):
            index = 3 * element[i]
            t[i] = self.q[index : index + 3]

        d_phi = self.d_tetrahedral_linear_shape_functions(element)
        F = t * d_phi

    def d_psi_neo_hookean_d_F(self, F: np.ndarray) -> np.ndarray:
        """Computes the gradient of the potential energy function psi.

        Args:
            F (np.ndarray): F The deformation gradient of reference and world space
            coordinates

        Returns:
            np.ndarray: The gradient of psi.
        """
        F1_1 = F[0, 0]
        F1_2 = F[0, 1]
        F1_3 = F[0, 2]
        F2_1 = F[1, 0]
        F2_2 = F[1, 1]
        F2_3 = F[1, 2]
        F3_1 = F[2, 0]
        F3_2 = F[2, 1]
        F3_3 = F[2, 2]

        # Stolen from https://github.com/dilevin/Bartels/blob/master/src/dpsi_neohookean_dF.cpp
        # Thanks, David!
        dw = np.zeros(9)
        dw[0] = (
            self.mu
            * (
                F1_1 * 2.0
                - ((F2_2 * F3_3 - F2_3 * F3_2) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            - self.lambda_
            * (F2_2 * F3_3 - F2_3 * F3_2)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[1] = (
            self.mu
            * (
                F2_1 * 2.0
                + ((F1_2 * F3_3 - F1_3 * F3_2) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            + self.lambda_
            * (F1_2 * F3_3 - F1_3 * F3_2)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[2] = (
            self.mu
            * (
                F3_1 * 2.0
                - ((F1_2 * F2_3 - F1_3 * F2_2) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            - self.lambda_
            * (F1_2 * F2_3 - F1_3 * F2_2)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[3] = (
            self.mu
            * (
                F1_2 * 2.0
                + ((F2_1 * F3_3 - F2_3 * F3_1) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            + self.lambda_
            * (F2_1 * F3_3 - F2_3 * F3_1)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[4] = (
            self.mu
            * (
                F2_2 * 2.0
                - ((F1_1 * F3_3 - F1_3 * F3_1) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            - self.lambda_
            * (F1_1 * F3_3 - F1_3 * F3_1)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[5] = (
            self.mu
            * (
                F3_2 * 2.0
                + ((F1_1 * F2_3 - F1_3 * F2_1) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            + self.lambda_
            * (F1_1 * F2_3 - F1_3 * F2_1)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[6] = (
            self.mu
            * (
                F1_3 * 2.0
                - ((F2_1 * F3_2 - F2_2 * F3_1) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            - self.lambda_
            * (F2_1 * F3_2 - F2_2 * F3_1)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[7] = (
            self.mu
            * (
                F2_3 * 2.0
                + ((F1_1 * F3_2 - F1_2 * F3_1) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            + self.lambda_
            * (F1_1 * F3_2 - F1_2 * F3_1)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        dw[8] = (
            self.mu
            * (
                F3_3 * 2.0
                - ((F1_1 * F2_2 - F1_2 * F2_1) * 2.0)
                / (
                    F1_1 * F2_2 * F3_3
                    - F1_1 * F2_3 * F3_2
                    - F1_2 * F2_1 * F3_3
                    + F1_2 * F2_3 * F3_1
                    + F1_3 * F2_1 * F3_2
                    - F1_3 * F2_2 * F3_1
                )
            )
            - self.lambda_
            * (F1_1 * F2_2 - F1_2 * F2_1)
            * (
                -F1_1 * F2_2 * F3_3
                + F1_1 * F2_3 * F3_2
                + F1_2 * F2_1 * F3_3
                - F1_2 * F2_3 * F3_1
                - F1_3 * F2_1 * F3_2
                + F1_3 * F2_2 * F3_1
                + 1.0
            )
            * 2.0
        )
        return dw

    def _d_v_linear_tetrahedral_d_q(
        self, element: np.ndarray, volume: float
    ) -> np.ndarray:
        """Computes the potential enery with respect to our deformed-space coordinates.

        Args:
            element (np.ndarray): The tetrahedral element coordinates
            volume (float): The volume of the tetrahedral

        Returns:
            np.ndarray: d_v partial derivative of potential enery with respect to q.
        """
        return self._quadrature(volume, element)


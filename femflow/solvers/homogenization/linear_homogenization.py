import copy
import math
from typing import Tuple, Union

import ilupp
import numpy as np
import numpy.matlib
from femflow.numerics.linear_algebra import sparse
from loguru import logger
from scipy.sparse.linalg import cg


class LinearHomogenization(object):
    def __init__(
        self, lambda_: Union[float, Tuple[float, float]], mu: Union[float, Tuple[float, float]], voxel: np.ndarray,
    ):
        assert voxel.ndim == 3, "Voxel input must be uniform 3D"

        self.cell_len_x, self.cell_len_y, self.cell_len_z = voxel.shape

        self.lambda_ = lambda_
        self.mu = mu

        if type(self.lambda_) == tuple:
            assert type(self.mu) == tuple, "Lambda and mu types must both be tuples!"

            if self.lambda_[0] == self.lambda_[1]:
                logger.warning("Lambda materials are the same! This could result in weird behavior")

            if self.mu[0] == self.mu[1]:
                logger.warning("Mu materials are the same! This could result in weird behavior")

            self.void_material = False
        else:
            self.void_material = True

        self.voxel = voxel

        self.constitutive_tensor = np.zeros((6, 6))

    def solve(self):
        nelx, nely, nelz = self.voxel.shape
        nel = nelx * nely * nelz
        dx = self.cell_len_x / nelx
        dy = self.cell_len_y / nely
        dz = self.cell_len_z / nelz

        # The stiffness matrix for linear FEA is broken down into two parts, one for lambda
        # and the other for mu, we'll separately compute them and then combine.
        ke_lambda, ke_mu, fe_lambda, fe_mu = self.compute_hexahedron(dx / 2, dy / 2, dz / 2)

        # Compute baseline degrees of freedom
        edof = self._compute_degrees_of_freedom(nel)
        # Impose periodic boundary conditions
        unique_nodes = self._compute_unique_nodes(nel)
        # Refine degrees of freedom to be only the active components (for void based meshes)
        edof = self._compute_unique_degrees_of_freedom(edof, unique_nodes)
        # Get rid of the unique nodes to save memory
        del unique_nodes
        # Number of degrees of freedom is 3 * elements (in euclidean coordinates in R3)
        ndof = 3 * nel

        if self.void_material:
            self.lambda_ = self.lambda_ * ((self.voxel == 1).astype(int))
            self.mu = self.mu * ((self.voxel == 1).astype(int))
        else:
            # Material properties for the composite vectors. We just sum them
            self.lambda_ = self.lambda_[0] * ((self.voxel == 1).astype(int)) + self.lambda_[1] * (
                (self.voxel == 2).astype(int)
            )
            self.mu = self.mu[0] * ((self.voxel == 1).astype(int)) + self.mu[1] * ((self.voxel == 2).astype(int))

        K = self._assemble_K(edof, ke_lambda, ke_mu, ndof)
        F = self._assemble_load(edof, fe_lambda, fe_mu, nel, ndof)
        ke = ke_mu + ke_lambda
        fe = fe_mu + fe_lambda
        del fe_mu, fe_lambda

        X = self._compute_displacement(K, F, edof, ndof)

        # Start homogenization
        # Displacement vectors from unit strain cases
        X0 = np.zeros((nel, 24, 6))

        # Element displacements for the 6 load tests
        X0_e = np.zeros((24, 6))

        indices = np.concatenate([np.array([3]), np.arange(6, 11), np.arange(12, 24)])
        X0_e[indices, :] = np.linalg.lstsq(ke[indices, :][:, indices], fe[indices, :])[0]
        X0[:, :, 0] = np.kron(X0_e[:, 0].transpose(), np.ones((nel, 1)))  # epsilon0_11 = (1,0,0,0,0,0)
        X0[:, :, 1] = np.kron(X0_e[:, 1].transpose(), np.ones((nel, 1)))  # epsilon0_22 = (0,1,0,0,0,0)
        X0[:, :, 2] = np.kron(X0_e[:, 2].transpose(), np.ones((nel, 1)))  # epsilon0_33 = (0,0,1,0,0,0)
        X0[:, :, 3] = np.kron(X0_e[:, 3].transpose(), np.ones((nel, 1)))  # epsilon0_12 = (0,0,0,1,0,0)
        X0[:, :, 4] = np.kron(X0_e[:, 4].transpose(), np.ones((nel, 1)))  # epsilon0_23 = (0,0,0,0,1,0)
        X0[:, :, 5] = np.kron(X0_e[:, 5].transpose(), np.ones((nel, 1)))  # epsilon0_13 = (0,0,0,0,0,1)

        # Fill Constitutive Tensor
        volume = self.cell_len_x * self.cell_len_y * self.cell_len_z

        edof -= 1
        for i in range(6):
            for j in range(6):
                dX = X0[:, :, j] - X[edof, j]
                sum_L = np.sum((np.matmul(X0[:, :, i] - X[edof, i], ke_lambda)) * dX, 1).reshape(nelx, nely, nelz)
                sum_M = np.sum((np.matmul(X0[:, :, i] - X[edof, i], ke_mu)) * dX, 1).reshape(nelx, nely, nelz)
                self.constitutive_tensor[i][j] = 1 / volume * np.sum(self.lambda_ * sum_L + self.mu * sum_M)
        logger.info("Homgenization complete")

    def _compute_degrees_of_freedom(self, n_el):
        n_el_x, n_el_y, n_el_z = self.voxel.shape

        self.number_of_nodes = (1 + n_el_x) * (1 + n_el_y) * (1 + n_el_z)

        # Applying the periodic boundary condition for periodic
        # volmes. Here we apply the node numbers and degrees of freedom for
        # this approximation.
        node_numbers = np.c_[np.arange(1, (self.number_of_nodes) + 1)]
        node_numbers = node_numbers.reshape(1 + n_el_x, 1 + n_el_y, 1 + n_el_z)

        sx, sy, sz = node_numbers.shape
        degrees_of_freedom = 3 * node_numbers[0 : sx - 1, 0 : sy - 1, 0 : sz - 1] + 1
        degrees_of_freedom = degrees_of_freedom.reshape(n_el, 1)

        mid = 3 * n_el_x + np.array([3, 4, 5, 0, 1, 2])
        add_x = np.concatenate((np.array([0, 1, 2]), mid, np.array([-3, -2, -1])))
        add_xy = 3 * (n_el_y + 1) * (n_el_x + 1) + add_x

        element_degrees_of_freedom = np.matlib.repmat(degrees_of_freedom, 1, 24) + np.matlib.repmat(
            np.concatenate([add_x, add_xy]), n_el, 1
        )

        return element_degrees_of_freedom

    def _compute_unique_nodes(self, n_el):
        n_el_x, n_el_y, n_el_z = self.voxel.shape
        number_unique_nodes = n_el

        unique_nodes_tensor = np.arange(1, number_unique_nodes + 1)
        unique_nodes_tensor = unique_nodes_tensor.reshape(n_el_x, n_el_y, n_el_z)
        back_border_cols = unique_nodes_tensor[:, :, 0]

        unt_indices = []
        for i, matrix in enumerate(unique_nodes_tensor):
            unt_indices.append(np.append(matrix, back_border_cols[i][..., None], 1))

        unique_nodes_tensor = np.array(unt_indices)

        left_border_cols = unique_nodes_tensor[:, 0, :]

        unt_indices = []
        for i, matrix in enumerate(unique_nodes_tensor):
            unt_indices.append(np.vstack([matrix, left_border_cols[i]]))

        unique_nodes_tensor = np.array(unt_indices)
        unique_nodes_tensor = np.concatenate((unique_nodes_tensor, unique_nodes_tensor[0][None]), axis=0)
        return unique_nodes_tensor

    def _compute_unique_degrees_of_freedom(self, edof, unt):
        _dof = np.zeros((3 * self.number_of_nodes, 1))
        _uniq = unt.reshape(np.prod(unt.shape))
        for i in range(0, _dof.shape[0], 3):
            idx = i // 3
            _dof[i] = 3 * _uniq[idx] - 2

        for i in range(1, _dof.shape[0], 3):
            idx = i // 3
            _dof[i] = 3 * _uniq[idx] - 1

        for i in range(2, _dof.shape[0], 3):
            idx = i // 3
            _dof[i] = 3 * _uniq[idx]

        edof = _dof[edof - 1]
        edof = np.squeeze(edof, axis=2)
        return edof.astype(int)

    def _compute_displacement(self, K, F, edof, ndof):
        # If we have a void-based single-material mesh, we have limited active dofs
        if self.void_material:
            # activedofs = edof[self._flat_1d((self.voxel == 1).astype(int))]
            indices = (self.voxel == 1).astype(int).flatten()

            activedofs = np.zeros((indices.size, edof.shape[1]))

            for i in range(indices.size):
                if indices[i] == 1:
                    activedofs[i] = edof[i, :]

            activedofs = np.sort(np.unique(activedofs)).astype(np.int32)
            activedofs = list(activedofs)
            del activedofs[0]
            activedofs = np.array(activedofs)
        else:
            activedofs = copy.deepcopy(edof)
            activedofs = np.sort(np.unique(self._flat_1d(activedofs)))

        # Subtract one from the dofs so indexing does not break
        activedofs -= 1

        end = activedofs.size
        K_sub = K[np.ix_(activedofs[3:end], activedofs[3:end])]
        L = self.ichol(K_sub)
        X = np.zeros((ndof, 6))

        for i in range(6):
            result, info = cg(A=K_sub, b=F[activedofs[3:end], i].todense(), tol=1e-10, maxiter=1000, M=L * L.T)

            if info > 0:
                raise RuntimeError("IChol solver failed")

            X[activedofs[3:end], i] = result

        return X

    def _assemble_K(self, edof, ke_lambda, ke_mu, ndof):
        # Index vectors
        stiffness_index_i = self._flat_1d(np.kron(edof, np.ones((24, 1))).transpose())
        stiffness_index_i -= 1
        stiffness_index_j = self._flat_1d(np.kron(edof, np.ones((1, 24))).transpose())
        stiffness_index_j -= 1
        # stiffness_entries = np.matmul(
        # 	self._flat_2d(ke_lambda), self._flat_2d(self.lambda_).conj().transpose()
        # ) + np.matmul(self._flat_2d(ke_mu), self._flat_2d(self.mu).conj().transpose())

        entry_lambda = np.multiply(
            np.expand_dims(ke_lambda.flatten("F"), axis=1), np.expand_dims(self.lambda_.flatten(), axis=1).conj().T
        )
        entry_mu = np.multiply(
            np.expand_dims(ke_mu.flatten("F"), axis=1), np.expand_dims(self.mu.flatten(), axis=1).conj().T
        )

        stiffness_entries = entry_lambda + entry_mu

        stiffness_entries = self._flat_1d(stiffness_entries)

        K = sparse(stiffness_index_i, stiffness_index_j, stiffness_entries, ndof, ndof)
        K = 1 / 2 * (K + K.transpose())
        return K

    def _assemble_load(self, edof, fe_lambda, fe_mu, n_el, ndof):
        load_index_i = self._flat_1d(np.matlib.repmat(edof.transpose(), 6, 1))
        load_index_i -= 1
        load_index_j = self._flat_1d(
            np.concatenate(
                (
                    np.ones((24, n_el)),
                    2 * np.ones((24, n_el)),
                    3 * np.ones((24, n_el)),
                    4 * np.ones((24, n_el)),
                    5 * np.ones((24, n_el)),
                    6 * np.ones((24, n_el)),
                )
            )
        )
        load_index_j -= 1

        entry_lambda = np.multiply(
            np.expand_dims(fe_lambda.flatten("F"), axis=1), np.expand_dims(self.lambda_.flatten(), axis=1).conj().T
        )
        entry_mu = np.multiply(
            np.expand_dims(fe_mu.flatten("F"), axis=1), np.expand_dims(self.mu.flatten(), axis=1).conj().T
        )

        load_entries = entry_lambda + entry_mu

        load_entries = load_entries.flatten("F")
        return sparse(load_index_i, load_index_j, load_entries, ndof, 6)

    @staticmethod
    def _flat_2d(v):
        v = copy.deepcopy(v)
        v = v.flatten("F")
        v.resize(v.shape[0], 1)
        return v

    @staticmethod
    def _flat_1d(v):
        v = copy.deepcopy(v)
        return v.flatten("F")

    @staticmethod
    def ichol(A):
        solver = ilupp.IChol0Preconditioner(A)
        return solver.factors()[0]

    @staticmethod
    def compute_hexahedron(a: float, b: float, c: float):
        """Compute hexahedron from 3d voxel shape grid

        Args:
            a (int): Dimension A (x direction)
            b (int): Dimension B (y direction)
            c (int): Dimension C (z direction)
        """
        # Constitutive matrix contributions - Mu
        C_mu = np.diag([2, 2, 2, 1, 1, 1])

        # Constitutive matrix contribution - Lambda
        C_lambda = np.zeros((6, 6))
        C_lambda[0:3, 0:3] = 1

        # Calculate the three gauss points in all 3 directions
        xx = np.array([-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)])
        yy = np.array([-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)])
        zz = np.array([-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)])
        ww = np.array([5 / 9, 8 / 9, 5 / 9])

        # Initialize
        ke_lambda = np.zeros((24, 24))
        fe_lambda = np.zeros((24, 6))

        ke_mu = np.zeros((24, 24))
        fe_mu = np.zeros((24, 6))

        # Begin integration over the volume
        for i in range(len(xx)):
            for j in range(len(yy)):
                for k in range(len(zz)):
                    # Integration point
                    x = xx[i]
                    y = yy[j]
                    z = zz[k]

                    # Compute the stress-strain displacement matrix
                    qx = [
                        -((y - 1) * (z - 1)) / 8,
                        ((y - 1) * (z - 1)) / 8,
                        -((y + 1) * (z - 1)) / 8,
                        ((y + 1) * (z - 1)) / 8,
                        ((y - 1) * (z + 1)) / 8,
                        -((y - 1) * (z + 1)) / 8,
                        ((y + 1) * (z + 1)) / 8,
                        -((y + 1) * (z + 1)) / 8,
                    ]

                    qy = [
                        -((x - 1) * (z - 1)) / 8,
                        ((x + 1) * (z - 1)) / 8,
                        -((x + 1) * (z - 1)) / 8,
                        ((x - 1) * (z - 1)) / 8,
                        ((x - 1) * (z + 1)) / 8,
                        -((x + 1) * (z + 1)) / 8,
                        ((x + 1) * (z + 1)) / 8,
                        -((x - 1) * (z + 1)) / 8,
                    ]

                    qz = [
                        -((x - 1) * (y - 1)) / 8,
                        ((x + 1) * (y - 1)) / 8,
                        -((x + 1) * (y + 1)) / 8,
                        ((x - 1) * (y + 1)) / 8,
                        ((x - 1) * (y - 1)) / 8,
                        -((x + 1) * (y - 1)) / 8,
                        ((x + 1) * (y + 1)) / 8,
                        -((x - 1) * (y + 1)) / 8,
                    ]
                    qq = np.array([qx, qy, qz])
                    dims = np.array(
                        [[-a, a, a, -a, -a, a, a, -a], [-b, -b, b, b, -b, -b, b, b], [-c, -c, -c, -c, c, c, c, c]]
                    ).transpose()

                    # Compute the jacobian to transform the coordinates to the deformed space (6 axial tests)
                    J = np.matmul(qq, dims)

                    qxyz = np.linalg.lstsq(J, qq)[0]
                    B_e = np.zeros((8, 6, 3))

                    for ii in range(B_e.shape[0]):
                        B_e[ii] = np.array(
                            [
                                [qxyz[0, ii], 0, 0],
                                [0, qxyz[1, ii], 0],
                                [0, 0, qxyz[2, ii]],
                                [qxyz[1, ii], qxyz[0, ii], 0],
                                [0, qxyz[2, ii], qxyz[1, ii]],
                                [qxyz[2, ii], 0, qxyz[0, ii]],
                            ]
                        )

                    B = np.hstack([B_e[ii] for ii in range(B_e.shape[0])])

                    weight = np.linalg.det(J) * ww[i] * ww[j] * ww[k]

                    # Element matrices
                    ke_lambda += weight * np.matmul(np.matmul(B.transpose(), C_lambda), B)
                    ke_mu += weight * np.matmul(np.matmul(B.transpose(), C_mu), B)

                    # Element loads
                    fe_lambda += weight * np.matmul(B.transpose(), C_lambda)
                    fe_mu += weight * np.matmul(B.transpose(), C_mu)

        return ke_lambda, ke_mu, fe_lambda, fe_mu

import os
import threading
from functools import cache
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from femflow.numerics.linear_algebra import matrix_to_vector, vector_to_matrix
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.solvers.mpm.parameters import Parameters
from femflow.viz.mesh import Mesh

from ..simulation_base import SimulationBase


class MPMSimulation(SimulationBase):
    def __init__(self, meshes: List[Mesh], save_positions=True):
        """Multiphysics engine for MPM.

        Args:
            meshes (List[Mesh]): The meshes to load.
            save_positions (bool) optional, default True: Whether or not to save the
                historical positions
        """
        super().__init__(meshes)
        self.save_positions = save_positions
        self.loaded = False
        self.running = False
        self.prev_positions = []

    @property
    @cache
    def full_mesh(self) -> Mesh:
        return np.sum(self.meshes)

    def load_previous_position(self, pos: int):
        if pos > len(self.prev_positions):
            raise IndexError("Position is out of bounds.")

        start = 0
        for mesh in self.meshes:
            end = start + mesh.vertices.size // 3
            mesh.replace(matrix_to_vector(self.prev_positions[pos][start:end]))
            start = end

    def load(self, **kwargs):
        self.mesh_params: List[Parameters] = kwargs["mesh_params"]

        if len(self.mesh_params) != len(self.meshes):
            raise ValueError("Each mesh must have a parameter set")

        dim = 3

        # Positions
        self.x = (
            vector_to_matrix(self.full_mesh.vertices.copy(), dim)
            * self.mesh_params[0].tightening_coeff
        )

        if self.save_positions:
            self.prev_positions = [self.x.copy() / self.mesh_params[0].tightening_coeff]

        # Momentum/Velocity
        self.v = np.zeros((len(self.x), dim), dtype=np.float64)

        # Deformation Gradient
        self.F = np.array([np.eye(dim, dtype=np.float64) for _ in range(len(self.x))])

        # Affine Momentum (MLS MPM)
        self.C = np.zeros((len(self.x), dim, dim), dtype=np.float64)

        # Volume (jacobian) for snow plasticity model
        self.Jp = np.ones((len(self.x), 1), dtype=np.float64)

        self.loaded = True
        logger.success("Simulation Loaded Successfully")

    def start(self, **kwargs):
        if not self.loaded:
            logger.error("Please load the simulation first")
            return
        n_timesteps: int = kwargs["n_timesteps"]
        threading.Thread(
            target=self._simulate_offline, args=((n_timesteps,)), daemon=True,
        ).start()

    def reset(self, **kwargs):
        pass

    def _simulate_offline(self, n_timesteps: int):
        self.running = True
        for _ in tqdm(range(n_timesteps)):
            start = 0
            for mesh, params in zip(self.meshes, self.mesh_params):
                end = start + mesh.vertices.size // 3
                solve_mls_mpm_3d(
                    params,
                    self.x[start:end],
                    self.v[start:end],
                    self.F[start:end],
                    self.C[start:end],
                    self.Jp[start:end],
                )
                start = end

            if self.save_positions:
                self.prev_positions.append(
                    self.x.copy() / self.mesh_params[0].tightening_coeff
                )

        if self.save_positions:
            logger.info("Saving displacements")
            if not os.path.exists("tmp"):
                os.mkdir("tmp")
            for i, displacement in enumerate(self.prev_positions):
                np.save(f"tmp/{i}", displacement)

        logger.success("Simulation done")
        self.running = False

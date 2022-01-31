import os
import threading

import numpy as np
from loguru import logger
from numba.typed import List as nb_list
from tqdm import tqdm

from femflow.numerics.linear_algebra import vector_to_matrix
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.solvers.mpm.particle import Particle
from femflow.viz.mesh import Mesh

from ..simulation_base import SimulationBase


class MPMSimulation(SimulationBase):
    def __init__(
        self,
        steps: int,
        dt: float,
        gyroid_mass: float,
        collider_mass: float,
        volume: float,
        gyroid_force: float,
        collider_force: float,
        gyroid_E: float,
        collider_E: float,
        gyroid_v: float,
        collider_v: float,
        hardening: float,
        grid_res: int,
        tightening_coeff: float,
        save_displacements=True,
    ):
        super().__init__()
        self.save_displacements = save_displacements
        self.loaded = False
        self.running = False
        self.displacements = []
        self.steps = steps
        self.dt = dt
        self.gyroid_mass = gyroid_mass
        self.collider_mass = collider_mass
        self.volume = volume
        self.gyroid_force = gyroid_force
        self.collider_force = collider_force
        self.gyroid_E = gyroid_E
        self.collider_E = collider_E
        self.gyroid_v = gyroid_v
        self.collider_v = collider_v
        self.hardening = hardening
        self.grid_res = grid_res
        self.tightening_coeff = tightening_coeff

        self.gyroid_mu_0 = self.gyroid_E / (2 * (1 + self.gyroid_v))
        self.gyroid_lambda_0 = (
            self.gyroid_E
            * self.gyroid_v
            / ((1 + self.gyroid_v) * (1 - 2 * self.gyroid_v))
        )

        self.dx = 1 / self.grid_res
        self.inv_dx = 1 / self.dx

    def load(self, **kwargs):
        if "mesh" not in kwargs:
            raise ValueError("'mesh' is a required parameter for the MPM Sim")

        # Always 3D for now.
        dim = 3

        self.mesh: Mesh = kwargs["mesh"]

        # Positions
        self.x = (
            vector_to_matrix(self.mesh.vertices.copy(), 3) * self.tightening_coeff
        ).astype(np.float64)

        if self.save_displacements:
            self.displacements = [self.x.copy() / self.tightening_coeff]

        # Momentum/Velocity
        self.v = np.zeros((len(self.x), dim), dtype=np.float64)

        # Deformation Gradient
        self.F = np.array([np.eye(dim, dtype=np.float64) for _ in range(len(self.x))])

        # Affine Momentum (MLS MPM)
        self.C = np.zeros((len(self.x), dim, dim), dtype=np.float64)

        # Volume (jacobian)
        self.Jp = np.ones((len(self.x), 1), dtype=np.float64)

        self.loaded = True
        logger.success("Simulation loaded")

    def start(self, **kwargs):
        if not self.loaded:
            logger.error("Please load the simulation first")
            return
        threading.Thread(target=self._simulate_offline, daemon=True,).start()

    def reset(self, **kwargs):
        self.mesh.reset_positions()
        self.load(**kwargs)

    def _simulate_offline(self):
        self.running = True
        particles = nb_list()
        [particles.append(Particle(pos, 1.0, 1.0, 1.0, 1.0)) for pos in self.x]

        for _ in tqdm(range(self.steps)):
            solve_mls_mpm_3d(
                self.grid_res,
                self.inv_dx,
                self.hardening,
                self.gyroid_mu_0,
                self.gyroid_lambda_0,
                self.gyroid_mass,
                self.dx,
                self.dt,
                self.volume,
                self.gyroid_force,
                particles,
                self.v,
                self.F,
                self.C,
                self.Jp,
            )
            if self.save_displacements:
                positions = np.array(
                    list(map(lambda p: p.pos.copy() / self.tightening_coeff, particles))
                )
                self.displacements.append(positions)
        logger.info("Saving displacements")
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        for i, displacement in enumerate(self.displacements):
            np.save(f"tmp/{i}", displacement)
        logger.success("Simulation done")
        self.running = False

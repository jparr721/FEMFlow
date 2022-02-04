import os
import threading
from typing import List

import numpy as np
from loguru import logger
from numba.typed import List as nb_list
from tqdm import tqdm

from femflow.numerics.fem import Ev_to_lambda, Ev_to_mu
from femflow.numerics.linear_algebra import vector_to_matrix
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.solvers.mpm.particle import Particle, map_particles_to_pos
from femflow.viz.mesh import Mesh

from ..simulation_base import SimulationBase


class MPMSimulation(SimulationBase):
    def __init__(
        self,
        outdir: str,
        steps: int,
        dt: float,
        gyroid_mass: float,
        collider_mass: float,
        volume: float,
        force: float,
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
        self.outdir = outdir
        self.save_displacements = save_displacements
        self.loaded = False
        self.running = False
        self.displacements = []
        self.steps = steps
        self.dt = dt
        self.gyroid_mass = gyroid_mass
        self.collider_mass = collider_mass
        self.volume = volume
        self.force = force
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
        # Always 3D for now.
        dim = 3

        # gyroid_mesh: Mesh = kwargs["gyroid_mesh"]

        self.particles = nb_list()
        # [
        #     self.particles.append(
        #         Particle(
        #             pos,
        #             self.force,
        #             self.gyroid_mass,
        #             Ev_to_lambda(self.gyroid_E, self.gyroid_v),
        #             Ev_to_mu(self.gyroid_E, self.gyroid_v),
        #         )
        #     )
        #     for pos in (
        #         vector_to_matrix(gyroid_mesh.vertices.copy(), 3) * self.tightening_coeff
        #     ).astype(np.float64)
        # ]

        collider_mesh: Mesh = kwargs["collider_mesh"]
        [
            self.particles.append(
                Particle(
                    pos,
                    self.force,
                    self.collider_mass,
                    Ev_to_lambda(self.collider_E, self.collider_v),
                    Ev_to_mu(self.collider_E, self.collider_v),
                )
            )
            for pos in (
                vector_to_matrix(collider_mesh.vertices.copy(), 3) * self.tightening_coeff
            ).astype(np.float64)
        ]

        n = len(self.particles)

        if self.save_displacements:
            self.displacements = [
                map_particles_to_pos(self.particles, self.tightening_coeff)
            ]

        # Momentum/Velocity
        self.v = np.zeros((n, dim), dtype=np.float64)

        # Deformation Gradient
        self.F = np.array([np.eye(dim, dtype=np.float64) for _ in range(n)])

        # Affine Momentum (MLS MPM)
        self.C = np.zeros((n, dim, dim), dtype=np.float64)

        # Volume (jacobian)
        self.Jp = np.ones((n, 1), dtype=np.float64)

        self.loaded = True
        logger.success("Simulation loaded")

    def start(self, **kwargs):
        if not self.loaded:
            logger.error("Please load the simulation first")
            return
        threading.Thread(target=self._simulate_offline, daemon=True,).start()

    def reset(self, **kwargs):
        self.load(**kwargs)

    def _simulate_offline(self):
        self.running = True

        for _ in tqdm(range(self.steps)):
            solve_mls_mpm_3d(
                self.grid_res,
                self.inv_dx,
                self.hardening,
                self.dx,
                self.dt,
                self.volume,
                self.force,
                self.particles,
                self.v,
                self.F,
                self.C,
                self.Jp,
            )
            if self.save_displacements:
                positions = map_particles_to_pos(self.particles, self.tightening_coeff)
                self.displacements.append(positions)
        logger.info("Saving displacements")
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        for i, displacement in enumerate(self.displacements):
            np.save(f"{self.outdir}/{i}", displacement)
        logger.success("Simulation done")
        self.running = False

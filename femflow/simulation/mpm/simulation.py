import threading

import numpy as np
from loguru import logger
from tqdm import tqdm

from femflow.numerics.linear_algebra import vector_to_matrix
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.solvers.mpm.parameters import Parameters
from femflow.viz.mesh import Mesh

from ..simulation_base import SimulationBase, SimulationRunType
from .primitives import (
    diamond,
    generate_cube_points,
    generate_implicit_points,
    gyroid,
    primitive,
)


class MPMSimulation(SimulationBase):
    def __init__(
        self,
        run_type: SimulationRunType = SimulationRunType.OFFLINE,
        save_displacements=True,
    ):
        super().__init__(run_type)
        self.save_displacements = save_displacements
        self.loaded = False
        self.running = False
        self.displacements = []

    def load(self, **kwargs):
        if "model" not in kwargs:
            self.model = "gyroid"
        else:
            self.model = kwargs["model"]

        if "mesh" not in kwargs:
            raise ValueError("'mesh' is a required parameter for the MPM Sim")

        # Always 3D for now.
        dim = 3

        self.mesh: Mesh = kwargs["mesh"]
        self.parameters: Parameters = kwargs["parameters"]

        # Positions
        # self.x = self._load_physical_model()
        self.x = (
            vector_to_matrix(self.mesh.vertices.copy(), 3)
            * self.parameters.tightening_coeff
        )

        if self.save_displacements:
            self.displacements = [self.x.copy() / self.parameters.tightening_coeff]

        # Add the vertices to the mesh for rendering
        # np.concatenate((mesh.vertices, self.x))

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
        if self.run_type == SimulationRunType.OFFLINE:
            n_timesteps: int = kwargs["n_timesteps"]
            # self._simulate_offline(n_timesteps)
            threading.Thread(
                target=self._simulate_offline, args=((n_timesteps,)), daemon=True,
            ).start()
        elif self.run_type == SimulationRunType.ONLINE:
            if self.save_displacements:
                logger.warning("Saving displacements not supported for online sims.")

            threading.Thread(target=self._simulate_online, daemon=True).start()

    def reset(self, **kwargs):
        pass

    def _load_physical_model(self) -> np.ndarray:
        if self.model == "gyroid":
            return generate_implicit_points(gyroid, 0.3, 0.3, 30)
        elif self.model == "diamond":
            return generate_implicit_points(diamond, 0.3, 0.3, 30)
        elif self.model == "primitive":
            return generate_implicit_points(primitive, 0.3, 0.3, 30)
        elif self.model == "cube":
            return generate_cube_points(np.zeros(3), 30)
        else:
            raise ValueError(f"Model {self.model} is not supported.")

    def _simulate_online(self):
        while self.running:
            solve_mls_mpm_3d(self.parameters, self.x, self.v, self.F, self.C, self.Jp)

    def _simulate_offline(self, n_timesteps: int):
        for _ in tqdm(range(n_timesteps)):
            solve_mls_mpm_3d(self.parameters, self.x, self.v, self.F, self.C, self.Jp)
            if self.save_displacements:
                self.displacements.append(
                    self.x.copy() / self.parameters.tightening_coeff
                )
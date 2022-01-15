from typing import List

import imgui
import numpy as np
import taichi as ti
from loguru import logger
from tqdm import tqdm

from femflow.numerics.fem import Ev_to_lame_coefficients
from femflow.solvers.mpm.grid_to_particle import grid_to_particle
from femflow.solvers.mpm.grid_velocity import grid_velocity
from femflow.solvers.mpm.parameters import MPMParameters
from femflow.solvers.mpm.particle import NeoHookeanParticle, make_particle
from femflow.solvers.mpm.particle_to_grid import particle_to_grid
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu

ti.init(arch=ti.gpu)


class MPMSimulationMenu(VisualizerMenu):
    def __init__(self):
        name = "MPM Simulation Options"
        flags = [imgui.TREE_NODE_DEFAULT_OPEN]
        super().__init__(name, flags)

        self._register_input("dt", 0.001)
        self._register_input("mass", 10)
        self._register_input("force", -100)
        self._register_input("youngs_modulus", 50000)
        self._register_input("poissons_ratio", 0.3)
        self._register_input("hardening", 10.0)
        self._register_input("grid_resoution", 100)

    def render(self, **kwargs) -> None:
        imgui.text("dt")
        self._generate_imgui_input("dt", imgui.input_float)
        imgui.text("Mass")
        self._generate_imgui_input("mass", imgui.input_float)
        imgui.text("Force")
        self._generate_imgui_input("force", imgui.input_float)
        imgui.text("Youngs Modulus")
        self._generate_imgui_input("youngs_modulus", imgui.input_int)
        imgui.text("Poissons Ratio")
        self._generate_imgui_input("poissons_ratio", imgui.input_float)
        imgui.text("Hardening")
        self._generate_imgui_input("hardening", imgui.input_float)
        imgui.text("Grid Resolution")
        self._generate_imgui_input("grid_resolution", imgui.input_int)


class MPMSimulation(object):
    def __init__(
        self,
        mass: float = 1.0,
        hardening: float = 10.0,
        E: float = 1e4,
        v: float = 0.2,
        gravity: float = -200.0,
        dt: float = 1e-4,
        grid_resolution: int = 80,
    ):
        mu, lambda_ = Ev_to_lame_coefficients(E, v)
        self.params = MPMParameters(
            mass,
            1,
            hardening,
            E,
            v,
            mu,
            lambda_,
            gravity,
            dt,
            1 / grid_resolution,
            grid_resolution,
            2,
            False,
        )

        self.grid = np.array([])
        self.particles: List[NeoHookeanParticle] = []

        self._initialize()
        logger.success("Simulation initialized successfully")

    def simulate(self, n_timesteps: int):
        total_steps = 0
        logger.info("Firing up simulation")
        gui = ti.GUI()
        while gui.running and not gui.get_event(gui.ESCAPE):
            for _ in tqdm(range(n_timesteps)):
                total_steps += 1
                # if total_steps >= 620:
                #     self.params.debug = True
                # if self.params.debug:
                #     logger.debug(f"TIMESTEP: {total_steps}")
                self._advance()

            gui.clear(0x112F41)
            gui.rect(
                np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F
            )
            all_particles = []
            for particle in self.particles:
                all_particles.append(particle.position)
            all_particles = np.array(all_particles)
            gui.circles(all_particles, radius=1.5, color=0xED553B)
            gui.show()

    def _advance(self):
        particle_to_grid(self.params, self.particles, self.grid)
        grid_velocity(self.params, self.grid)
        grid_to_particle(self.params, self.particles, self.grid)

    def _initialize(self):
        # Grid Layout is nxnx3 where each entry is [velocity_x, velocity_y, mass]
        self.grid = np.zeros(
            (self.params.grid_resolution, self.params.grid_resolution, 3),
            dtype=np.float64,
        )

        self._add_object(np.array((0.55, 0.45)), 0xED553B)
        # self._add_object(np.array((0.45, 0.65)), 0xED553B)
        # self._add_object(np.array((0.55, 0.85)), 0xED553B)

    def _add_object(self, center: np.ndarray, c: int):
        for _ in range(1):
            # pos = (np.random.rand(2) * 2.0 - np.ones(2)) * 0.08 + center
            self.particles.append(make_particle(center, np.zeros(2), c))

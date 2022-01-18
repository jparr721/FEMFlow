import imgui
import numpy as np
import taichi as ti
from tqdm import tqdm

from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_2d, solve_mls_mpm_3d
from femflow.solvers.mpm.parameters import Parameters
from femflow.solvers.mpm.particle import Particle
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


mass = 1.0
volume = 1.0
hardening = 10.0
E = 1e4
nu = 0.2
gravity = -200.0
dt = 1e-4
grid_resolution = 80
parameters = Parameters(mass, volume, hardening, E, nu, gravity, dt, grid_resolution)


def sim_2d():
    # for _ in range(50):
    #     center = np.array((0.55, 0.45))
    #     pos = (np.random.rand(2) * 2.0 - np.ones(2)) * 0.08 + center
    #     particles.append(Particle(pos, 0xED553B))

    n_particles = 50
    center = np.array((0.55, 0.45))
    x = np.array(
        [
            (np.random.rand(2) * 2.0 - np.ones(2)) * 0.08 + center
            for _ in range(n_particles)
        ],
        dtype=np.float64,
    )
    v = np.zeros((n_particles, 2), dtype=np.float64)
    F = np.array([np.eye(2, dtype=np.float64) for _ in range(n_particles)])
    C = np.zeros((n_particles, 2, 2), dtype=np.float64)
    Jp = np.ones((n_particles, 1), dtype=np.float64)

    gui = ti.GUI()
    while gui.running and not gui.get_event(gui.ESCAPE):
        for _ in tqdm(range(50)):
            solve_mls_mpm_2d(parameters, x, v, F, C, Jp)

        gui.clear(0x112F41)
        gui.rect(np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F)
        gui.circles(x, radius=1.5, color=0xED553B)
        gui.show()


def sim_3d():
    def map_position_2d(a: np.ndarray):
        phi, theta = np.radians(28), np.radians(32)

        a = a - 0.5
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        c, s = np.cos(phi), np.sin(phi)
        C, S = np.cos(theta), np.sin(theta)
        x, z = x * c + z * s, z * c - x * s
        u, v = x, y * C + z * S
        return np.array([u, v]).swapaxes(0, 1) + 0.5

    for _ in range(1):
        center = np.array((0.55, 0.45, 0.55))
        pos = (np.random.rand(3) * 2.0 - np.ones(3)) * 0.08 + center
        particles.append(Particle(pos, 0xED553B))

    gui = ti.GUI()
    while gui.running and not gui.get_event(gui.ESCAPE):
        for _ in tqdm(range(50)):
            solve_mls_mpm_3d(parameters, particles)

        gui.clear(0x112F41)
        all_particles = []
        for particle in particles:
            all_particles.append(particle.x)
        all_particles = np.array(all_particles)
        assert all_particles.shape[1] == 3
        all_particles = map_position_2d(all_particles)
        gui.circles(all_particles, radius=1.5, color=0xED553B)
        gui.show()


import time

import imgui
import numpy as np
import taichi as ti
from loguru import logger
from tqdm import tqdm

from femflow.meshing.implicit import gyroid_2d
from femflow.numerics.geometry import grid
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_2d, solve_mls_mpm_3d
from femflow.solvers.mpm.parameters import Parameters
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu


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


def draw_cube_2d(tl: np.ndarray, n: int = 10) -> np.ndarray:
    x = np.linspace(*tl, num=n)
    y = np.linspace(*tl, num=n)

    all_pts = []
    for row in x:
        for col in y:
            all_pts.append([row, col])
    return np.array(all_pts, dtype=np.float64)


def draw_gyroid_2d(tl: np.ndarray, n: int = 10) -> np.ndarray:
    g = grid(np.array((40, 40, 40)))

    def _fn(amplitude: float, pos: np.ndarray) -> float:
        two_pi = (2.0 * np.pi) / amplitude
        x, y, z = pos
        return (
            np.sin(two_pi * x) * np.cos(two_pi * y)
            + np.sin(two_pi * y) * np.cos(two_pi * z)
            + np.sin(two_pi * z) * np.cos(two_pi * x)
            - 0.3
        )

    inside = np.array([_fn(0.3, row) for row in g])
    cube = g[inside > 0.3]
    print(cube)
    print(cube.shape)
    return cube


def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), "bool")
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def draw_gyroid_3d(n: int = 10) -> np.ndarray:
    g = grid(np.array((n, n, n)))

    def _fn(amplitude: float, pos: np.ndarray) -> float:
        two_pi = (2.0 * np.pi) / amplitude
        x, y, z = pos
        return (
            np.sin(two_pi * x) * np.cos(two_pi * y)
            + np.sin(two_pi * y) * np.cos(two_pi * z)
            + np.sin(two_pi * z) * np.cos(two_pi * x)
            - 0.3
        )

    inside = np.array([_fn(0.3, row) for row in g])
    cube = g[inside > 0.03] * 0.15
    for row in cube:
        row[0] += 0.4
        row[1] += 0.4
    return cube


def draw_cube_3d(tl: np.ndarray, n: int = 10) -> np.ndarray:
    x = np.linspace(*tl, num=n)
    y = np.linspace(*tl, num=n)
    z = np.linspace(*tl, num=n)

    all_pts = []
    for layer in x:
        for row in y:
            for col in z:
                all_pts.append([layer, row, col])
    return np.array(all_pts, dtype=np.float64)


def _make_mpm_objects(lenx: int, dim: int):
    v = np.zeros((lenx, dim), dtype=np.float64)
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(lenx)])
    C = np.zeros((lenx, dim, dim), dtype=np.float64)
    Jp = np.ones((lenx, 1), dtype=np.float64)

    return v, F, C, Jp


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
    tl = np.array((0.4, 0.5))
    # x = draw_cube_2d(tl, 15)
    x = draw_gyroid_2d(tl, 20)
    print(x.shape)
    n_particles = len(x)
    v, F, C, Jp = _make_mpm_objects(n_particles, 2)

    # gui = ti.GUI(res=1024)
    # while gui.running and not gui.get_event(gui.ESCAPE):
    #     # for _ in tqdm(range(50)):
    #     #     solve_mls_mpm_2d(parameters, x, v, F, C, Jp)

    #     gui.clear(0x112F41)
    #     gui.rect(np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F)
    #     gui.circles(x, radius=1.5, color=0xED553B)
    #     gui.show()


particles = []


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

    tl = np.array((0.4, 0.5))
    # x = draw_cube_3d(tl, 20)
    x = draw_gyroid_3d(30)
    n_particles = len(x)
    v, F, C, Jp = _make_mpm_objects(n_particles, 3)

    outputs = [x.copy()]
    try:
        pass
        for _ in tqdm(range(5000)):
            solve_mls_mpm_3d(parameters, x, v, F, C, Jp)
            outputs.append(x.copy())
    except Exception as e:
        logger.error(f"Sim crashed out: {e}")

    ti.init(arch=ti.gpu)
    gui = ti.GUI(res=1024)
    i = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        if i >= len(outputs):
            i = 0

        gui.clear(0x112F41)
        gui.rect(np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F)
        gui.circles(map_position_2d(outputs[i]), radius=1.5, color=0xED553B)
        gui.show()
        i += 1


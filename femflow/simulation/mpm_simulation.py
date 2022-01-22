import os
from typing import List

import imgui
import numpy as np
import taichi as ti
from loguru import logger
from tqdm import tqdm

from femflow.numerics.geometry import grid
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.solvers.mpm.parameters import Parameters
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu
from femflow.viz.visualizer.visualizer_window import VisualizerWindow


class MPMSimulationConfigWindow(VisualizerMenu):
    def __init__(
        self, name="Sim Config", flags: List[int] = [imgui.TREE_NODE_DEFAULT_OPEN]
    ):
        super().__init__(name, flags)
        self._register_input("n_timesteps", 500)

    def render(self, **kwargs) -> None:
        imgui.text("Timestamps")
        self._generate_imgui_input("n_timesteps", imgui.input_int, step=100)

        if imgui.button(label="Start Sim"):
            start_sim_button_cb: Callable = self._unpack_kwarg(
                "start_sim_button_cb", callable, **kwargs
            )
            start_sim_button_cb()

        imgui.same_line()

        if imgui.button(label="Reset Sim"):
            reset_sim_button_cb: Callable = self._unpack_kwarg(
                "reset_sim_button_cb", callable, **kwargs
            )
            reset_sim_button_cb()


class MPMSimulationWindow(VisualizerWindow):
    def __init__(self):
        name = "MPM Simulation"
        flags = [imgui.TREE_NODE_DEFAULT_OPEN]
        super().__init__(name, flags)

        self._register_input("dt", 1e-4)
        self._register_input("mass", 1.0)
        self._register_input("volume", 1.0)
        self._register_input("force", -200)
        self._register_input("youngs_modulus", 1e4)
        self._register_input("poissons_ratio", 0.2)
        self._register_input("hardening", 0.7)
        self._register_input("grid_resolution", 80)
        self._register_input("model", 0)
        self.model_options = ["neo_hookean", "elastoplastic"]
        self.add_menu(MPMSimulationConfigWindow())

    @property
    def parameters(self):
        return Parameters(
            self.mass,
            self.volume,
            self.hardening,
            self.youngs_modulus,
            self.poissons_ratio,
            self.gravity,
            self.dt,
            self.grid_resolution,
            self.model,
        )

    def render(self, **kwargs) -> None:
        imgui.text("dt")
        self._generate_imgui_input("dt", imgui.input_float, format="%.6f")
        imgui.text("Mass")
        self._generate_imgui_input("mass", imgui.input_float)
        imgui.text("Volume")
        self._generate_imgui_input("volume", imgui.input_float)
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
        imgui.text("Material Model")
        self._generate_imgui_input("model", imgui.listbox, items=self.model_options)

        if imgui.button("Generate Simulation Model"):
            self._load_model()

    def resize(self, parent_width: int, parent_height: int, **kwargs):
        self.dimensions = (
            int(parent_width * 0.10) if parent_width >= 800 else 140,
            parent_height,
        )
        self.position = (0, 0)

    def _load_model(self):
        print("here")


def draw_cube_2d(tl: np.ndarray, n: int = 10) -> np.ndarray:
    x = np.linspace(*tl, num=n)
    y = np.linspace(*tl, num=n)

    all_pts = []
    for row in x:
        for col in y:
            all_pts.append([row, col])
    return np.array(all_pts, dtype=np.float64)


def draw_gyroid_3d(n: int = 10) -> Mesh:
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
    # cube = g[inside > 0.03] * 0.15
    cube = g[inside > 0.03]
    # for row in cube:
    #     row[0] += 0.4
    #     row[1] += 0.4
    return Mesh(cube)


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


def map_position_2d(a: np.ndarray):
    """Taichi gui only works in 2d for some reason

    Args:
        a (np.ndarray): a
    """
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


def sim_3d():

    tl = np.array((0.4, 0.5))
    # x = draw_cube_3d(tl, 10)
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

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for i, output in enumerate(outputs):
        np.save(f"tmp/{i}", output)

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


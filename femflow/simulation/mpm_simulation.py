import imgui
import numpy as np
import taichi as ti
from solvers.mpm.parameters import MPMParameters

from femflow.numerics.fem import Ev_to_lame_coefficients
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
        mass: float,
        hardening: float,
        E: float,
        v: float,
        gravity: float,
        dt: float,
        grid_resolution: int,
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
        )

        self.grid = np.zeros([])

        self._initialize()

    def _initialize(self):
        self.grid = np.zeros(
            (self.params.grid_resolution + 1, self.params.grid_resolution + 1, 3)
        )

    def advance(self):
        pass


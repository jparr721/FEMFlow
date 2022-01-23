from typing import Callable, List

import imgui

from femflow.numerics.linear_algebra import matrix_to_vector
from femflow.simulation.mpm.simulation import MPMSimulation
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
            start_sim_button_cb(n_timesteps=self.n_timesteps)

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
        self._register_input("grid_resolution", 64)
        self._register_input("model", 0)
        self._register_input("tightening_coeff", 0.5)
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
            self.force,
            self.dt,
            self.grid_resolution,
            self.model,
            self.tightening_coeff,
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
        imgui.text("Tightening Coefficient")
        self._generate_imgui_input("tightening_coeff", imgui.input_float)

        if imgui.button("Load Simulation"):
            load_sim_cb: Callable = self._unpack_kwarg("load_sim_cb", callable, **kwargs)
            mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
            load_sim_cb(parameters=self.parameters, mesh=mesh)

    def resize(self, parent_width: int, parent_height: int, **kwargs):
        self.dimensions = (
            int(parent_width * 0.10) if parent_width >= 800 else 140,
            parent_height,
        )
        self.position = (0, 0)


class MPMDisplacementsWindow(VisualizerWindow):
    def __init__(self):
        name = "Displacements"
        flags = [imgui.TREE_NODE_DEFAULT_OPEN]
        super().__init__(name, flags)

        self._register_input("current_timestep", 0)

    def render(self, **kwargs) -> None:
        sim: MPMSimulation = self._unpack_kwarg("sim", MPMSimulation, **kwargs)
        imgui.push_item_width(-1)
        self._generate_imgui_input(
            "current_timestep",
            imgui.slider_int,
            min_value=0,
            max_value=len(sim.displacements),
        )
        imgui.pop_item_width()

        if sim.loaded:
            sim.mesh.replace(matrix_to_vector(sim.displacements[self.current_timestep]))

    def resize(self, parent_width: int, parent_height: int, **kwargs):
        w = int(parent_width * 0.10) if parent_width >= 800 else 140
        self.dimensions = (
            parent_width - w,
            50,
        )
        self.position = (w, 0)

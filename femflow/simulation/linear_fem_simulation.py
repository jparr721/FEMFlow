from collections import defaultdict
from typing import List, Tuple, Union

import imgui
import numpy as np
from loguru import logger
from scipy.sparse import identity, csr_matrix
from tqdm import tqdm

from femflow.solvers.fea.linear_galerkin_nondynamic import LinearGalerkinNonDynamic
from femflow.solvers.fea.boundary_conditions import (
    basic_dirilecht_boundary_conditions,
    top_bottom_plate_dirilect_conditions,
)
from femflow.solvers.integrators.explicit_central_difference_method import (
    ExplicitCentralDifferenceMethod,
)
from femflow.solvers.material import (
    hookes_law_isotropic_constitutive_matrix,
    hookes_law_orthotropic_constitutive_matrix,
)
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu
from .parsers import *


class LinearFemSimulationMenu(VisualizerMenu):
    def __init__(
        self, name="Simulation Options", flags: List[int] = [imgui.TREE_NODE_DEFAULT_OPEN]
    ):
        super().__init__(name, flags)
        self.material_options = ["isotropic", "orthotropic"]

        self._register_input("dt", 0.001)
        self._register_input("mass", 10)
        self._register_input("force", -100)
        self._register_input("youngs_modulus", "50000")
        self._register_input("poissons_ratio", "0.3")
        self._register_input("shear_modulus", "1000")
        self._register_input("use_damping", False)
        self._register_input("material_type", 0)
        self._register_input("rayleigh_lambda", 0.0)
        self._register_input("rayleigh_mu", 0.0)

        # Initial material coefficients
        self.material_coefficients: Union[Tuple[str, str], Tuple[str, str, str]] = (
            self.youngs_modulus,
            self.poissons_ratio,
        )

    def render(self, **kwargs) -> None:
        imgui.text("dt")
        self._generate_imgui_input("dt", imgui.input_float)
        imgui.text("Mass")
        self._generate_imgui_input("mass", imgui.input_float)
        imgui.text("Force")
        self._generate_imgui_input("force", imgui.input_float)
        self._generate_imgui_input("use_damping", imgui.checkbox, use_key_as_label=True)

        if self.use_damping:
            imgui.text("Rayleigh Lambda")
            self._generate_imgui_input("rayleigh_lambda", imgui.input_float)
            imgui.text("Rayleigh Mu")
            self._generate_imgui_input("rayleigh_mu", imgui.input_float)

        imgui.text("Material Type")
        self._generate_imgui_input(
            "material_type", imgui.listbox, items=self.material_options
        )

        imgui.text("E")
        self._generate_imgui_input("youngs_modulus", imgui.input_text, buffer_length=512)
        imgui.text("v")
        self._generate_imgui_input("poissons_ratio", imgui.input_text, buffer_length=512)
        if self.material_type == 1:
            imgui.text("G")
            self._generate_imgui_input(
                "shear_modulus", imgui.input_text, buffer_length=512
            )
            self.material_coefficients = (
                self.youngs_modulus,
                self.poissons_ratio,
                self.shear_modulus,
            )
        else:
            self.material_coefficients = (self.youngs_modulus, self.poissons_ratio)


class LinearFemSimulation(object):
    def __init__(self, name="linear_galerkin"):
        self.name = name
        self.menu = LinearFemSimulationMenu()
        # Sim parameters
        self.boundary_conditions = defaultdict(np.ndarray)
        self.displacements: List[csr_matrix] = []
        self.loaded = False

    def load(self, mesh: Mesh):
        logger.info("Loading simulation with saved parameters")

        force_nodes, interior_nodes, _ = top_bottom_plate_dirilect_conditions(
            mesh.as_matrix(mesh.vertices, 3)
        )

        try:
            self.boundary_conditions = basic_dirilecht_boundary_conditions(
                np.array([0, self.menu.force, 0]), force_nodes, interior_nodes
            )
        except Exception as e:
            logger.error("Dirilect boundary condition assignment failed")
            logger.error(f"Boundary conditions had error: {e}")
            return

        if self.menu.material_type == 1:
            self.constitutive_matrix = hookes_law_orthotropic_constitutive_matrix(
                np.array(
                    parse_orthotropic_material_coefficients(
                        self.menu.material_coefficients
                    )
                )
            )
        else:
            self.constitutive_matrix = hookes_law_isotropic_constitutive_matrix(
                parse_isotropic_material_coefficients(self.menu.material_coefficients)
            )

        if len(mesh.tetrahedra) == 0:
            logger.error("Mesh is not tetrahedralized, cannot simulate")
            return

        self.reset(mesh)
        self.loaded = True

    def reset(self, mesh: Mesh):
        self.solver = LinearGalerkinNonDynamic(
            self.boundary_conditions,
            self.constitutive_matrix,
            mesh.vertices,
            mesh.tetrahedra,
        )
        logger.success("Solver created")

        mass_matrix = identity(self.solver.K_e.shape[0], format="csr") * self.menu.mass
        self.cd_integrator = ExplicitCentralDifferenceMethod(
            self.menu.dt,
            mass_matrix,
            self.solver.K_e,
            self.solver.U_e,
            self.solver.F_e,
            self.menu.rayleigh_lambda,
            self.menu.rayleigh_mu,
        )
        logger.success("Integrator created")
        self.displacements = [self.solver.U]

    def simulate(self, timesteps: int):
        for _ in tqdm(range(timesteps)):
            self.solver.U_e = self.cd_integrator.integrate(
                self.solver.F_e, self.solver.U_e
            )
            self.solver.solve()
            self.displacements.append(self.solver.U)
        logger.success("Simulation is done")

    def solve_static(self):
        self.solver.solve_static()
        self.displacements.append(self.solver.U)
        logger.success("Simulation is done")


class LinearFemSimulationHeadless(object):
    def __init__(
        self,
        name="linear_galerkin_headless",
        dt=0.001,
        mass=10,
        force=-100,
        youngs_modulus=50000,
        poissons_ratio=0.3,
        shear_modulus=1000,
        material_type=0,
        rayleigh_lambda=0.0,
        rayleigh_mu=0.0,
    ):
        self.name = name
        self.dt = dt
        self.mass = mass
        self.force = force
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.shear_modulus = shear_modulus
        self.material_type = material_type
        self.rayleigh_lambda = rayleigh_lambda
        self.rayleigh_mu = rayleigh_mu
        self.displacements = []

        self.force_nodes = np.array([])

    def load(self, mesh: Mesh):
        self.force_nodes, interior_nodes, _ = top_bottom_plate_dirilect_conditions(
            mesh.as_matrix(mesh.vertices, 3)
        )
        boundary_conditions = basic_dirilecht_boundary_conditions(
            np.array([0, self.force, 0]), self.force_nodes, interior_nodes
        )
        constitutive_matrix = hookes_law_isotropic_constitutive_matrix(
            np.array((self.youngs_modulus, self.poissons_ratio))
        )
        self.solver = LinearGalerkinNonDynamic(
            boundary_conditions, constitutive_matrix, mesh.vertices, mesh.tetrahedra
        )

        mass_matrix = identity(self.solver.K_e.shape[0], format="csr") * self.mass
        self.integrator = ExplicitCentralDifferenceMethod(
            self.dt, mass_matrix, self.solver.K_e, self.solver.U_e, self.solver.F_e
        )

    def solve_dynamic(self, timesteps: int):
        for _ in tqdm(range(timesteps)):
            self.solver.U_e = self.integrator.integrate(self.solver.F_e, self.solver.U_e)
            self.solver.solve()
            self.displacements.append(self.solver.U)

    def solve_static(self):
        self.solver.solve_static()
        self.displacements.append(self.solver.U)


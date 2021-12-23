from collections import defaultdict

import imgui
import numpy as np
from femflow.solvers.fea import galerkin2
from femflow.solvers.fea.boundary_conditions import (
    basic_dirilecht_boundary_conditions,
    top_bottom_plate_dirilect_conditions,
)
from femflow.solvers.integrators.explicit_central_difference_method import ExplicitCentralDifferenceMethod
from femflow.solvers.material import (
    hookes_law_isotropic_constitutive_matrix,
    hookes_law_orthotropic_constitutive_matrix,
)
from femflow.viz.mesh import Mesh
from loguru import logger
from tqdm import tqdm

from .environment import Environment


class LinearFemSimulation(Environment):
    def __init__(self, name="linear_galerkin"):
        super().__init__(name)
        self.dt = 0.001
        self.mass = 10
        self.force = -100
        self.youngs_modulus = "50000"
        self.poissons_ratio = "0.3"
        self.shear_modulus = "1000"
        self.use_damping = False
        self.material_type = 0
        self.material_options = ["isotropic", "orthotropic"]
        self.rayleigh_lambda = 0.0
        self.rayleigh_mu = 0.0

        # Sim parameters
        self.boundary_conditions = defaultdict(np.ndarray)
        self.displacements = []

    def load(self, mesh: Mesh):
        logger.info("Loading simulation with saved parameters")

        force_nodes, interior_nodes, _ = top_bottom_plate_dirilect_conditions(mesh.as_matrix(mesh.vertices, 3))

        try:
            self.boundary_conditions = basic_dirilecht_boundary_conditions(
                np.array([0, self.force, 0]), force_nodes, interior_nodes
            )
        except Exception as e:
            logger.error("Dirilect boundary condition assignment failed")
            logger.error(f"Boundary conditions had error: {e}")
            return

        self.U_e = np.zeros(len(self.boundary_conditions) * 3)
        self.U = np.zeros(mesh.vertices.size)
        self.displacements.append(self.U)

        if self.material_type == "orthotropic":
            if len(self.material_coefficients) != 3:
                logger.error("Unable to properly deconstruct material coefficients!")
                logger.error(f"Got options: {self.material_coefficients}")
                return

            youngs_modulus, poissons_ratio, shear_modulus = self.material_coefficients
            try:
                shear_modulus = "1000,1000,1000"
                _, shear_modulus = imgui.input_text(label="G")

                E_vals = youngs_modulus.split(",")

                # Copy value for all entries
                if len(E_vals) == 1:
                    value = float(E_vals[0])
                    youngs_modulus = np.array((value, value, value))
                elif len(E_vals) == 3:
                    values = map(float, E_vals)
                    youngs_modulus = np.array(E_vals)
                else:
                    logger.error("Invalid number of younds modulus' provided for orthotropic material")
                    return

                v_vals = poissons_ratio.split(",")
                if len(v_vals) == 1:
                    value = float(v_vals[0])
                    poissons_ratio = np.array((value, value, value, value, value, value))
                elif len(E_vals) == 3:
                    values = map(float, v_vals)
                    poissons_ratio = np.array((*values, *values))
                elif len(E_vals) == 6:
                    values = map(float, v_vals)
                    poissons_ratio = np.array(values)
                else:
                    logger.error("Invalid number of poissons ratios provided for orthotropic material")
                    return

                G_vals = shear_modulus.split(",")
                if len(G_vals) == 1:
                    value = float(G_vals[0])
                    shear_modulus = np.array((value, value, value))
                elif len(G_vals) == 3:
                    value = map(float, G_vals)
                    shear_modulus = np.array(G_vals)
                else:
                    logger.error("Invalid number of shear moduli provided for orthotropic material")
            except Exception as e:
                logger.error(
                    "Failed to parse youngs modulus, poissions ratio, and shear modulus for orthhotropic material"
                )
                logger.error(f"Stack trace was: {repr(e)}")
                return

            self.constitutive_matrix = hookes_law_orthotropic_constitutive_matrix(
                np.array((*youngs_modulus, *poissons_ratio, *shear_modulus))
            )

        else:
            if len(self.material_coefficients) != 2:
                logger.error("Unable to properly deconstruct material coefficients!")
                logger.error(f"Got options: {self.material_coefficients}")
                return
            youngs_modulus, poissons_ratio = self.material_coefficients
            try:
                self.material_coefficients = (youngs_modulus, poissons_ratio)
                self.constitutive_matrix = hookes_law_isotropic_constitutive_matrix(
                    np.array((float(youngs_modulus), float(poissons_ratio)))
                )
            except Exception as e:
                logger.error("Failed to parse youngs modulus and poissions ratio for isotropic material")
                logger.error(f"Stack trace was: {repr(e)}")
                return

        if len(mesh.tetrahedra) == 0:
            logger.error("Mesh is not tetrahedralized, cannot simulate")
            return

        self.reset(mesh)
        self.loaded = True

    def menu(self):
        imgui.text("dt")
        _, self.dt = imgui.input_double("##dt", self.dt)
        imgui.text("Mass")
        _, self.mass = imgui.input_double("##Mass", self.mass)
        imgui.text("Force")
        _, self.force = imgui.input_double("##Force", self.force)

        _, self.use_damping = imgui.checkbox("Use Damping", self.use_damping)

        if self.use_damping:
            imgui.text("Rayleigh Lambda")
            _, self.rayleigh_lambda = imgui.input_double("##lambda", self.rayleigh_lambda)
            imgui.text("Rayleigh Mu")
            _, self.rayleigh_mu = imgui.input_double("##mu", self.rayleigh_mu)

        imgui.text("Material Type")
        _, self.material_type = imgui.listbox("##Material Type", self.material_type, ["isotropic", "orthotropic"])

        imgui.text("E")
        _, self.youngs_modulus = imgui.input_text("##E", self.youngs_modulus, 512)
        imgui.text("v")
        _, self.poissons_ratio = imgui.input_text("##v", self.poissons_ratio, 512)
        if self.material_type == 1:
            imgui.text("G")
            _, self.shear_modulus = imgui.input_text("##G", self.shear_modulus, 512)

        if self.material_type == 0:
            self.material_coefficients = (self.youngs_modulus, self.poissons_ratio)
        else:
            self.material_coefficients = (self.youngs_modulus, self.poissons_ratio, self.shear_modulus)

    def reset(self, mesh: Mesh):
        self.displacements = [np.zeros(mesh.vertices.size)]
        self.solver = galerkin2.LinearGalerkinNonDynamic(
            self.boundary_conditions, self.constitutive_matrix, mesh.vertices, mesh.tetrahedra
        )

        mass_matrix = np.eye(self.solver.K_e.shape[0]) * self.mass
        self.cd_integrator = ExplicitCentralDifferenceMethod(
            self.dt,
            mass_matrix,
            self.solver.K_e,
            self.solver.U_e,
            self.solver.F_e,
            rayleigh_lambda=self.rayleigh_lambda,
            rayleigh_mu=self.rayleigh_mu,
        )

    def simulate(self, mesh: Mesh, timesteps: int):
        for _ in tqdm(range(timesteps)):
            self.U_e = self.cd_integrator.integrate(self.solver.F_e, self.solver.U_e)
            self.solver.U_e = self.U_e
            self.solver.solve()
            self.U = self.solver.U
            self.displacements.append(self.U)
        logger.success("Simulation is done")

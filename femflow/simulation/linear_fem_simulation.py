from collections import defaultdict

import imgui
import numpy as np
from loguru import logger
from solvers.fea.boundary_conditions import top_bottom_plate_dirilect_conditions
from solvers.fea.linear_galerkin_fea import (
    assemble_boundary_forces,
    assemble_element_stiffness_matrix,
    assemble_global_stiffness_matrix,
    assemble_shape_fn_matrix,
    compute_U_from_active_dofs,
)
from solvers.integrators.explicit_central_difference_method import ExplicitCentralDifferenceMethod
from solvers.material import hookes_law_isotropic_constitutive_matrix, hookes_law_orthotropic_constitutive_matrix
from viz.mesh import Mesh

from .environment import Environment


class LinearFemSimulation(Environment):
    def __init__(self, name="linear_galerkin"):
        super().__init__(name)
        self.dt = 0.28
        self.mass = 10
        self.youngs_modulus = "50000"
        self.poissons_ratio = "0.3"
        self.shear_modulus = "1000"
        self.use_damping = False
        self.material_type = 0
        self.material_options = ["isotropic", "orthotropic"]
        self.rayleigh_lambda = 0.5
        self.rayleigh_mu = 0.5

        # Sim parameters
        self.K_e = np.array([])
        self.F_e = np.array([])
        self.U_e = np.array([])
        self.U = np.array([])
        self.boundary_conditions = defaultdict(np.ndarray)
        self.displacements = []

    def load(self, mesh: Mesh):
        logger.info("Loading Simulation With Saved Parameters")

        self.boundary_conditions = top_bottom_plate_dirilect_conditions(mesh.as_matrix(mesh.vertices, 3))
        self.U_e = np.zeros(len(self.boundary_conditions) * 3)
        self.U = mesh.vertices.size
        self.displacements.append(self.U)

        if self.material_type == "orthotropic":
            if len(self.material_coefficients) != 3:
                logger.error("Unable to properly deconstruct material coefficients!")
                logger.error(f"Got options: {self.material_coefficients}")
                return None

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
                    return None

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
                    return None

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
                return None

            self.constitutive_matrix = hookes_law_orthotropic_constitutive_matrix(
                np.array((*youngs_modulus, *poissons_ratio, *shear_modulus))
            )

        else:
            if len(self.material_coefficients) != 2:
                logger.error("Unable to properly deconstruct material coefficients!")
                logger.error(f"Got options: {self.material_coefficients}")
                return None
            youngs_modulus, poissons_ratio = self.material_coefficients
            try:
                self.material_coefficients = (youngs_modulus, poissons_ratio)
                self.constitutive_matrix = hookes_law_isotropic_constitutive_matrix(
                    np.array((float(youngs_modulus), float(poissons_ratio)))
                )
            except Exception as e:
                logger.error("Failed to parse youngs modulus and poissions ratio for isotropic material")
                logger.error(f"Stack trace was: {repr(e)}")
                return None

        if mesh.tetrahedra is None:
            logger.error("Mesh is not tetrahedralized, cannot simulate")
            return None

    def menu(self):
        imgui.text("dt")
        _, self.dt = imgui.input_double("##dt", self.dt)
        imgui.text("Mass")
        _, self.mass = imgui.input_double("##Mass", self.mass)

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
        element_stiffnesses = []
        for row in mesh.tetrahedra:
            B = assemble_shape_fn_matrix(*mesh.vertices[row])
            element_stiffnesses.append(
                assemble_element_stiffness_matrix(row, mesh.vertices, B, self.constitutive_matrix)
            )

        K = assemble_global_stiffness_matrix(element_stiffnesses, 3 * len(mesh.vertices))
        self.K_e, self.F_e = assemble_boundary_forces(K, self.dirilect_boundary_conditions)

        self.U_e = np.zeros(len(self.dirilect_boundary_conditions) * 3)

        self.cd_integrator = ExplicitCentralDifferenceMethod(
            self.dt,
            self.point_mass,
            self.K_e,
            self.U_e,
            self.F_e,
            rayleigh_lambda=self.rayleigh_lambda,
            rayleigh_mu=self.rayleigh_mu,
        )

    def simulate(self, mesh: Mesh, timesteps: int):
        for i in range(timesteps):
            if i % 10 == 0:
                logger.info(f"Timestep: {i}")

            self.U_e = self.cd_integrator.integrate(self.F_e, self.U_e)
            self.U = compute_U_from_active_dofs(mesh.vertices.size, self.U_e, self.boundary_conditions)
            self.displacements.append(self.U)

        logger.success("Simulation is done")

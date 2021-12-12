from typing import Union

import imgui
import numpy as np
from loguru import logger
from solvers.fea.linear_galerkin_fea import (
    BoundaryConditions,
    assemble_boundary_forces,
    assemble_element_stiffness_matrix,
    assemble_global_stiffness_matrix,
    assemble_shape_fn_matrix,
    compute_U_from_active_dofs,
)
from solvers.integrators.explicit_central_difference_method import ExplicitCentralDifferenceMethod
from solvers.material import hookes_law_isotropic_constitutive_matrix, hookes_law_orthotropic_constitutive_matrix

from .environment import Environment

use_damping = False
material_type = 0
material_options = ["isotropic", "orthotropic"]


def make_linear_galerkin_parameter_menu() -> np.ndarray:
    global use_damping, material_type, material_options

    dt = 0.28
    mass = 10
    imgui.text("dt")
    _, dt = imgui.input_double("", dt)
    imgui.text("Mass")
    _, dt = imgui.input_double("", mass)

    _, use_damping = imgui.checkbox("Use Damping", use_damping)

    rayleigh_lambda = 0.0
    rayleigh_mu = 0.0
    if use_damping:
        imgui.text("Rayleigh Lambda")
        _, rayleigh_lambda = imgui.input_double("", rayleigh_lambda)
        imgui.text("Rayleigh Mu")
        _, rayleigh_mu = imgui.input_double("", rayleigh_mu)

    imgui.text("Material Type")
    _, material_type = imgui.listbox("", material_type, ["isotropic", "orthotropic"])

    constitutive_matrix = np.array([])
    youngs_modulus = "50000"
    poissons_ratio = "0.3"
    _, youngs_modulus = imgui.input_text("E", youngs_modulus, 512)
    _, poissons_ratio = imgui.input_text("v", poissons_ratio, 512)
    if material_type == "orthotropic":
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
            logger.error("Failed to parse youngs modulus, poissions ratio, and shear modulus for orthhotropic material")
            logger.error(f"Stack trace was: {repr(e)}")

        constitutive_matrix = hookes_law_orthotropic_constitutive_matrix(
            np.array((*youngs_modulus, *poissons_ratio, *shear_modulus))
        )

    else:
        try:
            youngs_modulus = float(youngs_modulus)
            poissons_ratio = float(poissons_ratio)
        except Exception as e:
            logger.error("Failed to parse youngs modulus and poissions ratio for isotropic material")
            logger.error(f"Stack trace was: {repr(e)}")

        constitutive_matrix = hookes_law_isotropic_constitutive_matrix(np.array((youngs_modulus, poissons_ratio)))

    return (dt, mass, constitutive_matrix, material_options[material_type], rayleigh_lambda, rayleigh_mu)


def make_linear_galerkin_simulation(
    dt: float,
    point_mass: float,
    v: np.ndarray,
    t: np.ndarray,
    constitutive_matrix: np.ndarray,
    dirilect_boundary_conditions: BoundaryConditions,
    rayleigh_lambda=0.5,
    rayleigh_mu=0.5,
) -> Union[Environment, None]:
    def reset_simulation():
        element_stiffnesses = []
        for row in t:
            B = assemble_shape_fn_matrix(*v[row])
            element_stiffnesses.append(assemble_element_stiffness_matrix(row, v, B, constitutive_matrix))

        K = assemble_global_stiffness_matrix(element_stiffnesses, 3 * len(v))
        K_e, F_e = assemble_boundary_forces(K, dirilect_boundary_conditions)

        U_e = np.zeros(len(dirilect_boundary_conditions) * 3)

        cd_integrator = ExplicitCentralDifferenceMethod(
            dt, point_mass, K_e, U_e, F_e, rayleigh_lambda=rayleigh_lambda, rayleigh_mu=rayleigh_mu
        )
        return cd_integrator

    def step_forward(U_e: np.ndarray):
        return compute_U_from_active_dofs(v.size, U_e, dirilect_boundary_conditions)

    def integrate(cd_integrator: ExplicitCentralDifferenceMethod, F_e: np.ndarray, U_e: np.ndarray) -> np.ndarray:
        return cd_integrator.integrate(F_e, U_e)

    return Environment(step_function=step_forward, integrator=integrate, reset_function=reset_simulation)

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


def make_linear_galerkin_parameter_menu(material_type: str) -> np.ndarray:
    _, youngs_modulus = imgui.input_text(label="E")
    _, poissons_ratio = imgui.input_text(label="v")
    if material_type == "orthotropic":
        try:
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

        return np.ndarray((*youngs_modulus, *poissons_ratio, *shear_modulus))

    else:
        try:
            youngs_modulus = float(youngs_modulus)
            poissons_ratio = float(poissons_ratio)
        except Exception as e:
            logger.error("Failed to parse youngs modulus and poissions ratio for isotropic material")
            logger.error(f"Stack trace was: {repr(e)}")

        return np.ndarray((youngs_modulus, poissons_ratio))


def make_linear_galerkin_simulation(
    dt: float,
    point_mass: float,
    v: np.ndarray,
    t: np.ndarray,
    material_coefficients: np.ndarray,
    dirilect_boundary_conditions: BoundaryConditions,
    material_type="isotropic",
    rayleigh_lambda=0.5,
    rayleigh_mu=0.5,
) -> Union[Environment, None]:
    def reset_simulation():
        if not (material_type == "isotropic" or material_type == "orthotropic"):
            logger.error("Material type specified is invalid! Exiting")
            return None

        D = None
        if material_type == "isotropic":
            D = hookes_law_isotropic_constitutive_matrix(material_coefficients)

        if material_type == "orthotropic":
            D = hookes_law_orthotropic_constitutive_matrix(material_coefficients)

        element_stiffnesses = []
        for row in t:
            B = assemble_shape_fn_matrix(*v[row])
            element_stiffnesses.append(assemble_element_stiffness_matrix(row, v, B, D))

        K = assemble_global_stiffness_matrix(element_stiffnesses, 3 * len(v))
        K_e, F_e = assemble_boundary_forces(K, dirilect_boundary_conditions)

        U_e = np.zeros(len(dirilect_boundary_conditions))

        cd_integrator = ExplicitCentralDifferenceMethod(
            dt, point_mass, K_e, U_e, F_e, rayleigh_lambda=rayleigh_lambda, rayleigh_mu=rayleigh_mu
        )
        return cd_integrator

    def start_simulation():
        pass

    def step_forward(U_e: np.ndarray):
        return compute_U_from_active_dofs(v.size, U_e, dirilect_boundary_conditions)

    def integrate(cd_integrator: ExplicitCentralDifferenceMethod, F_e: np.ndarray, U_e: np.ndarray) -> np.ndarray:
        return cd_integrator.integrate(F_e, U_e)

    return Environment(
        step_function=step_forward,
        integrator=integrate,
        reset_function=reset_simulation,
        start_function=start_simulation,
    )

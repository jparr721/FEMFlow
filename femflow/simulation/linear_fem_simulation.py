from typing import Tuple, Union

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

# Params
__imgui = {
    "dt": 0.28,
    "mass": 10,
    "youngs_modulus": "50000",
    "poissons_ratio": "0.3",
    "shear_modulus": "1000",
    "use_damping": False,
    "material_type": 0,
    "material_options": ["isotropic", "orthotropic"],
    "rayleigh_lambda": 0.5,
    "rayleigh_mu": 0.5,
}


def make_linear_galerkin_parameter_menu() -> np.ndarray:
    global __imgui

    imgui.text("dt")
    _, __imgui["dt"] = imgui.input_double("##dt", __imgui["dt"])
    imgui.text("Mass")
    _, __imgui["mass"] = imgui.input_double("##Mass", __imgui["mass"])

    _, __imgui["use_damping"] = imgui.checkbox("Use Damping", __imgui["use_damping"])

    if __imgui["use_damping"]:
        imgui.text("Rayleigh Lambda")
        _, __imgui["rayleigh_lambda"] = imgui.input_double("##lambda", __imgui["rayleigh_lambda"])
        imgui.text("Rayleigh Mu")
        _, __imgui["rayleigh_mu"] = imgui.input_double("##mu", __imgui["rayleigh_mu"])

    imgui.text("Material Type")
    _, __imgui["material_type"] = imgui.listbox(
        "##Material Type", __imgui["material_type"], ["isotropic", "orthotropic"]
    )

    imgui.text("E")
    _, __imgui["youngs_modulus"] = imgui.input_text("##E", __imgui["youngs_modulus"], 512)
    imgui.text("v")
    _, __imgui["poissons_ratio"] = imgui.input_text("##v", __imgui["poissons_ratio"], 512)
    if __imgui["material_type"] == "orthotropic":
        imgui.text("G")
        _, __imgui["shear_modulus"] = imgui.input_text("##G", __imgui["shear_modulus"], 512)

    material_coefficients = (__imgui["youngs_modulus"], __imgui["poissons_ratio"], __imgui["shear_modulus"])

    if imgui.button(label="Load"):
        logger.info("Loading Simulation With Saved Parameters")
        print(
            __imgui["dt"],
            __imgui["mass"],
            material_coefficients,
            __imgui["material_options"][__imgui["material_type"]],
            __imgui["rayleigh_lambda"],
            __imgui["rayleigh_mu"],
        )


def make_linear_galerkin_simulation(
    dt: float,
    point_mass: float,
    v: np.ndarray,
    t: np.ndarray,
    material_type: str,
    material_coefficients: Union[Tuple[str, str], Tuple[str, str, str]],
    dirilect_boundary_conditions: BoundaryConditions,
    rayleigh_lambda=0.5,
    rayleigh_mu=0.5,
) -> Union[Environment, None]:
    if material_type == "orthotropic":
        if len(material_coefficients) != 3:
            logger.error("Unable to properly deconstruct material coefficients!")
            logger.error(f"Got options: {material_coefficients}")
            return None

        youngs_modulus, poissons_ratio, shear_modulus = material_coefficients
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
            return None

        constitutive_matrix = hookes_law_orthotropic_constitutive_matrix(
            np.array((*youngs_modulus, *poissons_ratio, *shear_modulus))
        )

    else:
        if len(material_coefficients) != 2:
            logger.error("Unable to properly deconstruct material coefficients!")
            logger.error(f"Got options: {material_coefficients}")
        youngs_modulus, poissons_ratio = material_coefficients
        try:
            material_coefficients = (youngs_modulus, poissons_ratio)
            constitutive_matrix = hookes_law_isotropic_constitutive_matrix(
                np.array((float(youngs_modulus), float(poissons_ratio)))
            )
        except Exception as e:
            logger.error("Failed to parse youngs modulus and poissions ratio for isotropic material")
            logger.error(f"Stack trace was: {repr(e)}")
            return None

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

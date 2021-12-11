import numpy as np
from solvers.fea.linear_galerkin_fea import (
    BoundaryConditions,
    assemble_boundary_forces,
    assemble_element_stiffness_matrix,
    assemble_global_stiffness_matrix,
    assemble_shape_fn_matrix,
    compute_U,
)
from solvers.integrators.explicit_central_difference_method import ExplicitCentralDifferenceMethod
from solvers.material import hookes_law_isotropic_constitutive_matrix, hookes_law_orthotropic_constitutive_matrix

from .environment import Environment


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
) -> Environment:
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

    _, U_e = compute_U(K, K_e, F_e, dirilect_boundary_conditions)

    def step_function():
        pass

    integrator = ExplicitCentralDifferenceMethod(
        dt, point_mass, K_e, U_e, F_e, rayleigh_lambda=rayleigh_lambda, rayleigh_mu=rayleigh_mu
    )

    def integrator():
        integrator.integrate(F_e, U_e)

    return Environment(step_function, integrator)

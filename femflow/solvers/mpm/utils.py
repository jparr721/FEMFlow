import numba as nb
import numpy as np

from femflow.numerics.linear_algebra import polar_decomp_2d, polar_decomp_3d


@nb.njit
def constant_hardening(mu_0: float, lambda_0: float, e: float):
    """Compute the hardening of the plasticity model as
    F^n+1 = F^n+1_E + F_n+1_P
    Where each component is the elastic and plastic components of the hardening model.
    This is simplified as:
    mu(F_P) = mu_0 * e^epsilon(1 - J_p)
    lambda(F_P) = lambda_0 * e^epsilon(1 - J_p)
    J_p (volume) is provided by the particle, so we just compute the value of e and
    multiply through in this implementation.
    Args:
        mu_0 (float): The initial mu value
        lambda_0 (float): The initial mu value
        e (float): The hardness coefficient.
    Returns:
        Tuple[float, float]: The mu and lambda Lame's coefficients
    """
    return mu_0 * e, lambda_0 * e


@nb.njit
def snow_hardening(mu_0: float, lambda_0: float, h: float, Jp: float):
    """Compute the hardening of the plasticity model as
    F^n+1 = F^n+1_E + F_n+1_P
    Where each component is the elastic and plastic components of the hardening model.
    This is simplified as:
    mu(F_P) = mu_0 * e^epsilon(1 - J_p)
    lambda(F_P) = lambda_0 * e^epsilon(1 - J_p)
    J_p (volume) is provided by the particle, so we just compute the value of e and
    multiply through in this implementation.

    Args:
        mu_0 (float): The initial mu value
        lambda_0 (float): The initial mu value
        h (float): The hardness multiplier
        Jp (float): The volume of the particle p
        e (float): The hardness coefficient.
    Returns:
        Tuple[float, float]: The mu and lambda Lame's coefficients

    """
    e = np.exp(h * (1.0 - Jp))[0]
    return constant_hardening(mu_0, lambda_0, e)


@nb.njit
def fixed_corotated_stress_2d(
    F: np.ndarray,
    inv_dx: float,
    mu: float,
    lambda_: float,
    dt: float,
    volume: float,
    mass: float,
    C: np.ndarray,
):
    """Computes the fixed corotated stress of the particle following snow plasticity.
    It utilizes the following formula:
    P(F) = grad(strain_energy_density) with respect to deformation gradient;
    This translates to 2 * mu * (F - R) + lambda * (J - 1) * J * F^-T
    Where mu and lambda are our neo-hookean material coefficients, J is the
    jacobian, F is the deformation gradient, and R is the rotation component
    from the polar decomposition of the deformation gradient. This gives us our
    co-rotated stress responses in the snow plasticity model.
    Args:
        params (MPMParameters): The simulation parameters
        particle (NeoHookeanParticle): The simulation particle
    Returns:
        np.ndarray: The affine particle-in-a-cell momentum fused with cauchy stress
    """
    J = np.linalg.det(F)

    # F = r, s; Rotation Matrix and Symmetric Matrix
    r, _ = polar_decomp_2d(F)

    # Inverse density is our constant scaling factor for our APIC momentum/stres
    D_inv = 4 * inv_dx * inv_dx

    # Scaled cauchy stress
    PF = (2 * mu * (F - r) @ F.T) + lambda_ * (J - 1) * J
    stress = -(dt * volume) * (D_inv * PF)

    # Fused APIC momentum + MLS-MPM stress contribution
    # See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
    # Eqn 29
    return stress + mass * C


@nb.njit
def fixed_corotated_stress_3d(
    F: np.ndarray,
    inv_dx: float,
    mu: float,
    lambda_: float,
    dt: float,
    volume: float,
    mass: float,
    C: np.ndarray,
):
    """Computes the fixed corotated stress of the particle following snow plasticity.
    It utilizes the following formula:
    P(F) = grad(strain_energy_density) with respect to deformation gradient;
    This translates to 2 * mu * (F - R) + lambda * (J - 1) * J * F^-T
    Where mu and lambda are our neo-hookean material coefficients, J is the
    jacobian, F is the deformation gradient, and R is the rotation component
    from the polar decomposition of the deformation gradient. This gives us our
    co-rotated stress responses in the snow plasticity model.
    Args:
        params (MPMParameters): The simulation parameters
        particle (NeoHookeanParticle): The simulation particle
    Returns:
        np.ndarray: The affine particle-in-a-cell momentum fused with cauchy stress
    """
    J = np.linalg.det(F)

    # F = r, s; Rotation Matrix and Symmetric Matrix
    r, _ = polar_decomp_3d(F)

    # Inverse density is our constant scaling factor for our APIC momentum/stres
    D_inv = 4 * inv_dx * inv_dx

    # Scaled cauchy stress
    PF = (2 * mu * (F - r) @ F.T) + lambda_ * (J - 1) * J
    stress = -(dt * volume) * (D_inv * PF)

    # Fused APIC momentum + MLS-MPM stress contribution
    # See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
    # Eqn 29
    return stress + mass * C

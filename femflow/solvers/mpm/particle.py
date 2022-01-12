from dataclasses import dataclass

import numpy as np


@dataclass
class NeoHookeanParticle(object):
    # x
    position: np.ndarray

    # m
    mass: float

    # v
    velocity: np.ndarray

    # Affine Matrix B_p
    affine_momentum: np.ndarray

    # F
    deformation_gradient: np.ndarray

    # J_p, determinant of the deformation gradient
    volume: float

    color: np.ndarray

    # Lame's coefficients for the neo-hookean model
    lambda_: float
    mu: float


def make_particle(x: np.ndarray, v: np.ndarray, c: np.ndarray):
    return NeoHookeanParticle(
        position=x,
        mass=1,
        velocity=v,
        affine_momentum=np.zeros((2, 2)),
        deformation_gradient=np.eye(2),
        volume=1,
        color=c,
        lambda_=0,
        mu=0,
    )

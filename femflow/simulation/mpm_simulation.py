import os

import numpy as np
import taichi as ti
from loguru import logger
from tqdm import tqdm

from femflow.simulation.mpm.primitives import (
    generate_cube_points,
    generate_implicit_points,
    gyroid,
)
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.solvers.mpm.parameters import Parameters


def map_position_2d(a: np.ndarray):
    """Taichi gui only works in 2d for some reason

    Args:
        a (np.ndarray): a
    """
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


def scale(A):
    return (A - np.min(A)) / (np.max(A) - np.min(A))


def sim_3d():
    mass = 1.0
    volume = 1.0
    gravity = -200
    dt = 1e-4
    youngs_modulus = 1e4
    poissons_ratio = 0.2
    hardening = 0.7
    grid_resolution = 64
    model = "neo_hookean"
    parameters = Parameters(
        mass,
        volume,
        hardening,
        youngs_modulus,
        poissons_ratio,
        gravity,
        dt,
        grid_resolution,
        model,
    )

    x = generate_implicit_points(gyroid, 0.3, 0.3, 60) * 0.5
    # x[:, 1] += 0.5
    # x[:, 0] += 0.5
    # x = generate_cube_points(np.array((0.4, 0.5)), 10)
    dim = 3
    v = np.zeros((len(x), dim), dtype=np.float64)

    # Deformation Gradient
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(len(x))])

    # Affine Momentum (MLS MPM)
    C = np.zeros((len(x), dim, dim), dtype=np.float64)

    # Volume (jacobian)
    Jp = np.ones((len(x), 1), dtype=np.float64)

    outputs = [x.copy()]
    try:
        for _ in tqdm(range(2)):
            solve_mls_mpm_3d(parameters, x, v, F, C, Jp)
            outputs.append(x.copy())
    except Exception as e:
        logger.error(f"Sim crashed out: {e}")

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for i, output in enumerate(outputs):
        np.save(f"tmp/{i}", output)

    ti.init(arch=ti.gpu)
    gui = ti.GUI(res=1024)
    i = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        if i >= len(outputs):
            i = 0
        gui.clear(0x112F41)
        gui.rect(np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F)
        gui.circles(map_position_2d(outputs[i]), radius=1.5, color=0xED553B)
        gui.show()
        i += 1


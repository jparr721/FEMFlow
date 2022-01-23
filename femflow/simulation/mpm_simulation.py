import os

import numpy as np
import taichi as ti
from loguru import logger
from tqdm import tqdm

from femflow.numerics.geometry import grid
from femflow.solvers.mpm.mls_mpm import solve_mls_mpm_3d
from femflow.viz.mesh import Mesh


def draw_cube_2d(tl: np.ndarray, n: int = 10) -> np.ndarray:
    x = np.linspace(*tl, num=n)
    y = np.linspace(*tl, num=n)

    all_pts = []
    for row in x:
        for col in y:
            all_pts.append([row, col])
    return np.array(all_pts, dtype=np.float64)


def draw_gyroid_3d(n: int = 10) -> Mesh:
    g = grid(np.array((n, n, n)))

    def _fn(amplitude: float, pos: np.ndarray) -> float:
        two_pi = (2.0 * np.pi) / amplitude
        x, y, z = pos
        return (
            np.sin(two_pi * x) * np.cos(two_pi * y)
            + np.sin(two_pi * y) * np.cos(two_pi * z)
            + np.sin(two_pi * z) * np.cos(two_pi * x)
            - 0.3
        )

    inside = np.array([_fn(0.3, row) for row in g])
    # cube = g[inside > 0.03] * 0.15
    cube = g[inside > 0.03]
    # for row in cube:
    #     row[0] += 0.4
    #     row[1] += 0.4
    return Mesh(cube)


def draw_cube_3d(tl: np.ndarray, n: int = 10) -> np.ndarray:
    x = np.linspace(*tl, num=n)
    y = np.linspace(*tl, num=n)
    z = np.linspace(*tl, num=n)

    all_pts = []
    for layer in x:
        for row in y:
            for col in z:
                all_pts.append([layer, row, col])
    return np.array(all_pts, dtype=np.float64)


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


# def sim_3d():

#     tl = np.array((0.4, 0.5))
#     # x = draw_cube_3d(tl, 10)
#     x = draw_gyroid_3d(30)
#     n_particles = len(x)
#     v, F, C, Jp = _make_mpm_objects(n_particles, 3)

#     outputs = [x.copy()]
#     try:
#         pass
#         for _ in tqdm(range(5000)):
#             solve_mls_mpm_3d(parameters, x, v, F, C, Jp)
#             outputs.append(x.copy())
#     except Exception as e:
#         logger.error(f"Sim crashed out: {e}")

#     if not os.path.exists("tmp"):
#         os.mkdir("tmp")

#     for i, output in enumerate(outputs):
#         np.save(f"tmp/{i}", output)

#     ti.init(arch=ti.gpu)
#     gui = ti.GUI(res=1024)
#     i = 0
#     while gui.running and not gui.get_event(gui.ESCAPE):
#         if i >= len(outputs):
#             i = 0

#         gui.clear(0x112F41)
#         gui.rect(np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F)
#         gui.circles(map_position_2d(outputs[i]), radius=1.5, color=0xED553B)
#         gui.show()
#         i += 1


import numba as nb
import numpy as np
from numerics.bintensor3 import bintensor3
from numerics.geometry import grid

# void meshing::implicit_surfaces::ComputeImplicitGyroidMarchingCubes(Real amplitude, Real thickness,
#                                                                     unsigned int resolution,
#                                                                     const GyroidImplicitFunction &fn, MatrixXr &V,
#                                                                     MatrixXi &F, Tensor3r &scalar_field) {
#     MatrixXr GV;
#     igl::grid(RowVector3r(resolution, resolution, resolution), GV);
#     VectorXr GF(GV.rows());
#     igl::parallel_for(GV.rows(), [&](const int i) { GF(i) = fn(amplitude, GV.row(i)); });
#     scalar_field = Tensor3r::Expand(GF, resolution, resolution, resolution);

#     Tensor3r renderable_scalar_field = Tensor3r(scalar_field);
#     scalar_field = scalar_field.MakeBinary();

#     const MatrixXr zero = MatrixXr::Zero(resolution, resolution);
#     renderable_scalar_field.SetTop(0, zero);
#     renderable_scalar_field.SetTop(resolution - 1, zero);
#     renderable_scalar_field.SetLayer(0, zero);
#     renderable_scalar_field.SetLayer(resolution - 1, zero);
#     renderable_scalar_field.SetSide(0, zero);
#     renderable_scalar_field.SetSide(resolution - 1, zero);

#     GF = renderable_scalar_field.Vector();

#     igl::marching_cubes(GF, GV, resolution, resolution, resolution, thickness, V, F);
# }


@nb.njit(parallel=True)
def gyroid(amplitude: float, resolution: int) -> bintensor3:
    def _fn(amplitude: float, pos: np.ndarray) -> float:
        two_pi = (2.0 * np.pi) / amplitude
        x, y, z = pos
        return (
            np.sin(two_pi * x) * np.cos(two_pi * y)
            + np.sin(two_pi * y) * np.cos(two_pi * z)
            + np.sin(two_pi * z) * np.cos(two_pi * x)
        )

    gv = grid(np.array((resolution, resolution, resolution)))
    gf = np.zeros(gv.shape[0])
    for i in nb.prange(gv.shape[0]):
        row = gv[i]
        gf[i] = _fn(amplitude, row)

    t = bintensor3(gf.reshape((resolution, resolution, resolution)))
    t.padding(axis=0)
    t.padding(axis=1)
    t.padding(axis=2)
    return t

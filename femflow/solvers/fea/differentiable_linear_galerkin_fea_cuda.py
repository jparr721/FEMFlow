import numpy as np
from numerics.geometry import tet_volume


@ti.data_oriented
class DifferentiableLinearGalerkinFeaCuda(object):
    def __init__(self, v: np.ndarray, t: np.ndarray, D: np.ndarray):
        self.vertices = ti.Vector.field(3, ti.f32, v.size).from_numpy(v)
        self.tetrahedra = ti.Vector.field(4, ti.i32, t.size).from_numpy(t)
        self.D = ti.Vector.field
        self.KBuilder = ti.linalg.SparseMatrixBuilder(3 * v.size, 3 * v.size, max_num_triplets=100000)

    @ti.kernel
    def init_K(self):
        pass


def gui():
    gui = ti.GUI("Diff Galerkin", res=(512, 512))
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
        gui.show()


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    v = np.random.rand(30, 3)
    t = np.random.rand(40, 4)
    DifferentiableLinearGalerkinFeaCuda(v, t)
    # gui()

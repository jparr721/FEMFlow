# FEM Module

Each of these modules should be able to be loaded individually and used anywhere. These are heavily inspired by David Levin's [Bartels](https://github.com/dilevin/Bartels) library. He's a genius and you should check him out.

### Module Layout
- `tet/` Finite element functions relating to a _single_ tetrahedron. Lots of computations of the shape functions, potential energy, etc, are easier to compute one tet at a time.
- `tet_mesh/` The finite element functions relating to the _entire_ tetrahedral volume dV.

### Variables
- `q` The degrees of freedom of the system, in FEM they're _always_ the vertices of the tetrahedral mesh
- `F` The deformation gradient
- `psi` The potential enery function for a constitutive model
- `phi` The shape functions (linear or nonlinear)

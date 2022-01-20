# FEMFlow
FEM and MPM Solvers for physics based differentiable programming.

## Status
Active development. Many things are half baked and bug out.

## Run FEM
To run the _very_ rough visualizer, it's here.
```sh
$ conda env create --file environment.yml
$ ./scripts/fem
```

## Run MPM
The MPM Simulations are quite good and numerically stable, but not ready for ML pipelines yet. This one just requires numba and numpy, so you can use something other than conda if you want.
```sh
$ conda env create --file environment.yml
$ ./scripts/mpm [2d, 3d] # Default is 2d
# Use ./scripts/mpm --help to list the options
```

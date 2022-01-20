# FEMFlow
FEM and MPM Solvers for physics based differentiable programming.

## Status
Active development. Many things are half baked and bug out.

## Run FEM
To run the _very_ rough visualizer, it's here.
```sh
$ conda env create --file environment.yml
$ ./scripts/v fem
```

## Run MPM
The MPM Simulations are quite good and numerically stable, but not ready for ML pipelines yet.
```sh
$ conda env create --file environment.yml
$ ./scripts/v mpm [--2d, --3d] # Default is 2d
```

# FEMFlow
FEM and MPM Solvers for physics based differentiable programming.

## What Is This?
FEMFlow is a set of solver primitives and UI elements that enable to quick creation of physics-based simulations for the purposes of doing machine learning. The idea here is to limit the requirements on the scientist of having to build an entire rendering pipeline and wrangle all their datatypes and simply focus on just writing their research code. The provided modules allow for the creation quick UIs for MPM and FEM simulations. Additionally, there are interfaces for adding windows and menus without the overhead of having to write your own imgui handlers.

There are some built-in solvers that may or may not work, but should serve as an example for now (they're _very_ incomplete).

Python allows for people who don't want to write C++ to be able to do all of their work from an easier interface. With numba and numpy, you can get pretty close to what C++ can do for a lot of things.

## Status
Active development. Not ready for usage as there are really no docs available.

## Running
The `main.py` is a testbed for me to work on features and test them, and has useful examples. They are currently subject to change.

### Run FEM Window Example
To run the finite element visualizer, it's here.
```sh
$ conda env create --file environment.yml
$ conda activate femflow
$ ./scripts/fem
```

### Run MPM Window Example
To run the material point method visualizer, it's here.
```sh
$ conda env create --file environment.yml
$ conda activate femflow
$ ./scripts/mpm
```

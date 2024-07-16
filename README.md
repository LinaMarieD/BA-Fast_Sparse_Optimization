# Code Written for the Bachelor Thesis "Fast Sparse Optimization Using Consensus-Based ADMM"

This code was written for the numerical experiments of a bachelor thesis in mathematics that considered the optimization of sparse additive functions using a consensus-based alternating direction of multipliers (CB-ADMM) algorithm. Note that there exist many more efficient implementations of the ADMM algorithm. However, the code provided here is written in Python and therefore easily accessible to people familiar with Python.

## Requirements
The code was tested with Python 3.12.3. It requires the following Python packages, of which the listed versions were used for all experiments:
- Numpy 1.26.4
- SciPy 1.11.4
- PyTorch 2.2.0
- Pillow 10.3.0

## Repository Contents
- `experiment_scripts/`: Contains the Python scripts for running the different numerical experiments as well as example job scripts to run these scripts on an HPC cluster with a Slurm job scheduler.
  - `imaging_application_scripts/`: Scripts related to the imaging experiment comparing CB-ADMM to a block coordinate descent method on a denoising task.
  - `test_function_experiment_scripts/`: Scripts for running test function experiments comparing CB-ADMM to SciPy's BFGS algorithm.
  - `utils/`: Contains a script providing helpful functions used in different experiments.

- `function_generation/`: Contains the scripts for creating sparse additive functions and the set of test functions used in the thesis.
  - `SparseFunctions/`: Pre-generated sparse function data of smooth and non-smooth sparse additive functions that were used in the numerical experiments.

- `images/`: Contains the images used in the imaging experiments.


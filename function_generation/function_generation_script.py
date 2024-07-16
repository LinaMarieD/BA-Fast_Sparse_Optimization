#!/usr/bin/env python3
"""
Author: Lina Marie Dürrwald
Date: 15.04.2024
Description: Script to create a set of sparse additive test functions.

Example Usage:
    python3 function_generation_script.py --d 100 --N 100 --repeat 5 --max_size 5 --min_size 3 --output_directory "SparseFunctions_d=100_N=100"

Arguments:
    --d                : Dimensionality of the functions (int).
    --N                : Number of summands functions to generate (int).
    --repeat           : Number of functions to generate (int).
    --max_size         : Maximum size of the couplings (variables per summand) (int).
    --min_size         : Minimum size of the couplings (variables per summand) (int).
    --output_directory : Directory to save the generated functions (str).
"""

import os
import sys
import argparse
import numpy as np

# Import the module containing the function for generating sparse functions
import generate_sparse_functions as sp_f

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

def main(args):
    # Extract command-line arguments
    d = args.d
    N = args.N
    repeat = args.repeat
    max_size = args.max_size
    min_size = args.min_size
    frac = args.frac
    seed = args.seed

    ds = np.array([d] * repeat)
    Ns = np.array([N] * repeat)

    # Check if the output directory exists, if not, create it in the current dir
    directory = os.path.join(current_dir, "SparseFunctions", args.output_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate and save functions
    sp_f.SparseFunction.generate_and_save_functions(ds, Ns, max_size, min_size, frac, directory, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sparse functions.")
    parser.add_argument("--d", type=int, default=500, help="Dimension of the sparse functions")
    parser.add_argument("--N", type=int, default=500, help="Number of sparse functions to generate")
    parser.add_argument("--repeat", type=int, default=5, help="Number of sparse functions to generate")
    parser.add_argument("--max_size", type=int, default=5, help="Maximum size of coupling terms of sparse functions")
    parser.add_argument("--min_size", type=int, default=3, help="Minimum size of coupling terms of sparse functions")
    parser.add_argument("--frac", type=float, default="SparseFunctions", help="Fraction of dimensions to generate overlap from")
    parser.add_argument("--seed", type=int, default=1, help="Seed for random number generation")
    parser.add_argument("--output_directory", default="SparseFunctions", help="Directory to save generated functions")
    args = parser.parse_args()

    main(args)
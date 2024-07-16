#!/usr/bin/env python3
"""
Author: Lina Marie Dürrwald
Date: 15.04.2024
Description: Script to create multiple sets of sparse additive test functions.

Usage:
    python3 function_generation_wrapper.py 

Run after specifying the desired parameters in 'if __name__ == "__main__"'-block. 
"""

import os
import sys
import numpy as np

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import generate_sparse_functions as sp_f

def gen_func(d, N, repeat, max_size, min_size, frac, seed, smooth):
    ds = np.array([d] * repeat)
    Ns = np.array([N] * repeat)
    name = f"d={d}_N={N}_smooth={smooth}"

    # Check if the output directory exists, if not, create it in the current dir
    directory = os.path.join(current_dir, "SparseFunctions", name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate and save functions
    sp_f.SparseFunction.generate_and_save_functions(ds, Ns, max_size, min_size, frac, directory, name, seed, smooth)

if __name__ == "__main__":
    # create parameter arrays for function generation
    dimensions = np.array([20, 100, 500]) 
    fracs = np.array([0.2, 3/4, 6/7]) 
    repeat = 5
    max_size = 5 
    min_size = 3
    seed = 0

    for frac, dim in zip(fracs, dimensions):
        for smooth in [0,1]:
            N = int(dim) # shouldn't be greater than d, choose d for maximum overlap 
            gen_func(d=dim, N=N, repeat=repeat, max_size=max_size, min_size=min_size, frac=frac, seed=seed, smooth=smooth)


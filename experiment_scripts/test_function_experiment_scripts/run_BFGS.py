#!/usr/bin/python3
"""
Author: Lina Marie Dürrwald
Date: 07.05.2024
Description: Script to run SciPy methods (BFGS, called BB to refer to naive/blackbox optimization that doesn't consider the functions structure below) 
             on test functions. Used by the run_BFGS.sh job script.
"""

import numpy as np
import time
from scipy.optimize import minimize, Bounds
import os
import sys
import csv
import argparse 


# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
function_dir = os.path.join(os.path.dirname(current_dir), "function_generation")
utils_dir = os.path.join(current_dir, "utils")
sys.path.append(current_dir)
sys.path.append(function_dir)
sys.path.append(utils_dir)

from utils import *
import generate_sparse_functions as sp_f
import test_functions_for_grad_methods as tf

# specify output directory
output_dir="/work/duerrwald/Output/"

# Helper functions for saving the output
def write_to_csv_CB_vs_BB(filename, d, N, init_point_id, method, BB, x_opt_BB, runtime_BB, BB_succ, it, MAX_ITER_d_BB, tolerance, bounded):
    d_str = f"_d={d}"
    csv_filename = method + d_str + "_maxit=" + str(MAX_ITER_d_BB) + "_BFGS.csv" 
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Dimension (d)', 'N', 'Init Point ID', 'Result BB', 'Solution x (BB)', 'Runtime BB',  'BB Success', '# Iterations',  'Max. Iterations BB',  'Tolerance', 'Bounded'])

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, d, N, init_point_id, BB, x_opt_BB, runtime_BB, BB_succ, it, MAX_ITER_d_BB,  tolerance,  bounded])

def write_to_csv_seq_xbar(filename, d, N, init_point_id, method, times_BB, seq_xbar_BB, MAX_ITER, tolerance, bounded):
    d_str = f"_d={d}"
    csv_filename = method + d_str + "_maxit=" + str(MAX_ITER) + "_seq_xbar_BFGS.csv" 
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Dimension (d)', 'N', 'Init Point ID',  'Times BB', 'Sequence xbars BB', 'Max. Iterations Outer',  'Tolerance',  'Bounded'])

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, d, N, init_point_id, times_BB, seq_xbar_BB, MAX_ITER,  tolerance,  bounded])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test runtimes of the optimization of sparse additive test functions using consensus-based ADMM vs. naive BFGS optimization: Run BFGS.")
    parser.add_argument("--data_dir", default="d=20_N=20_smooth=0", type=str, help="Data directory")
    parser.add_argument("--filename", default="function_d=20_N=20_smooth=0_0.json", type=str, help="Filename")
    parser.add_argument("--method", type=str, default="BFGS", choices=["BFGS", "Newton-CG", "CG", "Nelder-Mead"], help="Optimization method")
    parser.add_argument("--init_point_id", type=int, default=0, help="Init Point ID")
    parser.add_argument("--MAX_ITER", type=int, default=50, help="Max number of iterations of naive algorithm to multiply with the dimension d") 
    parser.add_argument("--tolerance", type=float, default=1e-30, help="Tolerance")

    args = parser.parse_args()
    data_dir = args.data_dir
    filename = args.filename
    method = args.method
    init_point_id = args.init_point_id
    MAX_ITER = args.MAX_ITER
    tolerance = args.tolerance
    bounded = True if method == "Nelder-Mead" else False

    data_path = os.path.join(function_dir, "SparseFunctions", data_dir)
    func_path = os.path.join(data_path, filename)
    sparse_f = sp_f.SparseFunction.read(func_path)
    d = sparse_f.d
    N = sparse_f.N
    d_U = sparse_f.d_U
    cs = [np.array(params) for params in sparse_f.params]
    init_point = np.array(sparse_f.init_points[init_point_id])
    constraint = sparse_f.constraint
    rho = 1.0
    MAX_ITER_d_BB = min(MAX_ITER * d, 50000) 

    funcs = [tf.tf1, tf.tf2, tf.tf3, tf.tf4, tf.tf5, tf.tf6, tf.tf7, tf.tf8, tf.tf9, tf.tf10] if not bounded else [tf.tf1, tf.tf11, tf.tf3, tf.tf4, tf.tf5, tf.tf6, tf.tf7, tf.tf8, tf.tf9, tf.tf10] #use log_sum_exp only in bounded version
    f_list = []
    for i in range(N):
        f_list.append(funcs[sparse_f.f_list[i]])


    # define callback function to measure time each iteration takes in the blackbox algorithm
    times_BB = []
    seq_xbar_BB = []
    nit_BB = 0 # count iterations
    step = 10
    def get_callback(start_t):
        def callback_timer(xbar):
            global nit_BB
            if nit_BB % step == 0:
                time_elapsed = time.time() - start_t
                times_BB.append(time_elapsed)
                seq_xbar_BB.append(xbar) # save BB intermediate results 
            nit_BB += 1
        return callback_timer

    t = time.time() 
    res = minimize(sum_f([f_list[i](cs[i]) for i in range(len(f_list))], d_U), init_point, method=method,
                   bounds = None if not bounded else Bounds(constraint[0] * np.ones(d), constraint[1] * np.ones(d)),
                   options={'maxiter': MAX_ITER_d_BB}, tol=tolerance, callback=get_callback(t)) 

    BB_time = time.time()-t

    print("\n Time Naive BFGS: {}".format(time.time()-t))
    print(res)
    
    BB = sum_f([f_list[i](cs[i]) for i in range(len(f_list))], d_U)(res.x)
    print("-- Obj_func Naive BFGS: {}".format(BB))

    # Save results
    write_to_csv_CB_vs_BB(filename, d, N, init_point_id, method, BB, res.x, BB_time, res.success, res.nit,  MAX_ITER_d_BB,  tolerance, bounded)
    write_to_csv_seq_xbar(filename, d, N, init_point_id, method, times_BB, seq_xbar_BB, MAX_ITER_d_BB, tolerance, bounded)
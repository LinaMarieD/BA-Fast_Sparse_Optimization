#!/usr/bin/python3
"""
Author: Lina Marie Dürrwald
Date: 07.05.2024
Description: Script to run the CB-ADMM algorithm on test functions. Used by the run_CB-ADMM.sh job script.
"""

import torch
import numpy as np
import time
from scipy.optimize import minimize, Bounds
import copy
import os
import sys
import csv
import argparse 
import queue
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing.sharedctypes import RawArray
import ctypes 

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

# Helper functions
def worker(task_queue, method, max_iter_method, tol_method, bounded, constraint, gammas_array, xbar_array, *xbar_u_list): 
    # Create local numpy wrappers to RawArrays to store local solutions
    xbar_u_nps = []
    for xbar_u_array in xbar_u_list:
        xbar_us = np.frombuffer(xbar_u_array, dtype=np.float64)
        xbar_u_nps.append(xbar_us)

    # Create local numpy wrapper to RawArray to retrieve current xbar
    xbar = np.frombuffer(xbar_array, dtype=np.float64)

    # Create other arrays/inputs from buffer
    d = len(xbar)
    init_p_shm = SharedMemory(name="init_p_shm_name")
    init_point = np.ndarray((d,), dtype=np.float64, buffer= init_p_shm.buf)
    gammas = np.frombuffer(gammas_array, dtype=np.float64)

    # Get shared values
    method = method.value
    max_iter_method = max_iter_method.value 
    tol_method = tol_method.value 
    bounded = bounded.value 
    
    while True:
        try:
            work = task_queue.get(timeout=120)  # 120s timeout
        except queue.Empty:
            init_p_shm.close()
            return

        ind, d, first_time, tf, du, cu, zu = work

        du_shape = len(du)
        fu = tf(cu)
        xbar_u = xbar[du] 
        gammak = gammas[ind]
        init_p = init_point[du]
        reg = reg_outer(gammak) 
            
        gu = lambda val: fu(val) + reg(xbar_u + zu - val)
        gu_grad = lambda val: tf(cu, grad_=True)(val) + reg_outer(gammak, grad_=True)(torch.as_tensor(xbar_u + zu) - val)
        initk = copy.deepcopy(xbar_u) if not first_time else copy.deepcopy(init_p)  # warm start: IMPORTANT for speed up! 
        grad = compute_gradient_autograd(gu_grad) if method != "Nelder-Mead" else None
        res = minimize(gu, initk, method=method,
                       bounds=None if not bounded else Bounds(constraint[0] * np.ones(du_shape), constraint[1] * np.ones(du_shape)),
                       options={'maxiter': max_iter_method * du_shape}, tol=tol_method, jac=grad)

        # Write result to the correct raw array
        np.copyto(xbar_u_nps[ind], res.x)
        task_queue.task_done()


def write_to_csv_CB_vs_BB(filename, d, N, init_point_id, method, CB, x_opt, runtime_CB,it, MAX_ITER, max_iter_method, tolerance, tol_method, bounded, smooth, ncpus):
    d_str = f"_d={d}"
    csv_filename = method + d_str + "_maxit=" + str(MAX_ITER) + "_smooth=" + str(smooth) + "_cpus=" + str(ncpus) + "CB-ADMM.csv" 
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Dimension (d)', 'N', 'Init Point ID', 'Result CB', 'Solution x (CB)', 'Runtime CB', '# Iterations', 'Max. Iterations Outer CB', 'Max. Iterations Inner CB', 'Tolerance', 'Tolerance Inner', 'Bounded'])

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, d, N, init_point_id, CB, x_opt, runtime_CB, it, MAX_ITER, max_iter_method, tolerance, tol_method, bounded])

def write_to_csv_seq_xbar(filename, d, N, init_point_id, method, times_CB, seq_f, MAX_ITER, max_iter_method, tolerance, tol_method, bounded, smooth, ncpus):
    d_str = f"_d={d}"
    csv_filename = method + d_str + "_maxit=" + str(MAX_ITER) + "_smooth=" + str(smooth) + "_cpus=" + str(ncpus) + "CB-ADMM_seq_xbar.csv" 
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Dimension (d)', 'N', 'Init Point ID', 'Times CB', 'Sequence Results', 'Max. Iterations Outer', 'Max. Iterations Inner', 'Tolerance', 'Tolerance Inner', 'Bounded'])

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, d, N, init_point_id, times_CB, seq_f, MAX_ITER, max_iter_method, tolerance, tol_method, bounded])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test runtimes of the optimization of sparse additive test functions using consensus-based ADMM vs. naive BFGS optimization: Run CB-ADMM.")
    parser.add_argument("--data_dir", default="d=20_N=20_smooth=0", type=str, help="Data directory")
    parser.add_argument("--filename", default="function_d=20_N=20_smooth=0_0.json", type=str, help="Filename")
    parser.add_argument("--method", type=str, default="BFGS", choices=["BFGS", "Newton-CG", "CG", "Nelder-Mead"], help="Optimization method for evaluation of proximal operator.")
    parser.add_argument("--init_point_id", type=int, default=0, help="Init Point ID")
    parser.add_argument("--MAX_ITER", type=int, default=1000, help="Max. number of iterations of CB") 
    parser.add_argument("--max_iter_method", type=int, default=1000, help="Max. iterations method") 
    parser.add_argument("--tolerance", type=float, default=1e-30, help="Tolerance")
    parser.add_argument("--tol_method", type=float, default=1e-6, help="Tolerance of inner ADMM loop (method)")
    parser.add_argument("--smooth", type=bool, default=0, help="Function Category (smoothness)")

    args = parser.parse_args()
    data_dir = args.data_dir
    filename = args.filename
    method = args.method
    init_point_id = args.init_point_id
    MAX_ITER = args.MAX_ITER
    max_iter_method = args.max_iter_method
    tolerance = args.tolerance
    tol_method = args.tol_method
    smooth = args.smooth 
    bounded = True if method == "Nelder-Mead" else False
    print(bounded)

    # Get the number of available CPUs to use 
    ncpus = get_ncpus()
    print(f"Maximum number of CPUs allocated: {ncpus}")

    # Create values in shared memory
    manager = Manager()
    method_shm = manager.Value(ctypes.c_char, method)
    bounded_shm = manager.Value(ctypes.c_int , int(bounded))
    tol_method_shm = manager.Value(ctypes.c_float , tol_method)
    max_iter_method_shm = manager.Value(ctypes.c_int, max_iter_method)

    # Read function data
    data_path = os.path.join(function_dir, "SparseFunctions", data_dir)
    func_path = os.path.join(data_path, filename)
    sparse_f = sp_f.SparseFunction.read(func_path)

    d = sparse_f.d
    N = sparse_f.N
    d_U = sparse_f.d_U
    cs = [np.array(params) for params in sparse_f.params]
    constraint = ShareableList(sparse_f.constraint)
    rho = 10.0

    funcs = [tf.tf1, tf.tf2, tf.tf3, tf.tf4, tf.tf5, tf.tf6, tf.tf7, tf.tf8, tf.tf9, tf.tf10] if not bounded else [tf.tf1, tf.tf11, tf.tf3, tf.tf4, tf.tf5, tf.tf6, tf.tf7, tf.tf8, tf.tf9, tf.tf10] #use log_sum_exp only in bounded version
    f_list = []
    for i in range(N):
        f_list.append(funcs[sparse_f.f_list[i]])

    # Setup the task queue
    task_queue = multiprocessing.JoinableQueue()
    first_times = [True for _ in range(N)]

    # Create ctypes RawArrays to save worker outputs
    xbar_u_list = []
    for k in range(N):
        xbar_u_array = RawArray('d', len(d_U[k]))
        xbar_us = np.frombuffer(xbar_u_array, dtype=np.float64, count=len(d_U[k]))
        xbar_us.fill(0.0)
        xbar_u_list.append(xbar_u_array)

    xbar_array = RawArray('d', d)
    xbar_main = np.frombuffer(xbar_array, dtype=np.float64, count=d)
    xbar_main.fill(0.0)

    # Initialize Shareable Arrays
    try:
        init_p_shm = SharedMemory(create=True, size= d * np.float64().itemsize, name="init_p_shm_name")
    except FileExistsError:
        init_p_shm = SharedMemory(name="init_p_shm_name")
        init_p_shm.close()
        init_p_shm.unlink()
        init_p_shm = SharedMemory(create=True, size= d * np.float64().itemsize, name="init_p_shm_name")
    init_point = np.ndarray((d,), dtype=np.float64, buffer= init_p_shm.buf)
    init_point = np.array(sparse_f.init_points[init_point_id])

    gammas_array = RawArray('d', N)
    gammas = np.frombuffer(gammas_array, dtype=np.float64, count=N)
    gammas.fill(rho)

    # Count frequency of each variable
    count = np.zeros(d)
    for k in range(len(d_U)):
        for i in d_U[k]:
            count[i] += gammas[k]

    if N != len(f_list):
        exit(0)

    # Start as many workers as there are cpus available on the cluster node but max. as many as there are terms 
    processes = []
    for _ in range(min(N, ncpus)):
        p = Process(target=worker, args=(task_queue, method_shm, max_iter_method_shm, tol_method_shm, bounded_shm, constraint, gammas_array, xbar_array, *xbar_u_list))
        p.start()
        processes.append(p)

    # ADMM loop.
    max_time = 129600 # 36h
    step = 10 

    # Initializiation
    obj_old = 50
    obj_new = 100
    approx_err = 20
    xbar = np.zeros(d)
    zu = [np.zeros(len(du)) for du in d_U]
    seq_xbar = []
    seq_f = []
    times_CB = []
    
    it = 0
    MAX_ITER_d = min(MAX_ITER * d, 500000)
    start = time.time()
    while it <= MAX_ITER_d and approx_err > tolerance and time.time()-start < max_time: 
        # Put processes in the Queue 
        for i in range(N):
            task_queue.put((i, d, first_times[i], f_list[i], d_U[i], cs[i], zu[i]))
            first_times[i] = False
        task_queue.join() # Wait for all to finish before continuing
        
        local_solution = []
        l_sols_list = []
        local_solution = []
        for k in range(N):
            l_sol = np.zeros(d)
            out = xbar_u_list[k]
            l_sol[d_U[k]] = out
            local_solution += [gammas[k] * l_sol]
            l_sols_list.append(out)

        xbar = np.sum(np.array(local_solution), axis=0) / count
        np.copyto(xbar_main, xbar)
        obj_old = obj_new 
        obj_new = f_list[0](cs[0])(xbar[d_U[0]])
        zu[0] += xbar[d_U[0]] - l_sols_list[0]
        for k in range(1, len(d_U)):
            obj_new += f_list[k](cs[k])(xbar[d_U[k]])
            zu[k] += xbar[d_U[k]] - l_sols_list[k]
        approx_err = np.abs(obj_old - obj_new, dtype=np.float128)

        if it % step == 0:
            seq_xbar.append(xbar)
            t = time.time() - start
            times_CB.append(t)
            seq_f.append(obj_new[0])

        it += 1

    end = time.time()
    CB_time = end - start

    # Clean up
    for p in processes:
        p.terminate()
        p.join()

    # close the shared memory
    init_p_shm.close()
    init_p_shm.unlink()
    constraint.shm.close()
    constraint.shm.unlink()

    print("\n The optimal solution is: ", xbar)
    print("\nTime CB-ADMM: ", end - start)

    CB = sum_f([f_list[i](cs[i]) for i in range(len(f_list))], d_U)(xbar)
    approx_err = np.abs(obj_new - CB, dtype=np.float128)

    success = it <= MAX_ITER

    print("-- Obj_func CB-ADMM: {}".format(CB))
 
    write_to_csv_CB_vs_BB(filename, d, N, init_point_id, method, CB, xbar, CB_time, it-1, MAX_ITER_d, max_iter_method, tolerance, tol_method, bounded, smooth, ncpus)
    write_to_csv_seq_xbar(filename, d, N, init_point_id, method, times_CB, seq_f, MAX_ITER_d, max_iter_method, tolerance, tol_method, bounded, smooth, ncpus)

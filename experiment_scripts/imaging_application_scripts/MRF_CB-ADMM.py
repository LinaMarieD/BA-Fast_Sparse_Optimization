#!/usr/bin/python3
"""
Author: Lina Marie Dürrwald
Date: 07.05.2024
Description: Script to run the CB-ADMM algorithm on a Poisson-corrupted image to perform denoising.
             Used by the MRF_CB-ADMM.sh job script.
"""

import numpy as np
import time
from scipy.optimize import minimize
import copy
import os
import sys
import csv
import argparse 
from PIL import Image

import queue
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import RawArray
import ctypes 

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(current_dir, "utils")
sys.path.append(current_dir)
sys.path.append(utils_dir)

from utils import *

def data_fidelity(y): 
    # y is the value at pixel i in the observed image
    # x should be a numpy array 
    return lambda x: x[0] - y * np.log(x[0]) 

def regularizer(x): 
    return np.sum(np.abs(x-x[0]))

def reg_outer(gammak):
    # Returns regularizing term of ADMM for a given gammak
    def reg_inner(val):
        val = gammak * np.linalg.norm(val, ord=2)**2/2 
        return val
    return reg_inner

def write_to_csv_errors(filename, image_errors, runtimes, MAX_ITER, lam):
    csv_filename = "imaging_maxit=" + str(MAX_ITER) + "_lam=" + str(lam) + "_errors_CB-ADMM.csv" 
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'Error (L2)', 'Times',  'Max. Iterations'])

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, image_errors, runtimes, MAX_ITER])

def worker(task_queue, method, max_iter_method, tol_method, lam, gammas_array, xbar_array, *xbar_u_list): # gamma_array
    # Create local numpy wrappers to RawArrays to store local solutions
    xbar_u_nps = []
    for xbar_u_array in xbar_u_list:
        xbar_us = np.frombuffer(xbar_u_array, dtype=np.float64)
        xbar_u_nps.append(xbar_us)

    # Create local numpy wrapper to RawArray to retrieve current xbar
    xbar = np.frombuffer(xbar_array, dtype=np.float64)

    # Create other arrays/inputs from buffer
    d = len(xbar)
    image_shm = SharedMemory(name="image_shm_name")
    observed_image = np.ndarray((d,), dtype=np.float64, buffer=image_shm.buf)
    gammas = np.frombuffer(gammas_array, dtype=np.float64)

    # Get shared values
    method = method.value
    max_iter_method = max_iter_method.value 
    tol_method = tol_method.value 
    lam = lam.value
    
    while True:
        try:
            work = task_queue.get(timeout=120)  # 120s timeout
        except queue.Empty:
            image_shm.close()
            return

        ind, d, first_time, neighbors, zu = work

        du_shape = len(neighbors)
        y = observed_image[ind]
        xbar_u = xbar[neighbors] 
        init_p = observed_image[neighbors]
        initk = copy.deepcopy(xbar_u) if not first_time else copy.deepcopy(init_p)  # warm start: IMPORTANT for speed up! 
        fu = lambda x: data_fidelity(y)(x) + lam * regularizer(x)
        gammak = gammas[ind]
        reg = reg_outer(gammak) 
        
        # Evaluate proximal operator 
        gu = lambda val: fu(val) + reg(xbar_u + zu - val)
        res = minimize(gu, initk, method=method,
                       bounds=None, options={'maxiter': max_iter_method * du_shape}, tol=tol_method)
        
        # Write result to the correct raw array
        np.copyto(xbar_u_nps[ind], res.x)
        task_queue.task_done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Runtimes of CB-ADMM for Poisson Denoising.")
    parser.add_argument("--filename", default="160_lake_noisy.png", type=str, help="Filename")
    parser.add_argument("--method", type=str, default="BFGS", choices=["BFGS", "L-BFGS-B", "Newton-CG", "CG", "Nelder-Mead"], help="Optimization method")
    parser.add_argument("--MAX_ITER", type=int, default=2, help="Max. number of iterations of CB") 
    parser.add_argument("--max_iter_method", type=int, default=4500, help="Max. iterations method") 
    parser.add_argument("--tol_method", type=float, default=1e-6, help="Tolerance of inner ADMM loop (method)")
    parser.add_argument("--lam", type=float, default=0.1, help="Lambda value to multiply with Regularizer")

    args = parser.parse_args()
    filename = args.filename
    method = args.method
    MAX_ITER = args.MAX_ITER
    max_iter_method = args.max_iter_method
    tol_method = args.tol_method
    lam = args.lam

    # Get the number of available CPUs to use 
    ncpus = get_ncpus()
    print(f"Maximum number of CPUs allocated: {ncpus}")

    manager = Manager()
    method_shm = manager.Value(ctypes.c_char, method)
    tol_method_shm = manager.Value(ctypes.c_float , tol_method)
    max_iter_method_shm = manager.Value(ctypes.c_int, max_iter_method)
    lam_shm = manager.Value(ctypes.c_float , lam)

    # Read image file
    path = os.path.join(os.path.dirname(current_dir), "images/")
    output_dir = path # Change/Specify desired output path here

    print(path + filename)
    img_vector, L, M = image_to_vector(path + filename)
    img_vector = img_vector + 10 # add b=10 to avoid numerical issues with log-term 
    print(L, M, filename)

    d = L*M # dimension
    N = L*M # number summands
    neighbors = extract_3X3_blocks_indices(L, M)
    rho = 1.0

    # Set up the task queue
    task_queue = multiprocessing.JoinableQueue()
    first_times = [True for _ in range(N)]

    # Create ctypes RawArrays to save worker outputs
    xbar_u_list = []
    for k in range(N):
        xbar_u_array = RawArray('d', len(neighbors[k]))
        xbar_us = np.frombuffer(xbar_u_array, dtype=np.float64, count=len(neighbors[k]))
        xbar_us.fill(0.0)
        xbar_u_list.append(xbar_u_array)

    xbar_array = RawArray('d', d)
    xbar_main = np.frombuffer(xbar_array, dtype=np.float64, count=d)
    np.copyto(xbar_main, img_vector) # Initialize with noisy image

    try:
        y_image_shm = SharedMemory(create=True, size= d * np.float64().itemsize, name="image_shm_name")
    except FileExistsError:
        y_image_shm = SharedMemory(name="image_shm_name")
        y_image_shm.close()
        y_image_shm.unlink()
        y_image_shm = SharedMemory(create=True, size= d * np.float64().itemsize, name="image_shm_name")

    y_image = np.ndarray((d,), dtype=np.float64(), buffer= y_image_shm.buf)
    np.copyto(y_image, img_vector)
    gammas_array = RawArray('d', N)
    gammas = np.frombuffer(gammas_array, dtype=np.float64, count=N)
    gammas.fill(rho)

    # Count frequency of each variable
    count = np.zeros(d)
    for k in range(N):
        for i in neighbors[k]:
            count[i] += gammas[k]

    # Start as many workers as there are cpus available on the cluster node but max. as many as there are terms 
    processes = []
    for _ in range(min(N, ncpus)):
        p = Process(target=worker, args=(task_queue, method_shm, max_iter_method_shm, tol_method_shm, lam_shm, gammas_array, xbar_array, *xbar_u_list))
        p.start()
        processes.append(p)

    # ADMM loop.
    max_time = 36000 # 10h 
    step = 2

    # Initializiations
    xbar = copy.deepcopy(y_image)
    zu = [np.zeros(len(du)) for du in neighbors]
    
    it = 0
    images = [] # save all image arrays every step iterations in here
    times = []
    start = time.time()
    while it < MAX_ITER and time.time()-start < max_time: # run maximally 10h
        # Put processes in the Queue 
        for i in range(N):
            task_queue.put((i, d, first_times[i], neighbors[i], zu[i]))
            first_times[i] = False
        task_queue.join() # wait for all to finish before continuing
        
        local_solution = []
        l_sols_list = []
        local_solution = []
        for k in range(N):
            l_sol = np.zeros(d)
            out = xbar_u_list[k]
            l_sol[neighbors[k]] = out
            local_solution += [gammas[k] * l_sol]
            l_sols_list.append(out)

        xbar = np.sum(np.array(local_solution), axis=0) / count
        np.copyto(xbar_main, xbar)
        zu[0] += xbar[neighbors[0]] - l_sols_list[0]
        for k in range(1, len(neighbors)):
            zu[k] += xbar[neighbors[k]] - l_sols_list[k]

        if it % step==0:
            images.append(xbar)
            times.append(time.time() - start)

        it += 1

    end = time.time()
    CB_time = end - start

    # Clean up
    for p in processes:
        p.terminate()
        p.join()

    # Close the shared memory
    y_image_shm.close()
    y_image_shm.unlink()

    print("\nTime CB-ADMM: ", end - start)
    im_xbar = xbar.reshape(L,M) - 10 # substract previously added value of b=10
    image_sol = Image.fromarray(np.array(im_xbar, dtype=np.uint8))
    image_sol.save(output_dir + filename[:-9] + f'denoised_lambda={lam}_maxiter={MAX_ITER}_CB-ADMM.png', format='PNG')

    # Get error values over time 
    image_errors = []
    for im in images:
        image_errors.append(np.linalg.norm(xbar-im, ord=2)/d)

    write_to_csv_errors(filename, image_errors, times, it, lam)

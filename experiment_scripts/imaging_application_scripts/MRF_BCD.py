#!/usr/bin/python3
"""
Author: Lina Marie Dürrwald
Date: 07.05.2024
Description: Script to run the block coordinate descent (BCD) algorithm on a Poisson-corrupted image to perform denoising.
             Used by the MRF_BCD.sh job script.
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
import matplotlib.pyplot as plt


# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(current_dir, "utils")
sys.path.append(current_dir)
sys.path.append(utils_dir)
from utils import *


def data_fidelity(y):
    # y is the value at pixel i in the observed image
    return lambda x: x - y * np.log(x) # x is 1-dimensional

def regularizer(neighbors): # np.abs(x-x[0])
    return lambda x: np.sum(np.abs(x - neighbors)) # x is 1-dimensional unlike the regularizer function in the MRF_CB-ADMM script

def write_to_csv_errors(filename, image_errors, runtimes, MAX_ITER, lam):
    csv_filename = "imaging_maxit=" + str(MAX_ITER)  +"_lam=" + str(lam) + "_errors_BCD.csv" 
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'Error (L2)', 'Times',  'Max. Iterations'])

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, image_errors, runtimes, MAX_ITER])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Runtimes of BCD for Poisson Denoising.")
    parser.add_argument("--filename", default="lake_noisy.png", type=str, help="Filename")
    parser.add_argument("--method", type=str, default="BFGS", choices=["BFGS", "L-BFGS-B", "Newton-CG", "CG", "Nelder-Mead"], help="Optimization method")
    parser.add_argument("--MAX_ITER", type=int, default=200, help="Max. number of iterations of CB") 
    parser.add_argument("--max_iter_method", type=int, default=500, help="Max. iterations method") 
    parser.add_argument("--tol_method", type=float, default=1e-15, help="Tolerance of inner ADMM loop (method)")
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

    # Read image file
    path = os.path.join(os.path.dirname(current_dir), "images/")
    output_dir = path # Change/Specify desired output path here
    
    filename = 'image_noisy.png'
    print(path + filename)
    y_image, L, M = image_to_vector(path + filename)
    y_image = y_image + 10
    print(L, M, filename)

    d = L*M # dimension
    N = L*M # number summands
    neighbors = extract_3X3_blocks_indices(L, M)

    # Initializiations
    xbar_new = copy.deepcopy(y_image)
    xbar = copy.deepcopy(xbar_new)
    max_time = 36000 # 10h 
    step = 2
    it = 0
    start = time.time()
    images = [] # save all image arrays every step iterations in here
    times = []
    while it < MAX_ITER and time.time()-start < max_time: # run maximally 10h
        for i in range(d):
            y = y_image[i] 
            neighbors_i = neighbors[i]
            du_shape = len(neighbors_i)
            betas = np.ones(du_shape) 
            xbar_u = xbar[neighbors_i]
            x_hats = copy.deepcopy(xbar_u[1:]) # most recent estimates of neighbors 
            initk = copy.deepcopy(y)
            center = initk
            fu = lambda x: data_fidelity(y)(x) + lam * regularizer(x_hats)(x)
            res = minimize(fu, initk, method=method,
                           bounds=None, options={'maxiter': max_iter_method}, tol=tol_method)
            
            xbar_new[i] = res.x[0]
        
        xbar = copy.deepcopy(xbar_new)
        if it % step==0:
            images.append(xbar)
            times.append(time.time() - start)

        it += 1

    end = time.time()
    BCD_time = end - start

    print("\nTime BCD: ", end - start)
    im_xbar = xbar.reshape(L,M) - 10
    image_sol = Image.fromarray(np.array(im_xbar, dtype=np.uint8))
    image_sol.save(output_dir + filename[:-9] + f'denoised_lambda={lam}_maxiter={MAX_ITER}_BCD.png', format='PNG')

    # Get error values over time 
    image_errors = []
    for im in images:
        image_errors.append(np.linalg.norm(xbar-im, ord=2)/d)

    write_to_csv_errors(filename, image_errors, times, it, lam)
    plt.imshow(im_xbar, cmap='gray')
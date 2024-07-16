#!/usr/bin/python3
"""
Author: Lina Marie Dürrwald
Date: 22.05.2024
Description: Helper functions for running the numerical experiments.
"""

import os
import numpy as np
import torch 
from PIL import Image

def compute_gradient_autograd(func):
    '''
    :param X: input data in [-1, 1]^{N x d} where N is the number d-dimensional points
    :param func: target function
    :return: gradient of the target function as a function computed with autograd evaluated at X
    '''
    def gradient_f(X):
        def f(func):
            def get_random(X_):
                    return torch.as_tensor(func(X_))
            return get_random
        if type(X) == np.ndarray:
            X = torch.as_tensor(X)
        if X.ndim == 1:
            X = X.reshape(1, X.shape[0])
        gradient = torch.zeros_like(X.T)
        for i in range(X.shape[0]):
            X_i = X[i:i+1, :]
            X_i.required_grad = True
            gradient[:, i] = torch.autograd.functional.jacobian(f(func), X_i, create_graph=True)
        return gradient.squeeze().detach().cpu().numpy()

    return gradient_f

def reg_outer(gammak, grad_=False):
    # returns regularizing term of ADMM for a given gammak
    def reg_inner(val):
        if type(val) == np.ndarray:
            val = torch.as_tensor(val)
        val = gammak * torch.norm(val, p=2) ** 2 / 2
        return val.detach().numpy() if not grad_ else val
    return reg_inner

def sum_f(list_f, d_U_):
    def func(x_):
        x_ = x_.squeeze()
        f = np.array(list_f[0](x_[d_U_[0]]), dtype=np.float128)
        for i in range(1, len(list_f)):
            f += np.array(list_f[i](x_[d_U_[i]]), dtype=np.float128)
        return f
    return func

def get_ncpus():
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK']) - 1 
    else:
        return os.cpu_count() - 1 # leave one node free for OS to run

def constr_shareable_arrays(nested_list):
    flat_array = np.concatenate([np.array(nested_list[i]) for i in range(len(nested_list))])
    lengths = [len(sublist) for sublist in nested_list]
    cumulative_lengths = np.cumsum([0] + lengths)  
    return flat_array, cumulative_lengths

def reconstruct_nested_list(flat_array, cumulative_lengths, np_array=False): 
    if np_array:
        return [flat_array[cumulative_lengths[i]:cumulative_lengths[i+1]] for i in range(len(cumulative_lengths)-1)]
    else:
        return [list(flat_array[cumulative_lengths[i]:cumulative_lengths[i+1]]) for i in range(len(cumulative_lengths)-1)]
    
# Functions for imaging applications
def extract_3X3_blocks_indices(N, M):
    # Function to extract block indices of MRF structure (max. 8 neighbors, i.e. 9 pixels) for an NxM picture, the pixel itself is the first index/variable in the block.
    def get_block_indices(i, j):
        middle_index = i * M + j
        block_indices = [middle_index]
        for x in range(max(0, i - 1), min(N, i + 2)):
            for y in range(max(0, j - 1), min(M, j + 2)):
                index = x * M + y
                if index != middle_index:
                    block_indices.append(index)
        return block_indices

    all_blocks = []
    for i in range(N):
        for j in range(M):
            all_blocks.append(get_block_indices(i, j))

    return all_blocks

def image_to_vector(image_path):
    img = Image.open(image_path)
    N, M = img.size  
    img_array = np.array(img, dtype=np.float64)
    img_vector = img_array.flatten()
    return img_vector, N, M
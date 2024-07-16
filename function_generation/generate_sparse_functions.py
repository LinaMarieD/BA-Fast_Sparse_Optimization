#!/usr/bin/env python3
"""
Author: Lina Marie Dürrwald
Date: 08.04.2024
Description: Script/Module to enable the creation of sparse additive test functions and to be able to read and write them. 
             This script is used by function_generation_script.py to generate sparse additive functions.
"""

import numpy as np
import json
import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import generate_sparse_couplings as scpl

class SparseFunction:
    """
    Represents a sparse function with given parameters.
    """
    def __init__(self, d, N, max_size, min_size, frac, d_U=None, constraint=None, f_list=None, params=None, init_points=None, stats_d_U=None, smooth=False): #, f_list_names = None
        """
        Initializes a SparseFunction object.

        Parameters:
            d (int): Dimensionality of the function.
            N (int): Number of summands of the function, i.e., number of subsets in d_U (length of d_U).
            param_range (list): Range of values for randomly generated parameters. Default is [-100, 101].
            d_U (list of lists): List of subsets of indices defining the sparsity pattern. If not provided, generated randomly.
            alpha (list): List of coefficients corresponding to each subset in d_U. If not provided, randomly generated within param_range.
            constraint (list): List of length 2 of constraint values [a, b], constraining the input vector x to [a, b]^d.
                               If not provided, randomly generated as either [-1,1] or [0,1].
            f_list (list): List of functions corresponding to each subset in d_U. If not provided, generated based on random parameters.
            rho (float): Parameter rho for creating the gammas attribute. Default is 1.
            NUM_PROCS (int): Number of processors or processes used. Default is 4.
        
        Additionally initialized Attributes:
            gammas (np.ndarray): Parameter vector of length N for the augmented Lagrangian. Values are all set to rho
        """

        self.d = int(d)
        self.N = int(N)
        self.max_size = int(max_size)
        self.min_size = int(min_size)
        self.frac = frac
        self.d_U, self.stats_d_U = (d_U, stats_d_U) if d_U is not None else self._generate_couplings()
        self.smooth = smooth
        self.constraint = constraint if constraint is not None else [int(np.random.choice([-1,0])), 1]
        self.f_list = f_list if f_list is not None else [int(np.random.choice(range(5))) for i in range(N)] if self.smooth else [int(np.random.choice(range(9))) for i in range(N)]
        self.params = params if params is not None else self._generate_params()
        self.init_points = init_points if init_points is not None else self._generate_initpoints()
    
    def _generate_couplings(self):
        return scpl.generate_overlapping_couplings(d=self.d, N=self.N, max_size=self.max_size, min_size=self.min_size, frac = self.frac)

    def _generate_params(self):
        """
        Generate a list of random functions based on random parameters.
        """
        params = []
        for i in range(len(self.d_U)):
            coeff = int(self.constraint[1] - self.constraint[0])
            trans = int(np.abs(np.min(self.constraint)))
            params.append(list(coeff * np.random.rand(len(self.d_U[i])) - trans))
        return params
    
    def _generate_initpoints(self):
        """
        Generate a list of random initialization points to start the optimization of the function.
        """
        init_points = []
        for i in range(3):
            coeff = int(self.constraint[1] - self.constraint[0])
            trans = int(np.abs(np.min(self.constraint)))
            init_points.append(list(coeff * np.random.rand(self.d) - trans))
        return init_points 
    
    @classmethod
    def generate_n(cls, ds, Ns, max_size, min_size, frac, seed = 1, smooth = False):
        """
        Args:
            ds (np.ndarray): Array containing dimensions of functions to generate
            Ns (np.ndarray): Array containing # of summands of functions to generate
            max_size(int): Maximum # of variables in each summand
            min_size(int): Minimum # of variables in each summand

        Example:
            ds = np.linspace(20,210, 11)*10
            Ns = np.tile(np.linspace(3,12,10),10)
            fs = SparseFunction.generate_n(ds, Ns)
        """
        np.random.seed(seed)
        functions = []
        for (d,N) in zip(ds, Ns):
            func = SparseFunction(int(d),int(N), max_size, min_size, frac, smooth=smooth)
            functions.append(func)
        return functions
    
    @classmethod
    def generate_and_save_functions(cls, ds, Ns, max_size, min_size, frac, directory=".", name="test", seed=1, smooth=False):
        """
        Generate SparseFunction instances using the generate_n class method,
        save each function to JSON files in the specified directory, and name them by index.

        Parameters:
            ds (np.ndarray): Array containing dimensions of functions to generate.
            Ns (np.ndarray): Array containing # of summands of functions to generate.
            directory (str): Directory path where the files should be saved.
            seed (int): Seed for random number generation. Default is 1.

        Returns:
            None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        functions = SparseFunction.generate_n(ds, Ns, max_size, min_size, frac, seed, smooth)
        for i, func in enumerate(functions):
            filename = f"function_{name}_{i}.json"
            func.write(filename, directory)
    
    @classmethod
    def read_functions_from_directory(cls, directory):
        """
        Read SparseFunction instances from JSON files in the specified directory.

        Parameters:
            directory (str): Directory path from where the files should be read.

        Returns:
            list: List of SparseFunction instances read from the files.
        """
        functions = []
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                function = cls.read(filepath)
                functions.append(function)
        return functions

    
    def __str__(self):
        """
        String representation of SparseFunction object.
        """
        attributes = [
            f"d: {self.d}",
            f"N: {self.N}",
            f"max_size: {self.max_size}",
            f"min_size: {self.min_size}",
            f"smooth: {self.smooth}",
            f"d_U: {self.d_U}",
            f"constraint: {self.constraint}",
            f"f_list: {self.f_list}",
            f"params: {self.params}",
            f"stats_interactions: {self.stats_d_U}",
        ]
        return "\n".join(attributes)

    def __repr__(self):
        """
        Detailed representation of SparseFunction object.
        """
        return self.__str__()  # Call __str__ method to get the same output
    
    ### Functions to write to/read from file
    def write(self, filename, directory="."):
        """
        Write the attributes of the instance to a JSON file.

        Parameters:
            filename (str): Name of the JSON file to write.

        Returns:
            None
        """
        file_path = os.path.join(directory, filename)

        data = {
            'd': self.d,
            'N': self.N,
            'max_size': self.max_size,
            'min_size': self.min_size,
            'frac': self.frac,
            'smooth': self.smooth,
            'd_U': self.d_U,
            'constraint': self.constraint,
            'f_list': self.f_list,
            'params': self.params,
            'init_points': self.init_points,
            'stats_d_U': self.stats_d_U
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def read(cls, filename, directory="."):
        """
        Read the attributes of the instance from a JSON file.

        Parameters:
            filename (str): Name of the JSON file to read.

        Returns:
            SparseFunction: Instance of SparseFunction read from the file.
        """
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls(**data)
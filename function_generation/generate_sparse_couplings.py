#!/usr/bin/env python3
"""
Author: Lina Marie Dürrwald
Date: 03.04.2024
Description: Script/Module to enable the creation of sparse couplings, i.e. sparse subsets of variables, needed for the creation of sparse additive test functions. 
             The main function generate_overlapping_couplings() in this script is used by generate_sparse_functions.py.
"""

import copy
import numpy as np
import random


# Helper functions 
def check_subset(arr1, arr2):
    """
    Check if neither arr1 is a subset of (or equal to) arr2 nor arr2 is a subset of arr1.

    Parameters:
        arr1 (numpy.ndarray)
        arr2 (numpy.ndarray)

    Returns:
        bool: True if either arr1 is a subset of arr2 or arr2 is a subset of arr1, False otherwise (also False if arr1 == arr2).
    """
    if np.array_equal(arr1,arr2):
        return True 

    subset_check1 = np.isin(arr1, arr2).all()
    subset_check2 = np.isin(arr2, arr1).all()

    return subset_check1 or subset_check2

def redundancy_check_U(entry_indices, U):
    # Checks if generated indices are a subset or a superset of a previous coupling that was saved in U (if yes, it returns FALSE)
    copy_U = copy.deepcopy(U)
    for t_entries_el in copy_U:
        if check_subset(entry_indices, t_entries_el): 
            return False
    return True

# Helper functions to use in generate_couplings function
def generate_entry_indices(set_to_generate_from, min_size, max_size):
    """
    Generate entry indices based on remaining indices and maximal size.

    Parameters:
        remaining_indices (set): Set of remaining indices.
        max_size (int): Maximal size of the entry indices.

    Returns:
        list: Sorted list of entry indices.
    """
    cardinality_u_k = np.random.choice(range(min_size, max_size + 1))
    entry_indices = np.random.choice(list(set_to_generate_from), size=min(cardinality_u_k, len(set_to_generate_from)), replace=False)
    return sorted(entry_indices.tolist())

def add_new_indices(entry_indices, U, covered_indices, remaining_indices, prepend=False):
    """
    Add new indices to the list of couplings.

    Parameters:
        entry_indices (list): Entry indices to add.
        U (list): List of existing couplings.
        covered_indices (set): Set of covered indices.
        remaining_indices (set): Set of remaining indices.
        prepend

    Returns:
        tuple: Updated U, covered_indices, and remaining_indices.
    """
    if not prepend:
        U.append(entry_indices)
    else:
        U = [entry_indices] + U 
    covered_indices.update(entry_indices)
    remaining_indices -= set(entry_indices)
    return U, covered_indices, remaining_indices

def update_indexsets(U,d):
    covered_indices = set(np.concatenate(U))
    remaining_indices = set(range(d)) - covered_indices
    return covered_indices, remaining_indices 

def generate_and_add_indices(U, d, covered_indices, remaining_indices, min_size, max_size, prepend=False, generate_from_remaining=False):
    """
    Generate entry indices and add them to the list of couplings.

    Parameters:
        U (list): List of existing couplings.
        covered_indices (set): Set of covered indices.
        remaining_indices (set): Set of remaining indices.
        min_size (int): Minimal size of the entry indices.
        max_size (int): Maximal size of the entry indices.
        prepend (bool): Whether to prepend the new indices to U. Default is False (=append).
        generate_from_remaining (bool): Whether to generate new variable interaction sets from the remaining_indices set only. Default is False, then they are generated from range(d).

    Returns:
        tuple: chosen entry_indices, Updated U, covered_indices, and remaining_indices.
    """
    if generate_from_remaining:
        set_to_generate_from = remaining_indices
    else: 
        set_to_generate_from = range(d)
        
    entry_indices = generate_entry_indices(set_to_generate_from, min_size, max_size)
    check_item = False
    while not check_item:
        check_item = redundancy_check_U(entry_indices, U)
        if check_item:
            U, covered_indices, remaining_indices = add_new_indices(entry_indices, U, covered_indices, remaining_indices, prepend=prepend)
        else:
            entry_indices = generate_entry_indices(set_to_generate_from,  min_size, max_size)
    return U, covered_indices, remaining_indices

def generate_couplings(d, N, max_size, min_size=1):
    '''
    Generates randomly a set of couplings.

    Parameters:
        d (int): Dimensionality of the function.
        N (int): Number of summands of the function, i.e., number of subsets in d_U (length of d_U).
        max_size(int): Maximum # of variables in each summand
        min_size(int): Minimum # of variables in each summand, always >=1
    
    Returns: 
        list: A list of length N containing arrays which contain the coupled indices of each summand.
    '''

    assert (max_size <= d)
    
    U = []
    covered_indices = set()
    remaining_indices = set(range(d))

    for _ in range(N):
        U, covered_indices, remaining_indices = generate_and_add_indices(U, d, covered_indices, remaining_indices, min_size, max_size, prepend=False, generate_from_remaining=False)

        if remaining_indices: 
            # If the number of selected combinations is close to N and there are still many indices remaining, replace some combinations with new ones
            min_threshold = max(3,N/10)
            if len(U) >= max( min_threshold , N - min(min_threshold,int(len(remaining_indices)/(max_size - 1)))):
                print(f"Length selected combis: {len(U)}, remaining indices: {len(remaining_indices)}")

                # Calculate the number of combinations to replace
                min_replace = max(5,d/200)
                num_replace = int(max(1, min(min_replace,int(len(remaining_indices)/(max_size)))))
                print(f"Replace {num_replace}.")

                # Delete last num_replace entries and replace by new different ones
                slicer_ind = int(min(num_replace, len(U)-1))
                U = U[:-slicer_ind]
                covered_indices, remaining_indices = update_indexsets(U, d)
                for _ in range(num_replace):
                    U, covered_indices, remaining_indices = generate_and_add_indices(U, d, covered_indices, remaining_indices, min_size, max_size, prepend=True, generate_from_remaining=True)

    max_it = 200 
    it = 0                  
    while remaining_indices and it <= max_it: 
        print(f"Final while loop iteration # {it}")
        print(f"Length selected combis: {len(U)}, remaining indices: {len(remaining_indices)}")

        # replace one variable set
        U = U[:-1]
        covered_indices, remaining_indices = update_indexsets(U, d)
        U, covered_indices, remaining_indices = generate_and_add_indices(U, d, covered_indices, remaining_indices, min_size, max_size, prepend=True, generate_from_remaining=True)

        it += 1
    return U 

# Main function
def generate_overlapping_couplings(d, N, max_size, min_size=1, frac=4/5):
    """
    Wrapper function for function generate_couplings to create coupling with more overlap.

    Parameters:
        d (int): Dimensionality of the function.
        N (int): Number of summands of the function, i.e., number of subsets in d_U (length of d_U).
        max_size(int): Maximum # of variables in each summand
        min_size(int): Minimum # of variables in each summand
        frac(float): Fraction/portion of indices to take out of the initial generation, i.e. to append in the end.
    Returns: 
        list: A list of length N containing arrays which contain the coupled indices of each summand.
        dict: A dict counting the number of overlaps of size 1,2,... between terms.
    """

    d_star = int(d*(1-frac))
    print("Overlapping sets generated from", d_star, "indices.")
    min_size = min(max(1,min_size-2), max_size-2)
    U = generate_couplings(d_star, N, max_size-2, min_size)

    # Create random set of remaining indices
    d_complement = set(range(d)) - set(range(d_star))
    d_c_list = list(d_complement)
    random.shuffle(d_c_list)

    # add 2 elements of d_c_list to each coupling 
    for i,j  in zip(range(0,len(d_c_list),2),range(len(U))):
        U[j] += d_c_list[i:i+2]

    # extract "statistics"
    dict_count_interactions = overlap_dict_generator(U, N, d, max_size)

    return U, dict_count_interactions

def overlap_dict_generator(U, N, d, max_size):
    # returns a dict counting the number of overlaps of size 1,2,...max_size-2 between terms in U

    num_overlap = np.zeros((N,d))
    for i, coupling in enumerate(U):
        for entry in coupling:
            num_overlap[i][entry-1]=1

    # count interactions:
    max_possible_interactions = max_size - 2
    max_interaction_counter = np.zeros(max_possible_interactions+1)
    for i in range(N):
        max_sum = 1
        for j in range(i+1,N):
            sum = np.sum(num_overlap[i]*num_overlap[j])
            if sum > max_sum:
                max_sum = sum

        overlap_present = 0
        for j in range(1,max_possible_interactions+1):
            if max_sum >= j:
                max_interaction_counter[j] += 1
                overlap_present = 1
        if not overlap_present:
            max_interaction_counter[0] += 1

    dict_count_interactions = {i: max_interaction_counter[i] for i in range(max_possible_interactions+1)}
    dict_count_interactions.update({i: dict_count_interactions[i] - dict_count_interactions[i+1] for i in range(1,list(dict_count_interactions.keys())[-1])})

    return dict_count_interactions


if __name__ == '__main__':
    d = 500
    min_size = 3
    max_size = 5
    N = 500
    frac= 4/5
    U, dict_count_interactions = generate_overlapping_couplings(d, N, max_size, min_size=min_size, frac=frac)
    print(U)
    print("Length", len(U))
    print("Check contains all indices: ", len(set(np.concatenate(U))))
    print(dict_count_interactions)
#!/bin/bash --login
#SBATCH --job-name=Imaging_BCD
#SBATCH --output=Imaging_BCD_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -C cputype:epyc7302
#SBATCH --partition=smp

########## Slurm Job Script for Poisson Denoising with an MRF Prior using Block Coordinate Descent ############
#
# Author: Lina Marie Dürrwald
# Date: 09.06.2024
# Description: Script for Running the BCD Algorithm-Part of the CB-ADMM vs BCD Comparison.

# Example Usage:
#   sbatch MRF_BCD.sh -f "pepper_noisy.png"

# Arguments:
#   -f  : Filename of Image to denoise (should be a .png file located in ./images)

# Requirements:
#   - Ensure that any necessary modules are loaded before running the script by activating a suitable conda environment and replacing the respective line below.
#   - Ensure to change the cpytype specification according to your cluster and hardware requirements.

conda activate /work/duerrwald/anova

# Parse command line argument
while getopts "f:" opt; do
  case "${opt}" in
        f) filename=${OPTARG};;
        *) echo "Invalid option: -${OPTARG}" ;; 
    esac
done

# Initialize variables 
MAX_ITERs=(200) 
lambdas=(0.05)
max_iter_method=500 
tol_method=1e-15
method="BFGS"

# Iterate over different combinations of MAX_ITERs and lambdas
echo $filename
for MAX_ITER in "${MAX_ITERs[@]}"; do
    for lam in "${lambdas[@]}"; do
        echo $MAX_ITER
        echo $lam
        python3  block_coordinate_descent.py --filename="$filename" --method="$method" --MAX_ITER="$MAX_ITER" --max_iter_method="$max_iter_method" --tol_method="$tol_method" --lam="$lam"
    done
done




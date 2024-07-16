#!/bin/bash --login
#SBATCH --job-name=Imaging_CB-ADMM
#SBATCH --output=Imaging_CB-ADMM_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -C cputype:epyc7302
#SBATCH --cpus-per-task=32
#SBATCH --partition=smp

########## Slurm Job Script for Poisson Denoising with an MRF Prior using CB-ADMM ############
#
# Author: Lina Marie Dürrwald
# Date: 09.06.2024
# Description: Script for Running the CB-ADMM Algorithm-Part of the CB-ADMM vs BFGS Comparison.

# Example Usage:
#   sbatch MRF_CB-ADMM.sh -f "pepper_noisy.png"

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
max_iter_method=4500 # 500 * d
tol_method=1e-6
method="BFGS"

# Iterate over different combinations of MAX_ITERs and lambdas
echo $filename
for MAX_ITER in "${MAX_ITERs[@]}"; do
    for lam in "${lambdas[@]}"; do
        echo $MAX_ITER
        echo $lam
        python3  MRF_CB-ADMM.py --filename="$filename" --method="$method" --MAX_ITER="$MAX_ITER" --max_iter_method="$max_iter_method" --tol_method="$tol_method" --lam="$lam"
    done
done




#!/bin/bash --login
#SBATCH --job-name=CB-ADMM
#SBATCH --output=/work/duerrwald/Output/out_files/CB-ADMM_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH -C cputype:epyc7313
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=smp

########## Slurm Job Script for Running the CB-ADMM Algorithm ############
# Author: Lina Marie Dürrwald
# Date: 09.06.2024
# Description: Script for Running the CB-ADMM Algorithm-Part of the CB-ADMM vs BFGS Comparison.

# Example Usage:
#   method="BFGS"
#   id= 0
#   d=100
#   init_p=0
#   smooth=1
#   sbatch run_CB-ADMM.sh -m "$method" -f "$id" -d "$d" -i "$init_p" -s "$smooth"

# The above exmaple does the same as running: 
#   sbatch run_CB-ADMM.sh -m "BFGS" -f 0 -d 100 -i 0 -s 1

# Arguments:
#   -m  : SciPy Method to use for evaluation of the proximal operator (e.g. BFGS) (string).
#   -f  : File identifier (number of the test function) (integer).
#   -d  : Dimensionality of the test function (integer).
#   -i  : Number of initialization point to use (integer).
#   -s  : Smoothness parameter (integer: 0 or 1).

# Requirements:
#   - Ensure that any necessary modules are loaded before running the script by activating a suitable conda environment and replacing the respective line below.
#   - Ensure to change the cpytype specification according to your cluster and hardware requirements.

conda activate /work/duerrwald/anova

# Define the directory containing files 
parent_dir="../../function_generation/SparseFunctions/"

# Parse command line arguments
while getopts "m:f:d:i:s:" opt; do
  case "${opt}" in
        m) method=${OPTARG};;
        f) f_num=${OPTARG};;
        d) dim=${OPTARG};;
        i) init_point_id=${OPTARG};;
        s) smooth=${OPTARG};;
        *) echo "Invalid option: -${OPTARG}" ;;  # Handle invalid options
    esac
done

# Initialize variables 
MAX_ITER=1000 # 1000 * d
tolerance=1e-30
max_iter_method=1000 
tol_method=1e-6 # 1e-6 gives best and fastest results for d=100

# Iterate over bounds and filenames
file_str="d=${dim}" # select files by dim
file_str2="smooth=${smooth}" # select files by smoothness requirements of method
data_dir="d=${dim}_N=${dim}_smooth=${smooth}"
for filename_with_path in "$parent_dir/$data_dir"/*"$file_str"*"$file_str2"*.json; do
    filename=$(basename "$filename_with_path") 
    echo $filename
    echo $init_point_id

    # Extract the number from the filename (assuming the number is at the end before the extension)
    number=${filename##*_}  # Get the part after the last underscore
    number=${number%%.*}  # Remove the extension

    if [ "$f_num" == "$number" ]; then
        python3  run_CB-ADMM.py --data_dir="$data_dir" --filename="$filename" --method="$method" --init_point_id="$init_point_id" --MAX_ITER="$MAX_ITER" --max_iter_method="$max_iter_method" --tolerance="$tolerance" --tol_method="$tol_method" --smooth="$smooth"
    fi
done




#!/bin/bash --login
#SBATCH --job-name=Naive-BFGS
#SBATCH --output=/work/duerrwald/Output/out_files/Naive-BFGS_%j.out
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH -C cputype:epyc7313
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --partition=smp

########## Slurm Job Script for Running the Naive BFGS Algorithm ############
# Author: Lina Marie Dürrwald
# Date: 09.06.2024
# Description: Script for Running the Naive BFGS Algorithm-Part of the CB-ADMM vs BFGS Comparison. This script runs the BFGS, or an alternative 
#              SciPy method (if specified) with the specified parameters.

# Example Usage:
#   method="BFGS"
#   id= 0
#   d=100
#   init_p=0
#   smooth=1
#   sbatch run_BFGS.sh -m "$method" -f "$id" -d "$d" -i "$init_p" -s "$smooth"

# The above exmaple does the same as running: 
#   sbatch run_BFGS.sh -m "BFGS" -f 0 -d 100 -i 0 -s 1

# Arguments:
#   -m  : SciPy Method to use (e.g. BFGS) (string).
#   -f  : File identifier (number of the test function) (integer).
#   -d  : Dimensionality of the test function (integer).
#   -i  : Number of initialization point to use (integer).
#   -s  : Smoothness parameter (integer: 0 or 1).

# Requirements:
#   - Ensure that any necessary modules are loaded before running the script, by activating a suitable conda environment and replacing the respective line below.
#   - Ensure to change the cpytype specification according to your cluster and hardware requirements.


conda activate /work/duerrwald/anova 

# Define the directory containing files
parent_dir="../../function_generation/SparseFunctions/"

# Parse the function numbers from command line arguments
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

# initialize variables 
MAX_ITER=1000 # 1000 * d
tolerance=1e-30

# Iterate over bounds and filenames
file_str="d=${dim}" # select files by dim
file_str2="smooth=${smooth}" # select files by smoothness 
data_dir="d=${dim}_N=${dim}_smooth=${smooth}"
for filename_with_path in "$parent_dir/$data_dir"/*"$file_str"*"$file_str2"*.json; do
    filename=$(basename "$filename_with_path") 
    echo $filename
    echo $init_point_id

    # Extract the number from the filename (assuming the number is at the end before the extension)
    number=${filename##*_}  # Get the part after the last underscore
    number=${number%%.*}  # Remove the extension

    if [ "$f_num" == "$number" ]; then
        python3  run_BFGS.py --data_dir="$data_dir" --filename="$filename" --method="$method" --init_point_id="$init_point_id" --MAX_ITER="$MAX_ITER" --tolerance="$tolerance"
    fi
done




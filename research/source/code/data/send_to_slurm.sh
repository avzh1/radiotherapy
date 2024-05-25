#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 16                       # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

# Check if $1 variable is empty
if [ -z "$1" ]; then
    echo "Error: script is not provided."
    return -1
fi

# Run python script
jupyter nbconvert --to script $1.ipynb
python3 $1.py

rm -rf $1.py
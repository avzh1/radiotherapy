#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --nodelist loki          	# SLURM node
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Before I had an sbatch thing above which had --nodelist loki

# Check if $1 variable is empty
if [ -z "$1" ]; then
    echo "Error: DATASET_NAME_OR_ID is not provided."
    exit -1
fi

if [ -z "$2" ]; then
    echo "Error: FOLD is not provided."
    exit -1
fi

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

jupyter nbconvert --to script '1_fine_tune_structures.ipynb'
python3 1_fine_tune_structures.py $1 $2
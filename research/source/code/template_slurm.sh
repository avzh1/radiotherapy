#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 16                            # Number of CPU Cores
#SBATCH -p gpus                          # Partition (queue)
#SBATCH --gres gpu:1                     # gpu:n, where n = number of GPUs
#SBATCH --mem 20G                        # memory pool for all cores
#SBATCH --output=logs/slurm.%N.%j.log    # Standard output and error log

#SBATCH --job-name=pp.2.nnUNEt

# --nodelist lory           # SLURM node
# --nodelist loki          	# SLURM node

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

# Convert Python Script
# jupyter nbconvert --to script '*.ipynb'

# Your script here
# ...
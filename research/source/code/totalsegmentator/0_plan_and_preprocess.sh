#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 20G                   # memory pool for all cores
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

jupyter nbconvert --to script '0_plan_and_preprocess.ipynb'
ipython3 0_plan_and_preprocess.py 
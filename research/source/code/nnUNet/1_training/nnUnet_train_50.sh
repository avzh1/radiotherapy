#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --nodelist loki          	# SLURM node
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Check if $1 variable is empty
if [ -z "$1" ]; then
    echo "Error: DATASET_NAME_OR_ID is not provided."
    return -1
fi

# Source virtual environment (pip)
source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

# Set env variables
source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

# Run python script
# nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
nnUNetv2_train $1 3d_fullres 0 -tr nnUNetTrainer_50epochs --npz
nnUNetv2_train $1 3d_fullres 1 -tr nnUNetTrainer_50epochs --npz
nnUNetv2_train $1 3d_fullres 2 -tr nnUNetTrainer_50epochs --npz
nnUNetv2_train $1 3d_fullres 3 -tr nnUNetTrainer_50epochs --npz
nnUNetv2_train $1 3d_fullres 4 -tr nnUNetTrainer_50epochs --npz

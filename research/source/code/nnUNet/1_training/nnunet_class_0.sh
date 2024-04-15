#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpushigh                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --nodelist monal03          	# SLURM node
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Source virtual environment (pip)
source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

# Set env variables
source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

# Run python script
# nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
nnUNetv2_train 2 3d_fullres 0 --c
nnUNetv2_train 2 3d_fullres 1 --c
nnUNetv2_train 2 3d_fullres 2 --c
nnUNetv2_train 2 3d_fullres 3 --c
nnUNetv2_train 2 3d_fullres 4 --c

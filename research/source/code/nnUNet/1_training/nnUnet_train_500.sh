#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --output=logs/slurm.%N.%j.out
#SBATCH --error=logs/slurm.%N.%j.err
#SBATCH --job-name=nnUNet500

#--nodelist loki          	# SLURM node

# Check if $1 variable is empty
if [ -z "$1" ]; then
    echo "Error: DATASET_NAME_OR_ID is not provided."
    return -1
fi

# Source virtual environment (pip)
source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

# Set env variables
source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

# Lets try the biomedic3 location to see if this speeds up training...

# export nnUNet_raw=/vol/biomedic3/bglocker/nnUNet/nnUNet_raw
# export nnUNet_preprocessed=/vol/biomedic3/bglocker/nnUNet/nnUNet_preprocessed
# export nnUNet_results=/vol/biomedic3/bglocker/ugproj2324/az620/nnUNet_results

# Run python script
# nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
nnUNetv2_train $1 3d_fullres 0 -tr nnUNetTrainer_500epochs --npz
nnUNetv2_train $1 3d_fullres 1 -tr nnUNetTrainer_500epochs --npz
nnUNetv2_train $1 3d_fullres 2 -tr nnUNetTrainer_500epochs --npz
nnUNetv2_train $1 3d_fullres 3 -tr nnUNetTrainer_500epochs --npz
nnUNetv2_train $1 3d_fullres 4 -tr nnUNetTrainer_500epochs --npz

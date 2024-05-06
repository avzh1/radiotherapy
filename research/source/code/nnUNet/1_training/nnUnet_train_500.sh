#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 16                            # Number of CPU Cores
#SBATCH -p gpus                          # Partition (queue)
#SBATCH --gres gpu:1                     # gpu:n, where n = number of GPUs
#SBATCH --mem 20G                        # memory pool for all cores
#SBATCH --output=logs/training/slurm.train.7.%N.%j.log    # Standard output and error log
#SBATCH --nodelist loki

#SBATCH --job-name=7.nnUNEt

# --nodelist lory           # SLURM node
# --nodelist loki          	# SLURM node

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

# Check if $1 variable is empty
if [ -z "$1" ]; then
    echo "Error: DATASET_NAME_OR_ID is not provided."
    return -1
fi

# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 

# Run python script
# nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
nnUNetv2_train $1 3d_fullres 0 -tr nnUNetTrainer_500epochs --npz --val_best -p nnUNetResEncUNetLPlans --c
# nnUNetv2_train $1 3d_fullres 1 -tr nnUNetTrainer_500epochs --npz --val_best -p nnUNetResEncUNetLPlans --c
# nnUNetv2_train $1 3d_fullres 2 -tr nnUNetTrainer_500epochs --npz --val_best -p nnUNetResEncUNetLPlans --c
# nnUNetv2_train $1 3d_fullres 3 -tr nnUNetTrainer_500epochs --npz --val_best -p nnUNetResEncUNetLPlans --c
# nnUNetv2_train $1 3d_fullres 4 -tr nnUNetTrainer_500epochs --npz --val_best -p nnUNetResEncUNetLPlans --c


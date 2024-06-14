#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 16                            # Number of CPU Cores
#SBATCH -p gpus                          # Partition (queue)
#SBATCH --gres gpu:1                     # gpu:n, where n = number of GPUs
#SBATCH --mem 20G                        # memory pool for all cores
#SBATCH --output=logs/slurm.%N.%j.log    # Standard output and error log
#SBATCH --nodelist monal03

#SBATCH --job-name=infer.totbin

# --nodelist lory           # SLURM node
# --nodelist loki          	# SLURM node

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh


if [ -z "$1" ]; then
    echo "Error: FOLD is not provided."
    exit -1
fi

if [ -z "$2" ]; then
    echo "Error: TRAINER_NAME is not provided."
    exit -1
fi

# nnUNetTrainer_500epochs__totseg_nnUNetPlans__3d_fullres
# nnUNetTrainerCervical_500epochs__totseg_nnUNetPlans__3d_fullres

# Convert Python Script
jupyter nbconvert --to script '5_inference_custom_loss.ipynb'

# Run python script
python3 5_inference_custom_loss.py $1 $2
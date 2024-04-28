#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpushigh                     # Partition (queue)
#SBATCH --nodelist monal03          	# SLURM node
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --output=slurm.preprocess.%N.%j.log    # Standard output and error log

source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

# Run python script
echo '---------------'
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
echo '---------------'
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
echo '---------------'
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
echo '---------------'
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
echo '---------------'
nnUNetv2_plan_and_preprocess -d 5 --verify_dataset_integrity
echo '---------------'
nnUNetv2_plan_and_preprocess -d 6 --verify_dataset_integrity
echo '---------------'
nnUNetv2_plan_and_preprocess -d 7 --verify_dataset_integrity
echo '---------------'

# NOTE that this is pretty redundant, they are all the same dataset
#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

# Run python script
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 5 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 6 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 7 --verify_dataset_integrity

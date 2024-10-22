#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 16                       # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --output=logs/slurm.preprocess.%N.%j.log    # Standard output and error log

source /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/.venv/bin/activate

# Axis 1

python pre_CT_MR.py CT CTVn /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

python pre_CT_MR.py CT CTVp /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose
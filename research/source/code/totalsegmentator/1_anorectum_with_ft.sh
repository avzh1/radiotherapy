#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

export nnUNet_pretrained=$nnUNet_raw../../models/TotalSegmentator/.weights/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj
export nnUNet_results=${nnUNet_results}../TotalSegmentator_results

# ASSUMPTIONS:
# The data has been pre-processed and lives in the directory stored in environment variable $nnUNet_preprocessed which has been set above.
# We have a trained model stored 

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
echo $nnUNet_pretrained

# https://github.com/MIC-DKFZ/nnUNet/issues/774
# https://github.com/MIC-DKFZ/nnUNet/issues/1108

# 1. Preprocess the dataset for training with the pre-trained nnUNet model
nnUNetv2_plan_and_preprocess -d 1 --overwrite_plans_name TotalSegmentatorFineTuning preprocessor_name $nnUNet_pretrained

# 2. Train the model with the pre-processed weights

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 -pretrained_weights $nnUNet_pretrained/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth 


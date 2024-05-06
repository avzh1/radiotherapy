#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 16                       # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 20G                   # memory pool for all cores
#SBATCH --output=logs/slurm.planexperiment.%N.%j.log    # Standard output and error log

#SBATCH --job-name=planexperiment.nnUNEt

# --nodelist lory           # SLURM node
# --nodelist loki          	# SLURM node

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

############################
# Reason for the pl:
# INFO: You are using the old nnU-Net default planner. We have updated our recommendations. Please consider using those instead! Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
############################

# Run python script
nnUNetv2_plan_experiment -d 1 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
nnUNetv2_plan_experiment -d 2 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
nnUNetv2_plan_experiment -d 3 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
nnUNetv2_plan_experiment -d 4 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
nnUNetv2_plan_experiment -d 5 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
nnUNetv2_plan_experiment -d 6 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
nnUNetv2_plan_experiment -d 7 -np 4 4 4 # -pl nnUNetPlannerResEncL # -c 3d_lowres -pl nnUNetPlannerResEncM
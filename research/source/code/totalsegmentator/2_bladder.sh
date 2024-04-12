#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

source /vol/bitbucket/az620/radiotherapy/.venv/bin/activate

source /vol/bitbucket/az620/radiotherapy/data/data_vars.sh

# For some reason the python script isn't working, I'm going to do this manually for now

# jupyter nbconvert --to script '1_bladder.ipynb'
# python3 1_bladder.py 

TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_001_0000.nii.gz -o segmentations/bladder_001 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_002_0000.nii.gz -o segmentations/bladder_002 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_003_0000.nii.gz -o segmentations/bladder_003 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_004_0000.nii.gz -o segmentations/bladder_004 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_005_0000.nii.gz -o segmentations/bladder_005 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_006_0000.nii.gz -o segmentations/bladder_006 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_007_0000.nii.gz -o segmentations/bladder_007 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_008_0000.nii.gz -o segmentations/bladder_008 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_009_0000.nii.gz -o segmentations/bladder_009 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_010_0000.nii.gz -o segmentations/bladder_010 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_011_0000.nii.gz -o segmentations/bladder_011 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_012_0000.nii.gz -o segmentations/bladder_012 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_013_0000.nii.gz -o segmentations/bladder_013 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_014_0000.nii.gz -o segmentations/bladder_014 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_015_0000.nii.gz -o segmentations/bladder_015 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_016_0000.nii.gz -o segmentations/bladder_016 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_017_0000.nii.gz -o segmentations/bladder_017 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_018_0000.nii.gz -o segmentations/bladder_018 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_019_0000.nii.gz -o segmentations/bladder_019 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_020_0000.nii.gz -o segmentations/bladder_020 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_021_0000.nii.gz -o segmentations/bladder_021 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_022_0000.nii.gz -o segmentations/bladder_022 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_023_0000.nii.gz -o segmentations/bladder_023 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_024_0000.nii.gz -o segmentations/bladder_024 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_025_0000.nii.gz -o segmentations/bladder_025 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_026_0000.nii.gz -o segmentations/bladder_026 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_027_0000.nii.gz -o segmentations/bladder_027 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_028_0000.nii.gz -o segmentations/bladder_028 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_029_0000.nii.gz -o segmentations/bladder_029 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_030_0000.nii.gz -o segmentations/bladder_030 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_031_0000.nii.gz -o segmentations/bladder_031 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_032_0000.nii.gz -o segmentations/bladder_032 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_033_0000.nii.gz -o segmentations/bladder_033 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_034_0000.nii.gz -o segmentations/bladder_034 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_035_0000.nii.gz -o segmentations/bladder_035 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_036_0000.nii.gz -o segmentations/bladder_036 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_037_0000.nii.gz -o segmentations/bladder_037 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_038_0000.nii.gz -o segmentations/bladder_038 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_039_0000.nii.gz -o segmentations/bladder_039 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_040_0000.nii.gz -o segmentations/bladder_040 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_041_0000.nii.gz -o segmentations/bladder_041 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_042_0000.nii.gz -o segmentations/bladder_042 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_043_0000.nii.gz -o segmentations/bladder_043 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_044_0000.nii.gz -o segmentations/bladder_044 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_045_0000.nii.gz -o segmentations/bladder_045 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_046_0000.nii.gz -o segmentations/bladder_046 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_047_0000.nii.gz -o segmentations/bladder_047 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_048_0000.nii.gz -o segmentations/bladder_048 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_049_0000.nii.gz -o segmentations/bladder_049 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_050_0000.nii.gz -o segmentations/bladder_050 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_051_0000.nii.gz -o segmentations/bladder_051 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_052_0000.nii.gz -o segmentations/bladder_052 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_053_0000.nii.gz -o segmentations/bladder_053 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_054_0000.nii.gz -o segmentations/bladder_054 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_055_0000.nii.gz -o segmentations/bladder_055 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_056_0000.nii.gz -o segmentations/bladder_056 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_057_0000.nii.gz -o segmentations/bladder_057 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_058_0000.nii.gz -o segmentations/bladder_058 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_059_0000.nii.gz -o segmentations/bladder_059 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_060_0000.nii.gz -o segmentations/bladder_060 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_061_0000.nii.gz -o segmentations/bladder_061 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_062_0000.nii.gz -o segmentations/bladder_062 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_063_0000.nii.gz -o segmentations/bladder_063 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_064_0000.nii.gz -o segmentations/bladder_064 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_065_0000.nii.gz -o segmentations/bladder_065 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_066_0000.nii.gz -o segmentations/bladder_066 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_067_0000.nii.gz -o segmentations/bladder_067 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_068_0000.nii.gz -o segmentations/bladder_068 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_069_0000.nii.gz -o segmentations/bladder_069 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_070_0000.nii.gz -o segmentations/bladder_070 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_071_0000.nii.gz -o segmentations/bladder_071 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_072_0000.nii.gz -o segmentations/bladder_072 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_073_0000.nii.gz -o segmentations/bladder_073 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_074_0000.nii.gz -o segmentations/bladder_074 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_075_0000.nii.gz -o segmentations/bladder_075 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_076_0000.nii.gz -o segmentations/bladder_076 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_077_0000.nii.gz -o segmentations/bladder_077 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_078_0000.nii.gz -o segmentations/bladder_078 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_079_0000.nii.gz -o segmentations/bladder_079 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_080_0000.nii.gz -o segmentations/bladder_080 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_081_0000.nii.gz -o segmentations/bladder_081 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_082_0000.nii.gz -o segmentations/bladder_082 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_083_0000.nii.gz -o segmentations/bladder_083 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_084_0000.nii.gz -o segmentations/bladder_084 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_085_0000.nii.gz -o segmentations/bladder_085 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_086_0000.nii.gz -o segmentations/bladder_086 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_087_0000.nii.gz -o segmentations/bladder_087 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_088_0000.nii.gz -o segmentations/bladder_088 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_089_0000.nii.gz -o segmentations/bladder_089 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_090_0000.nii.gz -o segmentations/bladder_090 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_091_0000.nii.gz -o segmentations/bladder_091 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_092_0000.nii.gz -o segmentations/bladder_092 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_093_0000.nii.gz -o segmentations/bladder_093 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_094_0000.nii.gz -o segmentations/bladder_094 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_095_0000.nii.gz -o segmentations/bladder_095 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_096_0000.nii.gz -o segmentations/bladder_096 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_097_0000.nii.gz -o segmentations/bladder_097 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_098_0000.nii.gz -o segmentations/bladder_098 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_099_0000.nii.gz -o segmentations/bladder_099 --body_seg --roi_subset urinary_bladder
TotalSegmentator -i $nnUNet_raw/Dataset002_Bladder/imagesTr/zzAMLART_100_0000.nii.gz -o segmentations/bladder_100 --body_seg --roi_subset urinary_bladder

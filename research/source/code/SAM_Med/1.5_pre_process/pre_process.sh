#!/bin/bash

source /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/.venv/bin/activate

# Axis 0 

# python pre_CT_MR.py CT Anorectum /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# python pre_CT_MR.py CT Bladder /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# python pre_CT_MR.py CT CTVn /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# python pre_CT_MR.py CT CTVp /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# python pre_CT_MR.py CT Parametrium /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset005_Parametrium/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset005_Parametrium/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# python pre_CT_MR.py CT Uterus /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset006_Uterus/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset006_Uterus/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# python pre_CT_MR.py CT Vagina /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset007_Vagina/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset007_Vagina/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose

# # Axis 1

# python pre_CT_MR.py CT Anorectum /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# python pre_CT_MR.py CT Bladder /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# python pre_CT_MR.py CT CTVn /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# python pre_CT_MR.py CT CTVp /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# python pre_CT_MR.py CT Parametrium /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset005_Parametrium/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset005_Parametrium/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# python pre_CT_MR.py CT Uterus /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset006_Uterus/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset006_Uterus/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# python pre_CT_MR.py CT Vagina /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset007_Vagina/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset007_Vagina/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 1 --verbose

# # Axis 2

# python pre_CT_MR.py CT Anorectum /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

# python pre_CT_MR.py CT Bladder /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

python pre_CT_MR.py CT CTVn /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset003_CTVn/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

python pre_CT_MR.py CT CTVp /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset004_CTVp/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

python pre_CT_MR.py CT Parametrium /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset005_Parametrium/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset005_Parametrium/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

python pre_CT_MR.py CT Uterus /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset006_Uterus/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset006_Uterus/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

python pre_CT_MR.py CT Vagina /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset007_Vagina/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset007_Vagina/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 2 --verbose

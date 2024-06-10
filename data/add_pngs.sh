#!/bin/bash

# set up the data paths
source data_vars.sh

git add -f $nnUNet_results/**/**/**/*.png
git add -f $TotalSegmentator_results/**/**/**/*.png
git add -f $MedSAM_finetuned/**/**/*.png

git commit -m "update of learning curves for models"
git push
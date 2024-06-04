#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Structures
# 
# Assuming that the pre-processed data is available, run the fine-tuning process on the data
# we have for 50 epochs so that we may see it afterwards in due corse. We can increase the
# number of epochs later.

# In[ ]:


import os, sys
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..'))
if not dir1 in sys.path: sys.path.append(dir1)
from utils.environment import setup_data_vars
setup_data_vars()


# In[ ]:


def get_raw_and_gt_data_paths():
    
    setup_data_vars()

    classes = [os.environ.get('Anorectum')
             , os.environ.get('Bladder') 
             , os.environ.get('CTVn') 
             , os.environ.get('CTVp') 
             , os.environ.get('Parametrium') 
             , os.environ.get('Uterus') 
             , os.environ.get('Vagina')
             , os.environ.get('TotalBinary')
             , os.environ.get('TotalSegmentator')]

    raw_data = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingImages')) for x in classes]
    gt_labels = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingLabels')) for x in classes]

    return classes, raw_data, gt_labels


# In[ ]:


from totalsegmentator.config import setup_nnunet, setup_totalseg
from totalsegmentator.libs import download_pretrained_weights

def fetch_pretrained_totalsegmentator_model():
    """
    Fetch the pretrained TotalSegmentator model.

    The total segmentator model loads a separately trained nnUNet model for each new class
    However, it is not trained on the parametrium case. Therefore, we load the general
    model and attempt to finetune it on my case.
    """

    os.environ['TOTALSEG_HOME_DIR'] = os.path.join(os.environ.get('PROJECT_DIR'), 'models', 'TotalSegmentator', '.weights')

    setup_nnunet()
    setup_totalseg()

    # We assume that the model we are running will be finetuned with the 'total' task from
    # the original TotalSegmentator model because this contains the most information about
    # soft organ classification, most of which happens in the abdomen region, which
    # intuitively seems like the most logical knowledge to transfer into this task

    # From the totalsegmentator.python_api this task ID corresponds with the model trained
    # for organ segmentation. Note there is also potential for cropping the image.
    task_id = 291
    download_pretrained_weights(task_id)


# In[4]:


import sys
import argparse

if __name__ == '__main__':
    # Download the weights from TotalSegmentator
    fetch_pretrained_totalsegmentator_model()

    # Something in the fetch_pretrained_totalsegmentator_model overwrites the global variables
    setup_data_vars()
    classes, raw_data, gt_labels = get_raw_and_gt_data_paths()

    print('[DEBUG]: Obtained the environment variables. These are:')
    print(f'nnUNet_raw: {os.environ.get("nnUNet_raw")}')
    print(f'nnUNet_preprocessed: {os.environ.get("nnUNet_preprocessed")}')
    print(f'nnUNet_results: {os.environ.get("nnUNet_results")}')

    # Set the data to pre-train on the fingerprint of the training data.
    
    """
    TARGET_DATASET = the one you wish to fine tune on, Radiotherapy data
    SOURCE_DATASET = dataset you intend to run the pretraining on, TotalSegmentator

    1. nnUNetv2_plan_and_preprocess -d TARGET_DATASET (this has been done already)
    2. nnUNetv2_extract_fingerprint -d SOURCE_DATASET (this has been achieved by creating a dummy dataset id into which I copied the .json files obtained from the downloaded model)

    Path to the plans.json for TotalSegmentator:
    /vol/bitbucket/az620/radiotherapy/models/TotalSegmentator/.weights/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json

    3. Now I need to move the plans from the dummy dataset to the Radiotherapy dataset. Need to do this one at a time for each class

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=int, help='The dataset to fine tune on')
    parser.add_argument('fold', type=int, help='The fold to fine tune on')
    parser.add_argument('continue_trianing', type=bool, help='Whether to continue training from a checkpoint', default=False)
    args = parser.parse_args(
        # ['1', '0']
    )

    print('[DEBUG]: Arguments received:')
    print(f'Dataset: {args.dataset}')
    print(f'Fold: {args.fold}')
    print(f'Fold: {args.continue_trianing}')

    assert args.dataset is not None, "Please provide the dataset to fine tune on"
    assert args.dataset in range(1, len(classes) + 1), "Please provide a valid dataset to fine tune on"

    assert args.fold is not None, "Please provide the fold to fine tune on"
    assert args.fold in range(5), "Please provide a valid fold to fine tune on"

    TARGET_DATASET = args.dataset
    PATH_TO_CHECKPOINT = os.path.join(os.environ.get('PROJECT_DIR'), 'models', 'TotalSegmentator', '.weights', 'nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth')
    FOLD = args.fold
    CONFIG = '3d_fullres'

    os.environ['nnUNet_results'] = os.path.join(os.environ.get('DATA_DIR'), 'TotalSegmentator_results')

    # Run the training on the target dataset

    from nnunetv2.run.run_training import run_training_entry

    original_sys_argv = sys.argv

    print('-----------')
    print(f'FOLD: {FOLD}')
    print('-----------')

    # !nnUNetv2_train TARGET_DATASET CONFIG FOLD -pretrained_weights PATH_TO_CHECKPOINT
    sys.argv = [original_sys_argv[0], str(TARGET_DATASET), CONFIG, str(FOLD), '-tr', 'nnUNetTrainer_500epochs', '--npz', '-p', 'totseg_nnUNetPlans']
    if args.continue_trianing:
        sys.argv += ['--c']
    else:
        sys.argv += ['-pretrained_weights', PATH_TO_CHECKPOINT]

    run_training_entry()

    sys.argv = original_sys_argv


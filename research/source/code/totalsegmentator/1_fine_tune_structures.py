#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Structures
# 
# Assuming that the pre-processed data is available, run the fine-tuning process on the data
# we have for 50 epochs so that we may see it afterwards in due corse. We can increase the
# number of epochs later.

# In[1]:


import os 
import subprocess

def setup_data_vars(mine = True, overwrite = True):
    """
    From within any directory related to radiotherapy with backtrack into the data folder
    and execute the data_vars script. The assumption is that the datavars script will
    output the list of environment variables that need to be set. This function will set
    the environment variables for the current session.

    For the mean while, my model hasn't completely finished training, therefore, to get
    this task done, I will use Ben's pretrained nnUNet and then once mine has finished
    training I will use my own. For the mean while, this means that we can choose between
    using Ben's pretrained model or my own.
    """

    # If the environment variables are not set, assume that either a custom one has been
    # provided or resetting them again is a redundant task
    if os.environ.get('nnUNet_raw') is None or overwrite is True:
        # run the script in the data folder for specifying the environment variables
        if mine:
            cwd = os.getcwd().split('/')
            data_dir = os.path.join('/'.join(cwd[:cwd.index('radiotherapy') + 1]), 'data')

            # Assuming the data_vars.sh script echoes the environment variables
            script = os.path.join(data_dir, 'data_vars.sh')
            output = subprocess.run([script], capture_output=True)
            
            assert len(output.stdout) != 0, f"Please check {script} and make sure it echoes \
    the environment variables."

            output = output.stdout.decode('utf-8')
        else:
            data_dir = '/vol/biomedic3/bglocker/nnUNet'

            # Assuming this script won't change, it contains hard coded exports
            script = os.path.join(data_dir, 'exports')

            with open(script, 'r') as file:
                output = file.read()
        
        for line in output.split('\n'):
            if line != '':
                if mine:
                    line = line.split(': ')
                    os.environ[line[0]] = line[1]
                else:
                    line = line.split('=')
                    os.environ[line[0].split(' ')[1]] = line[1]

    assert os.environ.get('nnUNet_raw') is not None, "Environemnt variables not set. \
Please run the data_vars.sh script in the data folder."


# In[2]:


def get_raw_and_gt_data_paths():
    
    setup_data_vars()

    classes = [os.environ.get('data_Anorectum'), 
        os.environ.get('data_Bladder'), 
        os.environ.get('data_CTVn'), 
        os.environ.get('data_CTVp'), 
        os.environ.get('data_Parametrium'), 
        os.environ.get('data_Uterus'), 
        os.environ.get('data_Vagina')]

    raw_data = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingImages')) for x in classes]
    gt_labels = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingLabels')) for x in classes]

    return classes, raw_data, gt_labels


# In[3]:


from totalsegmentator.config import setup_nnunet, setup_totalseg
from totalsegmentator.libs import download_pretrained_weights

def fetch_pretrained_totalsegmentator_model():
    """
    Fetch the pretrained TotalSegmentator model.

    The total segmentator model loads a separately trained nnUNet model for each new class
    However, it is not trained on the parametrium case. Therefore, we load the general
    model and attempt to finetune it on my case.
    """

    os.environ['TOTALSEG_HOME_DIR'] = '/vol/bitbucket/az620/radiotherapy/models/TotalSegmentator/.weights'

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
    args = parser.parse_args()

    assert args.dataset is not None, "Please provide the dataset to fine tune on"
    assert args.dataset in range(1, len(classes) + 1), "Please provide a valid dataset to fine tune on"

    TARGET_DATASET = args.dataset
    PATH_TO_CHECKPOINT = '/vol/bitbucket/az620/radiotherapy/models/TotalSegmentator/.weights/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
    FOLD = 0
    CONFIG = '3d_fullres'

    # Run the training on the target dataset

    from nnunetv2.run.run_training import run_training_entry

    original_sys_argv = sys.argv

    for FOLD in range(5):
        print('-----------')
        print(f'FOLD: {FOLD}')
        print('-----------')

        # !nnUNetv2_train TARGET_DATASET CONFIG FOLD -pretrained_weights PATH_TO_CHECKPOINT
        sys.argv = [original_sys_argv[0], str(TARGET_DATASET), CONFIG, str(FOLD), '-pretrained_weights', PATH_TO_CHECKPOINT, '-tr', 'nnUNetTrainer_50epochs', '--npz', '-p', 'totseg_nnUNetPlans']
        run_training_entry()

    sys.argv = original_sys_argv


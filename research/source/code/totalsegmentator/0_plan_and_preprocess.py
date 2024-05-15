#!/usr/bin/env python
# coding: utf-8

# # Planning and Preprocessing Data for Transfer Learning
# 
# In order to fine tune the model on data that we have, the data must be transfored so that
# it matches the fingerprint of the data the original model was trained with. We can
# apparently fine-tune the model with the `plans.json` file that appears in each model's
# checkpoint once we download the weights for the model.
# 
# 

# In[8]:


import os, sys
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..'))
if not dir1 in sys.path: sys.path.append(dir1)


# In[9]:


from utils.environment import setup_data_vars
setup_data_vars()


# In[10]:


def get_raw_and_gt_data_paths():
    
    setup_data_vars()

    classes = [os.environ.get('Anorectum')
             , os.environ.get('Bladder') 
             , os.environ.get('CTVn') 
             , os.environ.get('CTVp') 
             , os.environ.get('Parametrium') 
             , os.environ.get('Uterus') 
             , os.environ.get('Vagina')]

    raw_data = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingImages')) for x in classes]
    gt_labels = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingLabels')) for x in classes]

    return classes, raw_data, gt_labels


# In[11]:


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


# In[15]:


import shutil
import sys

if __name__ == '__main__':
    # Download the weights from TotalSegmentator
    fetch_pretrained_totalsegmentator_model()

    # Something in the fetch_pretrained_totalsegmentator_model overwrites the global variables
    classes, raw_data, gt_labels = get_raw_and_gt_data_paths()
    
    source_file = os.path.join(os.environ.get('TOTALSEG_HOME_DIR'), 'nnunet', 'results', 'Dataset291_TotalSegmentator_part1_organs_1559subj', 'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres', 'plans.json')
    destination_file = os.path.join(os.environ.get('nnUNet_preprocessed'), 'Dataset008_TotalSegmentator', 'nnUNetPlans.json')

    assert os.path.exists(source_file), "The source file does not exist"
    shutil.copy(source_file, destination_file)

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

    print('RUNNING THE TRANSFER OF PLANS BETWEEN DATASETS......................')

    from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets

    starting_class = 1
    end_class = 7 # len(classes)

    for i in range(starting_class, end_class + 1):
        print(f'Transferring for {i}')

        move_plans_between_datasets(source_dataset_name_or_id=8
                                    , target_dataset_name_or_id=i
                                    , source_plans_identifier='nnUNetPlans'
                                    , target_plans_identifier=f'totseg_nnUNetPlans'
        )

    print('Now you can run the preprocessing on the source task:')

    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import preprocess_entry

    original_sys_argv = sys.argv

    for i in range(starting_class, end_class + 1):
        print('Preprocessing for class:', str(i), ' (', classes[i - 1], ')')

        sys.argv = [original_sys_argv[0], '-d', str(i), '-plans_name', 'totseg_nnUNetPlans', '-c', '3d_fullres', '--verbose', '-np', '4'] #, '-overwrite_plans_name', 'totseg_nnUNetPlans']
        preprocess_entry()
    
    sys.argv = original_sys_argv


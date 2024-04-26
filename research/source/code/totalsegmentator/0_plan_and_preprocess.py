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

# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


import sys

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

    print('RUNNING THE TRANSFER OF PLANS BETWEEN DATASETS......................')

    from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets

    starting_class = 5
    end_class = len(classes)

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


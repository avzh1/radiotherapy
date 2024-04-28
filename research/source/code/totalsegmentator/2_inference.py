#!/usr/bin/env python
# coding: utf-8

# # Run inference on the nnUNet Model we have fine-tuned
# 
# Assuming that the pre-processed data is available, and the model has been trained for a
# fold

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


def initialise_predictor(model_path, fold, device):

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=fold,
        checkpoint_name='checkpoint_final.pth',
    )

    return predictor


# In[6]:


import sys
import torch
import argparse
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   

    setup_data_vars()
    classes, raw_data, gt_labels = get_raw_and_gt_data_paths()

    print('[DEBUG]: Obtained the environment variables. These are:')
    print(f'nnUNet_raw: {os.environ.get("nnUNet_raw")}')
    print(f'nnUNet_preprocessed: {os.environ.get("nnUNet_preprocessed")}')
    print(f'nnUNet_results: {os.environ.get("nnUNet_results")}')

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=int, help='The dataset to run inference on')
    parser.add_argument('fold', type=int, help='The max number of nodes that were trained')
    args = parser.parse_args()
    
    assert args.dataset is not None, "Please provide the dataset to fine tune on"
    assert args.dataset in range(1, len(classes) + 1), "Please provide a valid dataset to fine tune on"

    assert args.fold is not None, "Please provide the fold to run inference on"
    assert args.fold in range(5), "Please provide a valid fold to run inference on"

    TARGET_DATASET = args.dataset
    FOLD = tuple(range(0, args.fold + 1))
    CONFIG = '3d_fullres'

    # TARGET_DATASET = 2
    # FOLD = tuple(range(0, 0 + 1))

    # Run inference
    model_name = 'nnUNetTrainer_50epochs__totseg_nnUNetPlans__3d_fullres'
    input_file = os.path.join(os.environ.get('nnUNet_raw'), classes[TARGET_DATASET - 1], os.environ.get('data_trainingImages'))
    model_path = os.path.join(os.environ.get('nnUNet_results'), classes[TARGET_DATASET - 1], model_name) 
    output_file = os.path.join(os.environ.get('nnUNet_raw'), '..', 'TotalSegmentator_inference', classes[TARGET_DATASET - 1], model_name)

    print('I am predicting on the dataset:', classes[TARGET_DATASET - 1])
    print('The Fold is:', FOLD)
    print('The config I\'m using is:', CONFIG)
    print('The model path is:', model_path)
    print('The input file is:', input_file)
    print('The output file is:', output_file)

    predictor = initialise_predictor(model_path, FOLD, device)
    predictor.predict_from_files(input_file,
                                 output_file,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


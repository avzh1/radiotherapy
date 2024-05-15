#!/usr/bin/env python
# coding: utf-8

# # Performance of nnUNet architecture

# Given a checkpoint of a model for a given class, evaluate the different metrics of the 
# model by performing forward inference on the model and extracting the different
# evaluation metrics on the data, such as DICE, Haussdorff distance, Jaccard etc.

# ## Setup env variables

# In[1]:


import os, sys
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..', '..'))
if not dir1 in sys.path: sys.path.append(dir1)


# In[2]:


from utils.environment import setup_data_vars


# ## Setup Inference Pipeline

# In[3]:


import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device


# In[4]:


def initialise_predictor(model_path, fold):

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
        os.path.join(os.environ.get('nnUNet_results'), model_path),
        use_folds=fold,
        checkpoint_name='checkpoint_best.pth',
    )

    return predictor


# In[5]:


def predict_file(input_file, output_file, predictor):
    """
    Predict the segmentation of a single file and save it to the output location.
    """

    if os.path.exists(input_file):
        print(f"{input_file} exists")
    else:
        print(f"{input_file} does not exist")

    predictor.predict_from_files(input_file,
                                 output_file,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    # predictor.predict_from_files([[file_name]],
    #                             [output_name],
    #                             save_probabilities=False, overwrite=overwrite,
    #                             num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                             folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


# Running inference on inputs: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/readme.md
# 
# Running on slurm requires freezing: https://stackoverflow.com/questions/24374288/where-to-put-freeze-support-in-a-python-script

# In[7]:


import multiprocessing
import argparse

if __name__ == '__main__':
    multiprocessing.freeze_support()

    setup_data_vars()

    classes = [os.environ.get('Anorectum')
             , os.environ.get('Bladder')
             , os.environ.get('CTVn')
             , os.environ.get('CTVp')
             , os.environ.get('Parametrium')
             , os.environ.get('Uterus')
             , os.environ.get('Vagina')]

    input_data_path = [os.path.join(os.environ.get('nnUNet_raw'), c, os.environ.get('data_trainingImages')) for c in classes]
    output_path = [os.path.join(os.environ.get('nnUNet_inference'), f'{c}_3dhighres') for c in classes]
    model_paths = [os.path.join(c, 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres') for c in classes]
    
    # for each model path, get the list of folds that contain checkpoints in them
    folds_per_model = [set() for _ in model_paths]

    for i,m in enumerate(model_paths):
        full_path = os.path.join(os.environ.get('nnUNet_results'), m)
        print(f'searching for folds for {classes[i]},', end=' ')
        _, subdirs, _ = next(os.walk(full_path))
        for folds in sorted(subdirs):
            for checkpoints in os.listdir(os.path.join(full_path, folds)):
                if '.pth' in checkpoints:
                    folds_per_model[i].add(folds.split('_')[1])
        print(f'found folds {folds_per_model[i]}')
    
    folds_per_model = [tuple(x) for x in folds_per_model]

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id', type=int, help='The dataset id to fine tune on')
    
    # import sys
    # original_args = sys.argv
    # sys.argv = [original_args[0], '3']
    
    args = parser.parse_args()

    input_class = input_data_path[args.dataset_id - 1]
    model_path = model_paths[args.dataset_id - 1]
    fold = folds_per_model[args.dataset_id - 1]

    input_location = os.path.join(os.environ.get('nnUNet_raw'), input_class)
    output_location = os.path.join(os.environ.get('nnUNet_inference'), f'{input_class}_3dhighres')
    
    print(f'initialising predictor with class {classes[args.dataset_id - 1]}, and folds {fold}')
    predictor = initialise_predictor(model_path, fold)
    
    print(f'predicting from {input_location} to {output_location}')
    predict_file(input_location, output_location, predictor)


# In[ ]:





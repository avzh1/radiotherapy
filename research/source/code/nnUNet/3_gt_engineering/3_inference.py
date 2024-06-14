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
                                 save_probabilities=True,
                                #  flatten_prediction=False,
                                 overwrite=False,
                                 num_processes_preprocessing=2,
                                 num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None,
                                 num_parts=1,
                                 part_id=0)

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    # predictor.predict_from_files([[file_name]],
    #                             [output_name],
    #                             save_probabilities=False, overwrite=overwrite,
    #                             num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                             folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


# Running inference on inputs: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/readme.md
# 
# Running on slurm requires freezing: https://stackoverflow.com/questions/24374288/where-to-put-freeze-support-in-a-python-script

# In[6]:


import multiprocessing

import argparse

if __name__ == '__main__':
    multiprocessing.freeze_support()

    setup_data_vars()

    parser = argparse.ArgumentParser()
    parser.add_argument('fold', type=int, help='fold to predict from, set to -1 to infer automatically')
    parser.add_argument('trainer_name', type=str, help='name of the trainer to use (e.g. nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres)')

    args = parser.parse_args()
    
    fold = args.fold
    trainer_name = args.trainer_name

    assert trainer_name in ['nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres', 'nnUNetTrainerCervical_500epochs__nnUNetPlans__3d_fullres']

    class_of_interest = os.environ.get('TotalBinary')
    input_data_path = os.path.join(os.environ.get('nnUNet_raw'), class_of_interest, os.environ.get('data_trainingImages'))
    output_path = os.path.join(os.environ.get('nnUNet_inference'), class_of_interest, trainer_name)
    model_path = os.path.join(class_of_interest, trainer_name)    

    if fold == -1:
        folds_for_model = set()
        full_path = os.path.join(os.environ.get('nnUNet_results'), model_path)
        _, subdirs, _ = next(os.walk(full_path))
        for folds in sorted(subdirs):
            for checkpoints in os.listdir(os.path.join(full_path, folds)):
                if '.pth' in checkpoints:
                    folds_for_model.add(folds.split('_')[1])

        fold = tuple(folds_for_model)
    else:
        fold = tuple([fold])
    
    print(f'initialising predictor with class {class_of_interest}, and folds {fold}')
    predictor = initialise_predictor(model_path, fold)

    setup_data_vars()
    
    print(f'predicting from {input_data_path} to {output_path}')
    predict_file(input_data_path, output_path, predictor)


# ## Custom predictor

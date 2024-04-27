# %% [markdown]
# # Performance of nnUNet architecture

# %% [markdown]
# Given a checkpoint of a model for a given class, evaluate the different metrics of the 
# model by performing forward inference on the model and extracting the different
# evaluation metrics on the data, such as DICE, Haussdorff distance, Jaccard etc.

# %% [markdown]
# ## Setup env variables

# %%
import sys
import os

sys.path.append('../')

from config import setup_data_vars

sys.path.append('2_inference/')

# %% [markdown]
# ## Setup Inference Pipeline

# %%
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device

# %%
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
        # use_folds=(0,1,2,3,4),
        use_folds=fold,
        checkpoint_name='checkpoint_best.pth',
    )

    return predictor

# %%
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

# %% [markdown]
# Running inference on inputs: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/readme.md
# 
# Running on slurm requires freezing: https://stackoverflow.com/questions/24374288/where-to-put-freeze-support-in-a-python-script

# %%
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    input_classes = ['Dataset001_Anorectum/imagesTr',
                   'Dataset002_Bladder/imagesTr',
                   'Dataset003_CTVn/imagesTr',
                   'Dataset004_CTVp/imagesTr',
                   'Dataset005_Parametrium/imagesTr',]
    
    model_paths = ['Dataset001_Anorectum/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres/',
                   'Dataset002_Bladder/nnUNetTrainer__nnUNetPlans__3d_fullres/',
                   'Dataset003_CTVn/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres/',
                   'Dataset004_CTVp/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres/',
                   'Dataset005_Parametrium/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres/',
                   ]

    folds = [(0,1,2), 
             (0,),
             (0,),
             (0,),
             (0,)
             ]
    
    input_class = input_classes[2]
    model_path = model_paths[2]
    fold = folds[2]

    input_location = os.path.join(os.environ.get('nnUNet_raw'), input_class)
    output_location = os.path.join('/vol/bitbucket/az620/radiotherapy/data/nnUNet_inference/', f'{input_class}_3dhighres')

    setup_data_vars(mine=True, overwrite=True)
    predictor = initialise_predictor(model_path, fold)
    predict_file(input_location, output_location, predictor)



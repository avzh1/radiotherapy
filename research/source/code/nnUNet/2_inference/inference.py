# %% [markdown]
# # Performance of nnUNet architecture

# %% [markdown]
# Given a checkpoint of a model for a given class, evaluate the different metrics of the 
# model by performing forward inference on the model and extracting the different
# evaluation metrics on the data, such as DICE, Haussdorff distance, Jaccard etc.

# %% [markdown]
# ## Setup env variables

# %%
import os 
import subprocess

def setup_data_vars(mine = False, overwrite = False):
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

# %% [markdown]
# ## Setup Inference Pipeline

# %%
model_path = 'Dataset001_Anorectum/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres/'

# %%
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device

# %%
def initialise_predictor():

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
        use_folds=(0,1),
        checkpoint_name='checkpoint_final.pth',
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

# %%
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    input_location = os.path.join(os.environ.get('nnUNet_raw'), 'Dataset001_Anorectum/imagesTr')
    output_location = os.path.join('/vol/bitbucket/az620/radiotherapy/data/nnUNet_inference/', 'Dataset001_Anorectum/imagesTr_3dhighres')

    # join = os.path.join

    # file_path = join(input_location, 'zzAMLART_001_0000.nii.gz')
    # output_path = join(output_location, 'zzAMLART_001.nii.gz')

    setup_data_vars(mine=True, overwrite=True)
    predictor = initialise_predictor()
    predict_file(input_location, output_location, predictor)



#!/usr/bin/env python
# coding: utf-8

# # Functions

# In[1]:


import os, sys
dir1 = os.path.dirname(os.path.abspath(''))
if not dir1 in sys.path: sys.path.append(dir1)


# In[2]:


from utils.environment import setup_data_vars
setup_data_vars()


# ## Calculate Metrics

# In[3]:


import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import itk
import os
import math

def calculate_for_MedSAM(prediction_path):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    # Get the number of axis that we've processed for this anatomy
    _, subdirs, _ = next(os.walk(prediction_path))
    subdirs = [int(x[len('axis'):]) for x in subdirs if 'axis' in x]

    anatomy = os.path.basename(prediction_path)

    ground_truth_path = os.path.join(os.environ.get('MedSAM_preprocessed'), 'gts', anatomy)

    dice = {}
    jaccard = {}
    volume_similarity = {}

    # Got through each axis and calculate the metrics
    for axis in subdirs:
        dice[axis] = []
        jaccard[axis] = []
        volume_similarity[axis] = []

        # Get the full list of predictions for this anatomy
        predictions = os.listdir(os.path.join(prediction_path, f'axis{axis}'))
        for pred_name in tqdm(predictions, desc=f'predicting for axis{axis}'):
            # Calculate the metrics
            pred_path = os.path.join(prediction_path, f'axis{axis}', pred_name)
            assert os.path.exists(pred_path), f"Prediction file {pred_path} does not exist"

            # Get information about prediction
            image_id = pred_name.split('-')[0]
            image_slice = pred_name.split('-')[1].split('.')[0]

            # Get ground truth
            gt = os.path.join(ground_truth_path, f'axis{axis}', f'CT_{anatomy}_zzAMLART_{image_id.zfill(3)}-{image_slice.zfill(3)}.npy')
            assert os.path.exists(gt), f"Ground truth file {gt} does not exist"

            # Calculate metrics
            prediction_array = np.load(pred_path)
            ground_truth_array = np.load(gt)

            try:
                ypred_sitk = sitk.GetImageFromArray(prediction_array)
                y_gt_sitk = sitk.GetImageFromArray(ground_truth_array)

                overlap_measures_filter.Execute(y_gt_sitk, ypred_sitk)
            
                dice[axis].append(overlap_measures_filter.GetDiceCoefficient())
                jaccard[axis].append(overlap_measures_filter.GetJaccardCoefficient())
                volume_similarity[axis].append(overlap_measures_filter.GetVolumeSimilarity())
            except Exception as e:
                print(f"Error calculating metrics for {pred_path} and {gt}")
                print(e)

    return dice, jaccard, volume_similarity   


# In[4]:


def calculate_metrics(input_path_gt: str, input_path_pred: dict, axis = None):
    """Will return the metrics for each prediction path

    Args:
        input_path_gt (str): a string to the ground truth 
        input_path_pred (dict): a dictionary containing the paths to the predictions
        axis (int, optional): the axis to calculate the metrics on. Defaults to None.

    Returns:
        dict: returns dictionary of metrics for the main prediction paths
    """

    # def read_slice(image, slice_index, axis):
    #     # https://examples.itk.org/src/filtering/imagegrid/processa2dsliceofa3dimage/documentation
    #     size = image.GetLargestPossibleRegion().GetSize()

    #     # Create an image region that will be used to extract a single slice along an axis
    #     extractRegion = image.GetLargestPossibleRegion()

    #     # Create an extract region set to extract a single slice along an axis
    #     extractRegion_size = [size[0], size[1], size[2]]
    #     extractRegion_size[axis] = 1
    #     extractRegion.SetSize(extractRegion_size)

    #     # Update extracting region to current slice
    #     extractRegion_index = [0] * 3
    #     extractRegion_index[axis] = slice_index
    #     extractRegion.SetIndex(extractRegion_index)

    #     # Extract Region
    #     extractFilter = itk.ExtractImageFilter.New(image)
    #     extractFilter.SetDirectionCollapseToSubmatrix()
    #     extractFilter.SetExtractionRegion(extractRegion)

    #     # Extract Data
    #     sliceImage = extractFilter.GetOutput()
    #     sliceImage.Update()
    #     return itk.array_from_image(sliceImage)


    ground_truth = [os.path.join(input_path_gt, file) for file in sorted(filter(lambda x: '.nii.gz' in x, os.listdir(input_path_gt)))]

    final_metrics = dict()
    for k, _ in input_path_pred.items(): final_metrics[k] = dict()

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    # Iterate over the models and their predictions
    for prediction_type, prediction_path in input_path_pred.items():

        dice = []
        jaccard = []
        volume_similarity = []

        if prediction_type in ['MedSAM']:
            dice, jaccard, volume_similarity = calculate_for_MedSAM(prediction_path)
            final_metrics[prediction_type]['dice'] = dice
            final_metrics[prediction_type]['jaccard'] = jaccard
            final_metrics[prediction_type]['volume_similarity'] = volume_similarity
            continue

        final_metrics[prediction_type]['dice'] = dice
        final_metrics[prediction_type]['jaccard'] = jaccard
        final_metrics[prediction_type]['volume_similarity'] = volume_similarity

        # Extract all .nii.gz files for metric calculation
        predictions = [os.path.join(prediction_path, file) for file in sorted(filter(lambda x: '.nii.gz' in x, os.listdir(prediction_path)))]

        # Iterate over pairs of predictions and ground truth
        for ypred, y_gt in tqdm(zip(predictions, ground_truth), desc=f'Calculating metrics for {prediction_type}', ncols=len(predictions)):
            if axis is not None:
                # # If we have specified an axis the process is a little bit different. We
                # # extract a slice from the image, and the same slice from the ground truth and
                # # use this to calculate the metric localised at this slice along the supplied
                # # axis
                # ypred_itk = itk.imread(ypred)
                # y_gt_itk = itk.imread(y_gt)
                
                # assert ypred_itk.GetLargestPossibleRegion().GetSize() == y_gt_itk.GetLargestPossibleRegion().GetSize(), "The images must be the same size"

                # size = y_gt_itk.GetLargestPossibleRegion().GetSize()

                # macro_dice = []
                # macro_jaccard = []
                # # macro_volume_similarity = []

                # for s in range(size[axis]):
                #     ypred_slice_array = read_slice(ypred_itk, s, axis)
                #     y_gt_slice_array = read_slice(y_gt_itk, s, axis)

                #     ypred_sitk = sitk.GetImageFromArray(ypred_slice_array)
                #     y_gt_sitk = sitk.GetImageFromArray(y_gt_slice_array)

                #     overlap_measures_filter.Execute(y_gt_sitk, ypred_sitk)
                    
                #     dice_score = overlap_measures_filter.GetDiceCoefficient()
                #     jaccard_score = overlap_measures_filter.GetJaccardCoefficient()
                #     # volume_similarity_score = overlap_measures_filter.GetVolumeSimilarity()

                #     if 0 <= dice_score <= 1:
                #         macro_dice.append(dice_score)
                #     if 0 <= jaccard_score <= 1:
                #         macro_jaccard.append(jaccard_score)
                #     # if not(math.isnan(volume_similarity_score) or math.isinf(volume_similarity_score)):
                #         # macro_volume_similarity.append(volume_similarity_score)

                # dice += macro_dice
                # jaccard += macro_jaccard
                # # volume_similarity += macro_volume_similarity
                pass
            else:
                # If we wish to process the metrics as a 3D whole image, then this is
                # trivially done without loading each slice separately
                ypred_sitk = sitk.ReadImage(ypred)
                y_gt_sitk = sitk.ReadImage(y_gt)

                overlap_measures_filter.Execute(y_gt_sitk, ypred_sitk)

                dice.append(overlap_measures_filter.GetDiceCoefficient())
                jaccard.append(overlap_measures_filter.GetJaccardCoefficient())
                volume_similarity.append(overlap_measures_filter.GetVolumeSimilarity())

        final_metrics[prediction_type]['dice'] = dice
        final_metrics[prediction_type]['jaccard'] = jaccard
        final_metrics[prediction_type]['volume_similarity'] = volume_similarity

    return final_metrics


# In[5]:


def fetch_metric_for_class(class_id: int, axis = None):
    """Prints a plot of the segmentations for the given class in the predefined format.
    This method acts in a factory pattern to generate the plot for each class. 

    Args:
        class_id (int): 1: Anorectum ... 5: Parametrium
    """

    setup_data_vars()

    classes = [os.environ.get('Anorectum')
             , os.environ.get('Bladder')
             , os.environ.get('CTVn')
             , os.environ.get('CTVp')
             , os.environ.get('Parametrium')
             , os.environ.get('Uterus')
             , os.environ.get('Vagina')]

    gt_labels = [os.path.join(os.environ.get('nnUNet_raw'), x, os.environ.get('data_trainingLabels')) for x in classes]
    # print('WARNING: using old nnUNet predictions. Change to new path when complete')

    anorectum = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('Anorectum'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('Anorectum'), 'imagesTr_3dhighres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('Anorectum'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'Anorectum'),
    }

    bladder = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('Bladder'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('Bladder'), 'imagesTr_3dhighres'),
        'total segmentator': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('Bladder'), 'nnUNetTrainer__nnUNetPlans__3d_fullres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('Bladder'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'Bladder'),
    }

    ctvn = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('CTVn'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('CTVn'), 'imagesTr_3dhighres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('CTVn'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'CTVn'),   
    }

    ctvp = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('CTVp'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('CTVp'), 'imagesTr_3dhighres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('CTVp'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'CTVp'),
    }

    parametrium = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('Parametrium'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('Parametrium'), 'imagesTr_3dhighres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('Parametrium'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'Parametrium'),
    }

    uterus = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('Uterus'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('Uterus'), 'imagesTr_3dhighres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('Uterus'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'Uterus'),
    }

    vagina = {
        'nnUNet': os.path.join(os.environ.get('nnUNet_inference'), os.environ.get('Vagina'), 'nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres'),
        # 'nnUNet (50 eps)': os.path.join(os.environ.get('OLD_DIR'), 'data', 'nnUNet_inference', os.environ.get('Vagina'), 'imagesTr_3dhighres'),
        'total segmentator (fine-tuned)': os.path.join(os.environ.get('TotalSegmentator_inference'), os.environ.get('Vagina'), 'nnUNetTrainer_250epochs__totseg_nnUNetPlans__3d_fullres'),
        # 'MedSAM': os.path.join(os.environ.get('MedSAM_results'), 'Vagina'),
    }

    predictions = [anorectum, bladder, ctvn, ctvp, parametrium, uterus, vagina]

    if 1 <= class_id <= len(predictions):
        return calculate_metrics(gt_labels[class_id - 1], predictions[class_id - 1], axis)
    raise ValueError("Invalid class_id. Please choose a class between 1 and 5.")


# ## Plot the metrics

# In[6]:


import shutil

def get_id_from_path(path: str, needs_num=True):
    """Given a path, assume that it is the full path that points to the file name. The
    file nam ehsould contain a number indicating the id number. It should appear first.

    Args:
        path (str): A path to the file name or the file name itsself. 
        
        needs_num (bool, optional): If the path needs a number. If it doesn't and no
        number was found return 0, otherwise return the number found. Defaults to True.

    Raises:
        ValueError: If a number is required and no number was found in the path

    Returns:
        int: the number found in the path
    """
    import re
    # Assume that it is the full path that points to the file name. The file name
    # should contain a number indicating the id number. It should appear first
    numbers = re.findall('\d+', path.split('/')[-1])
    if needs_num and len(numbers) == 0:
        raise ValueError(f"Could not find a number in {path}")
    if not needs_num and len(numbers) == 0:
        return 0
    return int(numbers[0])

def stats_about_metrics(metrics_dictionary: dict):

    return_metrics_dictionary = dict()
    return_metrics_dictionary['mean'] = dict()

    for model, metrics in metrics_dictionary.items():
        return_metrics_dictionary['mean'][model] = dict()
        for metric, data in metrics.items():
            return_metrics_dictionary['mean'][model][metric] = np.mean(data)

    return return_metrics_dictionary

def plot_metrics(metrics_dictionary: dict, organ_class: str, separate: bool = False, save: bool = False, showfliers = True, table = True, additional_title_context = ''):
    """Plots the metrics for the given dictionary of metrics and prints a table of mean
    metrics to the right

    Args:
        metrics_dictionary (dict): A dictionary containing a key value pair of model type
        and value of dictionary. This dictionary will have a key value pairing of metric
        type and a list of values for that metric.
        
        organ_class (str): For saving the figure, supply the name of the organ class
        
        separate (bool, optional): Whether we print each type of model type separately or
        together so that for each metric we plot the models side by side for better
        comparison. Defaults to False.

        save (bool, optional): Whether to save the figure or not. Defaults to False.

        showfliers (bool, optional): Whether to show the outliers in the boxplot. Defaults
        to True.

        table (bool, optional): Whether to show the table of mean metrics or not. Defaults
        to True.

        additional_title_context (str, optional): Additional context to add to the title
        of the plot. Defaults to ''.
    """

    def draw_table(metric_result, ax, model_type):
        # Fetch statistics
        mean = [np.mean(result) for result in metric_result]
        std = [np.std(result) for result in metric_result]
        median = [np.median(result) for result in metric_result]

        # Format the table
        cell_text = []
        cell_text.append([f'{mean[i]:.2f}' for i in range(len(mean))])
        cell_text.append([f'{std[i]:.2f}' for i in range(len(std))])
        cell_text.append([f'{median[i]:.2f}' for i in range(len(std))])
        rowLabels = [r'$\hat{x}$', r'$\sigma$', r'$med$']

        import textwrap as twp
        max_line_width = 18
        colLabels = [twp.fill(label, max_line_width) for label in model_type]

        # Plot the table
        table_height = 0.05 * len(rowLabels)
        ax_box = ax.get_position()
        ax.set_position([ax_box.x0, ax_box.y0, ax_box.width, ax_box.height * (1 - table_height)])
        #  bbox=[0., 1., 1., table_height / (1 - table_height)] the position before
        table = ax.table(cell_text, cellLoc='center', rowLabels=rowLabels, colLabels=colLabels if not separate else None, fontsize=100) #, bbox=[0., 1., 1., table_height / (1 - table_height)] )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        

    if separate and table:
        print('[WARNING]: Bug with table printing in the separate mode. Set table=False to avoid this.')

    augmented_metrics_dictionary = metrics_dictionary

    # If we have medsam predictions, for now, plot the different axese separately
    if 'MedSAM' in metrics_dictionary.keys():
        medsam_metrics = metrics_dictionary['MedSAM']
        for metric in medsam_metrics.keys():
            for axis in medsam_metrics[metric].keys():
                if f'MedSAM axis{axis}' not in augmented_metrics_dictionary.keys():
                    augmented_metrics_dictionary[f'MedSAM axis{axis}'] = dict()
                augmented_metrics_dictionary[f'MedSAM axis{axis}'][metric] = medsam_metrics[metric][axis]
        del augmented_metrics_dictionary['MedSAM']

    num_models = len(augmented_metrics_dictionary)

    nrows = num_models if separate == True else 1
    metrics_names = [list(augmented_metrics_dictionary[m].keys()) for m in augmented_metrics_dictionary.keys()]
    metrics_names = sorted(list(set([metric for model_metrics in metrics_names for metric in model_metrics])))
    ncols = len(metrics_names)

    plot_height = 5*1.6
    plot_width = 2*1.6

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(plot_width * (ncols + num_models - 1), nrows * plot_height))

    # Reshape axes to be a 2D array
    axes = np.reshape(axes, (nrows, ncols))

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            # << Fetch Data Collections For Plotting >>

            # Get the model type for printing. If we're not separating models, model_type
            # is a list of all model names
            model_type = list(augmented_metrics_dictionary.keys())[i] if separate else list(augmented_metrics_dictionary.keys())
            # Get the metric type for the current column. Its possible that this metric
            # doesn't exist for all models
            metrics_type = metrics_names[j]
            # Get the data for the current metric type while checking if it exists for the
            # current model
            getData = lambda m: augmented_metrics_dictionary[m][metrics_type] if metrics_type in augmented_metrics_dictionary[m].keys() else []
            metric_result = getData(model_type) if separate else [getData(model) for model in augmented_metrics_dictionary.keys()]

            # << Plot the Data >>
            nans = [float('nan'), float('nan')]
            metric_result = [val or nans for val in metric_result]
            # ax.violinplot(metric_result)
            ax.boxplot(metric_result, showfliers=showfliers, meanline=True, showmeans=True, patch_artist=True, widths=(.4))

            # << Plot Table with Mean Metrics >>
            
            if table:
                draw_table(np.reshape(metric_result, (1, -1)).tolist() if separate else metric_result, ax, model_type) 

            # << Axis Formatting >>

            ax.set_title(metrics_type.capitalize(), fontsize='x-large') # y=1.02,
            # TODO: Bug with separable mode. The x-axis labels are not being removed
            ax.set_xticklabels([] if table else model_type, rotation=30) #  if not separate else model_type, rotation=30
            ax.tick_params(axis='x', which='both', bottom=False)
    
    if separate:
        for ax, title in zip(axes[:,0], augmented_metrics_dictionary.keys()):
            ax.set_ylabel(title.capitalize(), fontsize='xx-large')
    
    # if organ class like Dataset000_classname, extract classname
    import re
    match = re.search(r'Dataset\d+_(\w+)', organ_class)
    if match:
        organ_class = match.group(1).lower()

    fig.tight_layout()
    fig.set_facecolor('silver')
    fig.suptitle(f'Segmentation metrics for the {organ_class.capitalize()} class {additional_title_context}', y=1, fontsize='xx-large', verticalalignment='center', horizontalalignment='center')  # Set the title of the whole plot
    fig.subplots_adjust(top=0.88)  # Adjust the plot to make room for the title

    # << Saving the Figure >>

    if save: 
        number = max([0] + [get_id_from_path(fn, False) for fn in os.listdir('metrics/') if fn.startswith(f'metrics{organ_class}')])

        file_name = lambda num: f'metrics{organ_class}_{f"{num}_"if num is not None else ""}{"separated" if separate else "combined"}_{"_".join(additional_title_context.split(" "))}.png'

        try:
            # Move the file
            os.makedirs('metrics/old', exist_ok=True)
            plt.savefig(f'metrics/{file_name(number + 1)}', bbox_inches='tight')
            shutil.move(f'metrics/{file_name(number)}', f'metrics/old/')
        except FileNotFoundError as e:
            print(f'WARNING: {e}')
    
    else:
        plt.show()

    


# # Plotting Section

# In[7]:


metrics = {}


# In[10]:


save = True
separate = False
table = True

for c in [
             os.environ.get('Anorectum'),
            #  os.environ.get('Bladder'),
            #  os.environ.get('CTVn'),
            #  os.environ.get('CTVp'),
            #  os.environ.get('Parametrium'),
            #  os.environ.get('Uterus'),
            #  os.environ.get('Vagina'),
             ]:
    id = int(c.split('_')[0][len('Dataset'):])
    print('Fetching metrics for ', c, ' id ', id)
    metrics[c] = fetch_metric_for_class(id)
    plot_metrics(metrics[c], c, save=save, separate=separate, table=table)

anorectum_metrics = None if os.environ.get('Anorectum') not in metrics.keys() else metrics[os.environ.get('Anorectum')]
bladder_metrics = None if os.environ.get('Bladder') not in metrics.keys() else metrics[os.environ.get('Bladder')]
ctvn_metrics = None if os.environ.get('CTVn') not in metrics.keys() else metrics[os.environ.get('CTVn')]
ctvp_metrics = None if os.environ.get('CTVp') not in metrics.keys() else metrics[os.environ.get('CTVp')]
parametrium_metrics = None if os.environ.get('Parametrium') not in metrics.keys() else metrics[os.environ.get('Parametrium')]
uterus_metrics = None if os.environ.get('Uterus') not in metrics.keys() else metrics[os.environ.get('Uterus')]
vagina_metrics = None if os.environ.get('Vagina') not in metrics.keys() else metrics[os.environ.get('Vagina')]

# anorectum_metrics_ax0 = fetch_metric_for_class(1, axis=0)
# bladder_metrics_ax0 = fetch_metric_for_class(2, axis=0)
# ctvn_metrics_ax0 = fetch_metric_for_class(3, axis=0)  
# ctvp_metrics_ax0 = fetch_metric_for_class(4, axis=0)
# parametrium_metrics_ax0 = fetch_metric_for_class(5, axis=0)

# anorectum_metrics_ax1 = fetch_metric_for_class(1, axis=1)
# bladder_metrics_ax1 = fetch_metric_for_class(2, axis=1)
# ctvn_metrics_ax1 = fetch_metric_for_class(3, axis=1)
# ctvp_metrics_ax1 = fetch_metric_for_class(4, axis=1)
# parametrium_metrics_ax1 = fetch_metric_for_class(5, axis=1)

# anorectum_metrics_ax2 = fetch_metric_for_class(1, axis=2)
# bladder_metrics_ax2 = fetch_metric_for_class(2, axis=2)
# ctvn_metrics_ax2 = fetch_metric_for_class(3, axis=2)
# ctvp_metrics_ax2 = fetch_metric_for_class(4, axis=2)
# parametrium_metrics_ax2 = fetch_metric_for_class(5, axis=2)


# In[9]:


save = True
separate = False
table = True

for anatomy, metrics_dict in metrics.items():
    plot_metrics(metrics_dict, anatomy, save=save, separate=separate, table=table)

# plot_metrics(anorectum_metrics, 'anorectum', save=save, separate=separate, table=table)
# plot_metrics(bladder_metrics, 'bladder', save=save, separate=separate, table=table, showfliers=False)
# plot_metrics(ctvn_metrics, 'ctvn', save=save, separate=separate, table=table)
# plot_metrics(ctvp_metrics, 'ctvp', save=save, separate=separate, table=table)
# plot_metrics(parametrium_metrics, 'parametrium', save=save, separate=separate, table=table)
# plot_metrics(uterus_metrics, 'uterus', save=save, separate=separate, table=table)
# plot_metrics(vagina_metrics, 'vagina', save=save, separate=separate, table=table)

# plot_metrics(anorectum_metrics_ax0, 'anorectum', save=save, separate=separate, table=table, additional_title_context='(Axis 0)')
# plot_metrics(bladder_metrics_ax0, 'bladder', save=save, separate=separate, table=table, additional_title_context='(Axis 0)')
# plot_metrics(ctvn_metrics_ax0, 'ctvn', save=save, separate=separate, table=table, additional_title_context='(Axis 0)')
# plot_metrics(ctvp_metrics_ax0, 'ctvp', save=save, separate=separate, table=table, additional_title_context='(Axis 0)')
# plot_metrics(parametrium_metrics_ax0, 'parametrium', save=save, separate=separate, table=table, additional_title_context='(Axis 0)')

# plot_metrics(anorectum_metrics_ax1, 'anorectum', save=save, separate=separate, table=table, additional_title_context='(Axis 1)')
# plot_metrics(bladder_metrics_ax1, 'bladder', save=save, separate=separate, table=table, additional_title_context='(Axis 1)')
# plot_metrics(ctvn_metrics_ax1, 'ctvn', save=save, separate=separate, table=table, additional_title_context='(Axis 1)')
# plot_metrics(ctvp_metrics_ax1, 'ctvp', save=save, separate=separate, table=table, additional_title_context='(Axis 1)')
# plot_metrics(parametrium_metrics_ax1, 'parametrium', save=save, separate=separate, table=table, additional_title_context='(Axis 1)')

# plot_metrics(anorectum_metrics_ax2, 'anorectum', save=save, separate=separate, table=table, additional_title_context='(Axis 2)')
# plot_metrics(bladder_metrics_ax2, 'bladder', save=save, separate=separate, table=table, additional_title_context='(Axis 2)')
# plot_metrics(ctvn_metrics_ax2, 'ctvn', save=save, separate=separate, table=table, additional_title_context='(Axis 2)')
# plot_metrics(ctvp_metrics_ax2, 'ctvp', save=save, separate=separate, table=table, additional_title_context='(Axis 2)')
# plot_metrics(parametrium_metrics_ax2, 'parametrium', save=save, separate=separate, table=table, additional_title_context='(Axis 2)')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


import os, sys
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..', '..'))
if not dir1 in sys.path: sys.path.append(dir1)

from utils.environment import setup_data_vars
setup_data_vars()


# In[ ]:


destination = os.path.join(os.environ.get('nnUNet_raw'), os.environ.get('TotalBinary'))
assert os.path.exists(destination), f"Destination folder {destination} does not exist"

gt_path_for_anatomy = lambda x: os.path.join(os.environ.get('nnUNet_raw'), os.environ.get(x), os.environ.get('data_trainingLabels'))
gt_path_for_each_anatomy = dict([(os.environ.get(x), gt_path_for_anatomy(x)) for x in ['Anorectum','Bladder','CTVn','CTVp','Parametrium','Uterus','Vagina']])
assert all([os.path.exists(x) for x in gt_path_for_each_anatomy.values()])


# ## Establish mapping from anatomy combinations to segmentation regions

# In[ ]:


from itertools import chain, combinations
import json

def powerset(iterable):
    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

n_labels = 7

labelscomb = list(powerset(range(1,n_labels + 1)))
multilabels = range(len(labelscomb))
labelscomb_to_multilabel = dict(zip(labelscomb, multilabels))

# So `labelscomb_to_multilabel` is basically a look-up table for converting a combination
# of labels to the huge integer value that will be used in the nnU-Net nifti label file.
# labelscomb_to_multilabel


# In[ ]:


# Writing labels for dataset.json
# labels = {'background': 0}
# labels = {**labels, **{str(biomarker_idx): [] for biomarker_idx in range(1,n_labels+1)}}
labels = {str(biomarker_idx): [] for biomarker_idx in range(1,n_labels+1)}

for biomarker_ids, multilabel in labelscomb_to_multilabel.items():
    for biomarker_idx in biomarker_ids:
        labels[str(biomarker_idx)].append(multilabel)


# In[ ]:


# rename the entries in the dictionary to the appropriate anatomy names
anatomies = ['anorectum','bladder','ctvn','ctvp','parametrium','uterus','vagina']

# Create a mapping dictionary
mapping_dict = {key: anatomies[int(key) - 1] for key in labels.keys()}

# Replace keys with corresponding positions in the array
labels = {mapping_dict[key]: value for key, value in labels.items()}
labels = {**{'background': 0}, **labels}


# In[ ]:


# Save to json file
dataset_dict = { 
    "channel_names": {  # formerly modalities
        "0": "CT",
    }, 
    "labels": labels, 
    "numTraining": 100, 
    "file_ending": ".nii.gz",
    "regions_class_order": list(range(1, n_labels + 1)),
}

with open(os.path.join(destination, 'dataset.json'), 'w') as f:
    f.write(json.dumps(dataset_dict, indent=4))


# ## Function for converting the 7 gt segmentations into one with region ids

# In[ ]:


import os
import numpy as np
import SimpleITK as sitk

def operation(x):
        labels_in_voxel = tuple(np.argwhere(x)[:, 0] + 1)
        return labelscomb_to_multilabel[labels_in_voxel]

def image_to_nnunet_multilabel(input_volume):
    # Note `input_volume` should be one-hot encoded, so input_volume.shape == (n_labels, D, H, W)
    input_volume_per_channel = input_volume.reshape(n_labels, -1).T # (n_labels, D x H x W)
    input_volume_translated = np.apply_along_axis(operation, 1, input_volume_per_channel)
    correct_shape = input_volume_translated.reshape(input_volume.shape[1:]) # (D x H x W)
    
    return correct_shape

def combine_gt(id: int):
    assert 0 <= id <= 100, 'assumed that there are only 100 ids'

    # read in each anatomy ground truth
    sample_name = f'zzAMLART_{id:03d}.nii.gz'

    gt_per_anatomy = dict([(k, sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(v, sample_name)))) for k, v in
                           gt_path_for_each_anatomy.items()])
    assert all([x.shape == gt_per_anatomy[os.environ.get('Anorectum')].shape for _, x in gt_per_anatomy.items()]), \
        'ground truths contain at least one element that isn\'t the same size!'

    # stack all the ground truths
    gt = np.stack([v for _, v in gt_per_anatomy.items()])  # (7, D, H, W)

    return gt


# In[ ]:


# quick test

# create an array of values between 0 and 1
gt_slice = np.random.rand(20, 40, 45)
# threshold these values to 0 if below 0.5 or 1 otherwise
gt_slice = np.where(gt_slice < 0.5, 0, 1)
# repeat this slice across 7 dimensions
gt = np.repeat(gt_slice[np.newaxis, :, :, :], 7, axis=0)
assert np.array_equal(gt[0], gt_slice)

output = image_to_nnunet_multilabel(gt)
assert gt_slice.shape == output.shape
assert np.array_equal(gt_slice * 127, output)


# ## Setup script for translating these ground truths

# In[ ]:


from tqdm import tqdm

for i in tqdm(range(1, 101)):
    output_file_name = f'zzAMLART_{i:03d}.nii.gz'
    if os.path.exists(os.path.join(destination, 'labelsTr', output_file_name)):
        continue

    # combine the ground truths for each anatomy
    gt = combine_gt(i)
    output = image_to_nnunet_multilabel(gt)

    # save the output as a .nii.gz file in the destination folder
    output = sitk.GetImageFromArray(output)
    output.CopyInformation(sitk.ReadImage(os.path.join(gt_path_for_each_anatomy[os.environ.get('Anorectum')], f'zzAMLART_{i:03d}.nii.gz')))
    sitk.WriteImage(output, os.path.join(destination, 'labelsTr', output_file_name))


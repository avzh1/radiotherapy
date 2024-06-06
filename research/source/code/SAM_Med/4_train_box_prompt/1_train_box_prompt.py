#!/usr/bin/env python
# coding: utf-8

# # Fine tune a checkpoint of MedSAM on point prompted Data

# Given a path to a MedSAM checkpoint, we want to fine tune it on pre-processed data
# (subject to modifications specified by the paper and the transformation script). This will
# be done initially on an anatomy-specific level.

# ## Argparse Setup

# In[ ]:


# Imports
import os
import sys
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader

# Add the setup_data_vars function as we will need it to find the directory for the training data.
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..', '..'))
if not dir1 in sys.path: sys.path.append(dir1)

from utils.environment import setup_data_vars
setup_data_vars()

# Add utility classes
dir2 = os.path.abspath(os.path.join(os.path.abspath(''), '..', '0_utils'))
if not dir2 in sys.path: sys.path.append(dir2)

from dataset import SAM_Dataset
from view_MedSAM_batch import display_batch
from medsam_model import MedSAM
from checkpoint_handler import CheckpointHandler
from dataload_handler import DataLoaderHandler
from logging_handler import LoggingHandler
from trainer import MedSAMTrainer


# In[ ]:


parser = argparse.ArgumentParser()

# Inspired by orginal code from the MedSAM/extensions/point_prompt

# 1. Add the anatomy on which we will fine-tune
parser.add_argument(
    '--anatomy',
    type=str,
    help='Anatomy on which to fine-tune the model. Note: this is case sensitive, please capitalize the first letter and accronyms such as CTVn or CTVp.',
    required=True
)

# 1.2 Add the model training type
parser.add_argument(
    '--model_training',
    type=str,
    help='Determines the type of model that is being trained. For example, if the model uses only points, the argument should be "point". If the model uses both points and bounding boxes, the argument should be "point_bbox".',
    required=True,
)

# 2. Path to the MedSAM checkpoint
parser.add_argument(
    '--checkpoint',
    type=str,
    help='Path to the checkpoint of the model to fine-tune',
    default=os.path.join(os.environ['PROJECT_DIR'], 'models', 'MedSAM', 'work_dir', 'MedSAM', 'medsam_vit_b.pth'),
    required=False
)

# 3. Path where we will be saving the checkpoints of the fine-tuned model
parser.add_argument(
    '--save_dir',
    type=str,
    help='Directory where the fine-tuned model will be saved',
    required=False,
    default=os.environ.get('MedSAM_finetuned')
)

# 4. Add the source directory for the data
parser.add_argument(
    '--img_dir',
    type=str,
    help='Directory containing the images for the slices of the anatomy',
    required=False,
)

# 5. Add the source directory for the gts
parser.add_argument(
    '--gt_dir',
    type=str,
    help='Directory containing the ground truth masks for the slices of the anatomy',
    required=False
)

# 6. Number of epochs for the fine-tuning
parser.add_argument(
    '--epochs',
    type=int,
    help='Number of epochs for the fine-tuning',
    required=False,
    default=300
)

# 7. Batch size for the fine-tuning
parser.add_argument(
    '--batch_size',
    type=int,
    help='Batch size for the fine-tuning',
    required=False,
    default=8
)

parser.add_argument(
    '--batches_per_epoch',
    type=int,
    help='Number of batches per epoch',
    required=False,
)

# 8. Learning rate for the fine-tuning
parser.add_argument(
    '--lr',
    type=float,
    help='Learning rate for the fine-tuning',
    required=False,
    default=0.00005
)

# 9. Number of workers for the data loader
parser.add_argument(
    '--num_workers',
    type=int,
    help='Number of workers for the data loader',
    required=False,
    default=16
)

parser.add_argument(
    '--weight_decay',
    type=float,
    help='Weight decay for the optimizer',
    required=False,
    default=0.01
)

# 11. Resume checkpoint
parser.add_argument(
    '--resume',
    type=bool,
    help='Whether to resume training using the latest checkpoint in the save_dir',
    required=False,
    default=True
)

parser.add_argument(
    '--lowres',
    type=bool,
    help='A flag for setting the source of the data. For now, if the flag is set to True, the data will be loaded from the lowres directory. Otherwise, we load it from the pure pre-processed directory.',
    required=False,
    default=True
)


# In[ ]:


print('Preparing to parse args!')


# In[ ]:


# Suppose for now we get the following set of required arguments:
# args = parser.parse_args([
#     '--anatomy', 'CTVn',
#     '--model_training', 'boxed_lowres_test',
# #     # '--checkpoint', os.path.join(os.environ['PROJECT_DIR'], 'models', 'MedSAM', 'work_dir', 'MedSAM', 'medsam_vit_b.pth'),
# #     # '--save_dir', os.path.join(os.environ['MedSAM_finetuned']),
#     '--epochs', '100',
#     '--batch_size', '2',
#     '--batches_per_epoch', '2', 
#     '--lowres', 'True',
# ])

args = parser.parse_args()


# ## Set up the vars

# In[ ]:


anatomy = args.anatomy
checkpoint_path = args.checkpoint
save_dir = args.save_dir
img_dir = args.img_dir
gt_dir = args.gt_dir
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
num_workers = args.num_workers
weight_decay = args.weight_decay
resume = args.resume
batches_per_epoch = args.batches_per_epoch
model_training = args.model_training
lowres = args.lowres

if img_dir is None:
    img_dir = os.environ['MedSAM_preprocessed_lowres'] if lowres else os.environ['MedSAM_preprocessed']
    img_dir = os.path.join(img_dir, 'imgs')
if gt_dir is None:
    gt_dir = os.environ['MedSAM_preprocessed_lowres'] if lowres else os.environ['MedSAM_preprocessed']
    gt_dir = os.path.join(gt_dir, 'gts', anatomy)

assert not lowres or ('lowres' in img_dir and 'lowres' in gt_dir) , 'Please make sure that the lowres flag is set correctly!'

save_dir = os.path.join(save_dir, model_training, anatomy)

# print all the args
print('Arguments:')
print(f'anatomy {anatomy}')
print(f'checkpoint {checkpoint_path}')
print(f'save_dir {save_dir}')
print(f'img_dir {img_dir}')
print(f'gt_dir {gt_dir}')
print(f'epochs {epochs}')
print(f'batch_size {batch_size}')
print(f'lr {lr}')
print(f'num_workers {num_workers}')
print(f'weight_decay {weight_decay}')
print(f'resume {resume}')
print(f'batches_per_epoch {batches_per_epoch}')
print(f'model_training {model_training}')
print(f'lowres {lowres}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


seed = 42

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# ## Set up Dataset class

# In[ ]:


# # quick test to see if the points are being generated correctly and transformations are also ok
# experimental_datset = SAM_Dataset(img_dir, gt_dir, [i for i in range(1, 2)], data_aug=False, max_box_points=1)
# dataloader = DataLoader(experimental_datset, batch_size=25, shuffle=True)
# # Get a batch of examples
# batch = next(iter(dataloader))
# # Display the batch
# display_batch(batch, show_points = False)


# ## Main Training Loop

# In[ ]:


loggingHandler = LoggingHandler(save_dir)
dataloaderHandler = DataLoaderHandler(save_dir, img_dir, gt_dir, batch_size, num_workers, True, 0, 5, 1)
checkpointHandler = CheckpointHandler(save_dir, checkpoint_path, device, lr=lr, weight_decay=weight_decay)


# In[9]:


myTrainer = MedSAMTrainer(
    loggingHandler, 
    dataloaderHandler, 
    checkpointHandler,
    device,
    epochs=epochs,
    resume=resume,
    batches_per_epoch=batches_per_epoch,
    use_boxes = True,
    use_positive_points = False,
)
myTrainer.run_training()


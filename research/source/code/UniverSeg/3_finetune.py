#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..'))
if not dir1 in sys.path: sys.path.append(dir1)
from utils.environment import setup_data_vars
setup_data_vars()


# In[2]:


# set random seed
import numpy as np
import random
import torch

torch.manual_seed(2147483648)
random.seed(2147483648)
np.random.seed(2147483648)


# In[ ]:


support_size = 80
batch_size = 3


# In[ ]:


print(f'Using support size {support_size} and batch size {batch_size}')


# In[ ]:


from torch.utils.data import Dataset
import random
import math
import numpy as np
import cv2
import torch

class UniverSegDataSet(Dataset):
    def __init__(self, support_size, training=True):

        self.medsam_gts = os.path.join(os.environ.get('MedSAM_preprocessed_lowres'), 'gts')
        self.medsam_imgs = os.path.join(os.environ.get('MedSAM_preprocessed_lowres'), 'imgs')

        self.support_size = support_size

        self.training = training
        self.anatomies = dict([(f, dict()) for f in os.listdir(self.medsam_gts) if f != 'TotalBinary'])

        min_samples = float('inf')

        for anatomy in self.anatomies.keys():
            for axis in [f for f in os.listdir(os.path.join(self.medsam_gts, anatomy)) if 'axis' in f]:
                subdir = os.path.join(anatomy, axis)

                samples = [f for f in os.listdir(os.path.join(self.medsam_gts, subdir)) if f.endswith('.npy')]
                random.shuffle(samples)

                if len(samples) < min_samples:
                    min_samples = len(samples)

                self.anatomies[anatomy][axis] = samples

        self.anatomy_keys = sorted(list(self.anatomies.keys()))
        self.list_of_axis = sorted(list(self.anatomies[self.anatomy_keys[0]].keys()))

        self.setup_sampler()

        self.training_length = min_samples * len(self.anatomy_keys) * len(self.list_of_axis)

    def set_training(self):
        self.training = True
        self.setup_sampler()

    def set_validation(self):
        self.training = False
        self.setup_sampler()

    def __len__(self):
        return self.training_length
    
    def setup_sampler(self):
        if self.training:
            self._sample_to_consider = lambda idx: (idx // (len(self.list_of_axis) * len(self.anatomy_keys)))
        else:
            self._sample_to_consider = lambda idx: -(idx // (len(self.list_of_axis) * len(self.anatomy_keys))) - 1

    def _read_image_and_gt(self, img_id, img_slice, anatomy, axis):
        img = np.load(os.path.join(self.medsam_imgs, f'axis{axis}', f'CT_zzAMLART_{img_id:03d}-{img_slice:03d}.npy'))
        gt = np.load(os.path.join(self.medsam_gts, anatomy, f'axis{axis}', f'CT_{anatomy}_zzAMLART_{img_id:03d}-{img_slice:03d}.npy'))
        return img, gt

    def _anatomy_to_consider(self, idx):
        return self.anatomy_keys[(idx // len(self.list_of_axis)) % len(self.anatomy_keys)]
    
    def _axis_to_consider(self, idx):
        return self.list_of_axis[idx % len(self.list_of_axis)]
    
    def __getitem__(self, idx):
        anatomy_to_consider = self._anatomy_to_consider(idx)
        axis_to_consider = self._axis_to_consider(idx)
        sample_to_consider = self._sample_to_consider(idx)

        ith_example = self.anatomies[anatomy_to_consider][axis_to_consider][sample_to_consider]

        get_id_from_img = lambda img_name: int(img_name.split('_')[3].split('-')[0])
        get_slice_from_img = lambda img_name: int(img_name.split('_')[3].split('-')[1].split('.')[0])

        ith_id = get_id_from_img(ith_example)
        ith_slice = get_slice_from_img(ith_example)

        # get a support set that doesn't contain the same id as the ith example
        support_set = random.sample([f for f in self.anatomies[anatomy_to_consider][axis_to_consider] if get_id_from_img(f) != ith_id], self.support_size)

        # read in the images and gts for the ith example and the support set and resize them appropriately
        ith_img, ith_gt = self._read_image_and_gt(ith_id, ith_slice, anatomy_to_consider, int(axis_to_consider[-1]))

        support_imgs = []
        support_gts = []

        for support_example in support_set:
            support_img, support_gt = self._read_image_and_gt(get_id_from_img(support_example), get_slice_from_img(support_example), anatomy_to_consider, int(axis_to_consider[-1]))

            support_imgs.append(support_img)
            support_gts.append(support_gt)

        # resize the images and gts to 128x128 we need for universeg

        ith_img = cv2.resize(ith_img, (128, 128), interpolation=cv2.INTER_LINEAR)
        ith_gt = cv2.resize(ith_gt, (128, 128), interpolation=cv2.INTER_NEAREST)

        support_imgs = [cv2.resize(support_img, (128, 128), interpolation=cv2.INTER_LINEAR) for support_img in support_imgs]
        support_gts = [cv2.resize(support_gt, (128, 128), interpolation=cv2.INTER_NEAREST) for support_gt in support_gts]

        # convert to torch tensors

        ith_img = torch.from_numpy(ith_img).float().unsqueeze(0)
        ith_gt = torch.from_numpy(ith_gt).float().unsqueeze(0)

        support_imgs = [torch.from_numpy(support_img).float().unsqueeze(0) for support_img in support_imgs]
        support_gts = [torch.from_numpy(support_gt).float().unsqueeze(0) for support_gt in support_gts]

        # stack the support images and gts
        support_imgs = torch.stack(support_imgs) # (S x 128 x 128)
        support_gts = torch.stack(support_gts) # (S x 128 x 128)

        assert support_imgs.shape == (self.support_size, 1, 128, 128), support_imgs.shape
        assert support_gts.shape == (self.support_size, 1, 128, 128), support_gts.shape
        assert ith_img.shape == (1, 128, 128), ith_img.shape
        assert ith_gt.shape == (1, 128, 128), ith_gt.shape

        return {
            'query_name': ith_example,
            'query_anatomy_axis': f'{anatomy_to_consider}_{axis_to_consider}', 
            'query': ith_img,
            'query_gt': ith_gt,
            'support_name': support_set,
            'support_imgs': support_imgs,
            'support_gts': support_gts
        }


# In[ ]:


from torch.utils.data import DataLoader

class UniversegDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super(UniversegDataLoader, self).__init__(dataset, **kwargs)

    def set_training(self):
        self.dataset.set_training()

    def set_validation(self):
        self.dataset.set_validation()


# In[ ]:


dataset = UniverSegDataSet(20)


# In[ ]:


# fetch a batch
dataloader = UniversegDataLoader(dataset, batch_size=1, shuffle=True)

dataloader.set_validation()

for batch in dataloader:
    print(batch['query'].shape)
    print(batch['query_gt'].shape)
    print(batch['support_imgs'].shape)
    print(batch['support_gts'].shape)
    print('query name:', batch['query_name'])
    print('support_name', batch['support_name'])
    break


# In[ ]:


from universeg import universeg


# In[ ]:


from platipy.imaging.label.comparison import compute_metric_total_apl, compute_surface_dsc, compute_metric_hd
import SimpleITK as sitk


# In[ ]:


import pandas as pd
import monai
import torch.nn as nn
from tqdm import tqdm
from time import time
import torch.optim as optim
import os

df = pd.DataFrame(columns=['epoch', 'train_or_val', 'batch', 'anatomy_axis', 'loss', 'time'])

save_dir = os.path.join('results_finetuned', 'finetuning')
os.makedirs(save_dir, exist_ok=True)


# In[ ]:


# load in the existing training csv

if os.path.exists(os.path.join(save_dir, 'training.csv')):
    df = pd.read_csv(os.path.join(save_dir, 'training.csv'))


# In[ ]:


df


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up model
model = universeg(pretrained=True)
model = model.to(device)

# set up optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01
)

# load the checkpoint
checkpoint = torch.load('results_finetuned/finetuning/model_checkpoint_latest.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# monitor best loss
best_loss = float('inf')
best_loss = checkpoint['best_loss']

# set up loss function
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

dataset = UniverSegDataSet(support_size=support_size, training=True)
dataloader = UniversegDataLoader(dataset, batch_size=batch_size, shuffle=True)

# set up training loop
for epoch in range(checkpoint['epoch'] + 1, 10):
    model.train()
    dataloader.set_training()
    for i, batch in tqdm(enumerate(dataloader), desc='Training', total=len(dataloader)):
        new_record = pd.DataFrame([
            {'epoch': epoch, 'train_or_val': 'train', 'batch': i, 'anatomy_axis': batch['query_anatomy_axis'], 'loss': 0, 'time': 0}
        ])

        # forward inference 
        start_time = time()
        
        names = batch['query_name']
        image = batch['query'].to(device)
        label = batch['query_gt']
        support_images = batch['support_imgs'].to(device)
        support_labels = batch['support_gts'].to(device)
        
        image = image.to(device)
        label = label.to(device)
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)

        optimizer.zero_grad()
        prediction_logits = model(image, support_images, support_labels)

        loss = seg_loss(prediction_logits, label) + ce_loss(prediction_logits, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        end_time = time()

        new_record['loss'] = loss.item()
        new_record['time'] = end_time - start_time

        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(f'{save_dir}/training.csv', index=False)

    epoch_loss_reduced = df[df['epoch'] == epoch]['loss'].mean()
    
    # save a checkpoint of the model if the loss is lower than the previous best
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch, 
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss
    }

    if epoch_loss_reduced < best_loss:
        best_loss = epoch_loss_reduced
        checkpoint['best_loss'] = best_loss
        torch.save(checkpoint, f'{save_dir}/model_checkpoint_best.pth')
    torch.save(checkpoint, f'{save_dir}/model_checkpoint_latest.pth')

    model.eval()
    dataloader.set_validation()

    validation_iterator = iter(dataloader)
    how_many_batches_validation = 100 # len(dataloader)
    for i in tqdm(range(how_many_batches_validation), desc='Validation', total=how_many_batches_validation):
        batch = next(validation_iterator)
        
        new_record = pd.DataFrame([
            {'epoch': epoch, 'train_or_val': 'val', 'batch': i, 'anatomy_axis': batch['query_anatomy_axis'], 'loss': 0, 'time': 0}
        ])

        start_time = time()

        # calculate the validation loss
        names = batch['query_name']
        image = batch['query'].to(device)
        label = batch['query_gt']
        support_images = batch['support_imgs'].to(device)
        support_labels = batch['support_gts'].to(device)

        image = image.to(device)
        label = label.to(device)
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)

        with torch.no_grad():
            prediction_logits = model(image, support_images, support_labels)
            loss = seg_loss(prediction_logits, label) + ce_loss(prediction_logits, label)

        new_record['loss'] = loss.item()

        end_time = time()

        new_record['time'] = end_time - start_time

        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(f'{save_dir}/training.csv', index=False)
    


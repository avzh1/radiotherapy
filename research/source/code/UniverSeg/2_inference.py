#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
dir1 = os.path.abspath(os.path.join(os.path.abspath(''), '..'))
if not dir1 in sys.path: sys.path.append(dir1)
from utils.environment import setup_data_vars
setup_data_vars()


# In[2]:


import argparse

# define argparser for anatomy, and axis

parser = argparse.ArgumentParser(description='Inference for UniverSeg')

parser.add_argument('anatomy', type=str, help='The anatomy to infer on')
parser.add_argument('axis', type=str, help='The axis to infer on (0,1,2)')

args = parser.parse_args(
    # ['Anorectum', '0']
)


# In[3]:


anatomy = args.anatomy # 'Anorectum'
support_size = 80
axis = int(args.axis) # 0
batch_size = 3


# In[4]:


print(f'Infering on {anatomy} along axis {axis}')
print(f'Using support size {support_size} and batch size {batch_size}')


# In[5]:


from torch.utils.data import Dataset
import random
import numpy as np
import cv2
import torch

class UniverSegDataSet(Dataset):
    def __init__(self, support_size, anatomy, axis):

        self.medsam_gts = os.path.join(os.environ.get('MedSAM_preprocessed_lowres'), 'gts', anatomy, f'axis{axis}')
        self.medsam_imgs = os.path.join(os.environ.get('MedSAM_preprocessed_lowres'), 'imgs', f'axis{axis}')

        self.support_size = support_size
        self.anatomy = anatomy
        self.axis = axis

        self.gts_samples = [f for f in os.listdir(self.medsam_gts) if f.endswith('.npy')]
        random.shuffle(self.gts_samples)

    def __len__(self):
        # return len(self.gts_samples)
        return 1000
    
    def _read_image_and_gt(self, img_id, img_slice):
        img = np.load(os.path.join(self.medsam_imgs, f'CT_zzAMLART_{img_id:03d}-{img_slice:03d}.npy'))
        gt = np.load(os.path.join(self.medsam_gts, f'CT_{self.anatomy}_zzAMLART_{img_id:03d}-{img_slice:03d}.npy'))
        return img, gt
    
    def __getitem__(self, idx):
        ith_example = self.gts_samples[idx]

        # get a support set that doesn't contain the same id as the ith example
        get_id_from_img = lambda img_name: int(img_name.split('_')[3].split('-')[0])
        get_slice_from_img = lambda img_name: int(img_name.split('_')[3].split('-')[1].split('.')[0])

        ith_id = get_id_from_img(ith_example)
        ith_slice = get_slice_from_img(ith_example)

        support_set = random.sample([f for f in self.gts_samples if get_id_from_img(f) != ith_id], self.support_size)

        # read in the images and gts for the ith example and the support set and resize them appropriately
        ith_img, ith_gt = self._read_image_and_gt(ith_id, ith_slice)

        support_imgs = []
        support_gts = []

        for support_example in support_set:
            support_img, support_gt = self._read_image_and_gt(get_id_from_img(support_example), get_slice_from_img(support_example))

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
            'query': ith_img,
            'query_gt': ith_gt,
            'support_imgs': support_imgs,
            'support_gts': support_gts
        }


# In[6]:


from torch.utils.data import DataLoader


my_dataset = UniverSegDataSet(support_size=support_size, anatomy=anatomy, axis=axis)
my_dataloder = DataLoader(my_dataset, batch_size=1, shuffle=True)

# get a batch of data

for i, batch in enumerate(my_dataloder):
    print(batch['query'].shape)
    print(batch['query_gt'].shape)
    print(batch['support_imgs'].shape)
    print(batch['support_gts'].shape)
    break


# In[7]:


from universeg import universeg

dataset = UniverSegDataSet(support_size=support_size, anatomy=anatomy, axis=axis)
dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)


# In[8]:


from platipy.imaging.label.comparison import compute_metric_total_apl, compute_surface_dsc, compute_metric_hd
import SimpleITK as sitk


# In[9]:


import pandas as pd
from tqdm import tqdm
import os

df = pd.DataFrame(columns=['name', 'dice', 'volume_similarity', 'apl', 'surface_distance', 'hausdorff_distance'])

save_dir = os.path.join('results_finetuned', anatomy, f'axis{axis}')
os.makedirs(save_dir, exist_ok=True)


# In[11]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

# Run the inference
model = universeg(pretrained=True)
model = model.to(device)

# load in parameters from the trained model
checkpoint = torch.load('results_finetuned/finetuning/model_checkpoint_best.pth')

model.load_state_dict(checkpoint['model'])

for i, batch in tqdm(enumerate(dataloader)):
    try:
        names = batch['query_name']
        image = batch['query'].to(device)
        label = batch['query_gt']
        support_images = batch['support_imgs'].to(device)
        support_labels = batch['support_gts'].to(device)
        
        image = image.to(device)
        # label = label.to(device)
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)

        prediction_logits = model(image, support_images, support_labels)
        prediction_soft = torch.sigmoid(prediction_logits)

        # threshold probabilities to get hard predictions
        prediction_hard = (prediction_soft > 0.5).float()

        # delete the variables from the gpu
        del image, support_images, support_labels, prediction_logits, prediction_soft

        # for each batch of predictions, compute the metrics
        for j in range(prediction_hard.shape[0]):
            # compute the metrics
            prediction = prediction_hard[j].cpu().detach().numpy()
            prediction = sitk.GetImageFromArray(prediction)
            label_sitk = sitk.GetImageFromArray(label[j])

            prediction = sitk.Cast(prediction, sitk.sitkUInt8)
            label_sitk = sitk.Cast(label_sitk, sitk.sitkUInt8)

            overlap_measures_filter.Execute(label_sitk, prediction)

            dice = overlap_measures_filter.GetDiceCoefficient()
            hd = compute_metric_hd(label_sitk, prediction)
            volume_similarity = overlap_measures_filter.GetVolumeSimilarity()
            surface_dsc = compute_surface_dsc(label_sitk, prediction)
            apl = compute_metric_total_apl(label_sitk, prediction)

            new_record = pd.DataFrame([
                {'name': names[j],  'dice': dice, 'volume_similarity': volume_similarity, 'apl': apl, 'surface_distance': surface_dsc, 'hausdorff_distance': hd}
            ])

            df = pd.concat([df, new_record], ignore_index=True)

        df.to_csv(f'{save_dir}/validation.csv', index=False)
    except Exception as e:
        print(e)
        continue


# In[ ]:


df


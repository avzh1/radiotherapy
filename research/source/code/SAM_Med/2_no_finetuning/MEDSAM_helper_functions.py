import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, H, W, box_1024 = None, include_points = None, exclude_points = None):
    assert box_1024 is not None or include_points is not None or exclude_points is not None, "At least one of box_1024, include_points or exclude_points must be provided"

    box_torch = None

    if box_1024 is not None:
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

    points = None
    labels = None

    if include_points is not None:
        include_points = torch.as_tensor(include_points, dtype=torch.float, device=img_embed.device)
        if len(include_points.shape) == 2:
            include_points = include_points[None, :, :] # (B, P, 2)

        points = include_points
        labels = torch.ones((include_points.shape[0], include_points.shape[1]))

    if exclude_points is not None:
        exclude_points = torch.as_tensor(exclude_points, dtype=torch.float, device=img_embed.device)
        if len(exclude_points.shape) == 2:
            exclude_points = exclude_points[None, :, :] # (B, P, 2)

        _points = exclude_points
        _labels = torch.zeros((exclude_points.shape[0], exclude_points.shape[1]))

        if points is None:
            points = _points
            labels = _labels
            # labels = torch.zeros_like(exclude_points[:, :])
        else:
            points = torch.cat([points, _points], dim=1)
            labels = torch.cat([labels, _labels], dim=1)    

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None if points is None else (points, labels),
        boxes=box_torch,
        masks=None,
    )
    
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


import cv2

def get_bounding_boxes(resized_gt, padding=10):
    # Find contours in the segmentation map
    contours, _ = cv2.findContours(resized_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store bounding boxes
    bounding_boxes = []

    # Loop through contours to find bounding boxes
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([x - padding, y - padding, x + w + padding, y + h + padding])  # Format: (x_min, y_min, x_max, y_max)

    return np.array(bounding_boxes)

def visualise_bounding_box_with_prediction(array_image
                                           , array_gt_label
                                           , bounding_boxes=None
                                           , predicted_mask=None
                                           , exclude_points = None
                                           , include_points = None
                                           , show_boxes_of_predictions=False
                                           , save_title = None
                                           , sup_title = None):
    
    ncols = sum([2, int(bounding_boxes is not None or exclude_points is not None or include_points is not None), int(predicted_mask is not None)])

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes[0].imshow(array_image, cmap='gray')
    axes[0].set_title('Raw Image')

    alpha_mask_gt = np.where(array_gt_label > 0, 1, 0).astype(np.float32)
    axes[1].imshow(array_image, cmap='gray')
    axes[1].imshow(array_gt_label, alpha=alpha_mask_gt, cmap='viridis')
    axes[1].set_title('Ground Truth')

    currcol = 2

    if bounding_boxes is not None or exclude_points is not None or include_points is not None:
        axes[2].imshow(array_image, cmap='gray')
        if bounding_boxes is not None:
            assert len(bounding_boxes) >= 1
            for box in bounding_boxes:
                box = list(box)
                x0, y0 = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
                axes[2].add_patch(
                    plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
                )
        if exclude_points is not None:
            for point in exclude_points.squeeze():
                axes[2].scatter(point[0], point[1], c='red', s=10)
        
        if include_points is not None:
            for point in include_points.squeeze():
                axes[2].scatter(point[0], point[1], c='lawngreen', s=10)
        
        axes[2].set_title('Bounding Box From Segmentation')
        currcol += 1

    if predicted_mask is not None:

        alpha_mask_pred = np.where(predicted_mask > 0, 1, 0).astype(np.float32)
        axes[currcol].imshow(array_image, cmap='gray')
        axes[currcol].imshow(predicted_mask, alpha=alpha_mask_pred, cmap='viridis')
        axes[currcol].set_title('Predicted Mask')

        if show_boxes_of_predictions:
            bounding_boxes_of_predictions = get_bounding_boxes(predicted_mask)

            for box in bounding_boxes_of_predictions:
                box = list(box)
                x0, y0 = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
                axes[currcol].add_patch(
                    plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=2)
                )
                
    plt.tight_layout()
    plt.suptitle(sup_title, y=1.05)
    
    if not save_title is None:
        save_path = '/'.join(save_title.split('/')[:-1])
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(save_title)
        plt.close(fig)
    else:
        plt.show()

import SimpleITK as sitk

def dice_similarity(gt, pred):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    ypred_sitk = sitk.GetImageFromArray(pred)
    y_gt_sitk = sitk.GetImageFromArray(gt)

    overlap_measures_filter.Execute(y_gt_sitk, ypred_sitk)
    return overlap_measures_filter.GetDiceCoefficient()


import torch
import bisect 
from torch.utils.data import Dataset, DataLoader

class SAM_Dataset(Dataset):

    def __init__(self, axis, anatomy, box_limit = 3, box_padding = 10):
        self.axis = axis
        self.anatomy = anatomy

        self.gt_slice_dir = os.path.join(os.environ.get('MedSAM_preprocessed'), 'gts', anatomy, f'axis{axis}')
        self.img_slice_dir = os.path.join(os.environ.get('MedSAM_preprocessed'), 'imgs', f'axis{axis}')
        
        self.slices_per_sample = {}
        for image_slice in os.listdir(self.gt_slice_dir):
            gt_id, gt_slice = self.get_id_and_slice_from_gt_path(image_slice)

            if gt_id not in self.slices_per_sample.keys():
                self.slices_per_sample[gt_id] = [gt_slice] # (min, max)
            else:
                bisect.insort(self.slices_per_sample[gt_id], gt_slice)

        self.box_limit = box_limit
        self.box_padding = box_padding

    def __len__(self):
        return sum([len(vs) for vs in self.slices_per_sample.values()])

    def __getitem__(self, idx):
        assert 0 <= idx < self.__len__(), f"Index {idx} is out of range for dataset of size {self.__len__()}"

        cursor = idx

        for img_ids, img_slices in self.slices_per_sample.items():
            if cursor - len(img_slices) >= 0:
                cursor = cursor - len(img_slices)
                continue
            else:
                curr_slice = img_slices[cursor]
                break

        gt_slice_path = os.path.join(self.gt_slice_dir, self.gt_slice_format(img_ids, curr_slice))
        img_slice_path = os.path.join(self.img_slice_dir, self.img_slice_format(img_ids, curr_slice))

        # load in the image
        gt_array = np.load(gt_slice_path)
        img_array = np.load(img_slice_path)

        # Get Bounding Boxes. If there are more than self.box_limit boxes, only take the
        # first self.box_limit boxes. If there are less than self.box_limit, duplicate the
        # last box to fill the rest (without loss of generality)
        bounding_boxes = get_bounding_boxes(gt_array, padding=self.box_padding)
        if len(bounding_boxes) > self.box_limit:
            bounding_boxes = bounding_boxes[:self.box_limit]
        else:
            while len(bounding_boxes) < self.box_limit:
                bounding_boxes = np.vstack([bounding_boxes, bounding_boxes[-1]])

        img_array = torch.tensor(img_array).float().permute(2, 0, 1)

        # return img_array, gt_array, and bounding box pair
        return img_array, gt_array, bounding_boxes, img_ids, curr_slice

    def gt_slice_format(self, img_num: int, slice_num: int) -> str:
        return 'CT_' + self.anatomy + '_zzAMLART_' + str(img_num).zfill(3) + '-' + str(slice_num).zfill(3) + '.npy'

    def img_slice_format(self, img_num, slice_num):
        return 'CT_zzAMLART_' + str(img_num).zfill(3) + '-' + str(slice_num).zfill(3) + '.npy'
    
    def get_id_and_slice_from_gt_path(self, path: str):
        get_id = lambda x: int(x.split('_')[3].split('-')[0])
        get_slice = lambda x: int(x.split('_')[3].split('-')[1].split('.')[0])

        return get_id(path), get_slice(path)
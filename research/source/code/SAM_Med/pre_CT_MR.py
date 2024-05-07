# Originally, you can find the code at
# https://github.com/bowang-lab/MedSAM/blob/main/pre_CT_MR.py For the purpose of this
# project, the code was modified to fit the needs of the project. Specifically, this was
# abstracted into a callable function with checkpointing so that it may be called from
# within a jupyter notebook.

# %% Import Packages

import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

from skimage import transform
from tqdm import tqdm
import cc3d

# %% Define Function

def pre_CT_MR(nii_path: str
            , gt_path: str
            , npy_path: str
            , anatomy: str
            , modality = "CT"
            , img_name_suffix = "_0000.nii.gz"
            , gt_name_suffix = ".nii.gz"
            # https://radiopaedia.org/articles/windowing-ct
            , WINDOW_LEVEL = 40 # only for CT images
            , WINDOW_WIDTH = 400 # only for CT images
            , image_size = 1024
            , voxel_num_thre2d = 100
            , voxel_num_thre3d = 1000
            ):

    # << SETUP DESTINATION >>
    
    prefix = modality + "_" + anatomy + "_"

    nyp_path_imgs = os.path.join(os.environ.get('PROJECT_DIR'), 'MedSAM_preprocessed', 'npy', 'imgs')
    nyp_path_gts = os.path.join(os.environ.get('PROJECT_DIR'), 'MedSAM_preprocessed', 'npy', 'gts', prefix[:-1])
    os.makedirs(nyp_path_imgs, exist_ok=True) # Nii images
    os.makedirs(nyp_path_gts, exist_ok=True) # Ground truth images

    # << ITERATE OVER ALL IMAGES AND CONVERT >>

    # Get the list of images to process
    gt_names = sorted(os.listdir(gt_path))
    img_names = sorted(os.listdir(nii_path))

    # Get the processed images for images and ground truth
    gt_processed_names = sorted([f for f in os.listdir(nyp_path_gts) if f.endswith(gt_name_suffix)])
    img_processed_names = sorted([f for f in os.listdir(nyp_path_imgs) if f.endswith(img_name_suffix)])

    # Get list of remaining images assuming each image id is unique
    # (for now, just set it to the list of all images)
    remaining_gt_names = gt_names
    remaining_img_names = img_names

    # Process the Images. Assume that the ground truth will be written last, therefore, in
    # the worst case we recalculate only one image redundantly.
    for gt_name in tqdm(remaining_gt_names):
        image_name = gt_name.split(gt_name_suffix)[0] + img_name_suffix

        # Read in Image
        gt_sitk = sitk.ReadImage(os.path.join(gt_path, name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
        
        # find non-zero slices
        z_index, _, _ = np.where(gt_data_ori > 0)
        z_index = np.unique(z_index)

        if len(z_index) > 0:
            # crop the ground truth with non-zero slices
            gt_roi = gt_data_ori[z_index, :, :]
            
            # load image and preprocess
            img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
            image_data = sitk.GetArrayFromImage(img_sitk)
            
            # nii preprocess start
            if modality == "CT":
                lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
                upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
            else:
                raise ValueError(f"Modality `{modality}` not supported")
        
            image_data_pre = np.uint8(image_data_pre)
            img_roi = image_data_pre[z_index, :, :]
            np.savez_compressed(os.path.join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
            
            # save the each CT image as npy file
            for i in range(img_roi.shape[0]):
                img_i = img_roi[i, :, :]
                img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                resize_img_skimg = transform.resize(
                    img_3c,
                    (image_size, image_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )
                resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                    resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)
                gt_i = gt_roi[i, :, :]
                resize_gt_skimg = transform.resize(
                    gt_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_gt_skimg = np.uint8(resize_gt_skimg)
                assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
                np.save(
                    join(
                        npy_path,
                        "imgs",
                        prefix
                        + gt_name.split(gt_name_suffix)[0]
                        + "-"
                        + str(i).zfill(3)
                        + ".npy",
                    ),
                    resize_img_skimg_01,
                )
                np.save(
                    join(
                        npy_path,
                        "gts",
                        prefix
                        + gt_name.split(gt_name_suffix)[0]
                        + "-"
                        + str(i).zfill(3)
                        + ".npy",
                    ),
                    resize_gt_skimg,
                )

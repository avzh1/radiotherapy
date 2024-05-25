# %% [markdown]
# # Pre-processing the data

# %% [markdown]
# Example Running:
# ```
# python pre_CT_MR.py CT Anorectum /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/imagesTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset001_Anorectum/labelsTr /vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed --axis 0 --verbose
# ```

# %%
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

import argparse

import SimpleITK as sitk
import matplotlib.pyplot as plt

# %%
parser = argparse.ArgumentParser(description="Preprocess CT/MR images")
parser.add_argument('modality', type=str, help='modality of the images, CT or MR')
parser.add_argument('anatomy', type=str, help='anatomy of the images')
parser.add_argument('nii_path', type=str, help='path to the nii images')
parser.add_argument('gt_path', type=str, help='path to the ground truth')
parser.add_argument('npy_path', type=str, help='path to save the npy files')
parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='suffix of the image name')
parser.add_argument('--gt_name_suffix', type=str, default='.nii.gz', help='suffix of the ground truth name')
parser.add_argument('--image_size', type=int, default=1024, help='size of the images')
parser.add_argument('--voxel_num_thre2d', type=int, default=100, help='threshold of the number of voxels in 2D')
parser.add_argument('--voxel_num_thre3d', type=int, default=1000, help='threshold of the number of voxels in 3D')
parser.add_argument('--WINDOW_LEVEL', type=int, default=40, help='window level for CT images')
parser.add_argument('--WINDOW_WIDTH', type=int, default=400, help='window width for CT images')
parser.add_argument('--axis', type=int, default=0, help='along which axis to preprocess image')
parser.add_argument('--verbose', action='store_true', help='print more information', default=False)

# %%
# import sys
# original_args = sys.argv

# sys.argv = [
#     sys.argv[0],
#     'CT',
#     'Bladder',
#     '/vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/imagesTr',
#     '/vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/nnUNet_raw/Dataset002_Bladder/labelsTr',
#     '/vol/biomedic3/bglocker/ugproj2324/az620/radiotherapy/data/MedSAM_preprocessed_lowres',
#     '--axis', '1', 
#     '--image_size', '512',
#     '--verbose'
# ]

# %%
args = parser.parse_args()

if args.verbose:
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

# %%
print('Preprocessing', args.modality, args.anatomy, args.axis)

# %%
modality = args.modality
assert modality in ["CT", "MR"]
anatomy = args.anatomy

img_name_suffix = args.img_name_suffix
gt_name_suffix = args.gt_name_suffix
prefix = modality + "_" + anatomy + "_"

nii_path = args.nii_path  # path to the nii images
gt_path = args.gt_path  # path to the ground truth
npy_path = args.npy_path

gt_save_dir = join(npy_path, "gts", anatomy, f'axis{str(args.axis)}')
img_save_dir = join(npy_path, "imgs", f'axis{str(args.axis)}')

os.makedirs(gt_save_dir, exist_ok=True)
os.makedirs(img_save_dir, exist_ok=True)

image_size = args.image_size
voxel_num_thre2d = args.voxel_num_thre2d
voxel_num_thre3d = args.voxel_num_thre3d

names = sorted(os.listdir(gt_path))

# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.WINDOW_LEVEL  # only for CT images
WINDOW_WIDTH = args.WINDOW_WIDTH  # only for CT images

# %%
def slice_at(i):
    slices = [slice(None)] * 3
    slices[args.axis] = i
    return tuple(slices)

# %%
# save preprocessed images and masks as npz files
for name in tqdm(names):

    if args.verbose:
        print(f'processing name {name}')

    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

    if args.verbose:
        print('dusting images')

    # exclude the objects with less than 1000 pixels in 3D
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )

    if args.verbose:
        print('gt_data_ori.shape', gt_data_ori.shape)

    # remove small objects with less than 100 pixels in 2D slices
    for slice_i in range(gt_data_ori.shape[args.axis]):
        slices = slice_at(slice_i)
        gt_i = gt_data_ori[slices]
        # remove small objects with less than 100 pixels
        # reason: for such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slices] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
        
    if args.verbose:
        print('finding non-zero slices')

    # find non-zero slices
    # For some reason vectorizing this operation doesn't work.
    slice_index = []
    for i in range(gt_data_ori.shape[args.axis]):
        my_slice = gt_data_ori[slice_at(i)]
        if np.any(my_slice):
            slice_index.append(i)

    if args.verbose:
        print(f'for name {name} the non zero slices for axis {args.axis} are {slice_index}')

    if len(slice_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[slice_at(slice_index)]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if modality == "CT":
            if args.verbose:
                print('normalizing Hosfield units')
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            raise NotImplementedError(f"modality {modality} is not implemented yet")

        if args.verbose:
            print('Saving slices...')

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[slice_at(slice_index)]

        # np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # # save the image and ground truth as nii files for sanity check;
        # # they can be removed
        # img_roi_sitk = sitk.GetImageFromArray(img_roi)
        # gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        # sitk.WriteImage(
        #     img_roi_sitk,
        #     join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        # )
        # sitk.WriteImage(
        #     gt_roi_sitk,
        #     join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        # )
        # save the each CT image as npy file
        for i, original_slice in zip(range(img_roi.shape[args.axis]), slice_index):
            img_save_path = join(
                    img_save_dir,
                    modality + "_"
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(original_slice).zfill(3)
                    + ".npy",
                )

            gt_save_path =  join(
                    gt_save_dir,
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(original_slice).zfill(3)
                    + ".npy",
                )
            
            if not os.path.isfile(img_save_path):
                img_i = img_roi[slice_at(i)]
                # img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                img_3c = img_i # don't repeat the channels when saving. This should be done in the dataloader as otherwise there is a lot of redundancy
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
                np.save(img_save_path, resize_img_skimg_01)
            elif args.verbose:
                print('file already found at ', img_save_path)
            

            if not os.path.isfile(gt_save_path):
                gt_i = gt_roi[slice_at(i)]
                resize_gt_skimg = transform.resize(
                    gt_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_gt_skimg = np.uint8(resize_gt_skimg)
                # assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape

                np.save(gt_save_path, resize_gt_skimg)
            elif args.verbose:
                print('file already found at ', gt_save_path)
    





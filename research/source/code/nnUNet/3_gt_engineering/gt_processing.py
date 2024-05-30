import os
import re
import numpy as np
import SimpleITK as sitk

combos = set()

def combine_gt(id: int, gt_path_for_each_anatomy):
    assert 0 <= id <= 100, 'assumed that there are only 100 ids'

    # read in each anatomy ground truth
    sample_name = f'zzAMLART_{id:03d}.nii.gz'

    gt_per_anatomy = dict([(k, sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(v, sample_name)))) for k, v in
                           gt_path_for_each_anatomy.items()])
    assert all([x.shape == gt_per_anatomy[os.environ.get('Anorectum')].shape for _, x in gt_per_anatomy.items()]), \
        'ground truths contain at least one element that isn\'t the same size!'

    for k, v in gt_per_anatomy.items():
        dataset_id = int(re.findall(r'\d+', k)[0])
        dataset_id = 5 if dataset_id == 7 else 7 if dataset_id == 5 else dataset_id
        gt_per_anatomy[k] = v * dataset_id

    # stack all the ground truths
    gt = np.stack([v for _, v in gt_per_anatomy.items()])  # (7, D, H, W)

    return np.apply_along_axis(reduce_fn, 0, gt)  # (D, H, W)

def reduce_fn(x):
    u = str(np.unique(x))
    if u not in combos:
        combos.add(u)
        print(u)
    return 0
import json
import SimpleITK as sitk

pathToData = "../../../../project/tmp/nnUNet_raw/Dataset001_Anorectum/imagesTr/"
fileName = "zzAMLART_075_0000.nii.gz"

# configure image path
image_path = f"{pathToData}{fileName}"

# read image
itk_image = sitk.ReadImage(image_path) 

# get metadata dict
header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}

# print(header)

# save dict in 'header.json'
with open("header.json", "w") as outfile:
    json.dump(header, outfile, indent=4)

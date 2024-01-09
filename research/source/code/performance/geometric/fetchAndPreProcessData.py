# A script for fetching the source labels and the prediction labels in a sensible format

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fetch_nii(path: str):
    """Fetches nii.gz file at some location

    Args:
        path (str): location of nii.gz file

    Returns:
        numpy.ndarray: 3D numpy array
    """
    
    sitk_nii = sitk.ReadImage(path)
    nii = sitk.GetArrayFromImage(sitk_nii)
    return nii

def view_nii(numpy_nii):
    """Attempts to print the 3D representation of the nii file

    Args:
        numpy_nii (3D numpy array): 3D numpy array of a figure 
    """

    # get non-zero indices
    indices = np.nonzero(numpy_nii)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the segmented structure points
    ax.scatter(indices[0], indices[1], indices[2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Segmented Structure Visualization')

    # My PC: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown plt.show()
    # Fix: Specify an Interactive Backend
    import matplotlib
    matplotlib.use('TkAgg')

    plt.show()

# Care: selecting the training image will give you many random numbers, the label will be 
source = fetch_nii('../../../../../project/tmp/nnUNet_raw/Dataset001_Anorectum/imagesTr/zzAMLART_075_0000.nii.gz')
# print(source.shape)
# view_nii(source)

values, count = np.unique(source, return_counts=True)
count_dict = dict(zip(values, count))

print(count_dict)
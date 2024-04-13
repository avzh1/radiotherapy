import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import os

def calculate_metrics(input_path_gt, input_path_pred):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    dice = []
    jaccard = []
    volume_similarity = []

    predictions = sorted(list(filter(lambda x: '.nii.gz' in x, os.listdir(input_path_pred))))

    ground_truth = sorted(list(filter(lambda x: '.nii.gz' in x, os.listdir(input_path_gt))))

    for ypred, y in tqdm(zip(predictions, ground_truth), desc='Calculating metrics', ncols=100):
        y_img_sitk = sitk.ReadImage(os.path.join(input_path_gt, y))
        ypred_img_sitk = sitk.ReadImage(os.path.join(input_path_pred, ypred))

        overlap_measures_filter.Execute(y_img_sitk, ypred_img_sitk)

        dice.append(overlap_measures_filter.GetDiceCoefficient())
        jaccard.append(overlap_measures_filter.GetJaccardCoefficient())
        volume_similarity.append(overlap_measures_filter.GetVolumeSimilarity())

    metrics = dict()

    metrics['dice'] = dice
    metrics['jaccard'] = jaccard
    metrics['volume_similarity'] = volume_similarity

    return metrics

import numpy as np

def calculate_stats(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    quartiles = np.percentile(data, [25, 50, 75])
    
    return mean, std_dev, quartiles

def plot_metrics(metrics_dictionary):
    # Create a figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_dictionary), figsize=(10, 6))

    for ax, (key, value) in zip(axes, metrics_dictionary.items()):
        ax.boxplot(value)
        ax.title.set_text(key)
        ax.set_xticklabels([])

        mean, std, quartiles = calculate_stats(value)

        text = f'Mean: {mean:.2f}\nStd Dev: {std:.2f}\nQ1: {quartiles[0]:.2f}\nQ2: {quartiles[1]:.2f}\nQ3: {quartiles[2]:.2f}'

        ax.text(0.5, -0.2, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Add a text box at the bottom
    # fig.text(0.5, 0.05, 'This is a text box', ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))     

    plt.show()

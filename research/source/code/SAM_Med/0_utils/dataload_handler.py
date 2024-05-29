# Imports
import re
import os
import sys
import json
import random
from torch.utils.data import DataLoader

from dataset import SAM_Dataset, image_id_from_file_name_regex, image_id_from_file_name_regex

class DataLoaderHandler():
    def __init__(self, 
            save_dir, 
            image_dir, 
            gt_dir,
            batch_size,
            num_workers,
            data_aug : bool,
            max_points : int,
            box_padding : int,
            max_box_points : int,
            training_split = 0.8, 
            validation_split = 0.2):

        # Where to save the data splits
        self.save_dir = save_dir
        
        # Where to get image and ground truth info for training
        self.image_dir = image_dir
        self.gt_dir = gt_dir

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug
        self.max_points = max_points
        self.box_padding = box_padding
        self.max_box_points = max_box_points
        
        # Splits for validation and training
        assert training_split + validation_split == 1
        self.training_split = training_split
        self.validation_split = validation_split

        # Final dataLoaders
        self.train_loader = None
        self.val_loader = None

    def save_splits_to_json(self, training_image_ids, validation_image_ids):

        data = {
            "training_image_ids": list(training_image_ids),
            "validation_image_ids": list(validation_image_ids)
        }
        with open(os.path.join(self.save_dir, 'data_splits.json'), 'w') as json_file:
            json.dump(data, json_file)

    def load_split_from_json(self):

        with open(os.path.join(self.save_dir, 'data_splits.json'), 'r') as json_file:
            data = json.load(json_file)
        self.training_split = set(data["training_image_ids"])
        self.validation_split = set(data["validation_image_ids"])

    def try_setup_data_split_from_save_with_fallback(self):

        # either load datasplits or setup anew
        if os.path.exists(os.path.join(self.save_dir, 'data_Splits.json')):
            self.load_split_from_json()
        else:
            self.setup_new_data_splits()

    def setup_new_data_splits(self):
        """Setup the data splits from scratch and save"""

        # get the image ids that have been processed. Use gt dir as reference
        axis0_slices = set(map(lambda x : int(re.search(image_id_from_file_name_regex, x).group(1)), os.listdir(os.path.join(self.gt_dir, 'axis0'))))
        axis1_slices = set(map(lambda x : int(re.search(image_id_from_file_name_regex, x).group(1)), os.listdir(os.path.join(self.gt_dir, 'axis1'))))
        axis2_slices = set(map(lambda x : int(re.search(image_id_from_file_name_regex, x).group(1)), os.listdir(os.path.join(self.gt_dir, 'axis2'))))

        if not axis0_slices == axis1_slices == axis2_slices:
            print('[WARNING]: The slices for the anatomy are not consistent across the three axes. Some axese are missing data, please check')
        
        # Split the data into training and validation
        self.training_image_ids = random.sample(list(axis0_slices), int(len(axis0_slices) * self.training_split))
        self.validation_image_ids = list(set(axis0_slices) - set(self.training_image_ids))
        assert set.intersection(set(self.training_image_ids), set(self.validation_image_ids)).__len__() == 0, 'Training and Validation sets are not disjoint'

        # Save the splits in a json file
        self.save_splits_to_json(self.training_image_ids, self.validation_image_ids)

    def setup_dataloaders(self):
        
        self.training_dataset = SAM_Dataset(self.image_dir, self.gt_dir, self.training_image_ids, data_aug = self.data_aug, max_points=self.max_points, box_padding=self.box_padding, max_box_points=self.max_box_points)
        self.validation_dataset = SAM_Dataset(self.image_dir, self.gt_dir, self.validation_image_ids, data_aug = self.data_aug, max_points=self.max_points, box_padding=self.box_padding, max_box_points=self.max_box_points)
        
        # Quick check
        assert set(map(lambda x : int(re.search(image_id_from_file_name_regex, x).group(1)), self.validation_dataset.axis0_imgs)) == set(self.validation_image_ids), 'DataSet incorrectly loaded image ids that don\'t match supplied validation set image ids'
        assert set(map(lambda x : int(re.search(image_id_from_file_name_regex, x).group(1)), self.training_dataset.axis0_imgs)) == set(self.training_image_ids), 'DataSet incorrectly loaded image ids that don\'t match supplied validation set image ids'

        self.train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
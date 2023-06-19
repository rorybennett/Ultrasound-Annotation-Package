"""
Had to change the labels from Tensor to LongTensor for some reason.
"""
import os

import numpy as np
import pandas as pd
import torch
from skimage import img_as_float32
from skimage import io
from torch.utils.data import Dataset

from IPV.Code import parameters as pms


class My_Dataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.csv_data = np.array(pd.read_csv(csv_file, header=None))
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        idx = idx - idx % 4
        if self.csv_data[idx, 0] == "none":
            return {'sample_name': "none"}

        image = []
        for i in range(len(pms.sub_patch_scales)):
            image.append(img_as_float32(io.imread(self.csv_data[idx + i, 1], True)))
            image[i] = np.array([image[i], image[i], image[i]])
        image = np.array(image)
        sample_name = self.csv_data[idx, 2]
        coords = self.csv_data[idx, 3:5]
        coords = np.array(coords, dtype=np.int32)
        labels = self.csv_data[idx, 5:]
        labels = np.array(labels, dtype=np.long)
        # labels =  labels - 1
        sample = {'image': image,
                  'sample_name': sample_name,
                  'coordinates': coords,
                  'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, sample_name, coords = sample['image'], sample['labels'], sample['sample_name'], sample[
            'coordinates']

        return {'image': torch.from_numpy(image),
                'sample_name': sample_name,
                'coordinates': coords,
                'labels': torch.LongTensor(label)}

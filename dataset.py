import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image

DATA_PATH = './data/floorplan/train-00/coco_vis'
BATCH_SIZE = 128
IMAGE_CHANNEL = 1 #All images in MNIST are single channel, which means the gray scale image. Therefore, a value of IMAGE_CHANNEL is 1.

# Data preprocessing
'''
dataset = dset.MNIST(root=DATA_PATH, download=True,
                     transform=transforms.Compose([
                     transforms.Resize(X_DIM),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))
'''

class floorPlanDataset(Dataset):
    def __init__(self, root_dir=None, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List of all images in our dataset
        self.img_list = []
        for folder in ['test-00', 'train-00', 'train-01']:

            img_dir = os.listdir(os.path.join('data', 'floorplan', folder, 'coco_vis'))
            img_dir = [os.path.join('data', 'floorplan', folder, 'coco_vis', img_path) for img_path in img_dir]
            self.img_list.extend(img_dir)
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        
        img = self.img_list[idx]
        img = read_image(img)

        if self.transform:
            print(f'pretransform: {img.shape}')
            img = self.transform(img[:3])
            print(f'posttransform: {img.shape}')

        return img

# Dataloader
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
#                                         shuffle=True)
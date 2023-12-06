import os
import pdb
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.utils import save_image

DATA_PATH = './data/floorplan/train-00/coco_vis'
BATCH_SIZE = 128
mean = [249.3592, 249.4293, 248.8701]
stds = [19.1668, 19.5032, 20.3175]

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
    def __init__(self, path, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        # List of all images in our dataset
        self.img_list = []

        for img in os.listdir(path):
            img = os.path.join(path, img)
            self.img_list.append(img)
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        # Read the ith image in our image list
        img_loc = self.img_list[idx]
        img = read_image(img_loc)

        if self.transform:
            img = self.transform(img.float())

        return img

# Dataloader
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
#                                         shuffle=True)
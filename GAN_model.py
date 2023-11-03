import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torchvision import models
import torch.optim as optim
import torch.utils.data
import pdb

resize_h = 500
resize_w = 500

dataset = 'floorplan'

CUDA = True
DATA_PATH = './data'
OUTPUT_PATH = 'output_examples/'
BATCH_SIZE = 64
IMAGE_CHANNEL = 3 if dataset == 'floorplan' else 1  # All images in MNIST are single channel, which means the gray scale image. Therefore, a value of IMAGE_CHANNEL is 1.
Z_DIM = 500  # Size of z latent vector (i.e. size of generator input). It is used to generate random numbers for the generator.
G_HIDDEN = 320  # Size of feature maps in the generator that are propagated through the generator.
X_DIM = resize_h  # An original image size in MNIST is 28x28. I will change 28x28 to 64x64 with a resize module for the network.
D_HIDDEN = 320  # Size of feature maps in the discriminator.
EPOCH_NUM = 5  # The number of times the entire training dataset is trained in the network. Lager epoch number is better, but you should be careful of overfitting.
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 16x16 images
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 6, kernel_size=(16, 16), stride=(16, 16)),
            nn.BatchNorm2d(G_HIDDEN * 6)
        )

        # 32 x 32 images
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 6, G_HIDDEN * 4, kernel_size=(17, 17), stride=(17, 17)),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
        )

        # 64 x 64 images
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, kernel_size=(33, 33), stride=(33,33)),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True)
        )

        # 128 x 128
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, kernel_size=(65, 65), stride=(65, 65)),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True)
        )

        # 256 x 256
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, kernel_size=(129, 129), stride=(129,129)),
            nn.Tanh()
        )

    def forward(self, input):
        pdb.set_trace()

        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        out = self.model(input).squeeze(-1)
        return out

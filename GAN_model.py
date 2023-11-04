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

dataset = 'floorplan'

CUDA = True
DATA_PATH = './data'
OUTPUT_PATH = 'output_examples/'
BATCH_SIZE = 32
IMAGE_CHANNEL = 3 if dataset == 'floorplan' else 1  # All images in MNIST are single channel, which means the gray scale image. Therefore, a value of IMAGE_CHANNEL is 1.
Z_DIM = 500  # Size of z latent vector (i.e. size of generator input). It is used to generate random numbers for the generator.
G_HIDDEN = 320  # Size of feature maps in the generator that are propagated through the generator.
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
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, kernel_size=(16, 16), stride=(1, 1)),
            nn.BatchNorm2d(G_HIDDEN * 8)
        )

        # 32 x 32 images
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, kernel_size=(17, 17), stride=(1, 1)),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(),
        )

        # 64 x 64 images
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, kernel_size=(33, 33), stride=(1,1)),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU()
        )

        # 128 x 128
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, kernel_size=(65, 65), stride=(2, 2)),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU()
        )

        # 256 x 256
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, kernel_size=(1, 1), stride=(1,1)),
            nn.ReLU()
        )

    def forward(self, input):

        pdb.set_trace()

        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x


# Define the Generator network
class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()

        self.upsample = torch.nn.Sequential(
            # Fully connected layer to start upsampling
            nn.Linear(500, 1024),
            nn.ReLU(True),
        )

        self.l1 = torch.nn.Sequential(
            # Upsampling layers
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.l2 = torch.nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.l3 = torch.nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.l4 = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.l5 = torch.nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.l6 = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.l7 = torch.nn.Sequential(
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
        )



    def forward(self, input):

        input = input.squeeze() # (N, 500)
        x = self.upsample(input)  # (N, 1024)
        x = x.unsqueeze(-1).unsqueeze(-1) # (N, 1024, 1, 1)
        x = self.l1(x) # (N, 512, 4, 4)
        x = self.l2(x) # (N, 256, 8, 8)
        x = self.l3(x) # (N, 128, 16, 16)
        x = self.l4(x) # (N, 64, 32, 32)
        x = self.l5(x) # (N, 32, 64, 64)
        x = self.l6(x) # (N, 16, 128, 128)
        x = self.l7(x) # (N, 3, 256, 256)

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

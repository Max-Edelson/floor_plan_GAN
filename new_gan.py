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

IMG_SIZE = 256
LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = IMG_SIZE // 64

        self.l0 = nn.Sequential(
            nn.Linear(LATENT_DIM, 512 * self.init_size ** 2),
            nn.PReLU()
        )
        self.tconv0 = nn.Sequential(
            nn.InstanceNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, groups=8)
        )
        self.se0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1, groups=8),
            nn.Hardswish(True),
            nn.Conv2d(64, 256, kernel_size=1, groups=8),
            nn.Sigmoid()
        )
        self.act0 = nn.Sequential(
            nn.PReLU(),
            nn.InstanceNorm2d(256)
        )

        self.tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, groups=4)
        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, kernel_size=1, groups=4),
            nn.Hardswish(True),
            nn.Conv2d(32, 128, kernel_size=1, groups=4),
            nn.Sigmoid()
        )
        self.act1 = nn.Sequential(
            nn.PReLU(),
            nn.InstanceNorm2d(128)
        )

        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, groups=4)
        self.se2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1, groups=4),
            nn.Hardswish(True),
            nn.Conv2d(16, 64, kernel_size=1, groups=4),
            nn.Sigmoid()
        )
        self.act2 = nn.Sequential(
            nn.PReLU(),
            nn.InstanceNorm2d(64)
        )

        self.hourglass3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, groups=4, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 64, kernel_size=3, groups=4, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(64)
        )

        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, groups=4)
        self.se4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, kernel_size=1, groups=4),
            nn.Hardswish(True),
            nn.Conv2d(8, 32, kernel_size=1, groups=4),
            nn.Sigmoid()
        )
        self.act4 = nn.Sequential(
            nn.PReLU(),
            nn.InstanceNorm2d(32)
        )

        self.hourglass5 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, groups=4, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(8),
            nn.Conv2d(8, 32, kernel_size=3, groups=4, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(32)
        )

        self.tconv6 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, groups=4)
        self.se6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, kernel_size=1, groups=4),
            nn.Hardswish(True),
            nn.Conv2d(4, 16, kernel_size=1, groups=4),
            nn.Sigmoid()
        )
        self.act6 = nn.Sequential(
            nn.PReLU(),
            nn.InstanceNorm2d(16)
        )

        self.hourglass7 = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, groups=4, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(4),
            nn.Conv2d(4, 16, kernel_size=3, groups=4, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(16)
        )

        self.tconv8 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, groups=4),
            nn.PReLU(),
            nn.InstanceNorm2d(16)
        )

        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, groups=4, padding=1)
        self.se9 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, kernel_size=1, groups=4),
            nn.Hardswish(True),
            nn.Conv2d(4, 16, kernel_size=1, groups=4),
            nn.Sigmoid()
        )
        self.act9 = nn.Sequential(
            nn.PReLU(),
            nn.InstanceNorm2d(16)
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l0(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        out = self.tconv0(out)
        out = self.act0(out * self.se0(out).expand_as(out))
        out = self.tconv1(out)
        out = self.act1(out * self.se1(out).expand_as(out)) # 16 x 16
        out = self.tconv2(out)
        out = self.act2(out * self.se2(out).expand_as(out))
        out = out + self.hourglass3(out) # 32 x 32
        out = self.tconv4(out)
        out = self.act4(out * self.se4(out).expand_as(out))
        out = out + self.hourglass5(out) # 64 x 64
        out = self.tconv6(out)
        out = self.act6(out * self.se6(out).expand_as(out))
        out = out + self.hourglass7(out) # 128 x 128
        out = self.tconv8(out) # 256 x 256
        out = self.conv9(out)
        out = self.act9(out * self.se9(out).expand_as(out))
        return self.final(out)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        #ds_size = IMG_SIZE // 2 ** 4
        #self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        # Adaptive pooling layer to get a fixed size output (e.g., 4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Linear layer for classification
        self.adv_layer = nn.Sequential(nn.Linear(128 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = self.adaptive_pool(out)  # This will ensure the output is 4x4 spatially
        out = out.view(out.shape[0], -1)  # Flatten the features
        validity = self.adv_layer(out)
        return validity

#!pip install torch-summary

# https://medium.com/@simple.schwarz/how-to-build-a-gan-for-generating-mnist-digits-in-pytorch-b9bf71269da8

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset import floorPlanDataset
import time
from GAN_model import Generator, Discriminator
import pdb

resize_h = 500
resize_w = 500

dataset = 'floorplan'

CUDA = True
DATA_PATH = './data'
OUTPUT_PATH = 'output_examples/'
BATCH_SIZE = 64
IMAGE_CHANNEL = 3 if dataset=='floorplan' else 1 #All images in MNIST are single channel, which means the gray scale image. Therefore, a value of IMAGE_CHANNEL is 1.
Z_DIM = 500 # Size of z latent vector (i.e. size of generator input). It is used to generate random numbers for the generator.
G_HIDDEN = 320 # Size of feature maps in the generator that are propagated through the generator.
X_DIM = resize_h # An original image size in MNIST is 28x28. I will change 28x28 to 64x64 with a resize module for the network.
D_HIDDEN = 320 # Size of feature maps in the discriminator.
EPOCH_NUM = 5 # The number of times the entire training dataset is trained in the network. Lager epoch number is better, but you should be careful of overfitting.
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1

CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
#print(f'device: {device}, torch.cuda.is_available(): {torch.cuda.is_available()}, CUDA: {CUDA}')
cudnn.benchmark = True

# Data preprocessing

#data = floorPlanDataset() #transform=transforms.Resize(size=(resize_h, resize_w)))


'''
img0 = data.__getitem__(0)
print(img0.shape)
plt.imshow(img0.T)
plt.savefig(OUTPUT_PATH + 'img0_original.png')
'''
'''
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
'''

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        print('here')
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_layer = torch.nn.Sequential(
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):

        raise AssertionError

        pdb.set_trace()

        d = self.layer1(input)
        d = self.layer2(d)
        d = self.layer3(d)
        d = self.layer4(d)
        d = self.output_layer(d)
        return d.view(-1, 1).squeeze(1)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_generator(path):
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(path))
    netG.eval()
    return netG

def load_discriminator(path):
    netD = Discriminator().to(device)
    netD.load_state_dict(torch.load(path))
    netD.eval()
    return netD
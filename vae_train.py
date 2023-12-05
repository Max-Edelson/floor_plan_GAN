import os
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import torch.distributions
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm as progress_bar
from dataset import floorPlanDataset
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from vae_models import VariationalAutoencoder
import time

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 500

def generate_images(timestr, vae, epoch, data):
    path = os.path.join('results', 'vae')
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(os.path.join(path, timestr)):
        os.mkdir(os.path.join(path, timestr))
        os.mkdir(os.path.join(path, timestr, 'generated_images'))
    
    path = os.path.join(path, timestr, 'generated_images')

    with torch.no_grad():
        x_hat = vae(data[:64]).cpu()[-1, 1, 256, 256].flatten()
        print(f'x_hat: {x_hat.size()}')
        save_image(x_hat, os.path.join(path, f'generated_{epoch}_examples.png'),
            nrow=8, value_range=(0,255), normalize=True)

def train(train_loader, latent_dims = 128):
    vae = VariationalAutoencoder(latent_dims).to(device)
    opt = torch.optim.Adam(vae.parameters())
    losses = []
    timestr = time.strftime("%Y%m%d-%H%M%S")

    vae.train()
    for epoch in range(epochs):
        data = None
        for i, (data) in progress_bar(enumerate(train_loader), total=len(train_loader)):
            data = data.to(device)
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\t'
                      % (epoch+1, epochs, i, len(train_loader), loss.item()))
        
        generate_images(timestr, vae, epoch, data)
    return vae

class ThresholdTransform(object):

  def __call__(self, x):
    return (x > 0).to(x.dtype)  # do not change the data type

if __name__ == '__main__':
    resize_h = 256
    resize_w = 256

    new_folder_name = 'binary_images'
    path = os.path.join('data', 'floorplan', new_folder_name)
    transform = transforms.Compose([
        ThresholdTransform(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Resize((resize_h, resize_w))
    ])
    data = floorPlanDataset(path=path, transform=transform)
    # Dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE,
                                                shuffle=True)
    train(dataloader)
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
import matplotlib.pyplot as plt
from dataset import floorPlanDataset
import time
# from GAN_model import Generator, Discriminator, Generator2
from new_gan import Generator, Discriminator
#from DCGAN import Generator, Discriminator
from tqdm import tqdm as progress_bar
import pandas as pd
from copy import deepcopy
import pdb
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd

dataset = 'floorplan'  # or 'mnist'

resize_h = 256
resize_w = 256

CUDA = True
DATA_PATH = './data'
OUTPUT_PATH = 'output_examples/'
BATCH_SIZE = 32
Z_DIM = 100  # Size of z latent vector (i.e. size of generator input). It is used to generate random numbers for the generator.
X_DIM = 28  # An original image size in MNIST is 28x28. I will change 28x28 to 64x64 with a resize module for the network.
EPOCH_NUM = 200  # The number of times the entire training dataset is trained in the network. Lager epoch number is better, but you should be careful of overfitting.
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
lambda_l1 =0
seed = 1
mean = [249.3592, 249.4293, 248.8701]
stds = [19.1668, 19.5032, 20.3175]
lambda_gp = 5
n_critic = 1

CUDA = CUDA and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")

class ThresholdTransform(object):

  def __call__(self, x):
    return (x > 0).to(x.dtype)  # do not change the data type


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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

def generate_images(G, epoch, timestr):
    path = os.path.join('results', 'wgan_gp_binary', timestr)

    if not os.path.exists(os.path.join(path)):
        os.mkdir(os.path.join(path))

    # Generate some examples
    noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)
    output = G(noise).cpu()

    save_image(output.data, os.path.join(path, 'generated_examples_' + str(epoch) +  '.png'),
               nrow=8, normalize=True)


def save_experiment(real_img_list, timestr, best_g_loss, best_d_loss, G_loss, D_loss, D, G):
    path = os.path.join('results', 'wgan_gp_binary', timestr)

    # Save discriminator and generator models
    save_model(D, os.path.join(path, 'Discriminator.pth'))
    save_model(G, os.path.join(path, 'Generator.pth'))

    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [lr],
        'Batch size': [BATCH_SIZE],
        'Z-dimension': [Z_DIM],
        'Minimum Generator Loss': [best_g_loss],
        'Minimum Discriminator Loss': [best_d_loss]
    }

    # Save statistics to a csv file
    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join(path, 'metrics.csv'), index=False, header=True)

    # save_image(real_img_list.data, os.path.join('results', timestr, 'real_examples.png'),
    #            nrow=8, normalize=True)

    plt.plot(G_loss)
    plt.title('Generator Loss during Training')
    plt.xlabel('# of Iterations')
    plt.ylabel('Generator Los')
    plt.savefig(os.path.join(path, 'generator_loss.png'))

    plt.clf()
    plt.plot(D_loss)
    plt.title('Discriminator Loss during Training')
    plt.xlabel('# of Iterations')
    plt.ylabel('Discriminator Loss')
    plt.savefig(os.path.join(path, 'discriminator_loss.png'))


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(dataloader):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    path = os.path.join('results', 'wgan_gp_binary')
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, timestr)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create the generator
    netG = Generator().to(device)

    # Create the discriminator
    netD = Discriminator().to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr)
    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    schedulerG = optim.lr_scheduler.OneCycleLR(optimizerG, max_lr=0.1, steps_per_epoch=len(dataloader), epochs=EPOCH_NUM)

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0

    best_G_Loss, best_D_Loss, best_g_model, best_d_model = None, None, None, None

    print("Starting Training Loop...")
    training_start = time.process_time()
    for epoch in range(EPOCH_NUM):

        epoch_G_Loss = 0
        epoch_D_Loss = 0

        for i, (data) in progress_bar(enumerate(dataloader), total=len(dataloader)):

            data = data.to(device)

            # Sample noise as generator input
            z = torch.randn(data.shape[0], Z_DIM, device=device)
            gen_imgs = netG(z)
            #print(f'gen_imgs shape: {gen_imgs.shape}')

            # -------------------
            # Train Discriminator
            # -------------------

            optimizerD.zero_grad()

            # Measure discriminator's ability to classify real samples
            #print(f'data shape: {data.shape}')
            real_output = netD(data)
            D_x = real_output.mean().item()

            # Fake images
            fake_output = netD(gen_imgs)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, data.data, gen_imgs.data)

            # Adverserial loss
            d_loss = -torch.mean(real_output) + torch.mean(fake_output) + lambda_gp*gradient_penalty

            d_loss.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            if i % n_critic == 0:

                # ---------------
                # Train Generator
                # ---------------

                # Generate a batch of images
                gen_imgs = netG(z)
                fake_output = netD(gen_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = -torch.mean(fake_output)
                D_G_z1 = fake_output.mean().item()

                g_loss.backward()
                optimizerG.step()
                schedulerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                          % (epoch, EPOCH_NUM, i, len(dataloader),
                             d_loss.item(), g_loss.item(), D_x, D_G_z1))

                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                epoch_D_Loss += d_loss.item()
                epoch_G_Loss += g_loss.item()

            iters += 1

        epoch_D_Loss /= len(dataloader)
        epoch_G_Loss /= len(dataloader)

        if best_D_Loss is None or epoch_D_Loss < best_D_Loss:
            best_D_Loss = epoch_D_Loss
            best_d_model = deepcopy(netD)
        if best_G_Loss is None or epoch_G_Loss < best_G_Loss:
            best_G_Loss = epoch_G_Loss
            best_g_model = deepcopy(netG)

        generate_images(best_g_model, epoch, timestr)

    total_training_time = time.process_time() - training_start
    print(f'Total training time (s): %.2f' % total_training_time)

    # Grab a batch of real images from the dataloader
    real_images = next(iter(dataloader))

    save_experiment(real_images, timestr, best_G_Loss, best_D_Loss, G_losses,
                    D_losses, best_d_model, best_g_model)


if __name__ == '__main__':
    new_folder_name = 'binary_images'
    path = os.path.join('data', 'floorplan', new_folder_name)
    transform = transforms.Compose([
        ThresholdTransform(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Resize((resize_h, resize_w))
    ])
    data = floorPlanDataset(path=path, transform=transform)
    #print(data[0].shape)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE,
                                             shuffle=True)
    train(dataloader)

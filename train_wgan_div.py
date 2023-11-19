import argparse
import os
import numpy as np
import math
import sys
from dataset import floorPlanDataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
mean = [249.3592, 249.4293, 248.8701]
stds = [19.1668, 19.5032, 20.3175]
from tqdm import tqdm as progress_bar
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch
import time
import pandas as pd
from copy import deepcopy

new_folder_name = 'binary'
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=1, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


k = 2
p = 6


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
timestr = time.strftime("%Y%m%d-%H%M%S")
os.mkdir(os.path.join('results', timestr))

class ThresholdTransform(object):

  def __call__(self, x):
    return (x > 0).to(x.dtype)  # do not change the data type

if cuda:
    generator.cuda()
    discriminator.cuda()

path = os.path.join('data', 'floorplan', new_folder_name)
transform = transforms.Compose([
    ThresholdTransform(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize(opt.img_size)
])
data = floorPlanDataset(path=path, transform=transform)

def generate_images(G, epoch, timestr, path):
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Generate some examples
    noise = torch.randn(32, 100, device=cuda)
    output = G(noise).cpu()

    save_image(output.data, os.path.join(path, 'generated_examples_' + str(epoch) +  '.png'),
               nrow=8, normalize=True)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_experiment(real_img_list, timestr, best_g_loss, best_d_loss, G_loss, D_loss, D, G, path):

    # Save discriminator and generator models
    save_model(D, os.path.join(path, 'Discriminator.pth'))
    save_model(G, os.path.join(path, 'Generator.pth'))

    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [opt.lr],
        'Batch size': [opt.batch_size],
        'Z-dimension': [opt.latent_dim],
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

# Dataloader

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

def train(dataloader, path):

    G_losses = []
    D_losses = []
    iters = 0
    best_G_Loss, best_D_Loss, best_g_model, best_d_model = None, None, None, None

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    schedulerG =  optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=0.1, steps_per_epoch=len(dataloader), epochs=opt.n_epochs)

    batches_done = 0
    print("Starting Training Loop...")
    training_start = time.process_time()
    for epoch in range(opt.n_epochs):
        epoch_G_Loss = 0
        epoch_D_Loss = 0
        fake_imgs=None
        for i, (imgs) in progress_bar(enumerate(dataloader), total=len(dataloader)):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor), requires_grad=True)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)

            # Compute W-div gradient penalty
            real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(
                real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(
                fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                schedulerG.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                if batches_done % opt.sample_interval == 0 and False:
                    output = fake_imgs.cpu()

                    img_name = os.path.join('results', 'train_wgan_div', timestr, str(batches_done) + '.png')

                    save_image(output.data, img_name, nrow=int(opt.batch_size/20), normalize=True)
                
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                epoch_D_Loss += d_loss.item()
                epoch_G_Loss += g_loss.item()

            batches_done += 1
            real_images = next(iter(dataloader))
        epoch_D_Loss /= len(dataloader)
        epoch_G_Loss /= len(dataloader)

        if best_D_Loss is None or epoch_D_Loss < best_D_Loss:
            best_D_Loss = epoch_D_Loss
            best_d_model = deepcopy(discriminator)
        if best_G_Loss is None or epoch_G_Loss < best_G_Loss:
            best_G_Loss = epoch_G_Loss
            best_g_model = deepcopy(generator)

        generate_images(best_g_model, epoch, timestr, path)
        save_image(fake_imgs.cpu().data, os.path.join('results', 'train_wgan_div', timestr, str(epoch) + '.png'), nrow=int(opt.batch_size/20), normalize=True)

    total_training_time = time.process_time() - training_start
    print(f'Total training time (s): %.2f' % total_training_time)
    save_experiment(real_images, timestr, best_G_Loss, best_D_Loss, G_losses,
        D_losses, best_d_model, best_g_model, path)

if __name__ == '__main__':
    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                         shuffle=True)
    
    path = os.path.join('results', 'train_wgan_div')
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join('results', 'train_wgan_div', timestr)
    if not os.path.exists(path):
        os.mkdir(path)
    

    train(dataloader, path)

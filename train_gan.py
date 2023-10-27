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
from tqdm import tqdm as progress_bar
import pandas as pd
from copy import deepcopy

dataset='floorplan' # or 'mnist'

resize_h = 500
resize_w = 500

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


def save_experiment(fake_img_list, real_img_list, timestr, best_g_loss, best_d_loss, G_loss, D_loss, D, G):

    os.mkdir('results', timestr)

    # Save discriminator and generator models
    save_experiment(D, os.path.join('results', timestr, 'Discriminator.pth'))
    save_experiment(G, os.path.join('results', timestr, 'Generator.pth'))

    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [lr],
        'Batch size': [BATCH_SIZE],
        'Z-dimension': [Z_DIM],
        'G_Hidden': [G_HIDDEN],
        'D_Hidden': [D_HIDDEN],
        'Minimum Generator Loss': [best_g_loss],
        'Minimum Discriminator Loss': [best_d_loss]
    }

    # Save statistics to a csv file
    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join('results', timestr, 'metrics.csv', index=False, header=True))

     # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.imsave(os.path.join('results', timestr, 'fake_images.png'))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.imsave(os.path.join('results', timestr, 'real_images.png'))

    plt.plot(G_loss)
    plt.title('Generator Loss during Training')
    plt.xtitle('# of Iterations')
    plt.ytitle('Generator Los')
    plt.imsave(os.path.join('results', timestr, 'generator_loss.png'))

    plt.plot(D_loss)
    plt.title('Discriminator Loss during Training')
    plt.xtitle('# of Iterations')
    plt.ytitle('Discriminator Loss')
    plt.imsave(os.path.join('results', timestr, 'discriminator_loss.png'))
    

def train(dataloader):

    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Create the generator
    #netG = torch.compile(Generator()).to(device)
    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    #netD = torch.compile(Discriminator()).to(device)
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that I will use to visualize the progression of the generator
    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
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
            #print(f'data: {data.shape}')

            # (1) Update the discriminator with real data
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device).float()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            print(f'output.shape: {output.shape}, label: {label.shape}')

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)

            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(FAKE_LABEL)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            # (3) Update the generator with fake data
            netG.zero_grad()
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, EPOCH_NUM, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            epoch_D_Loss += errD.item()
            epoch_G_Loss += errG.item()

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(viz_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        epoch_D_Loss /= len(dataloader)
        epoch_G_Loss /= len(dataloader)   
        
        if best_D_Loss in None or epoch_D_Loss < best_D_Loss:
            best_D_Loss = epoch_D_Loss
            best_d_model = deepcopy(netD)
        if best_G_Loss in None or epoch_G_Loss < best_G_Loss:
            best_G_Loss = epoch_G_Loss
            best_G_model = deepcopy(netG)
        

    total_training_time = time.process_time() - training_start
    print(f'Total training time (s): %.2f' % total_training_time)

    # Grab a batch of real images from the dataloader
    real_images = next(iter(dataloader))

    save_experiment(img_list, real_images, timestr, best_G_Loss, best_D_Loss, G_losses, D_losses, best_d_model, best_g_model)

    
if __name__ == '__main__':
    '''data = dset.MNIST(root=DATA_PATH, download=True,
                     transform=transforms.Compose([
                     transforms.Resize(X_DIM),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))'''
    new_folder_name='resized_500x500'
    path = os.path.join('data', 'floorplan', new_folder_name)
    data = floorPlanDataset(path=path) 
    # Dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0)
    train(dataloader)

import torch
import numpy as np
import os
import argparse
import torch.nn as nn
import time
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from tokenizer import TextDataset, Tokenizer
from SVG_GAN import Generator, Discriminator
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm as progress_bar
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib as mpl
import torch.autograd as autograd
mpl.use('TkAgg')  # !IMPORTANT
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# https://drive.google.com/file/d/1qTTV3NbAkgIgwRBmFfJLITfrVNaIOtBp/view?usp=sharing

def save_model(path, model):

    # Save model to path
    torch.save(model, path)

def generate_images(timestr, netG, epoch, args, tokenizer):

    # TODO: Generate examples with tokens, then untokenize the example
    if not os.path.exists(os.path.join('results', timestr)):
        os.mkdir(os.path.join('results', timestr))
        os.mkdir(os.path.join('results', timestr, 'generated_images'))

    # Generate some examples
    noise = torch.randn(args.batch_size, tokenizer.max_seq_len, args.noise_dim, device=device)

    with torch.no_grad():

        batch_output = netG(noise)
        batch_output = batch_output.detach().cpu()[:10]

        untokenized_data = []

        for i in range(len(batch_output)):
            

            output = batch_output[i]
            idx = 0

            # Untokenize till end point
            while output[idx] != tokenizer.end_token and idx < len(output):
                untokenized_data.append(tokenizer.get_token(output[idx].item()))
                idx += 1
            
            untokenized_str = ''.join(untokenized_data)

            f = open(os.path.join('results', timestr, 'generated_images', str(epoch) + '_' + str(i) + '.svg'), 'w')
            f.write(untokenized_str)
            f.close()
            



def save_experiment(args, timestr, best_g_loss, best_d_loss, G_loss, D_loss, D, G):

    path = os.path.join('results', timestr)

    if not os.path.exists(path):
        os.mkdir(path)

    # Save discriminator and generator models
    save_model(os.path.join(path, 'Discriminator.pth'), D)
    save_model(os.path.join(path, 'Generator.pth'), G)

    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [args.lr],
        'Batch size': [args.batch_size],
        'Z-dimension': [args.noise_dim],
        'Minimum Generator Loss': [best_g_loss],
        'Minimum Discriminator Loss': [best_d_loss],
    }

    # Save statistics to a csv file
    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join(path, 'metrics.csv'), index=False, header=True)

    plt.plot(G_loss)
    plt.title('Generator Loss during Training')
    plt.xlabel('# of Iterations')
    plt.ylabel('Generator Loss')
    plt.savefig(os.path.join(path, 'generator_loss.png'))

    plt.clf()
    plt.plot(D_loss)
    plt.title('Discriminator Loss during Training')
    plt.xlabel('# of Iterations')
    plt.ylabel('Discriminator Loss')
    plt.savefig(os.path.join(path, 'discriminator_loss.png'))

def train(timestr, netG, netD, args, train_loader, tokenizer):

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # SWA Models
    swa_G = AveragedModel(netG)
    swa_D = AveragedModel(netD)

    # SWA Schedulers
    swa_D_scheduler = SWALR(optimizerD, swa_lr=args.lr)
    swa_G_scheduler = SWALR(optimizerG, swa_lr=args.lr)

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    best_G_Loss, best_D_Loss, best_g_model, best_d_model = None, None, None, None
    best_D_x, best_D_G = 0, 0

    for epoch in range(args.epochs):

        epoch_G_Loss = 0
        epoch_D_Loss = 0
        epoch_D_x = 0
        epoch_D_G = 0

        netD.train()
        netG.train()

        for i, (data) in progress_bar(enumerate(train_loader), total=len(train_loader)):

            data = data.to(device)

            # ---------------
            # Train Discriminator
            # ---------------
            optimizerD.zero_grad()

            # Sample noise as generator input
            z = torch.randn(data.shape[0], tokenizer.max_seq_len, args.noise_dim, device=device)

            # Generate a batch of images
            fake_imgs_soft = netG(z)

            # Get the max from the sample
            fake_imgs_hard = torch.argmax(fake_imgs_soft, dim=-1)

            real_validity = torch.mean(netD(data))
            fake_validity = torch.mean(netD(fake_imgs_hard.detach()))

            # Adversarial loss
            d_loss = -real_validity + fake_validity

            d_loss.backward()
            optimizerD.step()

            # Update SWA parameters
            swa_D.update_parameters(netD)
            swa_D_scheduler.step()

            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            # -------------------
            # Train Generator
            # -------------------

            optimizerG.zero_grad()

            # Adversarial loss
            g_loss = -torch.mean(netD(fake_imgs_hard))

            # Update weights
            g_loss.backward()
            optimizerG.step()

            # Update SWA parameters
            swa_G.update_parameters(netG)
            swa_G_scheduler.step()

            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t D(x): %.4f\t D(G(z)): %.4f'
                      % (epoch+1, args.epochs, i, len(train_loader),
                         d_loss.item(), g_loss.item(), real_validity.item(), fake_validity.item()))

            epoch_D_Loss += d_loss.item()
            epoch_G_Loss += g_loss.item()

        # Get averaged epoch statistics
        epoch_D_Loss /= len(train_loader)
        epoch_G_Loss /= len(train_loader)
        epoch_D_G /= len(train_loader)
        epoch_D_x /= len(train_loader)

        G_losses.append(epoch_G_Loss)
        D_losses.append(-epoch_D_Loss)

        if best_D_Loss is None or epoch_D_Loss < best_D_Loss:
            best_D_Loss = epoch_D_Loss
            best_d_model = deepcopy(netD.state_dict())
        if best_G_Loss is None or epoch_G_Loss < best_G_Loss:
            best_G_Loss = epoch_G_Loss
            best_g_model = deepcopy(netG.state_dict())

        best_D_G = max(epoch_D_G, best_D_G)
        best_D_x = max(epoch_D_x, best_D_x)

        if epoch % 5 == 0:

            # Generate images for this epoch
            generate_images(timestr, netG, epoch, args, tokenizer)

    save_experiment(args, timestr, best_G_Loss, best_D_Loss, G_losses,
                    D_losses, best_d_model, best_g_model)


def params():

    """
    Loads the hyperparameters passed into the command line
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=2e-4, type=float,
                        help="Model learning rate starting point.")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=1e-5, type=float,
                        help="L2 Regularization")
    parser.add_argument("--epochs", default=1,  type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--discriminator-hidden", default=64, type=int)
    parser.add_argument("--discriminator-layers", default=2, type=int)
    parser.add_argument("--generator-hidden", default=128, type=int)
    parser.add_argument("--generator-layers", default=2, type=int)
    parser.add_argument("--noise-dim", default=100, type=int)
    parser.add_argument("--clip-value", type=float, default=1.5, help="lower and upper clip value for disc. weights")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=181)
    args = parser.parse_args()

    return args

# Initialize the model with Xavier/Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':

    args = params()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    token_limit=30000
    dataset_type='cubicasa5k'
    tokenizer_meta_data = os.path.join('data', 'tokenizer_data', dataset_type + '_vocab_data_' + str(token_limit) + '.json')
    tokenizer = Tokenizer(dataset_type=dataset_type, tokenizer_meta_data=tokenizer_meta_data, token_limit=token_limit, readInMetadata=True)

    train_dataset = TextDataset(tokenizer=tokenizer, dataset_type=dataset_type, token_limit=token_limit)

    # end_token = tokenizer.ctr
    # tokenizer.end_token = end_token
    # tokenizer.ctr += 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create the Generator and Discriminator networks
    netG = Generator(
        vocab_size=tokenizer.end_token+1,
        embedding_dim=args.embedding_dim,
        latent_dim=args.noise_dim,
        max_sequence_length=tokenizer.max_seq_len
    )

    netD = Discriminator(
       vocab_size=tokenizer.end_token+1,
       embedding_dim=args.embedding_dim
    )

    netG.apply(init_weights)
    netD.apply(init_weights)

    netG = netG.to(device)
    netD = netD.to(device)

    # Train the model
    train(timestr, netG, netD, args, train_loader, tokenizer)


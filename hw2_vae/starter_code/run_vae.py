import os
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
from dataset import floorPlanDataset

class ThresholdTransform(object):

  def __call__(self, x):
    return (x > 0).to(x.dtype)  # do not change the data type

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize_h = 256
resize_w = 256

new_folder_name = 'binary_images'
path = os.path.join('data', 'floorplan', new_folder_name)
print(f'path: {path}')
transform = transforms.Compose([
    ThresholdTransform(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize((resize_h, resize_w))
])
data = floorPlanDataset(path=path, transform=transform)
# Dataloader
train_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)

vae_original = VAE(z_dim=args.z, name=model_name).to(device)

if args.train:
    
    stats = [] # i, nelbos, kls, recs
    for i in range(5):
        name = model_name + str(i+1)
        vae = VAE(z_dim=args.z, name=name).to(device)
        print(f'Model name: |{name}|')
        writer = ut.prepare_writer(name, overwrite_existing=True)
        train(model=vae,
              train_loader=train_loader,
              labeled_subset=None,
              device=device,
              tqdm=tqdm.tqdm,
              writer=writer,
              iter_max=args.iter_max,
              iter_save=args.iter_save)
        #nelbo, kl, rec = ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)
        #stats.append((i, nelbo, kl, rec))
        #ut.save_model_by_name(vae, args.iter_max)
        
    stats = sorted(stats, key=lambda x: x[3])
    
    vae = VAE(z_dim=args.z, name=model_name + str(stats[0][0])).to(device)
    
    ut.load_model_by_name(vae, global_step=args.iter_max)
    
    X = vae.sample_x(200).reshape(200,28,28).detach().numpy()
    
    M = 10
    fig, axs = plt.subplots(20, M, figsize=(40,20))

    for i in range(20):
        for j in range(M):
            axs[i,j].imshow(X[i*10+j], cmap='gray')
            axs[i,j].axis('off')

    plt.show()
    
    print(*stats, sep='\n')

else:
    ut.load_model_by_name(vae_original, global_step=args.iter_max)
    X = vae_original.sample_x(200).reshape(200,28,28).detach().numpy()
    
    M = 10
    #dim = int(np.sqrt(.shape[1]))

    fig, axs = plt.subplots(20, M, figsize=(40,20))

    for i in range(20):
        for j in range(M):
            axs[i,j].imshow(X[i*10+j], cmap='gray')
            axs[i,j].axis('off')

    plt.show()
   
        
    #ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)
    

import torchvision.transforms as transforms
import torchvision.datasets as dset
from dataset import floorPlanDataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

resize_h, resize_w = 500, 500

new_folder_name='resized_500x500'
path = os.path.join('data', 'floorplan', new_folder_name,'')

transform=transforms.Resize(size=(resize_h, resize_w))

data = floorPlanDataset(transform=transform) 

for idx in range(len(data)):
    img = data[idx].squeeze(dim=0).numpy()[:3].astype(np.uint8)
    #img = img.squeeze(dim=0).numpy()[:3].astype(np.uint8) # [1,4,2000,2000] -> [3,resize_h,resize_w]
    img = np.transpose(img, (1,2,0)) # [3,resize_h,resize_w] -> [resize_h, resize_w, 3]
    im = Image.fromarray(img)
    im.save(f'{path}{idx}.jpeg')

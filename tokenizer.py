import re
import os
from torch.utils.data import Dataset
import torch

token_to_id = {} # TODO This has to be saved with each model, as mappings may vary from model to model
ctr = 0

start_token = re.escape('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
pattern = f'(style="[^"]*")|([\w:-]+=)|(rgb|none|rotate|layer|matrix|default|visible|px|em|block|crosshair|translate)|(<)(\w+)|(/>)|(</\w+>)|({start_token})|(.)'
# Requires that we remove text, desc, class, id, and extra spaces in pre-processing
def custom_tokenizer(text):
    matches = re.findall(pattern, text, re.DOTALL)
    res = []
    for group in matches:
        if group[0]:
            styles = re.findall(r'(style=)(")|(rgb|none|rotate|layer|matrix|default|visible|px|em|block|crosshair|translate)|([a-zA-Z0-9-]*:)|(.)', group[0], re.DOTALL)
            for match in styles:
                for i in match:
                    if i not in [None, '']:
                        res.append(i)
            continue
        for match in group:
            if match not in [None, '']:
                res.append(match)

    return res

def get_id(token):
    global ctr
    if token in token_to_id:
        return token_to_id[token]
    token_to_id[token] = ctr
    ctr += 1
    return token_to_id[token]

class TextDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all files in the directory
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Open the file corresponding to idx
        with open(os.path.join(self.root_dir, self.files[idx]), 'r') as f:
            data_point = f.read()

        # Apply the custom tokenizer
        data_point = custom_tokenizer(data_point)

        # Convert tokens to tensor
        data_point = torch.tensor([get_id(token) for token in data_point])

        if self.transform:
            data_point = self.transform(data_point)

        return data_point
    

# Usage example below

# from torch.utils.data import DataLoader

# # Create a dataset
# dataset = TextDataset(root_dir='/path/to/your/files')

# # Create a DataLoader
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Rules for tokenization
# < > "space" " . # ; : , ( ) -
# header line + svg opening tag with properties as start token
# <tag - tag is a token
# property=
# rgb
# rotate
# closing tags </svg>
# font-family value
# All caps as tokens with underscores
# none
# layer

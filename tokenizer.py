import re
import os
from torch.utils.data import Dataset
import torch
import json 

class Tokenizer(object):
    def __init__(self, tokenizer_meta_data=os.join('data', 'tokenizer_meta_data.json')):
        self.token_to_id = {} # TODO This has to be saved with each model, as mappings may vary from model to model
        self.ctr = 0
        if os.path.isfile(tokenizer_meta_data): # tokenizer_meta_data file already exists. Load it in
            meta_data = json.load(open(tokenizer_meta_data,))
            self.token_to_id = meta_data['token_to_id']
            self.ctr = meta_data['ctr']

        self.start_token = re.escape('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
        self.pattern = f'(style="[^"]*")|([\w:-]+=)|(rgb|none|rotate|layer|matrix|default|visible|px|em|block|crosshair|translate)|(<)(\w+)|(/>)|(</\w+>)|({self.start_token})|(.)'
    
    # Requires that we remove text, desc, class, id, and extra spaces in pre-processing
    def tokenize(self, text):
        matches = re.findall(self.pattern, text, re.DOTALL)
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

        token_tensor = torch.tensor([self.get_id(token) for token in res])
        return token_tensor

    def get_id(self, token):
    #    global ctr
        if token in self.token_to_id:
            return self.token_to_id[token]
        self.token_to_id[token] = self.ctr
        self.ctr += 1
        return self.token_to_id[token]
    
    def save_tokenizer_meta_data(self, path=os.join('data/tokenizer_meta_data.json')):
        meta_data = {'token_to_id': self.token_to_id,
                     'ctr':         self.ctr}
        data = json.dumps(meta_data, indent=4)
 
        with open(path, "w") as outfile:
            outfile.write(data)
        outfile.close()

class TextDataset(Dataset, ):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all files in the directory
        self.files = os.listdir(root_dir)
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Open the file corresponding to idx
        with open(os.path.join(self.root_dir, self.files[idx]), 'r') as f:
            data_point = f.read()

        # Apply the custom tokenizer
        data_point = self.tokenizer.tokenize(data_point)
        self.tokenizer.save_tokenizer_meta_data()

        # Convert tokens to tensor
        #data_point = 

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

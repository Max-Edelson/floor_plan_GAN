import re
import os
from torch.utils.data import Dataset
import torch
import json 
from collections import defaultdict
import numpy as np

class Tokenizer(object):
    def __init__(self, tokenizer_meta_data=os.path.join('data', 'tokenizer_data','cubicasa_vocab_data.json')):
        self.token_to_id = {} # TODO This has to be saved with each model, as mappings may vary from model to model
        self.id_to_token = {}
        self.tokens_per_document = defaultdict(dict) # key = document -> returns dict of token counts per that document
        self.global_token_count = {}
        self.ctr = 0
        if os.path.isfile(tokenizer_meta_data): # tokenizer_meta_data file already exists. Load it in
            meta_data = json.load(open(tokenizer_meta_data,))
            self.token_to_id = meta_data['token_to_id']
            self.id_to_token = meta_data['id_to_token']
            self.tokens_per_document = meta_data['tokens_per_document']
            self.ctr = meta_data['ctr']

        self.start_token = re.escape('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
        self.pattern = f'(style="[^"]*")|([\w:-]+=)|(rgb|none|rotate|layer|matrix|default|visible|px|em|block|crosshair|translate)|(<)(\w+)|(/>)|(</\w+>)|({self.start_token})|(.)'
    
    # Requires that we remove text, desc, class, id, and extra spaces in pre-processing
    def tokenize(self, text, file=None):
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

        token_tensor = torch.tensor([self.get_id(token, file) for token in res])
        return token_tensor

    def get_id(self, token, file=None):
    #    global ctr
        if token in self.token_to_id:
            return self.token_to_id[token]
        self.token_to_id[token] = self.ctr
        if token in self.global_token_count: self.global_token_count[token] += 1  
        else: self.global_token_count[token] = 1
        if file is not None:
            if file not in self.tokens_per_document:
                self.tokens_per_document[file] = {}
            if token in self.tokens_per_document[file]: self.tokens_per_document[file][token] += 1
            else: self.tokens_per_document[file][token] = 1
        self.id_to_token[self.ctr] = token
        self.ctr += 1
        return self.token_to_id[token]
    
    def get_token(self, id):
        return self.id_to_token[id]
    
    def save_tokenizer_meta_data(self, path=os.path.join('data', 'tokenizer_data','cubicasa_vocab_data.json')):
        meta_data = {'token_to_id': self.token_to_id,
                     'ctr':         self.ctr}
        data = json.dumps(meta_data, indent=4)
 
        with open(path, "w") as outfile:
            outfile.write(data)
        outfile.close()

    def write_vocab_to_file(self, path=os.path.join('data', 'tokenizer_data','cubicasa.vocab')):
        num_docs = len(self.tokens_per_document)
        num_tokens = len(self.global_token_count)
        d_t_cnt = np.zeros((num_docs, num_tokens)) # d x t
        t_cnt = np.zeros((num_tokens,)) # t x 1
        log_freqs = np.zeros((num_tokens,)) # term_freq x log(num_docs/num_docs_containing_token)

        for d_idx, doc in enumerate(self.tokens_per_document):
            for t_idx, token in enumerate(self.global_token_count):
                if token in self.tokens_per_document[doc]:
                    d_t_cnt[d_idx, t_idx] = self.tokens_per_document[doc][token]
                else: d_t_cnt[d_idx, t_idx] = 0
        for t_idx, token in enumerate(self.global_token_count):
                t_cnt[t_idx] = self.global_token_count[token]
        #           t x 1     t x 1
        log_freqs = t_cnt * np.log(num_docs/np.count_nonzero(d_t_cnt, axis=0))

        output = "[PAD]\t0\n[EOS]\t0\n[UNK]\t0\n[CLS]\t0\n[SEP]\t0\n[MASK]\t0\n"
        for t_idx, token in enumerate(self.global_token_count):
            output += token + "\t" + str(log_freqs[t_idx]) + '\n'

        with open(path, "w") as outfile:
            outfile.write(output)
        outfile.close()
        self.save_tokenizer_meta_data()

class TextDataset(Dataset, ):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all files in the directory
        self.files = os.listdir(root_dir)
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.files)
    
    def tokenize_all_data(self):
        for file in self.files:
            if '.svg' not in file: continue
            with open(os.path.join(self.root_dir, file), 'r') as f:
                data_point = f.read()
            data_point = self.tokenizer.tokenize(data_point, file)
        self.tokenizer.write_vocab_to_file()

    def __getitem__(self, idx):
        # Open the file corresponding to idx
        with open(os.path.join(self.root_dir, self.files[idx]), 'r') as f:
            data_point = f.read()

        file = self.files[idx]

        # Apply the custom tokenizer
        data_point = self.tokenizer.tokenize(data_point, file)
        #self.tokenizer.save_tokenizer_meta_data()

        # Convert tokens to tensor
        #data_point = 

        if self.transform:
            data_point = self.transform(data_point)
            

        return data_point
    
#dataset = TextDataset(root_dir='data/cubicasa5k/svgs')
#dataset.tokenize_all_data()

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

import re
import os
from torch.utils.data import Dataset
import torch
import json 
from collections import defaultdict
import numpy as np
import time
import statistics

class Tokenizer(object):
    def __init__(self, dataset_type, tokenizer_meta_data, readInMetadata=True):
        self.token_to_id = {} # TODO This has to be saved with each model, as mappings may vary from model to model
        self.id_to_token = {}
        self.tokens_per_document = defaultdict(dict) # key = document -> returns dict of token counts per that document
        self.global_token_count = {}
        self.ctr = 1 # 0 used by pytorch for padding
        self.end_token = None
        self.max_seq_len = 0
        if os.path.isfile(tokenizer_meta_data) and readInMetadata: # tokenizer_meta_data file already exists. Load it in
            meta_data = json.load(open(tokenizer_meta_data,))
            self.token_to_id = meta_data['token_to_id']
            self.id_to_token = meta_data['id_to_token']
            #self.tokens_per_document = meta_data['tokens_per_document']
            self.ctr = meta_data['ctr']
            self.end_token = meta_data['end_token']
            self.max_seq_len = meta_data['max_seq_len']
        #self.start_token = re.escape('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
        
        self.start_token1 = re.escape('<?xml version="1.0" encoding="UTF-8"?>\n<svg style="background-color: #000;" version="1.1" viewBox="0 0 100.0 100.0" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">')
        self.start_token2 = re.escape('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
        #self.pattern =   r'(fill-opacity:\s*.*?[;\s/])|(stroke-opacity:\s*.*?[;\s/])|(stroke-width:\s*.*?[;\s/])|(pointer-events:\s*.*?[;\s/])'
        self.pattern = None
        if dataset_type == 'floorplan':
            self.pattern =  rf'({self.start_token1})|({self.start_token2})|(font-family=\s*.*?)[;\s/]|([\w:-]+=)|(fill-opacity:\s*.*?[;\s/])|(stroke-opacity:\s*.*?[;\s/])|(stroke-width:\s*.*?[;\s/])|(pointer-events:\s*.*?[;\s/])|(rgb)|(none)|(rotate)|(layer)|(matrix)|(default)|(visible)|\d+(px)|\d+(em)|(block)|(crosshair)|(translate)|(<\w+)|(</\w+>)|(>)|(\s/>)|(.)'
        elif dataset_type == 'cubicasa5k':
            self.pattern =  rf'({self.start_token1})|({self.start_token2})|(stroke\s*[:=]\s*"([^"]*)")|(fill\s*[:=]\s*"([^"]*)")|([\w:-]+=)|(fill-opacity:\s*.*?[;\s/])|(stroke-opacity:\s*.*?[;\s/])|(stroke-width:\s*.*?[;\s/])|(pointer-events:\s*.*?[;\s/])|(rgb)|(none)|(rotate)|(layer)|(matrix)|(default)|(visible)|\d+(px)|\d+(em)|(block)|(crosshair)|(translate)|(<\w+)|(</\w+>)|(>)|(\s/>)|(.)'

        #self.pattern = f'(style="[^"]*")|([\w:-]+=)|(rgb|none|rotate|layer|matrix|default|visible|px|em|block|crosshair|translate)|(<)(\w+)|(/>)|(</\w+>)|({self.start_token})|(.)'
    
    def remove_non_ascii(self, input_string):
        # Define a regular expression pattern to match non-ASCII characters
        non_ascii_pattern = re.compile('[^\x00-\x7F]')

        # Use the pattern to find and replace non-ASCII characters with an empty string
        result_string = re.sub(non_ascii_pattern, '', input_string)

        return result_string

    # Requires that we remove text, desc, class, id, and extra spaces in pre-processing
    def tokenize(self, text, file=None):
        text = self.remove_non_ascii(text)
        matches = re.findall(self.pattern, text, re.DOTALL)
        #print(*matches, sep='\n')      

        res = []
        for tupl in matches:
            for b in tupl:
                if b != '':
                    res.append(b)
                    break
       #print(text)
        #print(*(res[:200]), sep='\n')
        #print(res)
        '''for group in matches:
            if group[0]:
                styles = re.findall(r'(style=)(")|(rgb|none|rotate|layer|matrix|default|visible|px|em|block|crosshair|translate)|([a-zA-Z0-9-]*:)|(.)', group[0], re.DOTALL)
                #print(styles)
                for match in styles:
                    for i in match:
                        if i not in [None, '']:
                            res.append(i)
                continue
            for match in group:
                if match not in [None, '']:
                    res.append(match)'''

        self.max_seq_len = max(self.max_seq_len, len(res))
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
    
    def save_tokenizer_meta_data(self, path):
        meta_data = {'token_to_id': self.token_to_id,
                     'id_to_token': self.id_to_token,
                     'ctr':         self.ctr,
                     'end_token':   self.end_token,
                     'max_seq_len': self.max_seq_len}
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
    def __init__(self, tokenizer, transform=None, dataset_type='cubicasa5k', token_limit=50000):
        #self.root_dir = root_dir
        self.transform = transform
        self.seq_limit = 100000

        # List all files in the directory
        self.files_json = f'{dataset_type}_svgs_{token_limit}.json' #os.listdir(root_dir)
        #tokenizer_meta_data = os.path.join('data', 'tokenizer_data', dataset_type + '_vocab_data_' + str(token_limit) + '.json')
        self.files = None
        with open(self.files_json, 'r') as openfile:
            self.files = json.load(openfile)['small_files'] # gives a list of file names
        self.tokenizer = tokenizer
        self.end_token = tokenizer.end_token
        self.num_tokens = len(self.tokenizer.id_to_token) + 2 # one for end token, one for padding

    def __len__(self):
        return len(self.files)

    def pad_data(self, x, max_seq):
        #print(f'max_seq: {max_seq}')
        padded_data = torch.zeros(max_seq+1)
        seq_len = x.shape[0]
        padded_data[:seq_len] = x
        padded_data[seq_len] = self.end_token
        #print(f'padded data: {padded_data.shape}')
        return padded_data

    # padded_ex is of shape 1, max_seq_len
    # return 1, max_seq_len, num tokens
    def one_hot_encode(self, padded_ex, num_tokens):
        padded_ex = padded_ex.squeeze()
        eye = torch.eye(num_tokens)
        padded_ex = padded_ex.int()
    #    print(padded_ex)
        eye = eye[padded_ex]
        eye = eye.unsqueeze(0)
        return eye

    def __getitem__(self, idx):
        # Open the file corresponding to idx
        with open(self.files[idx], 'r') as f:
            text = f.read()

        file = self.files[idx]

        # Apply the custom tokenizer. returns a tensor
        data_point = self.tokenizer.tokenize(text)

        if self.transform:
            data_point = self.transform(data_point)

        data_point = self.pad_data(data_point, self.tokenizer.max_seq_len)
        data_point = self.one_hot_encode(padded_ex=data_point, num_tokens=self.num_tokens)    

        return data_point

if __name__ == '__main__':
    tokenized_data = []
    token_limit = 30000
    dataset_type='floorplan'  #'cubicasa5k'
    tokenizer_meta_data = os.path.join('data', 'tokenizer_data', dataset_type + '_vocab_data_' + str(token_limit) + '.json')

    tokenizer = Tokenizer(dataset_type=dataset_type, tokenizer_meta_data=tokenizer_meta_data, readInMetadata=False)
    #dataset = os.listdir(os.path.join('data', dataset_type, 'no_text_2'))
    dataset = os.listdir(os.path.join('data', dataset_type, 'svgs'))
    #dataset = None
    #with open(f'{dataset_type}_svgs<{token_limit}.json', 'r') as openfile:
    #    dataset = json.load(openfile)['small_files']
    
    small_files_list = []

    lengths = []

    t0 = time.time()
    for file in dataset:
        #file = os.path.join('data', dataset_type, 'no_text_2', file)
        file = os.path.join('data', dataset_type, 'svgs', file)
        #print(f'file: {file}')
        with open(file, 'r') as f:
            text = ''.join(f.readlines())
        ids = tokenizer.tokenize(text)
        lengths.append(ids.shape[0])
        if ids.shape[0] < token_limit:
            tokenized_data.append((ids, ids.shape[0]))
            small_files_list.append(file)
    t1 = time.time()

    print(f'mean length: {statistics.mean(lengths)}, median length: {statistics.median(lengths)}, length stdev: {statistics.stdev(lengths)}')

    end_token = tokenizer.ctr
    tokenizer.end_token = end_token
    tokenizer.ctr += 1
    tokenizer.save_tokenizer_meta_data(path=tokenizer_meta_data)
    print(f'end_token: {end_token}')
    small_file_obj = json.dumps({'small_files': small_files_list}, indent=4)
 
    # Writing to sample.json
    with open(f'{dataset_type}_svgs_{token_limit}.json', "w") as outfile:
        outfile.write(small_file_obj)
    outfile.close()

    tokenized_data.sort(reverse=True, key=lambda x: x[1])
    print(f'Max sequence length: {tokenized_data[0][1]}. Tokenization took {(t1-t0)/60} minutes. Contains {len(tokenized_data)} examples. end_token: {end_token}. Vocab Size: {tokenizer.ctr}.')
    

    #dataset = TextDataset()
    #preprocessed_item = dataset[0]


    '''
    un_tokenized = ''
    for id in ids:
        un_tokenized += tokenizer.get_token(id.item())

    f = open('1_model_untokenized.svg', 'w')
    f.write(un_tokenized)
    f.close()
    '''
    #print(un_tokenized)    
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

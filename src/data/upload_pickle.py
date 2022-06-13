from yaml import safe_load
import torchvision
import numpy as np
import torch
import pickle

import sys
sys.path.append('/home/dsi/ziniroi/roi-aviad/src/features')

from feature_extractor import RTF


# configs' args upload:
stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
dictionary = safe_load(stream)
data_init = dictionary.get("Data")

batch_size = 2

def loader(input):
    mix_index = 0
    target_index = 1
    
    fs = 16000
    seconds = 3
    samples = seconds * fs

    item = pickle.load(open(input, 'rb'))
    mix = item[mix_index]
    length = mix[:].shape[1]
    
    min_index = min(length,samples)
    mix_slice = mix[:][0:min_index]
    mix_out = torch.zeros([mix[:].shape[0],samples])
    mix_out[:,0:min_index] = torch.tensor(mix_slice[:,0:min_index]) 
    
    target = item[target_index]
    target_slice = torch.tensor(target[:][0:min_index])
    target_out = torch.zeros([target_slice.shape[0],samples])
    target_out[:,0:min_index] = target_slice[:,0,0:min_index]

    return mix_out ,target_out






root = data_init.get("results_path").get("train")
train_data = torchvision.datasets.DatasetFolder(root=root, loader=loader, extensions='.p', transform=RTF)  #, is_valid_file=is_valid_file
train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True)

def train_loader():
    root = data_init.get("results_path").get("train")
    data = torchvision.datasets.DatasetFolder(root=root, loader=loader, extensions='.p', transform=RTF)  #, is_valid_file=is_valid_file
    t_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True
    )
    return t_loader

def is_valid_file(input):
    item = pickle.load(open(input, 'rb'))
    return not(np.isnan(item).any())
'''
for batch in train_loader:
    
    x = batch
    all_dim = x[0][0].shape
    print(all_dim)
'''
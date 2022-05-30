# import os
from yaml import safe_load
# import random
import torch
import torchvision
import pickle
import sys
sys.path.append('/home/dsi/ziniroi/roi-aviad/src/features')
from feature_extractor import RTF
import numpy as np

# configs' args upload:
stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
dictionary = safe_load(stream)
data_init = dictionary.get("Data")

batch_size = 2

def mix_loader(input):
    mix_index = 0
    item = pickle.load(open(input, 'rb'))
    return item[mix_index]

def is_valid_file(input):
    item = pickle.load(open(input, 'rb'))
    return not(np.isnan(item).any())

def collate_batch(batch):
  
  data = []
  
  for i in batch:
    processed = torch.tensor(loader, dtype=torch.int64)
    data.append(processed)
  
  list = pad_sequence(data, batch_first=True, padding_value=0)
  
  return list.to(device)

# root = data_init.get("results_path").get("train")
train_data = torchvision.datasets.DatasetFolder(root='/home/dsi/ziniroi/roi-aviad/data/raw', loader=mix_loader, extensions='.p', transform=RTF)#, is_valid_file=is_valid_file
train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        collate_fn=torch.nn.utils.rnn.pad_sequence,
        shuffle = True)






indxs = 0
max_num = 0
examples_num = 0
for i in train_loader:
    all_dim = i[0][0].shape # indexs 0 is real indexs 1 is imag
    # print(all_dim)
    sh = all_dim[3]
    if indxs == 0:
        min_num = sh
        indxs = 1

    min_num = np.copy(np.min([sh,min_num]))
    max_num = np.copy(np.max([sh,max_num]))
    examples_num += 1
print(min_num,max_num)
print(examples_num)
# min = 199 ,max = 1614
# min = 170 max = 2047
# 25500




import sys
sys.path.append('/home/dsi/ziniroi/roi-aviad/src/features')

import os
import torch
import pickle
import numpy as np
from feature_extractor import RTF_mix, RTF_target ,H_transform
from torch.utils.data import Dataset
from scipy import signal

class OldDataset(Dataset):
    def __init__(self, data_dir, transform=RTF_mix, target_transform=RTF_target):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.files_list = os.listdir(data_dir)
        self.len = len(self.files_list)
        # According to data generation:
        self.mix_index = 0
        self.target_index = 1
        self.fs = 16000

        # Initiated cutting:
        self.seconds = 3
        self.samples = self.seconds * self.fs
    #def get_len(self):
    #    return self.len
    def loader(self,dir):
        # Loading pickle file:
        item = pickle.load(open(dir, 'rb'))

        # mix:
        mix = np.array(item[self.mix_index])
        length = mix.shape[1]

        min_index = min(length,self.samples)
        mix_slice = mix[0:min_index]
        mix_out = torch.zeros([mix.shape[0],self.samples])
        mix_out[:,0:min_index] = torch.tensor(mix_slice[:,0:min_index]) 
    
        # target:
        target = np.array(item[self.target_index])
        target_slice = torch.tensor(target[:,0:min_index])
        target_out = torch.zeros([target_slice.shape[0],self.samples])
        target_out[:,0:min_index] = target_slice[:,0,0:min_index]

        return mix_out ,target_out

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        dir = os.path.join(self.data_dir, self.files_list[idx])
        mix, target = self.loader(dir)

        if self.transform:
            mix = self.transform(mix)
        if self.target_transform:
            target = self.target_transform(target)
        return mix, target

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=RTF_mix, target_transform=H_transform):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.files_list = os.listdir(data_dir)
        self.len = len(self.files_list)
        # According to data generation:
        self.mix_index = 0
        self.target_index = 2
        self.fs = 16000

        # Initiated cutting:
        self.seconds = 3
        self.samples = self.seconds * self.fs

    def get_len(self):
        return self.len

    def loader(self,dir):
        # Loading pickle file:
        item = pickle.load(open(dir, 'rb'))

        # mix:
        mix = np.array(item[self.mix_index])
        length = mix.shape[1]

        min_index = min(length,self.samples)
        mix_slice = mix[0:min_index]
        mix_out = torch.zeros([mix.shape[0],self.samples])
        mix_out[:,0:min_index] = torch.tensor(mix_slice[:,0:min_index])
        
        # option 1: target:
        target = np.array(item[self.target_index])
        #print(target.shape)
        target_out = torch.zeros(target.shape[0]*target.shape[1],target.shape[2])
        target_out[:4,:] = torch.from_numpy(target[0,:,:])
        target_out[4:,:] = torch.from_numpy(target[1,:,:])
        '''
        # option 2: target:
        target = np.array(item[self.target_index])
        #print(target.shape)
        #target_out = torch.zeros_like(target)

        target_out = torch.from_numpy(target)
        '''

        return mix_out ,target_out

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        dir = os.path.join(self.data_dir, self.files_list[idx])
        mix, target = self.loader(dir)

        if self.transform:
            mix = self.transform(mix)
        if self.target_transform:
            target = self.target_transform(target,self.fs)
        return mix, target


'''
def loader_(dir='/home/dsi/ziniroi/roi-aviad/data/raw/train/train_scenario_2.p'):
    target_index = 2
    # Loading pickle file:
    item = pickle.load(open(dir, 'rb'))
    
    # target:
    target = np.array(item[target_index])
    print(target.shape)
    target_out = torch.zeros(target.shape[0]*target.shape[1],target.shape[2])
    target_out[:4,:] = torch.from_numpy(target[0,:,:])
    target_out[4:,:] = torch.from_numpy(target[1,:,:])
    return target_out

h=loader_()
H = H_transform(h)
print(H.shape)
print(H)

#print(np.shape(ph))
'''
'''
data_set = CustomDataset('/home/dsi/ziniroi/roi-aviad/data/raw/train')

train_dataloader = DataLoader(data_set, batch_size=3, shuffle=True)
'''




'''
for i in train_dataloader:
    mix, target = i

    print('mix: '+str(mix.shape))
    print('target: '+str(target.shape))
    break
'''
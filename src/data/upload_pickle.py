#from msilib.schema import tables
from yaml import safe_load
import torchvision
import numpy as np
import torch
import pickle
import sys

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
    #out= target_out[0,:]
    #print("out:",out.shape)
    print("mix_out:",mix_out.shape)
    out = dict[5, target_out[0,:]]
    output = [mix_out ,out]
    print(len(output))
    return output







'''
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True)
'''
def train_loader():
    sys.path.append('/home/dsi/ziniroi/roi-aviad/src/features')
    # configs' args upload:
    stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
    dictionary = safe_load(stream)
    data_init = dictionary.get("Data")
    from feature_extractor import RTF, RTF1
    batch_size = 2
    root = data_init.get("results_path").get("train")

    train_data = torchvision.datasets.DatasetFolder(
    root=root,
    loader=loader,
    extensions='.p',
    transform=RTF,
    target_transform=RTF1  
     )  #, is_valid_file=is_valid_file

    t_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True
    )
    return t_loader

def val_loader():
    sys.path.append('/home/dsi/ziniroi/roi-aviad/src/features')
    # configs' args upload:
    stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
    dictionary = safe_load(stream)
    data_init = dictionary.get("Data")
    from feature_extractor import RTF, RTF1
    batch_size = 2

    root = data_init.get("results_path").get("train")

    train_data = torchvision.datasets.DatasetFolder(
    root=root,
    loader=loader,
    extensions='.p',
    transform=RTF,
    target_transform=RTF1 
    )  

    root = data_init.get("results_path").get("train")
    #data = torchvision.datasets.DatasetFolder(root=root, loader=loader, extensions='.p', transform=RTF)  #, is_valid_file=is_valid_file
    t_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = False

    )
    return t_loader





'''
    for batch in t_loader:
        
        x = batch[0]
        print(len(x),len(x[0]))
        #print(x)
        
        #print(x[0])
        
        print('***********************************************')
        #print(x[1])
        all_dim = x[0].shape
        print(all_dim)
        break
    
    return t_loader
'''
def is_valid_file(input):
    item = pickle.load(open(input, 'rb'))
    return not(np.isnan(item).any())



'''

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
'''
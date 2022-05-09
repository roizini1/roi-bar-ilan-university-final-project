import os
import nprirgen
from yaml import safe_load
# import random
import torch
import pickle
from feature_extractor import RTF
# configs' args upload:
stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
dictionary = safe_load(stream)
data_init = dictionary.get("Data")

#scenario_list = os.listdir(data_init.get("results_path").get("train"))
#def Data_Loader(train_list,):
#    shuffled_list = random.shuffle(scenario_list)
batch_size = 32

def pickle_loader(input):
    item = pickle.load(open(input, 'rb'))
    return item.values

train_data= torch.torchvision.datasets.DatasetFolder(root=data_init.get("results_path").get("train"), loader=pickle_loader, extensions='.pickle', transform=RTF)
train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True)





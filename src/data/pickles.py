import pickle
import numpy as np
import time
import torch
item = pickle.load(open('/home/dsi/ziniroi/roi-aviad/data/raw/train to download/train_scenario_1.p', 'rb'))
item2 = pickle.load(open('/home/dsi/ziniroi/roi-aviad/data/raw/train to download/train_scenario_2.p', 'rb'))
    
fs = 16000
seconds = 3
samples = seconds*fs

item3 = torch.tensor(item2[0])
#list = []

item4 = item3[:,0:samples]
#time.sleep(1000)
print(item4.shape)
# print(np.shape(item2[1]))
import os
import numpy as np
import pickle
data_dir = '/home/dsi/ziniroi/roi-aviad/data/raw/train'
files_list = os.listdir(data_dir)
len = len(files_list)

def loader(dir):
        # Loading pickle file:
        item = pickle.load(open(dir, 'rb'))
        return item
index = 0
counter = 0
for file in files_list:
    file_dir = os.path.join(data_dir, file)
    if os.path.exists(file_dir):
        index+=1
    if os.path.getsize(file_dir)>0:
        mix,target,h,doa = loader(file_dir)
        #print('iteration sizes-> mix: '+str(np.shape(mix))+' target: '+str(np.shape(target))+' h: '+str(np.shape(h))+' doa: '+str(np.shape(doa)))
        if np.isnan(mix).any() or np.isnan(target).any() or np.isnan(h).any() or np.isnan(doa).any():
            os.remove(file_dir)
            print('Nan file')
            counter+=1
    else:
        os.remove(file_dir)
        print('empty file')
        counter+=1
print('{} total files got checked'.format(index))
print('{} total files'.format(len))
print('{} deleted files in total'.format(counter))
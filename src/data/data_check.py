import os
import pickle
import pathlib
import numpy as np

from yaml import safe_load
stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
dictionary = safe_load(stream)
df = dictionary.get("Data") # data parameters

dataset_path = pathlib.Path(df.get("results_path").get(df.get("data_mode")))
files = [e for e in dataset_path.iterdir() if e.is_file()] #####

cnt=0
for i in range(len(files)):
    file=files[i]
    if os.path.getsize(file) > 0:  
        with open(file, 'rb') as f:
            z, s = pickle.load(f)
            if np.isnan(z).any() or np.isnan(s).any():
                os.remove(str(file.parent) + '/' + file.name)
                print('NaN OCCURED')
                cnt +=1
    else:
        os.remove(str(file.parent) + '/' + file.name)
        print('NO SIZE')
        cnt +=1


print('{} deleted files'.format(cnt))
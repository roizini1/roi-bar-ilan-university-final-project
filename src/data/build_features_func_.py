import os
import pickle
import pathlib
import numpy as np
from scipy import signal

def build_features_func(raw_data_dir=['/home/dsi/ziniroi/roi-aviad/data/raw/train'], features_dir='/home/dsi/ziniroi/roi-aviad/data/processed/features', features_type='1'):
    print("Generate features...")
   
    for indx in range(np.shape(raw_data_dir)[0]):
        data_dir = raw_data_dir[indx]

        if os.path.exists(data_dir):
            data_dir = pathlib.Path(data_dir)

            for pickle_dir in data_dir.iterdir():
                
                data = pickle.load(open(pickle_dir, "rb"))  # read pickle file
                ##############np.max(np.abs(data[0]))
                signal = data[0] / np.max(np.abs(data[0])) # signal is all normalized mixs' signals, data[0] is all mixed signals
                Sig_mag ,Sig_phase = stft_mag_phase(signal)
                
                features_file = pathlib.Path(features_dir)/ 'stft_mag_phase_{}'.format(os.path.normpath(os.path.basename(pickle_dir)))
                
                with open(features_file, 'wb') as f:
                    pickle.dump((Sig_mag, Sig_phase) , f)
        else:
            print('path does not exist:' + data_dir)


def stft_mag_phase(sig):
    eps = np.exp(-16)
    Signal_mag = [] # signal magnitude - log scale
    Signal_phase = []
    for i in range(sig.shape[0]):
        Sig = signal.stft(sig[i, :], nperseg=512, window="hamming", noverlap=512 * 0.75)[2]
        mag = Sig_log_abs = np.log(np.abs(Sig) + eps)
        phase = Sig_phase = np.angle(Sig)
        Signal_mag.append(mag)
        Signal_phase.append(phase)

    return Signal_mag ,Signal_phase


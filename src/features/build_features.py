import pathlib
import pickle
import numpy as np
from scipy import signal
from yaml import safe_load





stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
dictionary = safe_load(stream)
df = dictionary.get("Data") # data parameters

print("Generate features...")

eps = np.exp(-16)
mode = df.get("data_mode") # 'train', 'val', 'test' 
data_dir = pathlib.Path(df.get("results_path").get("raw")) / mode
spectrograms_dir = pathlib.Path(df.get("results_path").get("featuers")) / '/spectrograms'#/{0}'.format(mode)'
spectrograms_dir.mkdir(parents=True, exist_ok=True)


mix_signals = []
for file_path in data_dir.iterdir():
    mix_signals.append(file_path)
    if mix_signals[-1] == spectrograms_dir: del mix_signals[-1]

for idx, s in enumerate(mix_signals):
    data = pickle.load(open(s, "rb")) # read pickle file
    signal = data[0] / np.max(np.abs(signal)) # signal is all normalized mixs' signals, data[0] is all mixed signals
    
    Signal_mag = [] # signal magnitude - log scale
    Signal_phase = []
    for i in range(signal.shape[1]):
        Sig = signal.stft(signal[:, i], nperseg=512, window="hamming", noverlap=512 * 0.75)[2]
        mag = Sig_log_abs = np.log(np.abs(Sig) + eps)
        phase = Sig_phase = np.angle(Sig)
        Signal_mag.append(mag)
        Signal_phase.append(phase)


    spectrogram_file = spectrograms_dir / 'mix_spectogram_{}.p'.format(idx)
    with open(spectrogram_file, 'wb') as f:
        pickle.dump(Signal_mag, Signal_phase, f)

print('done')

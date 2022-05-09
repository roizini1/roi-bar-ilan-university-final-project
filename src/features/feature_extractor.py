import numpy as np
from scipy import signal

def RTF(data):
    '''Relative Transfer Function'''
    '''Recived data is in shape->(mics num, samples)'''
    stft_sig = []
    for i in range(data.shape[0]):
        sig = data[i,:] #  i is mic number
        sig = sig / max(abs(sig)) # normalized data
        stft_sig.appand(signal.stft(sig))
    
    for i in range(data.shape[0]):
        ref_mic_stft = stft_sig[0]
        stft_sig[i] = stft_sig[i] / ref_mic_stft
    
    return np.real(stft_sig), np.imag(stft_sig)
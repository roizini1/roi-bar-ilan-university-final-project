import numpy as np
from scipy import signal

def RTF(data):
    '''Relative Transfer Function'''
    '''Recived data is in shape->(mics num, samples)'''
    stft_sig = []
    mics_num = data.shape[0] # mics num

    for i in range(mics_num):
        sig = data[i,:] #  i is mic number
        sig = np.array(sig / max(abs(sig)),dtype='f') # normalized data
        stft_calc = signal.stft(sig,nperseg=512, window="hamming", noverlap=512 * 0.75)[2]  ##might be better with other params
        #print(np.shape(stft_calc))
        stft_sig.append(np.array(stft_calc))
    
    
    ref_mic_stft =  np.array(stft_sig[0])
    for i in range(mics_num):
        stft_sig[i] =  np.array(stft_sig[i]) /  ref_mic_stft
    # print(ref_mic_stft.shape)
    
    
    return np.real(stft_sig), np.imag(stft_sig)
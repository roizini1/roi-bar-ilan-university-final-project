import numpy as np
import torch
from scipy import signal

def RTF_mix(data):
    '''Relative Transfer Function'''
    '''Recived data is in shape->(mics num, samples)'''
    #print("RTF input data shape: ",data.shape)
    
    eps = 10**-6
    stft_mix = []
    mix = data

    mics_num = mix.shape[0] # mics num
    
    # mix handle:
    i = 0
    for i in range(mics_num):
        sig = mix[i,:] #  i is mic number
        
        sig = sig / max(abs(sig)) # normalized data
        stft_calc = signal.stft(sig,nperseg=512, window="hamming", noverlap=512 * 0.75)[2]  ##might be better with other params
        stft_mix.append(stft_calc)

    stft_shape = stft_mix[0].shape

    rtf = torch.zeros([mics_num,2,stft_shape[0],stft_shape[1]],dtype=torch.cfloat)
    ref_mic_stft = stft_mix[0]

    i = 0
    for i in range(mics_num):
        temp = torch.tensor(stft_mix[i] / (eps+ref_mic_stft),dtype=torch.cfloat)

        rtf[i,0,:,:]= temp.real
        rtf[i,1,:,:]=temp.imag

    return  rtf

def RTF_target(data):
    target = data
    return target
from scipy import signal
import torch
import numpy as np
def RTF_mix(mix):
    '''Relative Transfer Function'''
    '''Recived data is in shape->(mics num, samples)'''
    '''RTF output is [channel,stft dim 1,stft dim 2]'''
    
    eps = 10**-6
    stft_mix = []
    

    mics_num = mix.shape[0] # mics num
    
    # mix handle:
    i = 0
    for i in range(mics_num):
        sig = mix[i,:] #  i is mic number
        
        #sig = sig / max(abs(sig)) # normalized data ####################
        stft_calc = signal.stft(sig,nperseg=512, window="hamming", noverlap=512 * 0.75)[2]  ##might be better with other params
        stft_mix.append(stft_calc)

    stft_shape = stft_mix[0].shape
    channels_num = (mics_num-1)*2

    rtf = torch.zeros([channels_num,stft_shape[0],stft_shape[1]])
    ref_mic_index = 0

    i = 0
    for i in range(mics_num-1):
        temp = torch.tensor(stft_mix[i+1] / (stft_mix[ref_mic_index]+eps),dtype=torch.cfloat)

        rtf[2*i,:,:]= temp.real
        rtf[2*i+1,:,:]=temp.imag

    return torch.tensor(rtf,dtype=torch.float)

def RTF_target(target):
    '''STFT calculation'''
    stft_target = []
    targets_num = target.shape[0] # mics num
    
    # target handle:
    
    i = 0
    for i in range(targets_num):
        sig = target[i,:] #  i is mic number
        
        #sig = sig / max(abs(sig)) # normalized data ####################
        stft_calc = signal.stft(sig,nperseg=512, window="hamming", noverlap=512 * 0.75)[2]  ##might be better with other params
        stft_target.append(stft_calc)
        
    target_out = np.zeros([targets_num*2,stft_target[0].shape[0],stft_target[0].shape[1]])
    for i in range(targets_num):
        target_out[2*i,:,:] = stft_target[i].real
        target_out[2*i+1,:,:] = stft_target[i].imag


    return torch.tensor(target_out,dtype=torch.float32)
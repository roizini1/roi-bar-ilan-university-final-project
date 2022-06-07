import numpy as np
import torch
from scipy import signal

def RTF(data):
    '''Relative Transfer Function'''
    '''Recived data is in shape->(mics num, samples)'''
    stft_mix = []
    # stft_target = []
    mix , target = data
    # mix = data
    # print(mix.shape,target.shape)
    mics_num = mix.shape[0] # mics num
    # num_of_speakers = target.shape[0]
    
    # mix handle:
    i = 0
    for i in range(mics_num):
        sig = mix[i,:] #  i is mic number
        
        sig = sig / max(abs(sig)) # normalized data
        stft_calc = signal.stft(sig,nperseg=512, window="hamming", noverlap=512 * 0.75)[2]  ##might be better with other params
        stft_mix.append(stft_calc)
    
    # print('mix shape: '+str(stft_mix[0].shape))
    stft_shape = stft_mix[0].shape

    #j = 0
    #for j in range(num_of_speakers):
    #    sig = target[j,0,:] #  i is mic number
    #    sig = sig / max(abs(sig)) # normalized data
    #    stft_calc = signal.stft(sig,nperseg=512, window="hamming", noverlap=512 * 0.75)[2]  ##might be better with other params
    #    stft_target.append(stft_calc)
    

    rtf = torch.zeros([mics_num,2,stft_shape[0],stft_shape[1]],dtype=torch.cfloat)
    ref_mic_stft = stft_mix[0]

    i = 0
    for i in range(mics_num):
        temp = torch.tensor(stft_mix[i] /  ref_mic_stft,dtype=torch.cfloat)
        # print(temp.shape)
        rtf[i,0,:] = temp.real
        rtf[i,1,:] = temp.imag
    
    #real_imag_mix =                                        #torch.cat((torch.tensor(rtf,dtype=torch.cfloat).real, torch.tensor(rtf,dtype=torch.cfloat).imag,2))
    
    # real_imag_target = torch.cat((torch.tensor(stft_target,dtype=torch.cfloat).real(), torch.tensor(stft_target,dtype=torch.cfloat).imag()),2)
    return  rtf , target         #real_imag_mix , real_imag_target
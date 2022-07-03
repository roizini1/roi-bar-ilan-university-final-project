import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import distutils.version
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from Conv4dPyTorch import Conv4d
import pytorch_lightning as pl
# from dataset import DirDataset
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
import torch
from scipy import signal
import sys
from asteroid.losses import PITLossWrapper
from torchmetrics import ScaleInvariantSignalDistortionRatio
sys.path.append('/home/dsi/ziniroi/roi-aviad/src/data')

from upload_pickle import train_loader, val_loader
class Unet(pl.LightningModule):
    def __init__(self): #, hparams
        super(Unet, self).__init__()
        #print(hparams)
        #self.hparams = hparams

        self.n_channels =  4 #hparams.n_channels
        self.n_classes = 1 #hparams.n_classes
        self.bilinear = True

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool3d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()
                '''
                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    '''
                self.up = nn.ConvTranspose3d(in_channels , in_channels//2, (1,1,1))

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                #print(x1.shape, x2.shape)                
                x1 = self.up(x1)
                #print(x1.shape, x2.shape)                

                # [?, C, H, W]
                diffX = x2.size()[2] - x1.size()[2]
                diffY = x2.size()[3] - x1.size()[3]
                diffZ = x2.size()[4] - x1.size()[4]
                
                x1 = F.pad(x1, [
                                diffZ // 2, diffZ - diffZ//2 ,
                                diffY // 2, diffY - diffY//2,
                                diffX // 2, diffX - diffX//2,
                                ])
                              
                #print(x1.shape, x2.shape)                
                x = torch.cat([x2, x1], dim=1) ## why 1?
                
                return self.conv(x)
        #self.inc = nn.Conv3d(4, 1, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        #self.inc = Conv4d(8, 1, kernel_size=(3, 1,1, 1), padding=(0, 0, 0, 0), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        print("before double_conv")
        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        '''
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        '''
        self.up4 = up(128, 64)
        self.out = nn.Conv3d(64, self.n_classes, kernel_size=(1,1,1))

    def forward(self, x):
        #print(len(x),.shape,x[1].shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        '''
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        '''
        x = self.up4(x2, x1)
        return self.out(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        print(y_hat.size())
        sig = torch.complex(y_hat[0,0,0,:,:],y_hat[0,0,1,:,:]) #  i is mic number
        _, stft_calc = signal.istft(sig)  ##might be better with other params
        print(stft_calc[0::2].shape)
        print(y.shape)
        si_sdr = ScaleInvariantSignalDistortionRatio()
        loss=si_sdr(stft_calc[0::2], y)       
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        print(y_hat.size())
        sig = torch.complex(y_hat[0,0,0,:,:],y_hat[0,0,1,:,:]) #  i is mic number
        _, stft_calc = signal.istft(sig)  ##might be better with other params
        print(stft_calc[0::2].shape)
        print(y)
        si_sdr = ScaleInvariantSignalDistortionRatio()
        loss=si_sdr(stft_calc[0::2], y)
        #loss_func = PITLossWrapper(MultiSrcNegSDR("sisdr"),
         #                   pit_from='perm_avg')
        #loss = loss_func(torch.from_numpy(stft_calc), y)
        #loss = F.cross_entropy(torch.from_numpy(stft_calc), y)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)

    def __dataloader(self):
        return {
            'train': train_loader(),
            'val': val_loader()#val_loader
        }

    #@pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    #@pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    #@staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=8)
        parser.add_argument('--n_classes', type=int, default=48000)
        return parser


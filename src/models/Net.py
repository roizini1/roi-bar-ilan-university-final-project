
from new_loader import CustomDataset
from Unet_f import Unet
from torch.utils.data import DataLoader, random_split

from torchmetrics import MeanSquaredError,PermutationInvariantTraining

import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

class Unet_Model(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_hp = hparams.Training
        self.net_hp = hparams.Net_hp

        self.in_channels = self.net_hp.n_channels
        self.out_channels = self.net_hp.n_classes
        
        self.Unet = Unet(self.in_channels, self.out_channels,out_sz=(2048))
        
        self.lr = self.train_hp.lr
        self.batch_size = self.train_hp.batch_size
        self.train_data_dir = self.hparams.db.train_data_dir
        #self.dataset = self.dataset_def()
        
        self.metric = PermutationInvariantTraining(F.mse_loss, eval_func='min')
    '''
    def my_mse(self,y_hat,y):
        mse = torch.sum((y_hat-y)**2)
        return mse
    '''

    def forward(self, x):
        return self.Unet(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self.Unet(x)
        #print(y_hat.type)
        #print(y.type)
        
        loss =  self.metric(y_hat,y)

        tensorboard_logs = {'train_loss': loss}
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)#,sync_dist=True)
        return {'loss': loss, 'log': tensorboard_logs}
        


    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    '''
    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")
    '''
    
    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.Unet(x)
        loss = self.metric(y_hat,y)
        
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return {f"{prefix}_loss": loss}
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
    
    def dataset_def(self):
        
        dataset = CustomDataset(self.train_data_dir)
        n_val = int(dataset.get_len() * 0.1)
        #n_val = int(np.copy(n_val))

        n_train = dataset.get_len() - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,pin_memory=True,shuffle=True,num_workers=4) #pin_memory=True,
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, pin_memory=True, shuffle=False,num_workers=4)#pin_memory=True,

        return {
            'train': train_loader,
            'val': val_loader
        }

    #@pl.data_loader
    def train_dataloader(self):
        return self.dataset_def()['train']
    
    #@pl.data_loader
    def val_dataloader(self):
        return self.dataset_def()['val']
    

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()


class Unet_old(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_hp = hparams.Training
        self.net_hp = hparams.Net_hp

        self.lr = self.train_hp.lr
        self.batch_size = self.train_hp.batch_size
        self.train_data_dir = self.hparams.db.train_data_dir

        self.n_channels = self.net_hp.n_channels
        self.n_classes = self.net_hp.n_classes
        
        self.bilinear = True
        
        self.metric = PermutationInvariantTraining(F.mse_loss, eval_func='min')


        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # (N,C,H,W)
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                #if bilinear:
                #    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)
        class down_to_H_size(nn.Module):
            def __init__(self,out_sz):
                super().__init__()
                self.linear = nn.Linear(488*1055,out_sz) # 488*1055 is rtf size and 2048 is h length
    
    def forward(self, x):
        return self.linear(x)
        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.unet_out = nn.Conv2d(64, self.n_classes, kernel_size=1)
        self.out = down_to_H_size(2048)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        unet_output = self.unet_out(x)
        return self.out(unet_output)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.metric(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss_func = PITLossWrapper(multisrc_mse, pit_from='perm_avg')
        loss = loss_func(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    #
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    #

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()


    def __dataloader(self):
        
        dataset = CustomDataset(self.train_data_dir)
        n_val = int(dataset.get_len() * 0.1)
        #n_val = int(np.copy(n_val))

        n_train = dataset.get_len() - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, pin_memory=True, shuffle=True,num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, pin_memory=True, shuffle=False,num_workers=4)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    #@pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    #@pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']
 

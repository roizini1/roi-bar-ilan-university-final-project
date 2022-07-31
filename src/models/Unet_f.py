import torch
from torchvision.transforms import CenterCrop
#import torch.nn.functional as F
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels,chs=[64,128,256,512,1024]):
        super().__init__()
        
        self.chs=chs.insert(0, in_channels)

        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.chs = (1024,512,256,128,64)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(self.chs[i], self.chs[i+1], 3, 3) for i in range(len(self.chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(self.chs[i], self.chs[i+1]) for i in range(len(self.chs)-1)]) 
        #self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.conv = nn.Conv2d(self.chs[len(self.chs)-1], out_channels, kernel_size=1)
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        conv_out = self.conv(x)
        return conv_out
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class Unet(nn.Module):
    def __init__(self,in_channels, out_channels, out_sz):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)
        self.To_H = down_to_H_size(out_channels,out_sz)
        
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        dec_out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        return self.To_H(dec_out)

class down_to_H_size(nn.Module):
    def __init__(self,input_dim,out_sz):
        super().__init__()
        self.down = Encoder(input_dim,chs=[64,128,64,input_dim])
        self.flatten = nn.Flatten(start_dim=-2,end_dim=-1)
        self.linear = nn.Linear(53*124,out_sz) # 53 x 124 is rtf size and 2048 is h length
    
    def forward(self, x):
        out = self.down(x)
        flat = self.flatten(out[len(out)-1])
        lin = self.linear(flat)
        #print(lin.size())
        return lin

'''
class down_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
'''

'''
class Unet_old(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_hp = hparams.Training
        self.net_hp = hparams.Net_hp

        self.lr = self.train_hp.lr
        self.batch_size = self.train_hp.batch_size
        self.train_data_dir = self.train_hp.train_data_dir

        self.n_channels = self.net_hp.n_channels
        self.n_classes = self.net_hp.n_classes
        
        self.bilinear = True
        self.loss_function = F.mse_loss
        

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
        loss_func = PITLossWrapper(multisrc_mse, pit_from='perm_avg')
        loss = loss_func(y_hat, y)
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
'''
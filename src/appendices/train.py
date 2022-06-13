from gc import callbacks
import os
from argparse import ArgumentParser

import numpy as np
#from sqlalchemy import true
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main():
    model = Unet() #hparams

    
    os.makedirs('/home/dsi/ziniroi/roi-aviad/models', exist_ok=True) # model dir - log_dir

    try:
        log_dir = sorted(os.listdir('/home/dsi/ziniroi/roi-aviad/models'))[-1]
    except IndexError:
        log_dir = os.path.join('/home/dsi/ziniroi/roi-aviad/models', 'version_0')
    print(log_dir)
    checkpoint_callback = ModelCheckpoint()
    '''
        dirpath=os.path.join(log_dir, 'checkpoints'),
        verbose=True,
    )
    '''
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=True,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback,stop_callback],max_epochs=1#,strategy ='cpu'#gpus=1
    )

    trainer.fit(model)


if __name__ == '__main__':
    
    '''
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='unet\lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()
    '''
    main()
import os
import torch
#from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from Net import Unet_Model,Unet_old
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar,ModelCheckpoint, EarlyStopping
#import GPUtil
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing
#available_gpus = torch.cuda.device_count()


import logging
import hydra

from config_classes import Config_class
from omegaconf import OmegaConf



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name = "Config", node = Config_class)
logger = logging.getLogger(__name__)

import pickle
torch.cuda.empty_cache()

@hydra.main(config_path="./conf", config_name="config")
def main(cfg: Config_class):# -> Trainer:
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    model = Unet_Model(cfg) # Unet_old(cfg)


    os.makedirs(cfg.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(cfg.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(cfg.log_dir, 'version_0')
    
    checkpoint_dirpath = os.path.join(log_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # by default is none whichsaves only the last epoch
        save_top_k=-1,
        filename='checkpoint_{epoch:02d}-{val_loss:.2f}',
        dirpath = checkpoint_dirpath,
        verbose=True
    )
    stop_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min') # for val_loss mode is min 
    #logger_dirpath = os.path.join(log_dir,'lightning_logs')
    #tb_logger = TensorBoardLogger(save_dir=logger_dirpath)
    
    #logger = TensorBoardLogger("/home/dsi/ziniroi/roi-aviad/src/lightning_logs", name="my_model")

    
    trainer = Trainer(
        accelerator="gpu",
        #devices=2,
        #auto_select_gpus=True,
        #accelerator='ddp',
        #strategy = "ddp",
        #gpus=1,
        #auto_scale_batch_size="power",
        devices=[2],
        max_epochs=100,

        enable_checkpointing=True,
        callbacks=[checkpoint_callback, RichProgressBar(),stop_callback],
        logger=True,#tb_logger,
        
        #detect_anomaly=True,
        check_val_every_n_epoch=1
    )
    #trainer.tune(model)
    
    trainer.fit(model)


if __name__ == '__main__':
    main()
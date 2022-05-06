
import torchaudio
import hydra
from termcolor import colored
import sys
sys.path.append("/workspace/inputs/aviad/extraction/src")
from data.mydataloader import EXDataModule #, CreateFeatures
from data.choose_model import fetch_model_according_to_conf
from torch2pl import Pl_module
import numpy as np
import os
from pytorch_lightning import Trainer, plugins
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
import logging
from pytorch_lightning.plugins import DDPPlugin
logging.getLogger('lightning').setLevel(logging.WARNING)
import shutil




warnings.filterwarnings("ignore", category=UserWarning)
torchaudio.set_audio_backend("sox_io")
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=sys.maxsize)

#############
# ======================================== main section ==================================================
@hydra.main(config_path="/workspace/inputs/aviad/extraction/conf/", config_name="config.yaml")
def main(hp):
    print(colored('GPU ID:', 'yellow'), colored(hp.train.cuda_visible_devices, 'green'))

    if hp.debug:
        print(colored('This is a Debug Mode', 'yellow'))

    on_epoch_end_dir =  os.path.join(os.getcwd(),'on_epoch_end')

    model_def = fetch_model_according_to_conf(hp)
    model = Pl_module(model_def,hp,on_epoch_end_dir)
    dm = EXDataModule(hp)


    finetuning_path = hp.finetuning_path if  hp.finetuning else None

    shutil.copy('/workspace/inputs/aviad/extraction/src/models/model_def.py', 'save_model_def.py')
    shutil.copy('/workspace/inputs/aviad/extraction/src/models/utils_classes.py', 'save_utils_classes.py')
    shutil.copy('/workspace/inputs/aviad/extraction/src/data/mydataloader.py', 'save_mydataloader.py')


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        # dirpath= curr_file_abs_path + '/../model_ckpts/',
        filename='{epoch:02d},{val_loss:.2f}',
        save_last=False,  # when true, always saves the model at the end of the epoch to a file
        save_top_k=2,  # the best k models according to the quantuty monitored will be saved
        mode='min',  # for monitor='val_loss' this should be 'min'
        verbose=True
    )

    # monitor a metric and stop when it stops improving
    earlystopping_callback = EarlyStopping(monitor='val_loss',  # quntity to be monitored
                                           patience=hp.train.patience)  # number of check with no improvement after which training will be stopped

    if hp.train.cuda_visible_devices != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = hp.train.cuda_visible_devices

    checkpoints_path = os.getcwd()+'/checkpoints/'
    trainer = Trainer(gpus=hp.train.num_of_gpus,
                      accelerator='ddp',
                     fast_dev_run=hp.debug,
                      check_val_every_n_epoch=hp.train.check_val_every_n_epoch,
                      default_root_dir=checkpoints_path,
                      callbacks=[earlystopping_callback, checkpoint_callback],
                      progress_bar_refresh_rate=10, # How often to refresh progress bar (in steps)
                      plugins=DDPPlugin(find_unused_parameters=False),
                      # precision=precision
                      resume_from_checkpoint = finetuning_path
                      )

    # if hp.finetuning:
    #      trainer.fit(net, dm,checkpoints_path=finetuning_path)
    # else:
    # trainer.tune(model, datamodule=dm)
    trainer.fit(model, dm)
    # trainer.test(model=net) # if I have  def test_step
    checkpoint_callback.best_model_path



if __name__ == '__main__':

    main()

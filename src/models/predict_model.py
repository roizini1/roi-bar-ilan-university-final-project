from Unet_roi_try import Unet
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import torch
def main(hparams):

    PATH = '/home/dsi/ziniroi/roi-aviad/src/lightning_logs/version_0/checkpoints/checkpoint_epoch=00-val_loss=0.53.ckpt'
    model = Unet(hparams)
    #trainer = Trainer()
    #model.load_state_dict(torch.load(PATH))
    model.load_from_checkpoint(PATH,kwargs = hparams)
    #trainer.fit(model, ckpt_path=PATH)
    print(model.learning_rate)
    model.eval()
    y_hat = model(x)
    return y_hat


if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset_dir', default='/home/dsi/ziniroi/roi-aviad/data/raw/train')
    parent_parser.add_argument('--log_dir', default='/home/dsi/ziniroi/roi-aviad/src/lightning_logs')
    parent_parser.add_argument('--batch_size', default=4)
    parent_parser.add_argument('--learning_rate',default=0.1)

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
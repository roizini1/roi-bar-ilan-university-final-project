import logging
import hydra

from config_classes import Config_class
from omegaconf import OmegaConf

from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name = "Config", node = Config_class)
logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: Config_class):# -> Trainer:
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    #print(cfg.db.train_data_dir)
    # model def:
    #model = Unet(hparams)


if __name__ == "__main__":
    main()


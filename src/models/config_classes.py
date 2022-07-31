from dataclasses import dataclass

@dataclass
class Train_hp:
    epochs: int
    lr: float
    batch_size: int

@dataclass
class Network:
    n_channels: int
    n_classes: int

@dataclass
class data_base:
    train: str
    #val: str
    #test: str
'''
@dataclass
class network:
'''

@dataclass
class Config_class:
    db: data_base
    Training: Train_hp
    Net_hp: Network
    log_dir: str

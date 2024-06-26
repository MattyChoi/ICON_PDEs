import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import hydra
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def get_predict_params(cfg: DictConfig):
    params = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(params, Dict)

    if cfg.log:
        # logging using tensorboard
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html
        logger = TensorBoardLogger(save_dir="./")

        # pytorch lightning callbacks
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#:~:text=A%20callback%20is%20a%20self,your%20lightning%20module%20to%20run.
        callbacks = []

        # add callbacks
        if cfg.callbacks:
            for _, callback in cfg.callbacks.items():
                callbacks.append(hydra.utils.instantiate(callback))

        params["logger"] = logger
        params["callbacks"] = callbacks

    else:
        params["logger"] = False

    return params


def pred(cfg: DictConfig):
    # set random seed
    seed_everything(cfg.seed)

    # build model
    pretrained = cfg.pretrained
    task = hydra.utils.get_class(cfg.tasks).load_from_checkpoint(
        pretrained, dataset=cfg.dataset, map_location=None
    )

    # build data for model to test on
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        dataset=cfg.dataset,
        transform=cfg.transform,
        _recursive_=False,
    )
    
    # create the Trainer object with all the wanted configurations
    params = get_predict_params(cfg)
    trainer = Trainer(**params)

    # test the model
    trainer.test(model=task, datamodule=data_module)



@hydra.main(version_base=None, config_path="../config", config_name="test_defaults")
def run(cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR", "1")
    pred(cfg)

if __name__ == "__main__":
    run()
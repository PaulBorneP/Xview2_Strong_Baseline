import os

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import wandb


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> pl.Trainer:
    pl.seed_everything(cfg.seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    data_module = hydra.utils.instantiate(cfg.data)
    network = hydra.utils.instantiate(cfg.network)
    wandb.init(project=cfg.logger.project,
               config=config, 
               group=cfg.group, 
               name=cfg.name)
    wandb.watch(network,log_freq=1000)
    trainer = hydra.utils.instantiate(cfg.trainer)
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(config)
    trainer = trainer(logger=logger)
    trainer.fit(network, data_module)
    trainer.test(network, data_module)
    trainer.predict(network, data_module)

    return trainer


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["WANDB_MODE"] ="disabled"
    main()
from typing import Any, Tuple, Dict, Union, Optional, Type
import os

import cv2
from lightning.pytorch import LightningModule
import torch
import torchmetrics

from legacy.refactor_utils import get_class_weights


class SLModule(LightningModule):
    """supervised learning module"""

    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 train_metric: Type[torchmetrics.Metric],
                 val_metric: Type[torchmetrics.Metric],
                 test_metric: Type[torchmetrics.Metric],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 model_dir: str,
                 name: str,
                 class_weights: Optional[str] = None,
                 pretrain_path: Optional[str] = None) -> None:

        super().__init__()
        self.model = model
        if pretrain_path is not None:
            # load the model from a pth file
            print("Loading model from {}".format(pretrain_path))
            state_dict = torch.load(pretrain_path)
            state_dict = {k.replace("model.", ""): v for k,
                          v in state_dict.items()}
            # remove parameters that are not in the model
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            print(incompatible)
            assert len(state_dict) > 0, "No parameters loaded"
            # assert weights were loaded correctly
            for k in state_dict:
                assert torch.all(torch.eq(self.model.state_dict()[
                                 k], state_dict[k])), "Weights not loaded correctly"

        self.loss_fn = loss_fn
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_dir = model_dir
        self.name = name
        self.class_weights = get_class_weights(
            class_weights) if class_weights is not None else None

        # TODO: not working
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self,
                      batch: Dict[str, Union[torch.Tensor, Any, str]],
                      batch_idx: int) -> torch.Tensor:
        x: torch.Tensor = batch["img"]  # type: ignore
        y: torch.Tensor = batch["msk"]  # type: ignore
        y_hat = self.forward(x)
        if self.class_weights is not None:
            loss_dict = {}
            for i in range(y_hat.shape[1]):
                loss_dict[f"channel_{i}"] = self.loss_fn(
                    y_hat[:, i, ...], y[:, i, ...])
            loss = torch.stack(
                [loss_dict[f"channel_{i}"]*self.class_weights[i] for i in range(y_hat.shape[1])], dim=0).sum()
        else:
            loss = self.loss_fn(y_hat, y)
        self.log("train_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=x.shape[0])
        self.train_metric.update(y_hat, y)
        # is this ok ? https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        #log per channel F1
        self.log(f"train_{self.train_metric.__class__.__name__}",
                 self.train_metric,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=x.shape[0])
        return loss

    def validation_step(self,
                        batch: Dict[str, Union[torch.Tensor, Any, str]],
                        batch_idx: torch.Tensor) -> torch.Tensor:
        x, y = batch["img"], batch["msk"]  # type: ignore
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=x.shape[0])
        self.val_metric.update(y_hat, y)
        # is this ok ? https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        self.log(f"val_{self.val_metric.__class__.__name__}",
                 self.val_metric,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=x.shape[0])
        return loss

    def on_predict_start(self) -> None:
        os.makedirs(os.path.join(self.model_dir,
                    f"{self.name}", "submission"), exist_ok=True)

    @staticmethod
    def pred_one_image(pred):
        y_sigm = torch.sigmoid(pred)
        y_pred = y_sigm.cpu().numpy().transpose(1, 2, 0)
        loc_pred = y_pred[..., 0]
        # loc mask in {0,1}
        loc_msk = (loc_pred > 0.38).astype('uint8')
        # dmg mask in {0,1,2,3,4}
        dmg_msk = y_pred[..., 1:] .argmax(
            axis=2) + 1  # get 4-class ids per pixel
        dmg_msk = dmg_msk * loc_msk
        loc_msk = loc_msk.astype('uint8')
        dmg_msk = dmg_msk.astype('uint8')
        return loc_msk, dmg_msk

    def predict_step(self,
                     batch: Dict[str, Union[torch.Tensor, Any, str]],
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        x, fns = batch["img"],  batch["fn"]
        y_hat = self.forward(x)
        for pred, fn in zip(y_hat.unbind(dim=0), fns):
            file_name = fn.split('/')[-1]
            loc_msk, msk_dmg = self.pred_one_image(pred)
            cv2.imwrite(os.path.join(self.model_dir, f"{self.name}", "submission", file_name.replace(
                '_pre_disaster', '_localization_disaster_prediction')), loc_msk)
            cv2.imwrite(os.path.join(self.model_dir, f"{self.name}", "submission",  file_name.replace(
                '_pre_disaster', '_damage_disaster_prediction')), msk_dmg)

    def test_step(self,
                  batch: Dict[str, Union[torch.Tensor, Any, str]],
                  batch_idx: torch.Tensor) -> torch.Tensor:
        x, y = batch["img"], batch["msk"]
        y_hat = self.forward(x)
        y_sigm = torch.sigmoid(y_hat)
        loc_pred = y_sigm[:, 0, ...]
        # loc mask in {0,1}
        loc_msk = (loc_pred > 0.38)
        # dmg mask in {0,1,2,3,4}
        dmg_msk = y_sigm[:, 1:, ...].argmax(
            axis=1) + 1  # get 4-class ids per pixel
        dmg_msk = dmg_msk * loc_msk
        # hot encode dmg_msk shape (batch_size, 5, h, w)
        hot_dmg_msk = torch.zeros(y_hat.shape, dtype=y_hat.dtype,device=y_hat.device)
        for i in range(5):
            hot_dmg_msk[:, i, ...] = dmg_msk == i
        hot_dmg_msk[:, 0, ...] = loc_msk
        for i in range(y_hat.shape[1]):
            self.test_metric[i].update(hot_dmg_msk[:, i, ...], y[:, i, ...])
            self.log(f"test_channel_{i}_F1",
                    self.test_metric[i],
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    on_epoch=True,
                    batch_size=x.shape[0])        

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer,
                                            torch.optim.lr_scheduler._LRScheduler]:

        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.scheduler(optimizer)
        return ([optimizer], [scheduler])


if __name__ == "__main__":
    import hydra
    import os
    import pytorch_lightning as pl
    import numpy as np

    hydra.initialize(config_path="../conf")
    cfg = hydra.compose(config_name="supervised_config")
    pl.seed_everything(cfg.seed)

    network = hydra.utils.instantiate(cfg.network)

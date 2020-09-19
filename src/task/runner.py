from omegaconf import DictConfig
from pytorch_lightning import EvalResult, LightningModule, TrainResult
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .metric import acc, cross_entropy


class ClassificationRunner(LightningModule):
    def __init__(self, model: nn.Module, runner_config: DictConfig):
        super().__init__()
        self.model = model
        self.hparams.update(runner_config.optimizer.params)
        self.hparams.update(runner_config.scheduler.params)
        self.hparams.update(runner_config.trainer.params)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(params=self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = ExponentialLR(opt, gamma=self.hparams.gamma)
        return [opt], [scheduler]

    def on_after_backward(self) -> None:
        with torch.no_grad():
            norm = self.model._fc.weight.norm(p=2, dim=-1, keepdim=True)

            if any(norm > self.hparams.scale_factor):
                self.model._fc.weight.div_(norm).mul_(self.hparams.scale_factor)

    def training_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self.model(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_acc = acc(mb_labels_hat, y_mb)
        result = TrainResult(minimize=mb_loss)
        result.log_dict(
            {"tr_loss": mb_loss, "tr_acc": mb_acc},
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self.model(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_acc = acc(mb_labels_hat, y_mb)
        result = EvalResult(checkpoint_on=mb_loss)
        result.log_dict(
            {"val_loss": mb_loss, "val_acc": mb_acc},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({"val_loss": "loss", "val_acc": "acc"})
        return result

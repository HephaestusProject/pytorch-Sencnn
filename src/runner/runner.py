import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.core import LightningModule
from .metric import cross_entropy


class Runner(LightningModule):
    def __init__(self, model: nn.Module, runner_config: DictConfig):
        super().__init__()
        self.model = model
        self.hparams.update(runner_config.dataloader.params)
        self.hparams.update(runner_config.optimizer.params)
        self.hparams.update(runner_config.scheduler.params)
        self.hparams.update(runner_config.trainer.params)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        opt = Adam(params=self.model.parameters(),
                   lr=self.hparams.learning_rate)
        scheduler = ExponentialLR(opt, gamma=self.hparams.gamma)
        return [opt], [scheduler]

    # def on_after_backward(self) -> None:
    #     with torch.no_grad():
    #         norm = self.model._fc.weight.norm(p=2, dim=-1, keepdim=True)
    #
    #         if any(norm > self.hparams.scale_factor):
    #             self.model._fc.weight.div_(norm).mul_(self.hparams.scale_factor)

    def training_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_acc = (y_mb == mb_labels_hat).float().mean()
        log = {"loss_per_step": mb_loss.unsqueeze(0), "acc_per_step": mb_acc.unsqueeze(0)}
        return {"loss": mb_loss, "log": log}

    def training_epoch_end(self, outputs):
        avg_tr_loss = torch.stack([x["log"]["loss_per_step"] for x in outputs]).mean()
        avg_tr_acc = torch.stack([x["log"]["acc_per_step"] for x in outputs]).mean()
        tqdm_dict = {"avg_tr_loss": avg_tr_loss, "avg_tr_acc": avg_tr_acc}
        log = {"avg_tr_loss": avg_tr_loss, "avg_tr_acc": avg_tr_acc}
        return {"progress_bar": tqdm_dict, "log": log}

    def validation_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_n_correct_pred = torch.sum(y_mb == mb_labels_hat).item()
        return {"loss": mb_loss, "n_correct_pred": mb_n_correct_pred, "n_pred": len(x_mb)}

    def validation_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["loss"] * x["n_pred"] for x in outputs]).sum()
        avg_val_loss = total_loss / total_count
        avg_val_acc = torch.tensor(total_n_correct_pred / total_count)
        tqdm_dict = {"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc}
        log = {"avg_val_loss": avg_val_loss, "avg_tr_acc": avg_val_acc}
        return {"progress_bar": tqdm_dict, "log": log}

    def test_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_n_correct_pred = torch.sum(y_mb == mb_labels_hat).item()
        return {"loss": mb_loss, "n_correct_pred": mb_n_correct_pred, "n_pred": len(x_mb)}

    def test_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["loss"] * x["n_pred"] for x in outputs]).sum().item()
        avg_test_loss = total_loss / total_count
        avg_test_acc = total_n_correct_pred / total_count
        return {"loss": avg_test_loss, "acc": avg_test_acc}


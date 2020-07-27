import torch
import torch.nn as nn

from torch import optim
from pytorch_lightning.core import LightningModule
from argparse import ArgumentParser
from .metric import cross_entropy


class Runner(LightningModule):
    def __init__(self, model: nn.Module, hparams):
        super().__init__()
        self.model = model
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.Adam(params=self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=self.hparams.patience)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_acc = (y_mb == mb_labels_hat).float().mean()
        tensorboard_logs = {"tr_loss": mb_loss, "tr_acc": mb_acc}
        return {"loss": mb_loss, "tr_acc": mb_acc, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_mb_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_mb_acc = torch.stack([x["tr_acc"] for x in outputs]).mean()
        tqdm_dict = {"tr_acc": avg_mb_acc, "tr_loss": avg_mb_loss}
        tensorboard_logs = {"tr_loss": avg_mb_loss, "tr_acc": avg_mb_acc}
        return {**tqdm_dict, "log": tensorboard_logs, "progress_bar": tqdm_dict}

    def validation_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_n_correct_pred = torch.sum(y_mb == mb_labels_hat).item()
        return {"val_loss": mb_loss, "n_correct_pred": mb_n_correct_pred, "n_pred": len(x_mb)}

    def validation_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["val_loss"] * x["n_pred"] for x in outputs]).sum()
        val_loss = total_loss / total_count
        val_acc = total_n_correct_pred / total_count
        tqdm_dict = {"val_acc": val_acc, "val_loss": val_loss}
        tensorboard_logs = {"val_loss": val_loss, "val_acc": val_acc}
        return {**tqdm_dict, "log": tensorboard_logs, "progress_bar": tqdm_dict}

    def test_step(self, batch, batch_idx):
        x_mb, y_mb = batch
        y_hat_mb = self(x_mb)
        mb_loss = cross_entropy(y_hat_mb, y_mb)
        mb_labels_hat = torch.argmax(y_hat_mb, dim=1)
        mb_n_correct_pred = torch.sum(y_mb == mb_labels_hat).item()
        return {"test_loss": mb_loss, "n_correct_pred": mb_n_correct_pred, "n_pred": len(x_mb)}

    def test_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["test_loss"] * x["n_pred"] for x in outputs]).sum()
        test_loss = total_loss / total_count
        test_acc = total_n_correct_pred / total_count
        return {"test_loss": test_loss, "test_acc": test_acc}

    @staticmethod
    def add_runner_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--patience', default=5, type=int)
        return parser

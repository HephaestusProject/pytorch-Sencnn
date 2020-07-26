import pickle
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from omegaconf import DictConfig
from runner.data import CORPUS_FACTORY
from .metric import cross_entropy


class Runner(LightningModule):
    def __init__(self, model: nn.Module, dconf: DictConfig, pconf: DictConfig, rconf: DictConfig):
        super().__init__()
        self._model = model
        self._dconf = dconf
        self._pconf = pconf
        self.hparams = rconf

    def forward(self, x):
        return self._model(x)

    def prepare_data(self) -> None:
        with open(self._pconf.path, mode="rb") as io:
            preprocessor = pickle.load(io)
        corpus = CORPUS_FACTORY[self._dconf.name]
        self._train_ds = corpus(self._dconf.path.train, preprocessor.encode)
        self._valid_ds = corpus(self._dconf.path.validation, preprocessor.encode)
        self._test_ds = corpus(self._dconf.path.test, preprocessor.encode)

    def train_dataloader(self):
        return DataLoader(self._train_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=4,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self._valid_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=4,
                          drop_last=False,
                          shuffle=False)

    def configure_optimizers(self):
        opt = optim.Adam(params=self._model.parameters(), lr=self.hparams.learning_rate)
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
        tensorboard_logs = {"tr_loss": avg_mb_loss, "tr_acc": avg_mb_acc}
        return {"tr_loss": avg_mb_loss, "tr_acc": avg_mb_acc, "log": tensorboard_logs}

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
        tensorboard_logs = {"val_loss": val_loss, "val_acc": val_acc}
        return {"val_loss": val_loss, "val_acc": val_acc, "log": tensorboard_logs}

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
        total_loss = torch.stack([x["val_loss"] * x["n_pred"] for x in outputs]).sum()
        test_loss = total_loss / total_count
        test_acc = total_n_correct_pred / total_count
        return {"test_loss": test_loss, "test_acc": test_acc}
import pickle

from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.net import SenCNN
from src.runner.runner import Runner
from src.utils.data import CORPUS_REGISTRY
from src.utils.preprocessing import PreProcessor


def get_preprocessor(filepath: str) -> PreProcessor:
    with open(filepath, mode="rb") as io:
        preprocessor = pickle.load(io)
    return preprocessor


def get_data_loaders(ds_conf: str,
                     dl_conf: DictConfig,
                     preprocessor: PreProcessor) -> DataLoader:
    dataset = CORPUS_REGISTRY[ds_conf.type]
    tr_ds = dataset(ds_conf.path.train, preprocessor.encode)
    tr_dl = DataLoader(tr_ds,
                       batch_size=dl_conf.params.batch_size,
                       num_workers=dl_conf.params.num_workers,
                       drop_last=True,
                       shuffle=True)
    val_ds = dataset(ds_conf.path.validation, preprocessor.encode)
    val_dl = DataLoader(val_ds,
                        batch_size=dl_conf.params.batch_size,
                        num_workers=dl_conf.params.num_workers,
                        drop_last=False,
                        shuffle=False)
    return tr_dl, val_dl


def main(args) -> None:
    conf_dir = Path("conf")
    mconf_dir = conf_dir / "model"
    dconf_dir = conf_dir / "dataset"
    pconf_dir = conf_dir / "preprocessor"
    rconf_dir = conf_dir / "runner"

    mconf = OmegaConf.load(mconf_dir / args.model_config)
    dconf = OmegaConf.load(dconf_dir / args.dataset_config)
    pconf = OmegaConf.load(pconf_dir / args.preprocessor_config)
    rconf = OmegaConf.load(rconf_dir / args.runner_config)

    preprocessor = get_preprocessor(pconf.path)
    tr_dl, val_dl = get_data_loaders(dconf, rconf.dataloader, preprocessor)
    model = SenCNN(mconf.params.num_classes, mconf.params.dropout_rate, preprocessor.vocab)
    runner = Runner(model, rconf)
    version = f"{rconf.type}"
    tb_logger = TensorBoardLogger(save_dir="exp",
                                  name=mconf.type,
                                  version=version)

    prefix = f"exp/{mconf.type}/{version}/"
    suffix = "{epoch:02d}-{tr_loss:.2f}-{val_loss:.2f}-{tr_acc:.2f}-{val_acc:.2f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_top_k=2,
                                          save_weights_only=True)

    trainer = Trainer(**rconf.trainer.params, logger=tb_logger, checkpoint_callback=checkpoint_callback)
    trainer.fit(runner, train_dataloader=tr_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_config", default="nsmc.yaml", type=str)
    parser.add_argument("--model_config", default="sencnn.yaml", type=str)
    parser.add_argument("--preprocessor_config", default="mecab_10_32.yaml", type=str)
    parser.add_argument("--runner_config", default="v0.yaml", type=str)
    args = parser.parse_args()
    main(args)

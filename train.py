import pickle

from typing import Tuple, Union, List
from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from src.model.net import SenCNN
from src.runner.runner import Runner
from src.utils.corpus import CorpusRegistry
from src.utils.preprocessing import PreProcessor
from pprint import pprint


def get_config(args: Namespace) -> DictConfig:
    config_dir = Path("conf")
    model_config_dir = config_dir / "model"
    dataset_config_dir = config_dir / "dataset"
    preprocessor_config_dir = config_dir / "preprocessor"
    runner_config_dir = config_dir / "runner"

    config = OmegaConf.create()
    model_config = OmegaConf.load(model_config_dir / f"{args.model}.yaml")
    dataset_config = OmegaConf.load(dataset_config_dir / f"{args.dataset}.yaml")
    preprocessor_config = OmegaConf.load(preprocessor_config_dir / f"{args.preprocessor}.yaml")
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(model=model_config, dataset=dataset_config, preprocessor=preprocessor_config, runner=runner_config)
    return config


def get_logger_and_callbacks(args: Namespace) -> Tuple[LightningLoggerBase, Union[Callback, List[Callback]]]:
    logger = TensorBoardLogger(save_dir="exp",
                               name=args.model,
                               version=args.runner)

    prefix = f"exp/{args.model}/{args.runner}/"
    suffix = "{epoch:02d}-{avg_tr_loss:.4f}-{avg_tr_acc:.4f}-{avg_val_loss:.4f}-{avg_val_acc:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_top_k=1,
                                          monitor="avg_val_acc",
                                          save_weights_only=True,
                                          verbose=True)
    return logger, checkpoint_callback


def get_preprocessor(preprocessor_config: DictConfig) -> PreProcessor:
    with open(preprocessor_config.path, mode="rb") as io:
        preprocessor = pickle.load(io)
    return preprocessor


def get_data_loaders(dataset_config: DictConfig,
                     dataloader_config: DictConfig,
                     preprocessor: PreProcessor) -> Tuple[DataLoader, DataLoader]:
    dataset = CorpusRegistry.get(dataset_config.type)
    tr_ds = dataset(dataset_config.path.train, preprocessor.encode)
    tr_dl = DataLoader(tr_ds,
                       batch_size=dataloader_config.params.batch_size,
                       num_workers=dataloader_config.params.num_workers,
                       drop_last=True,
                       shuffle=True)
    val_ds = dataset(dataset_config.path.validation, preprocessor.encode)
    val_dl = DataLoader(val_ds,
                        batch_size=dataloader_config.params.batch_size,
                        num_workers=dataloader_config.params.num_workers,
                        drop_last=False,
                        shuffle=False)
    return tr_dl, val_dl


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    config = get_config(args)
    logger, checkpoint_callback = get_logger_and_callbacks(args)

    preprocessor = get_preprocessor(config.preprocessor)
    tr_dl, val_dl = get_data_loaders(config.dataset, config.runner.dataloader, preprocessor)
    model = SenCNN(preprocessor.vocab, **config.model.params)
    runner = Runner(model, config.runner)

    trainer = Trainer(**config.runner.trainer.params,
                      logger=logger,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(runner, train_dataloader=tr_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="nsmc", type=str)
    parser.add_argument("--model", default="nsmc_classifier", type=str)
    parser.add_argument("--preprocessor", default="mecab_5_32", type=str)
    parser.add_argument("--runner", default="nsmc_v0", type=str)
    parser.add_argument("--reproduce", default=False, action="store_true")
    args = parser.parse_args()
    main(args)

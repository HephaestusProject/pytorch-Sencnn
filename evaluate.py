import pickle
import torch
import json

from typing import Tuple
from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from src.model.net import SenCNN
from src.runner.task import ClassificationTask
from src.utils.corpus import CorpusRegistry
from src.utils.preprocessing import PreProcessor


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


def get_preprocessor(preprocessor_config: DictConfig) -> PreProcessor:
    with open(preprocessor_config.path, mode="rb") as io:
        preprocessor = pickle.load(io)
    return preprocessor


def get_data_loaders(dataset_config: DictConfig,
                     dataloader_config: DictConfig,
                     preprocessor: PreProcessor) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = CorpusRegistry.get(dataset_config.type)
    tr_ds = dataset(dataset_config.path.train, preprocessor.encode)
    tr_dl = DataLoader(tr_ds,
                       batch_size=dataloader_config.params.batch_size,
                       num_workers=dataloader_config.params.num_workers,
                       drop_last=False,
                       shuffle=False)
    val_ds = dataset(dataset_config.path.validation, preprocessor.encode)
    val_dl = DataLoader(val_ds,
                        batch_size=dataloader_config.params.batch_size,
                        num_workers=dataloader_config.params.num_workers,
                        drop_last=False,
                        shuffle=False)
    tst_ds = dataset(dataset_config.path.test, preprocessor.encode)
    tst_dl = DataLoader(tst_ds,
                        batch_size=dataloader_config.params.batch_size,
                        num_workers=dataloader_config.params.num_workers,
                        drop_last=False,
                        shuffle=False)

    return tr_dl, val_dl, tst_dl


def main(args) -> None:
    seed_everything(42)
    config = get_config(args)
    preprocessor = get_preprocessor(config.preprocessor)
    tr_dl, val_dl, tst_dl = get_data_loaders(config.dataset, config.runner.dataloader, preprocessor)

    # restore runner
    model = SenCNN(preprocessor.vocab, **config.model.params)
    runner = ClassificationTask(model, config.runner)

    checkpoint_path = f"exp/{args.model}/{args.runner}/{args.checkpoint}.ckpt"
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))

    results = {}
    trainer = Trainer(**config.runner.trainer.params,
                      logger=False,
                      checkpoint_callback=False)

    tr_result = trainer.test(runner, test_dataloaders=tr_dl)
    val_result = trainer.test(runner, test_dataloaders=val_dl)
    tst_result = trainer.test(runner, test_dataloaders=tst_dl)
    results.update(checkpoint=args.checkpoint, train=tr_result, validation=val_result, test=tst_result)

    with open(f"exp/{args.model}/{args.runner}/results.json", mode="w") as io:
        json.dump(results, io, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="nsmc", type=str, )
    parser.add_argument("--model", default="nsmc_classifier", type=str)
    parser.add_argument("--preprocessor", default="mecab_5_32", type=str)
    parser.add_argument("--runner", default="nsmc_v0", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args)

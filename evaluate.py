import pickle
import torch
import json

from typing import Tuple
from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from src.model.net import SenCNN
from src.runner.runner import Runner
from src.utils.corpus import CorpusRegistry
from src.utils.preprocessing import PreProcessor


def get_config(args: Namespace) -> DictConfig:
    conf_dir = Path("conf")
    mconf_dir = conf_dir / "model"
    dconf_dir = conf_dir / "dataset"
    pconf_dir = conf_dir / "preprocessor"
    rconf_dir = conf_dir / "runner"

    conf = OmegaConf.create()
    mconf = OmegaConf.load(mconf_dir / f"{args.model}.yaml")
    dconf = OmegaConf.load(dconf_dir / f"{args.dataset}.yaml")
    pconf = OmegaConf.load(pconf_dir / f"{args.preprocessor}.yaml")
    rconf = OmegaConf.load(rconf_dir / f"{args.runner}.yaml")
    conf.update(model=mconf, dataset=dconf, preprocessor=pconf, runner=rconf)
    return conf


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
    config = get_config(args)
    preprocessor = get_preprocessor(config.preprocessor)
    tr_dl, val_dl, tst_dl = get_data_loaders(config.dataset, config.runner.dataloader, preprocessor)

    # restore runner
    model = SenCNN(preprocessor.vocab, **config.model.params)
    runner = Runner(model, config.runner)

    checkpoint_path = f"exp/{args.model}/{args.runner}/{args.checkpoint}.ckpt"
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))

    results = {}
    trainer = Trainer(gpus=1)
    tr_result = trainer.test(runner, test_dataloaders=tr_dl)
    val_result = trainer.test(runner, test_dataloaders=val_dl)
    tst_result = trainer.test(runner, test_dataloaders=tst_dl)
    results.update(train=tr_result, validation=val_result, test=tst_result)

    with open(f"exp/{args.model}/{args.runner}/results.json", mode="w") as io:
        json.dump(results, io, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="nsmc", type=str, )
    parser.add_argument("--model", default="sencnn", type=str)
    parser.add_argument("--preprocessor", default="mecab_10_32", type=str)
    parser.add_argument("--runner", default="v0", type=str)
    parser.add_argument("--checkpoint", default="epoch=04-tr_loss=0.33-val_loss=0.33-tr_acc=0.87-val_acc=0.87", type=str)
    args = parser.parse_args()
    main(args)

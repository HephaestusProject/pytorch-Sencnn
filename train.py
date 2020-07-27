import pickle

from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from src.model.net import SenCNN
from src.runner.runner import Runner
from src.utils.data import CORPUS_FACTORY
from src.utils.preprocessing import PreProcessor


def get_preprocessor(filepath: str) -> PreProcessor:
    with open(filepath, mode="rb") as io:
        preprocessor = pickle.load(io)
    return preprocessor


def get_data_loader(dconf: DictConfig,
                    flag: str,
                    preprocessor: PreProcessor,
                    batch_size: int,
                    num_workers: int) -> DataLoader:
    dataset = CORPUS_FACTORY[dconf.name]
    mode = True if flag == "train" else False
    ds = dataset(dconf.path[flag], preprocessor.encode)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=mode,
                    shuffle=mode)
    return dl


def main(args) -> None:
    conf_dir = Path("conf")
    mconf_dir = conf_dir / "model"
    dconf_dir = conf_dir / "dataset"
    pconf_dir = conf_dir / "preprocessor"

    mconf = OmegaConf.load(mconf_dir / args.model_config)
    dconf = OmegaConf.load(dconf_dir / args.dataset_config)
    pconf = OmegaConf.load(pconf_dir / args.preprocessor_config)

    preprocessor = get_preprocessor(pconf.path)
    tr_dl = get_data_loader(dconf, "train", preprocessor, args.batch_size, args.num_workers)
    val_dl = get_data_loader(dconf, "validation", preprocessor, args.batch_size, args.num_workers)

    model = SenCNN(mconf.num_classes, mconf.dropout_rate, preprocessor.vocab)
    runner = Runner(model, args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(runner, train_dataloader=tr_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset_config", default="nsmc.yaml", type=str)
    parser.add_argument("--model_config", default="sencnn.yaml", type=str)
    parser.add_argument("--preprocessor_config", default="mecab_10_32.yaml", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser = Runner.add_runner_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)

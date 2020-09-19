from argparse import ArgumentParser, Namespace
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.net import SenCNN
from src.task.pipeline import DataPipeline
from src.task.runner import ClassificationRunner


def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("conf")
    child_config_dir = parent_config_dir / args.dataset
    model_config_dir = child_config_dir / "model"
    pipeline_config_dir = child_config_dir / "pipeline"
    preprocessor_config_dir = child_config_dir / "preprocessor"
    runner_config_dir = child_config_dir / "runner"

    config = OmegaConf.create()
    model_config = OmegaConf.load(model_config_dir / f"{args.model}.yaml")
    pipeline_config = OmegaConf.load(pipeline_config_dir / f"{args.pipeline}.yaml")
    preprocessor_config = OmegaConf.load(
        preprocessor_config_dir / f"{args.preprocessor}.yaml"
    )
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(
        model=model_config,
        pipeline=pipeline_config,
        preprocessor=preprocessor_config,
        runner=runner_config,
    )
    return config


def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/{args.dataset}", name=args.model, version=args.runner
    )
    return logger


def get_checkpoint_callback(args: Namespace) -> ModelCheckpoint:
    prefix = f"exp/{args.dataset}/{args.model}/{args.runner}/"
    suffix = "{epoch:02d}-{val_acc:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_top_k=1,
        monitor="val_loss",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    config = get_config(args)
    logger = get_tensorboard_logger(args)
    checkpoint_callback = get_checkpoint_callback(args)

    pipeline = DataPipeline(
        pipline_config=config.pipeline, preprocessor_config=config.preprocessor
    )
    model = SenCNN(pipeline.preprocessor.vocab, **config.model.params)
    runner = ClassificationRunner(model, config.runner)

    trainer = Trainer(
        **config.runner.trainer.params,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(runner, datamodule=pipeline)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", default="nsmc", type=str, choices=["nsmc", "trec6"]
    )
    parser.add_argument(
        "--model", default="sencnn", type=str, help="configuration of model"
    )
    parser.add_argument(
        "--pipeline", default="pv00", type=str, help="configuration of pipeline"
    )
    parser.add_argument(
        "--runner", default="rv00", type=str, help="configuration of runner"
    )
    parser.add_argument(
        "--preprocessor",
        default="mecab_5_32",
        type=str,
        choices=["mecab_5_32", "basic_2_32"],
    )
    parser.add_argument("--reproduce", default=False, action="store_true")
    args = parser.parse_args()
    main(args)

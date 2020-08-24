from typing import Union, Tuple, List
from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from src.model.net import SenCNN
from src.task.pipeline import DataPipeline
from src.task.runner import ClassificationTaskRunner


def get_config(args: Namespace) -> DictConfig:
    config_dir = Path("conf")
    model_config_dir = config_dir / "model"
    pipeline_config_dir = config_dir / "pipeline"
    preprocessor_config_dir = config_dir / "preprocessor"
    runner_config_dir = config_dir / "runner"

    config = OmegaConf.create()
    model_config = OmegaConf.load(model_config_dir / f"{args.model}.yaml")
    pipeline_config = OmegaConf.load(pipeline_config_dir / f"{args.pipeline}.yaml")
    preprocessor_config = OmegaConf.load(preprocessor_config_dir / f"{args.preprocessor}.yaml")
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(model=model_config, pipeline=pipeline_config, preprocessor=preprocessor_config, runner=runner_config)
    return config


def get_loggers(args: Namespace) -> Union[LightningLoggerBase, List[LightningLoggerBase]]:
    logger = TensorBoardLogger(save_dir=f"exp{args.pipeline}",
                               name=args.model,
                               version=args.runner.split("_")[-1])
    return logger


def get_callbacks(args: Namespace) -> Union[ModelCheckpoint, Tuple[ModelCheckpoint, List[Callback]]]:
    prefix = f"exp/{args.pipeline}/{args.model}/{args.runner.split('_')[-1]}/"
    suffix = "{epoch:02d}-{val_acc:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_top_k=1,
                                          monitor="val_loss",
                                          save_weights_only=True,
                                          verbose=True)
    return checkpoint_callback


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    config = get_config(args)
    logger = get_loggers(args)
    checkpoint_callback = get_callbacks(args)

    pipeline = DataPipeline(pipline_config=config.pipeline, preprocessor_config=config.preprocessor)
    model = SenCNN(pipeline.preprocessor.vocab, **config.model.params)
    runner = ClassificationTaskRunner(model, config.runner)

    trainer = Trainer(**config.runner.trainer.params,
                      logger=logger,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(runner, datamodule=pipeline)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="sencnn", type=str)
    parser.add_argument("--pipeline", default="nsmc", type=str)
    parser.add_argument("--runner", default="nsmc_v00", type=str)
    parser.add_argument("--preprocessor", default="mecab_5_32", type=str)
    parser.add_argument("--reproduce", default=False, action="store_true")
    args = parser.parse_args([])
    main(args)

import torch
import json

from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from src.model.net import SenCNN
from src.task.runner import ClassificationRunner
from src.task.pipeline import DataPipeline


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
    preprocessor_config = OmegaConf.load(preprocessor_config_dir / f"{args.preprocessor}.yaml")
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(model=model_config, pipeline=pipeline_config, preprocessor=preprocessor_config, runner=runner_config)
    return config


def main(args) -> None:
    seed_everything(42)
    config = get_config(args)

    # prepare dataloader
    pipeline = DataPipeline(pipline_config=config.pipeline, preprocessor_config=config.preprocessor)
    pipeline.setup()

    train_dataloader = pipeline.get_dataloader(pipeline.train_dataset, shuffle=False, drop_last=False,
                                               **pipeline.pipeline_config.dataloader.params)
    val_dataloader = pipeline.get_dataloader(pipeline.val_dataset, shuffle=False, drop_last=False,
                                             **pipeline.pipeline_config.dataloader.params)
    test_dataloader = pipeline.get_dataloader(pipeline.test_dataset, shuffle=False, drop_last=False,
                                              **pipeline.pipeline_config.dataloader.params)

    # restore runner
    model = SenCNN(pipeline.preprocessor.vocab, **config.model.params)
    runner = ClassificationRunner(model, config.runner)
    checkpoint_path = f"exp/{args.dataset}/{args.model}/{args.runner}/{args.checkpoint}.ckpt"
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))

    trainer = Trainer(**config.runner.trainer.params,
                      logger=False,
                      checkpoint_callback=False)
    results = {}
    train_result = trainer.test(runner, test_dataloaders=train_dataloader)
    val_result = trainer.test(runner, test_dataloaders=val_dataloader)
    test_result = trainer.test(runner, test_dataloaders=test_dataloader)
    results.update(checkpoint=args.checkpoint, train=train_result, validation=val_result, test=test_result)

    with open(f"exp/{args.dataset}/{args.model}/{args.runner}/results.json", mode="w") as io:
        json.dump(results, io, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="nsmc", type=str, choices=["nsmc", "trec6"])
    parser.add_argument("--model", default="sencnn", type=str)
    parser.add_argument("--pipeline", default="pv00", type=str)
    parser.add_argument("--runner", default="rv00", type=str)
    parser.add_argument("--preprocessor", default="mecab_5_32", type=str)
    parser.add_argument("--checkpoint", default="epoch=01-val_acc=0.8650", type=str)
    args = parser.parse_args()
    main(args)

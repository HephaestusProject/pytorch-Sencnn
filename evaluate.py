from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
import torch

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


def main(args) -> None:
    seed_everything(42)
    config = get_config(args)

    # prepare dataloader
    pipeline = DataPipeline(
        pipline_config=config.pipeline, preprocessor_config=config.preprocessor
    )

    dataset = pipeline.get_dataset(
        pipeline.dataset_builder,
        config.pipeline.dataset.path.get(args.type),
        pipeline.preprocessor.encode,
    )
    dataloader = pipeline.get_dataloader(
        dataset,
        shuffle=False,
        drop_last=False,
        **pipeline.pipeline_config.dataloader.params,
    )

    # restore runner
    model = SenCNN(pipeline.preprocessor.vocab, **config.model.params)
    runner = ClassificationRunner(model, config.runner)
    checkpoint_path = (
        f"exp/{args.dataset}/{args.model}/{args.runner}/{args.checkpoint}.ckpt"
    )
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))

    trainer = Trainer(
        **config.runner.trainer.params, logger=False, checkpoint_callback=False
    )
    results_path = Path(f"exp/{args.dataset}/{args.model}/{args.runner}/results.json")

    if results_path.exists():
        with open(results_path, mode="r") as io:
            results = json.load(io)

        result = trainer.test(runner, test_dataloaders=dataloader)
        results.update({"checkpoint": args.checkpoint, f"{args.type}": result})

    else:
        results = {}
        result = trainer.test(runner, test_dataloaders=dataloader)
        results.update({"checkpoint": args.checkpoint, f"{args.type}": result})

    with open(
        f"exp/{args.dataset}/{args.model}/{args.runner}/results.json", mode="w"
    ) as io:
        json.dump(results, io, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", default="nsmc", type=str, choices=["nsmc", "trec6"]
    )
    parser.add_argument(
        "--type", default="test", type=str, choices=["train", "validation", "test"]
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
    parser.add_argument("--checkpoint", default="epoch=02-val_acc=0.8653", type=str)
    args = parser.parse_args()
    main(args)

"""
    This script was made by Nick at 19/07/20.
    To implement code for training your model.
"""
import pickle

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from src.model.net import SenCNN
from src.runner.runner import Runner


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    with open(cfg.preprocessor.path, mode="rb") as io:
        preprocessor = pickle.load(io)

    model = SenCNN(cfg.model.num_classes, cfg.model.dropout_ratio, preprocessor.vocab)
    runner = Runner(model, cfg.dataset, cfg.preprocessor, cfg.runner)
    trainer = Trainer(max_epochs=cfg.runner.epochs, early_stop_callback=True, gpus=[0, 1])
    trainer.fit(runner)


if __name__ == "__main__":
    main()

import pickle

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.net import SenCNN
from src.runner.runner import Runner


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    with open(cfg.preprocessor.path, mode="rb") as io:
        preprocessor = pickle.load(io)

    model = SenCNN(cfg.model.num_classes, cfg.model.dropout_ratio, preprocessor.vocab)
    runner = Runner(model, cfg.dataset, cfg.preprocessor, cfg.runner)
    trainer = Trainer(max_epochs=cfg.runner.max_epochs, gpus=cfg.runner.gpus)
    trainer.fit(runner)


if __name__ == "__main__":
    main()

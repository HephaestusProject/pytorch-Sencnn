import pickle

from omegaconf import DictConfig
from typing import Optional, Callable
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from ..utils.corpus import CorpusRegistry
from ..utils.preprocessing import PreProcessor


class DataPipeline(LightningDataModule):
    def __init__(self, pipline_config: DictConfig, preprocessor_config: DictConfig) -> None:
        super(DataPipeline, self).__init__()
        self.pipeline_config = pipline_config
        self.preprocessor_config = preprocessor_config
        self.dataset_builder = CorpusRegistry.get(self.pipeline_config.dataset.type)
        self.preprocessor = DataPipeline.get_preprocessor(self.preprocessor_config)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                          self.pipeline_config.dataset.path.train,
                                                          self.preprocessor.encode)

            self.val_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                        self.pipeline_config.dataset.path.validation,
                                                        self.preprocessor.encode)

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                         self.pipeline_config.dataset.path.test,
                                                         self.preprocessor.encode)

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.train_dataset,
                                           batch_size=self.pipeline_config.dataloader.params.batch_size,
                                           num_workers=self.pipeline_config.dataloader.params.num_workers,
                                           drop_last=True,
                                           shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.val_dataset,
                                           batch_size=self.pipeline_config.dataloader.params.batch_size,
                                           num_workers=self.pipeline_config.dataloader.params.num_workers,
                                           drop_last=False,
                                           shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.test_dataset,
                                           batch_size=self.pipeline_config.dataloader.params.batch_size,
                                           num_workers=self.pipeline_config.dataloader.params.num_workers,
                                           drop_last=False,
                                           shuffle=False)

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, filepath: str, encode_fn: Callable) -> Dataset:
        dataset = dataset_builder(filepath, encode_fn)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool,
                       **kwargs) -> DataLoader:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          **kwargs)

    @classmethod
    def get_preprocessor(cls, preprocessor_config: DictConfig) -> PreProcessor:
        with open(preprocessor_config.path, mode="rb") as io:
            preprocessor = pickle.load(io)
        return preprocessor
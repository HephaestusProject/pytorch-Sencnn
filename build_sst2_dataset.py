import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf
from typing import Union, Dict


def extract_data(string: str) -> Dict[str, Union[int, str]]:
    split_string = string.strip().split(" ")
    label = int(split_string[0])
    sentence = " ".join(split_string[1:])
    return {"label": label, "document": sentence}


config_dir = Path("conf")
dataset_config_dir = config_dir / "dataset"
parent_dir = Path("dataset")
child_dir = parent_dir / "sst2"
dataset_config_path = dataset_config_dir / "sst2.yaml"
dataset_config = OmegaConf.load(dataset_config_path)

# refine and save dataset
for key, path in dataset_config.path.items():

    with open(path, mode="r", encoding="utf-8") as io:
        data = io.readlines()
    data = pd.DataFrame([extract_data(_) for _ in data])
    data.to_csv(path, sep="\t", index=False)

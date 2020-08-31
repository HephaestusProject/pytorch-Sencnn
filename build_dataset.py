import argparse
import pandas as pd

from argparse import Namespace
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


def create_dataset(filepath: Path) -> pd.DataFrame:
    with open(filepath, mode="rb") as io:
        list_of_sentences = io.readlines()
        data = []

    for sentence in list_of_sentences:
        try:
            decoded_sentence = sentence.strip().decode("utf-8")
            label = int(decoded_sentence[0])
            document = decoded_sentence[2:]
            data.append({"document": document, "label": label})
        except UnicodeDecodeError:
            continue

    return pd.DataFrame(data)


def main(args: Namespace) -> None:
    parent_config_dir = Path("conf")
    child_config_dir = parent_config_dir / args.dataset

    parent_dataset_dir = Path("dataset")
    child_dataset_dir = parent_dataset_dir / args.dataset

    pipeline_config_dir = child_config_dir / "pipeline"
    pipeline_config_path = pipeline_config_dir / f"{args.pipeline}.yaml"
    pipeline_config = OmegaConf.load(pipeline_config_path)

    if args.dataset == "nsmc":
        # loading dataset
        dataset = pd.read_csv(pipeline_config.dataset.path.train, sep="\t").loc[
            :, ["document", "label"]
        ]
        dataset = dataset.loc[dataset["document"].isna().apply(lambda elm: not elm), :]
        train, validation = train_test_split(
            dataset, test_size=args.valid_ratio, random_state=args.seed
        )
        test = pd.read_csv(pipeline_config.dataset.path.test, sep="\t").loc[:, ["document", "label"]]
        test = test.loc[test["document"].isna().apply(lambda elm: not elm), :]

    elif args.dataset == "trec6":

        dataset = create_dataset(pipeline_config.dataset.path.train)
        dataset = dataset.loc[dataset["document"].isna().apply(lambda elm: not elm), :]

        train, validation = train_test_split(
            dataset, test_size=args.valid_ratio, random_state=args.seed
        )

        test = create_dataset(pipeline_config.dataset.path.test)
        test = test.loc[dataset["document"].isna().apply(lambda elm: not elm), :]

    path_dict = {
        "train": str(child_dataset_dir / "train.txt"),
        "validation": str(child_dataset_dir / "validation.txt"),
        "test": str(child_dataset_dir / "test.txt")
    }

    pipeline_config.dataset.path.update(path_dict)
    OmegaConf.save(pipeline_config, pipeline_config_path)

    train.to_csv(pipeline_config.dataset.path.train, sep="\t", index=False)
    validation.to_csv(pipeline_config.dataset.path.validation, sep="\t", index=False)
    test.to_csv(pipeline_config.dataset.path.test, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsmc", choices=["nsmc", "trec6"])
    parser.add_argument("--pipeline", type=str, default="pv00")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


def create_dataset(filepath):
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


def main(args):
    config_dir = Path("conf")
    dataset_config_dir = config_dir / "dataset"
    parent_dir = Path("dataset")
    child_dir = parent_dir / args.dataset
    dataset_config_path = dataset_config_dir / f"{args.dataset}.yaml"
    dataset_config = OmegaConf.load(dataset_config_path)

    if args.dataset == "nsmc":
        # loading dataset
        dataset = pd.read_csv(dataset_config.path.train, sep="\t").loc[
            :, ["document", "label"]
        ]
        dataset = dataset.loc[dataset["document"].isna().apply(lambda elm: not elm), :]
        train, validation = train_test_split(
            dataset, test_size=args.valid_ratio, random_state=args.seed
        )
        test = pd.read_csv(dataset_config.path.test, sep="\t").loc[:, ["document", "label"]]
        test = test.loc[test["document"].isna().apply(lambda elm: not elm), :]

        path_dict = {
            "train": str(child_dir / "train.txt"),
            "validation": str(child_dir / "validation.txt"),
            "test": str(child_dir / "test.txt")
        }

        dataset_config.path.update(path_dict)
        OmegaConf.save(dataset_config, dataset_config_path)

        train.to_csv(dataset_config.path.train, sep="\t", index=False)
        validation.to_csv(dataset_config.path.validation, sep="\t", index=False)
        test.to_csv(dataset_config.path.test, sep="\t", index=False)

    elif args.dataset == "trec6":

        config_dir = Path("conf")
        dataset_config_dir = config_dir / "dataset"
        parent_dir = Path("dataset")
        child_dir = parent_dir / "trec6"
        dataset_config_path = dataset_config_dir / "trec6.yaml"
        dataset_config = OmegaConf.load(dataset_config_path)
        dataset = create_dataset(dataset_config.path.train)
        dataset = dataset.loc[dataset["document"].isna().apply(lambda elm: not elm), :]

        train, validation = train_test_split(
            dataset, test_size=args.valid_ratio, random_state=args.seed
        )

        test = create_dataset(dataset_config.path.test)
        test = test.loc[dataset["document"].isna().apply(lambda elm: not elm), :]

        path_dict = {
            "train": str(child_dir / "train.txt"),
            "validation": str(child_dir / "validation.txt"),
            "test": str(child_dir / "test.txt")
        }
        dataset_config.path.update(path_dict)
        OmegaConf.save(dataset_config, dataset_config_path)

        train.to_csv(dataset_config.path.train, sep="\t", index=False)
        validation.to_csv(dataset_config.path.validation, sep="\t", index=False)
        test.to_csv(dataset_config.path.test, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsmc", choices=["nsmc", "trec6"])
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

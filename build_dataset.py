import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


def main(args):
    conf_dir = Path("conf")
    dconf_dir = conf_dir / "dataset"
    parent_dir = Path("dataset")
    child_dir = parent_dir / args.dataset
    dconf_path = dconf_dir / f"{args.dataset}.yaml"
    dconf = OmegaConf.load(dconf_path)

    # loading dataset
    dataset = pd.read_csv(dconf.path.train, sep="\t").loc[
        :, ["document", "label"]
    ]
    dataset = dataset.loc[dataset["document"].isna().apply(lambda elm: not elm), :]
    train, validation = train_test_split(
        dataset, test_size=args.valid_ratio, random_state=args.seed
    )
    test = pd.read_csv(dconf.path.test, sep="\t").loc[:, ["document", "label"]]
    test = test.loc[test["document"].isna().apply(lambda elm: not elm), :]

    path_dict = {
        "train": str(child_dir / "train.txt"),
        "validation": str(child_dir / "validation.txt"),
        "test": str(child_dir / "test.txt")
    }

    dconf.path.update(path_dict)
    OmegaConf.save(dconf, dconf_path)

    train.to_csv(dconf.path.train, sep="\t", index=False)
    validation.to_csv(dconf.path.validation, sep="\t", index=False)
    test.to_csv(dconf.path.test, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsmc")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    main(args)

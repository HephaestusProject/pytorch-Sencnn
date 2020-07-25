import argparse
import itertools
import pickle
from collections import Counter
from pathlib import Path

import gluonnlp as nlp
import pandas as pd
from omegaconf import OmegaConf

from src.utils.preprocessing import PadSequence, PreProcessor
from src.utils.tokenization import TOKENIZATION_FACTORY
from src.utils.vocab import Vocab


def main(args):
    conf_dataset_dir = Path("conf/dataset")
    conf_preprocessor_dir = Path("conf/preprocessor")
    dataset_conf = OmegaConf.load(conf_dataset_dir / f"{args.dataset}.yaml")
    preprocessor_conf = OmegaConf.load(
        conf_preprocessor_dir / f"{args.preprocessor}.yaml"
    )
    parent_dir = Path("preprocessor")
    child_dir = parent_dir / args.preprocessor
    # loading dataset
    train = pd.read_csv(dataset_conf.refinement.train, sep="\t").loc[
        :, ["document", "label"]
    ]

    # extracting morph in sentences
    tokenize_fn = TOKENIZATION_FACTORY[preprocessor_conf.type]
    list_of_tokens = train["document"].apply(tokenize_fn).tolist()

    # generating the vocab
    token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
    intermediate_vocab = nlp.Vocab(
        counter=token_counter,
        min_freq=preprocessor_conf.min_freq,
        bos_token=None,
        eos_token=None,
    )

    # connecting SISG embedding with vocab
    embedding_source = nlp.embedding.create("fasttext", source="wiki.ko")
    intermediate_vocab.set_embedding(embedding_source)
    embedding = intermediate_vocab.embedding.idx_to_vec.asnumpy()

    vocab = Vocab(
        intermediate_vocab.idx_to_token,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token=None,
        eos_token=None,
        cls_token=None,
        sep_token=None,
        mask_token=None,
    )
    vocab.embedding = embedding

    preprocessor = PreProcessor(
        vocab,
        tokenize_fn=tokenize_fn,
        pad_fn=PadSequence(length=preprocessor_conf.max_len, pad_val=vocab.pad_token),
    )

    # saving vocab
    if not child_dir.exists():
        child_dir.mkdir(parents=True)

    with open(child_dir / "preprocessor.pkl", mode="wb") as io:
        pickle.dump(preprocessor, io)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsmc")
    parser.add_argument("--preprocessor", type=str, default="mecab_10_32")

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
    conf_dir = Path("conf")
    dconf_dir = conf_dir / "dataset"
    pconf_dir = conf_dir / "preprocessor"
    dconf = OmegaConf.load(dconf_dir / f"{args.dataset}.yaml")
    pconf_path = pconf_dir / f"{args.preprocessor}.yaml"
    pconf = OmegaConf.load(pconf_path)

    parent_dir = Path("preprocessor")
    child_dir = parent_dir / args.preprocessor
    # loading dataset
    train = pd.read_csv(dconf.path.train, sep="\t").loc[
        :, ["document", "label"]
    ]

    # extracting morph in sentences
    tokenize_fn = TOKENIZATION_FACTORY[pconf.params.tokenizer]
    list_of_tokens = train["document"].apply(tokenize_fn).tolist()

    # generating the vocab
    token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
    intermediate_vocab = nlp.Vocab(
        counter=token_counter,
        min_freq=pconf.params.min_freq,
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
        pad_fn=PadSequence(length=pconf.params.max_len, pad_val=vocab.pad_token),
    )

    # saving vocab
    if not child_dir.exists():
        child_dir.mkdir(parents=True)

    path_dict = {"path": str(child_dir / "preprocessor.pkl")}
    pconf.update(path_dict)
    OmegaConf.save(pconf, pconf_path)

    with open(pconf.path, mode="wb") as io:
        pickle.dump(preprocessor, io)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsmc")
    parser.add_argument("--preprocessor", type=str, default="mecab_10_32")
    args = parser.parse_args()
    main(args)

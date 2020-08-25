import argparse
import itertools
import pickle
import gluonnlp as nlp
import pandas as pd
import numpy as np

from argparse import Namespace
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf
from src.utils.preprocessing import PadSequence, PreProcessor
from src.utils.tokenization import TokenizationRegistry
from src.utils.vocab import Vocab


def main(args: Namespace) -> None:
    config_dir = Path("conf")

    task_config_dir = config_dir / args.task
    task_pipeline_config_dir = task_config_dir / "pipeline"
    task_preprocessor_config_dir = task_config_dir / "preprocessor"

    task_pipeline_config_path = task_pipeline_config_dir /  f"{args.pipeline}.yaml"
    task_pipeline_config = OmegaConf.load(task_pipeline_config_path)

    # task_pipeline_config = OmegaConf.load(task_dataset_dir / f"{args.task}.yaml")
    task_preprocessor_config_path = task_preprocessor_config_dir / f"{args.preprocessor}.yaml"
    task_preprocessor_config = OmegaConf.load(task_preprocessor_config_path)


    # loading dataset
    train = pd.read_csv(task_pipeline_config.dataset.path.train, sep="\t").loc[
        :, ["document", "label"]
    ]

    # extracting morph in sentences
    tokenize_fn = TokenizationRegistry[task_preprocessor_config.params.tokenizer]
    list_of_tokens = train["document"].apply(tokenize_fn).tolist()

    # generating the vocab
    token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
    intermediate_vocab = nlp.Vocab(
        counter=token_counter,
        min_freq=task_preprocessor_config.params.min_freq,
        bos_token=None,
        eos_token=None,
    )

    # connecting SISG embedding with vocab
    embedding_source = nlp.embedding.create(task_preprocessor_config.params.embedding_name,
                                            source=task_preprocessor_config.params.embedding_source)

    intermediate_vocab.set_embedding(embedding_source)
    embedding = intermediate_vocab.embedding.idx_to_vec.asnumpy()

    # init vector
    zero_vector_indices = np.delete(np.where(embedding.sum(axis=-1) == 0)[0],
                                    [intermediate_vocab.to_indices(intermediate_vocab.unknown_token),
                                    intermediate_vocab.to_indices(intermediate_vocab.padding_token)])
    non_zero_vector_indices = np.delete(np.arange(0, len(intermediate_vocab)),
                              np.append(zero_vector_indices,
                                        [intermediate_vocab.to_indices(intermediate_vocab.unknown_token),
                                         intermediate_vocab.to_indices(intermediate_vocab.padding_token)]))
    vars_of_dim = np.var(embedding[non_zero_vector_indices], axis=0)

    initialized_vectors = np.random.uniform(-vars_of_dim, vars_of_dim, size=embedding[zero_vector_indices].shape)
    embedding[zero_vector_indices] = initialized_vectors

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
        pad_fn=PadSequence(length=task_preprocessor_config.params.max_len, pad_val=vocab.pad_token),
    )


    # saving vocab
    preprocessor_dir = Path("preprocessor")
    task_preprocessor_dir = preprocessor_dir / args.task / args.preprocessor

    if not task_preprocessor_dir.exists():
        task_preprocessor_dir.mkdir(parents=True)

    path_dict = {"path": str(task_preprocessor_dir / "preprocessor.pkl")}
    task_preprocessor_config.update(path_dict)
    OmegaConf.save(task_preprocessor_config, task_preprocessor_config_path)

    with open(task_preprocessor_config.path, mode="wb") as io:
        pickle.dump(preprocessor, io)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="nsmc", choices=["nsmc", "trec6"])
    parser.add_argument("--pipeline", type=str, default="pv00")
    parser.add_argument("--preprocessor", type=str, default="mecab_5_32")
    args = parser.parse_args()
    main(args)

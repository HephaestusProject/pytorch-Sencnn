import itertools
import os
from pathlib import Path

import pytest

from src.utils.tokenization import mecab_tokenize
from src.utils.vocab import Vocab


def setup_function():
    test_dir = Path.cwd() / "tests"
    nsmc_samples_path = test_dir / "nsmc_samples.txt"
    trec6_samples_path = test_dir / "trec6_samples.txt"

    list_of_nsmc_samples = [
        "document\tlabel\n",
        "재미없을거 같네요 하하하\t0\n정말 좋네요 이 영화 강추!!!!\t1",
    ]
    list_of_trec6_samples = [
        "document\tlabel\n",
        "How big is a quart ?\t5\n",
        "How old is Jeremy Piven ?\t5",
    ]

    with open(nsmc_samples_path, mode="w", encoding="utf-8") as io:
        for nsmc_sample in list_of_nsmc_samples:
            io.write(nsmc_sample)

    with open(trec6_samples_path, mode="w", encoding="utf-8") as io:
        for trec6_sample in list_of_trec6_samples:
            io.write(trec6_sample)


def teardown_function():
    test_dir = Path.cwd() / "tests"
    nsmc_samples_path = test_dir / "nsmc_samples.txt"
    trec6_samples_path = test_dir / "trec6_samples.txt"

    os.remove(nsmc_samples_path)
    os.remove(trec6_samples_path)


@pytest.fixture
def filepath_of_resources():
    test_dir = Path.cwd() / "tests"
    nsmc_samples_path = test_dir / "nsmc_samples.txt"
    trec6_samples_path = test_dir / "trec6_samples.txt"

    class ResourcePaths:
        pass

    resource_paths = ResourcePaths
    resource_paths.nsmc_samples_path = nsmc_samples_path
    resource_paths.trec6_samples_path = trec6_samples_path
    return resource_paths


def test_Vocab(filepath_of_resources):
    list_of_nsmc_samples = []
    with open(
        filepath_of_resources.nsmc_samples_path, mode="r", encoding="utf-8"
    ) as nsmc_samples:
        for idx, nsmc_sample in enumerate(nsmc_samples):
            if idx == 0:
                continue
            list_of_nsmc_samples.append(nsmc_sample.strip().split("\t")[0])
    footprint = itertools.chain.from_iterable(
        [mecab_tokenize(nsmc_sample) for nsmc_sample in list_of_nsmc_samples]
    )
    set_of_tokens = set()

    for token in footprint:
        set_of_tokens.add(token)
    list_of_tokens = sorted(set_of_tokens)

    vocab = Vocab(
        list_of_tokens=list_of_tokens,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token=None,
        eos_token=None,
        cls_token=None,
        sep_token=None,
        mask_token=None,
    )

    assert vocab.pad_token == "<pad>"
    assert vocab.unk_token == "<unk>"
    assert vocab.idx_to_token == [
        "<unk>",
        "<pad>",
        "!",
        "!!!",
        "강추",
        "같",
        "거",
        "네요",
        "영화",
        "을",
        "이",
        "재미없",
        "정말",
        "좋",
        "하하하",
    ]

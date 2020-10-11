import itertools

import pytest

from src.utils.preprocessing import PadSequence, PreProcessor
from src.utils.tokenization import mecab_tokenize
from src.utils.vocab import Vocab


@pytest.fixture(scope="module")
def sample_Vocab(filepath_of_each_samples):
    list_of_nsmc_samples = []
    with open(
        filepath_of_each_samples.nsmc_samples_filepath, mode="r", encoding="utf-8"
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
    return vocab


@pytest.fixture(scope="module")
def sample_PadSequence(sample_Vocab):
    padder = PadSequence(length=8, pad_val=sample_Vocab.pad_token)
    return padder


def test_PadSequence():
    pad_sequence = PadSequence(length=8, pad_val=0)
    assert pad_sequence([1, 1, 1]) == [1, 1, 1, 0, 0, 0, 0, 0]


def test_PreProcessor(sample_Vocab, sample_PadSequence):
    preprocessor = PreProcessor(
        vocab=sample_Vocab, tokenize_fn=mecab_tokenize, pad_fn=sample_PadSequence
    )

    assert preprocessor.tokenize("안녕하세요.") == ["안녕", "하", "세요", "."]
    assert preprocessor.pad(preprocessor.tokenize("안녕하세요.")) == [
        "안녕",
        "하",
        "세요",
        ".",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
    ]
    assert preprocessor.encode("안녕하세요.") == [0, 0, 0, 0, 1, 1, 1, 1]

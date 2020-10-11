import itertools

from src.utils.tokenization import mecab_tokenize
from src.utils.vocab import Vocab


def test_Vocab(filepath_of_each_samples):
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

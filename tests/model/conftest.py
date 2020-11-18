import numpy as np
import pytest
import torch

from src.utils.vocab import Vocab


# 각각의 module에서 공통적으로 사용하는 fixture의 경우 conftest.py에 정의하고 사용한다.
@pytest.fixture(scope="package")
def sample_inputs():
    sequence_of_unk_tokens = [0] * 5  # integer of unk token = 0
    inputs = torch.tensor([sequence_of_unk_tokens]).reshape(
        -1, len(sequence_of_unk_tokens)
    )
    return inputs


@pytest.fixture(scope="package")
def sample_vocab():
    vocab = Vocab(
        list_of_tokens=None,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token=None,
        bos_token=None,
        cls_token=None,
        sep_token=None,
        mask_token=None,
        reserved_tokens=None,
        token_to_idx=None,
    )

    embedding = np.repeat(0, 1500).reshape(-1, 300).astype(np.float32)
    vocab.embedding = embedding
    return vocab

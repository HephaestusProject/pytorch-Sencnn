import pytest

from src.utils.preprocessing import PadSequence, PreProcessor
from src.utils.tokenization import basic_tokenize

# @pytest.fixture()
# def vocab():
#
#
#
# def test_PadSequence():
#     pad_sequence = PadSequence(length=8, pad_val=0)
#     assert pad_sequence([1, 1, 1]) == [1, 1, 1, 0, 0, 0, 0, 0]
#
#
# def test_PreProcessor():

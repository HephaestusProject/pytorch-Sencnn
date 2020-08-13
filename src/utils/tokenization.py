import re

from konlpy.tag import Mecab, Okt
from typing import List

mecab_tokenize = Mecab().morphs
okt_tokenize = Okt().morphs


def basic_tokenize(sentence: str) -> List[str]:

    def clean_str_sst(string):
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    return clean_str_sst(sentence).split(" ")


TokenizationRegistry = {"mecab": mecab_tokenize, "okt": okt_tokenize, "basic": basic_tokenize}

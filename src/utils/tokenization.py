import re

from konlpy.tag import Mecab, Okt
from typing import List

mecab_tokenize = Mecab().morphs
okt_tokenize = Okt().morphs


def basic_tokenize(sentence: str) -> List[str]:
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    return clean_str(sentence).split(" ")


TokenizationRegistry = {"mecab": mecab_tokenize, "okt": okt_tokenize, "basic": basic_tokenize}

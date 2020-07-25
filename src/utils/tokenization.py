from konlpy.tag import Mecab, Okt

mecab_tokenize = Mecab().morphs
okt_tokenize = Okt().morphs

TOKENIZATION_FACTORY = {"mecab": mecab_tokenize, "okt": okt_tokenize}

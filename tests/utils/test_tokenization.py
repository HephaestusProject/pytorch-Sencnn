from src.utils.tokenization import basic_tokenize, mecab_tokenize


def test_mecab_tokenize():
    assert mecab_tokenize("재미없을거 같네요 하하하") == ["재미없", "을", "거", "같", "네요", "하하하"]
    assert mecab_tokenize("정말 좋네요 이 영화 강추!!!!") == [
        "정말",
        "좋",
        "네요",
        "이",
        "영화",
        "강추",
        "!",
        "!!!",
    ]


def test_basic_tokenize():
    assert basic_tokenize("How big is a quart ?") == [
        "how",
        "big",
        "is",
        "a",
        "quart",
        "?",
    ]
    assert basic_tokenize("How old is Jeremy Piven ?") == [
        "how",
        "old",
        "is",
        "jeremy",
        "piven",
        "?",
    ]

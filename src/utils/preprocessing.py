from typing import Callable, List, Union

from .vocab import Vocab


class PreProcessor:
    """PreProcessor class"""

    def __init__(
        self,
        vocab: Vocab,
        tokenize_fn: Callable[[str], List[str]],
        pad_fn: Callable[[List[int]], List[int]] = None,
    ) -> None:
        """Instantiating PreProcessor class

        Args:
            vocab (Vocab): the instance of Vocab created from specific split_fn
            tokenize_fn (Callable): a function that can act as a splitter
            pad_fn (Callable): a function that can act as a padder
        """
        self._vocab = vocab
        self._tokenize = tokenize_fn
        self._pad = pad_fn

    def tokenize(self, string: str) -> List[str]:
        list_of_tokens = self._tokenize(string)
        return list_of_tokens

    def pad(self, list_of_tokens: List[str]):
        list_of_tokens = self._pad(list_of_tokens) if self._pad else list_of_tokens
        return list_of_tokens

    def encode(self, string) -> List[int]:
        list_of_indices = self._vocab.to_indices(
            self.pad(self.tokenize(string))
        )
        return list_of_indices

    @property
    def vocab(self):
        return self._vocab


class PadSequence:
    """PadSequence class"""

    def __init__(
        self, length: int, pad_val: Union[int, str], clip: bool = True
    ) -> None:
        """Instantiating PadSequence class
        Args:
            length (int): the maximum length to pad/clip the sequence
            pad_val (int): the pad value
            clip (bool): whether to clip the length, if sample length is longer than maximum length
        """
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample):
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[: self._length]
            else:
                return sample
        else:
            return sample + [self._pad_val for _ in range(self._length - sample_length)]
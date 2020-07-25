import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Callable


class NSMCCorpus(Dataset):
    """NSMCCorpus class"""

    def __init__(self, filepath: str, encode_fn: Callable[[str], List[int]]) -> None:
        """Instantiating NSMCCorpus class

        Args:

            filepath (str): filepath
            encode_fn (Callable): a function that can act as a encoder
        """
        self._corpus = pd.read_csv(filepath, sep="\t").loc[:, ["document", "label"]]
        self._encode_fn = encode_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens2indices = torch.tensor(
            self._encode_fn(self._corpus.iloc[idx]["document"])
        )
        label = torch.tensor(self._corpus.iloc[idx]["label"])
        return tokens2indices, label


CORPUS_FACTORY = {"nsmc": NSMCCorpus}

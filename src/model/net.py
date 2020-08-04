import torch
import torch.nn as nn
from .ops import MultiChannelEmbedding, ConvolutionLayer, MaxOverTimePooling
from ..utils.vocab import Vocab


class SenCNN(nn.Module):
    """SenCNN class"""

    def __init__(self, vocab: Vocab, num_classes: int, dropout_rate: float) -> None:
        """Instantiating SenCNN class

        Args:
            vocab (Vocab): the instance of Vocab
            num_classes (int): the number of classes
            dropout_rate (float): ratio of dropout
        """
        super(SenCNN, self).__init__()
        self._embed = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(vocab.embedding.shape[-1], 300)
        self._pooling = MaxOverTimePooling()
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(300, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._embed(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)
        return score

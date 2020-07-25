import torch
import torch.nn as nn
from .ops import MultiChannelEmbedding, ConvolutionLayer, MaxOverTimePooling
from ..utils.vocab import Vocab


class SenCNN(nn.Module):
    """SenCNN class"""

    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        """Instantiating SenCNN class

        Args:
            num_classes (int): the number of classes
            vocab (Vocab): the instance of Vocab
        """
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(self._embedding._static.size()[1], 300)
        self._pooling = MaxOverTimePooling()
        self._dropout = nn.Dropout()
        self._fc = nn.Linear(300, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._embedding(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)
        return score

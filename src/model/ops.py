import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.vocab import Vocab
from typing import Tuple


class MultiChannelEmbedding(nn.Module):
    """MultiChannelEmbedding class"""

    def __init__(self, vocab: Vocab) -> None:
        """Instantiating MultiChannelEmbedding class

        Args:
            vocab (Vocab): the instance of Vocab
        """
        super(MultiChannelEmbedding, self).__init__()
        self._static = nn.Embedding.from_pretrained(
            torch.from_numpy(vocab.embedding),
            freeze=True,
            padding_idx=vocab.to_indices(vocab.pad_token),
        )
        self._non_static = nn.Embedding.from_pretrained(
            torch.from_numpy(vocab.embedding),
            freeze=False,
            padding_idx=vocab.to_indices(vocab.pad_token),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        static = self._static(x).permute(0, 2, 1)
        non_static = self._non_static(x).permute(0, 2, 1)
        return static, non_static


class ConvolutionLayer(nn.Module):
    """ConvolutionLayer class"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Instantiating ConvolutionLayer class

        Args:
            in_channels (int): the number of channels from input feature map
            out_channels (int): the number of channels from output feature map
        """
        super(ConvolutionLayer, self).__init__()
        self._tri_gram_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels // 3, kernel_size=3
        )
        self._tetra_gram_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels // 3, kernel_size=4
        )
        self._penta_gram_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels // 3, kernel_size=5
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        static, non_static = x
        tri_feature_map = F.relu(self._tri_gram_conv(static)) + F.relu(self._tri_gram_conv(non_static))
        tetra_feature_map = F.relu(self._tetra_gram_conv(static)) + F.relu(
            self._tetra_gram_conv(non_static)
        )
        penta_feature_map = F.relu(self._penta_gram_conv(static)) + F.relu(
            self._penta_gram_conv(non_static)
        )
        return tri_feature_map, tetra_feature_map, penta_feature_map


class MaxOverTimePooling(nn.Module):
    """MaxOverTimePooling class"""

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        tri_feature_map, tetra_feature_map, penta_feature_map = x
        fmap = torch.cat(
            [
                tri_feature_map.max(dim=-1)[0],
                tetra_feature_map.max(dim=-1)[0],
                penta_feature_map.max(dim=-1)[0],
            ],
            dim=-1,
        )
        return fmap

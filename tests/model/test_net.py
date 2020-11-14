import torch

from src.model.net import SenCNN


def test_SenCNN(sample_vocab, sample_inputs):
    model = SenCNN(vocab=sample_vocab, num_classes=2, dropout_rate=0)
    output = model(sample_inputs)

    assert output.shape == torch.Size([1, 2])

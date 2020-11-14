import numpy as np
import torch

from src.model.ops import ConvolutionLayer, MaxOverTimePooling, MultiChannelEmbedding


def test_MultiChannelEmbedding(sample_inputs, sample_vocab):
    embedding = torch.tensor(np.repeat(0, 1500).astype(np.float32)).reshape(
        -1, 300, sample_inputs.shape[-1]
    )
    ops = MultiChannelEmbedding(vocab=sample_vocab)
    static, non_static = ops(sample_inputs)
    assert (static.numpy() == embedding.numpy()).all()
    assert (non_static.detach().numpy() == embedding.numpy()).all()
    assert static.grad is None


def test_ConvolutionLayer(sample_inputs, sample_vocab):
    ops_0 = MultiChannelEmbedding(vocab=sample_vocab)
    ops_0_output = ops_0(sample_inputs)
    ops_1 = ConvolutionLayer(sample_vocab.embedding.shape[-1], 300)
    outputs = ops_1(ops_0_output)

    assert outputs[0].shape == torch.Size([1, 100, 3])
    assert outputs[1].shape == torch.Size([1, 100, 2])
    assert outputs[2].shape == torch.Size([1, 100, 1])


def test_MaxOverTimePooling(sample_inputs, sample_vocab):
    ops_0 = MultiChannelEmbedding(vocab=sample_vocab)
    ops_0_output = ops_0(sample_inputs)

    ops_1 = ConvolutionLayer(sample_vocab.embedding.shape[-1], 300)
    ops_1_outputs = ops_1(ops_0_output)

    ops_2 = MaxOverTimePooling()
    outputs = ops_2(ops_1_outputs)

    assert outputs.shape == torch.Size([1, 300])

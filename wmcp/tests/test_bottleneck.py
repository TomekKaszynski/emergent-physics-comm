"""Tests for wmcp.bottleneck."""

import torch
import pytest
from wmcp.bottleneck import GumbelSoftmaxBottleneck


def test_bottleneck_output_shape():
    b = GumbelSoftmaxBottleneck(vocab_size=3, n_heads=2, hidden_dim=64)
    b.eval()
    h = torch.randn(8, 64)
    msg, logits = b(h)
    assert msg.shape == (8, 6)  # 2 heads * 3 vocab
    assert len(logits) == 2


def test_bottleneck_one_hot():
    b = GumbelSoftmaxBottleneck(vocab_size=5, n_heads=3, hidden_dim=32)
    b.eval()
    h = torch.randn(4, 32)
    msg, _ = b(h)
    reshaped = msg.view(4, 3, 5)
    for i in range(4):
        for j in range(3):
            assert abs(reshaped[i, j].sum().item() - 1.0) < 1e-5


def test_bottleneck_decode_tokens():
    b = GumbelSoftmaxBottleneck(vocab_size=3, n_heads=2, hidden_dim=64)
    b.eval()
    h = torch.randn(8, 64)
    msg, _ = b(h)
    tokens = b.decode_tokens(msg)
    assert tokens.shape == (8, 2)
    assert tokens.min() >= 0
    assert tokens.max() <= 2


def test_bottleneck_training_mode():
    b = GumbelSoftmaxBottleneck(vocab_size=3, n_heads=2, hidden_dim=64)
    b.train()
    h = torch.randn(4, 64, requires_grad=True)
    msg, _ = b(h, tau=1.0, hard=True)
    loss = msg.sum()
    loss.backward()
    assert h.grad is not None


def test_bottleneck_different_tau():
    b = GumbelSoftmaxBottleneck(vocab_size=3, n_heads=2, hidden_dim=64)
    b.train()
    h = torch.randn(4, 64)
    msg_high, _ = b(h, tau=10.0, hard=False)
    msg_low, _ = b(h, tau=0.1, hard=False)
    # Higher tau → more uniform, lower tau → more peaked
    assert msg_high.std() < msg_low.std()

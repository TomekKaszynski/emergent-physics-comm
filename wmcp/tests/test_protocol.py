"""Tests for wmcp.protocol."""

import torch
import pytest
from wmcp.protocol import Protocol, AgentSender, Receiver


def test_protocol_init():
    p = Protocol([(1024, 4), (384, 4)], vocab_size=3, n_heads=2)
    assert p.n_agents == 2
    assert p.vocab_size == 3
    assert p.n_heads == 2
    assert p.msg_dim == 2 * 2 * 3


def test_protocol_encode():
    p = Protocol([(1024, 4), (384, 4)], vocab_size=3, n_heads=2)
    p.eval()
    views = [torch.randn(8, 4, 1024), torch.randn(8, 4, 384)]
    msg, logits = p.encode(views)
    assert msg.shape == (8, 12)  # 2 agents * 2 heads * 3 vocab
    assert len(logits) == 4  # 2 agents * 2 heads


def test_protocol_communicate():
    p = Protocol([(1024, 4), (384, 4)], vocab_size=3, n_heads=2)
    p.eval()
    va = [torch.randn(4, 4, 1024), torch.randn(4, 4, 384)]
    vb = [torch.randn(4, 4, 1024), torch.randn(4, 4, 384)]
    pred = p.communicate(va, vb)
    assert pred.shape == (4,)


def test_protocol_extract_tokens():
    p = Protocol([(1024, 4), (384, 4)], vocab_size=3, n_heads=2)
    views = [torch.randn(8, 4, 1024), torch.randn(8, 4, 384)]
    tokens = p.extract_tokens(views)
    assert tokens.shape == (8, 4)
    assert tokens.min() >= 0
    assert tokens.max() <= 2


def test_protocol_info():
    p = Protocol([(1024, 4), (384, 4)], vocab_size=3)
    info = p.info
    assert info["wmcp_version"] == "0.1.0"
    assert info["K"] == 3
    assert info["n_agents"] == 2


def test_protocol_reset_receivers():
    p = Protocol([(1024, 4)], vocab_size=3, n_receivers=2)
    old_params = list(p.receivers[0].parameters())[0].clone()
    p.reset_receivers()
    new_params = list(p.receivers[0].parameters())[0]
    assert not torch.equal(old_params, new_params)


def test_agent_sender():
    s = AgentSender(input_dim=768, hidden_dim=64, vocab_size=5, n_heads=3, n_frames=2)
    s.eval()
    x = torch.randn(4, 2, 768)
    msg, logits = s(x)
    assert msg.shape == (4, 15)  # 3 heads * 5 vocab
    assert len(logits) == 3


def test_receiver():
    r = Receiver(msg_dim=12, hidden_dim=64)
    ma = torch.randn(4, 12)
    mb = torch.randn(4, 12)
    out = r(ma, mb)
    assert out.shape == (4,)

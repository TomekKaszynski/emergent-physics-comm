"""Tests for wmcp.compliance — using synthetic data."""

import numpy as np
import torch
import pytest
from wmcp.protocol import Protocol
from wmcp.compliance import validate_protocol


def test_validate_protocol_runs():
    """Compliance suite runs without errors on random data."""
    p = Protocol([(64, 2), (32, 2)], hidden_dim=16, vocab_size=3, n_heads=2, n_receivers=1)
    n = 50
    views = [torch.randn(n, 2, 64), torch.randn(n, 2, 32)]
    mass = np.random.RandomState(42).uniform(1, 100, n)
    names = [f"obj_{i % 10}" for i in range(n)]

    result = validate_protocol(p, views, mass, names)
    assert "tests" in result
    assert "n_pass" in result
    assert result["n_total"] == 9
    # On random data, not all tests will pass, but it should run
    assert isinstance(result["all_pass"], bool)


def test_validate_returns_details():
    p = Protocol([(64, 2)], hidden_dim=16, vocab_size=3, n_heads=2, n_receivers=1)
    n = 30
    views = [torch.randn(n, 2, 64)]
    mass = np.linspace(1, 100, n)
    names = [f"obj_{i % 6}" for i in range(n)]

    result = validate_protocol(p, views, mass, names)
    for test in result["tests"]:
        assert "name" in test
        assert "passed" in test
        assert "detail" in test

"""Tests for wmcp.metrics."""

import numpy as np
import pytest
from wmcp.metrics import (mutual_information, compute_posdis, compute_topsim,
                           compute_bosdis, compute_mi_matrix, make_attributes)


def test_mutual_information_identical():
    x = np.array([0, 1, 2, 0, 1, 2])
    mi = mutual_information(x, x)
    assert mi > 0


def test_mutual_information_independent():
    rng = np.random.RandomState(42)
    x = rng.randint(0, 3, 10000)
    y = rng.randint(0, 3, 10000)
    mi = mutual_information(x, y)
    assert mi < 0.05  # Near zero for independent


def test_compute_posdis_perfect():
    # Position 0 encodes attribute 0, position 1 encodes attribute 1
    tokens = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                       [2, 0], [2, 1], [2, 2]])
    attrs = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                      [2, 0], [2, 1], [2, 2]])
    posdis, mi, ent = compute_posdis(tokens, attrs, vocab_size=3)
    assert posdis > 0.9  # Should be close to 1.0


def test_compute_posdis_random():
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 3, (100, 4))
    attrs = rng.randint(0, 3, (100, 2))
    posdis, mi, ent = compute_posdis(tokens, attrs, vocab_size=3)
    assert 0 <= posdis <= 1
    assert mi.shape == (4, 2)
    assert len(ent) == 4


def test_compute_topsim():
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 3, (100, 2))
    p1 = rng.randint(0, 5, 100)
    p2 = rng.randint(0, 5, 100)
    ts = compute_topsim(tokens, p1, p2)
    assert -1 <= ts <= 1


def test_compute_bosdis():
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 3, (100, 4))
    attrs = rng.randint(0, 3, (100, 2))
    bd = compute_bosdis(tokens, attrs, vocab_size=3)
    assert 0 <= bd <= 1


def test_compute_mi_matrix_shape():
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 3, (50, 4))
    attrs = rng.randint(0, 3, (50, 3))
    mi = compute_mi_matrix(tokens, attrs)
    assert mi.shape == (4, 3)


def test_make_attributes():
    mass = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    names = ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
    attrs = make_attributes(mass, names)
    assert attrs.shape == (10, 2)
    assert attrs.dtype in [np.int64, np.intp]

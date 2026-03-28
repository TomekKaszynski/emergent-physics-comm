"""Bandwidth-adaptive protocol — adjust vocabulary size based on constraints."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List


def compute_bandwidth(K: int, L: int, n_agents: int) -> Dict:
    """Compute bandwidth metrics for a given configuration.

    Args:
        K: Vocabulary size per position.
        L: Positions per agent.
        n_agents: Number of agents.

    Returns:
        Dict with bits_per_agent, total_bits, message_space, compression vs common dims.
    """
    bits_per_agent = L * math.log2(K)
    total_bits = n_agents * bits_per_agent
    message_space = K ** (n_agents * L)

    # Compression vs common encoder dims
    common_dims = {"V-JEPA 2": 1024, "DINOv2": 384, "CLIP": 768}
    compression = {name: (dim * 32) / total_bits for name, dim in common_dims.items()}

    return {
        "K": K,
        "L": L,
        "n_agents": n_agents,
        "bits_per_agent": float(bits_per_agent),
        "total_bits": float(total_bits),
        "message_space": int(message_space),
        "compression_ratios": compression,
    }


def minimum_viable_K(target_accuracy: float = 0.7) -> Dict:
    """Report minimum K for each accuracy threshold based on Phase 92c data.

    Returns empirically validated K-accuracy mapping.
    """
    # From Phase 92c sweep (spring, 2-agent, 10 seeds)
    k_accuracy = {
        2: {"acc": 0.72, "posdis": 0.65, "bits": 2.0},
        3: {"acc": 0.78, "posdis": 0.76, "bits": 3.17},
        5: {"acc": 0.82, "posdis": 0.78, "bits": 4.64},
        8: {"acc": 0.78, "posdis": 0.76, "bits": 6.0},
        16: {"acc": 0.82, "posdis": 0.80, "bits": 8.0},
        32: {"acc": 0.76, "posdis": 0.77, "bits": 10.0},
    }
    return k_accuracy


def bandwidth_sweep() -> List[Dict]:
    """Generate bandwidth efficiency data across K values."""
    results = []
    for K in [2, 3, 4, 5, 8, 16, 32]:
        bw = compute_bandwidth(K, L=2, n_agents=2)
        results.append(bw)
    return results

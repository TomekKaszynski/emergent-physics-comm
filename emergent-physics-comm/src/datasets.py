"""
Dataset loading and splitting utilities.

Handles pre-extracted foundation model features (DINOv2, V-JEPA 2) and
Latin square holdout splits for compositional generalization evaluation.
"""

import numpy as np
import torch


# Default Latin square holdout: 5 cells from 5x5 grid, one per row and column
DEFAULT_HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}


def load_features(path, weights_only=True):
    """Load pre-extracted foundation model features.

    Supports two formats:
    1. Keys: features, mass_bins/e_bins, rest_bins/f_bins (Phase 79 DINOv2 ViT-S format)
    2. Keys: features, index (list of dicts with mass_ratio_bin/restitution_bin)

    Args:
        path: Path to .pt file.
        weights_only: Whether to use safe loading (default: True).

    Returns:
        features: (N, T, D) float32 tensor.
        prop1_bins: (N,) int array (mass_ratio or elasticity bins).
        prop2_bins: (N,) int array (restitution or friction bins).
    """
    data = torch.load(path, map_location='cpu', weights_only=weights_only)

    features = data['features']
    if features.dtype == torch.float16:
        features = features.float()

    # Try direct bin arrays first (Phase 79 DINOv2 ViT-S format)
    if 'mass_bins' in data:
        prop1_bins = np.array(data['mass_bins'])
        prop2_bins = np.array(data['rest_bins'])
    elif 'e_bins' in data:
        prop1_bins = np.array(data['e_bins'])
        prop2_bins = np.array(data['f_bins'])
    elif 'index' in data:
        # ViT-L format: labels in index dicts
        index = data['index']
        if 'mass_ratio_bin' in index[0]:
            prop1_bins = np.array([s['mass_ratio_bin'] for s in index])
            prop2_bins = np.array([s['restitution_bin'] for s in index])
        else:
            prop1_bins = np.array([s.get('e_bin', s.get('elasticity_bin', 0)) for s in index])
            prop2_bins = np.array([s.get('f_bin', s.get('friction_bin', 0)) for s in index])
    else:
        raise ValueError(f"Cannot find property labels in feature file. Keys: {list(data.keys())}")

    return features, prop1_bins, prop2_bins


def create_splits(prop1_bins, prop2_bins, holdout_cells=None):
    """Create train/holdout split using Latin square pattern.

    Args:
        prop1_bins: (N,) integer property 1 bins.
        prop2_bins: (N,) integer property 2 bins.
        holdout_cells: Set of (bin1, bin2) tuples to hold out.
            Defaults to DEFAULT_HOLDOUT_CELLS.

    Returns:
        train_ids: Array of training indices.
        holdout_ids: Array of holdout indices.
    """
    if holdout_cells is None:
        holdout_cells = DEFAULT_HOLDOUT_CELLS

    train_ids, holdout_ids = [], []
    for i in range(len(prop1_bins)):
        if (int(prop1_bins[i]), int(prop2_bins[i])) in holdout_cells:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


def sample_pairs(ids, batch_size, rng):
    """Sample random pairs of scene indices for comparison task.

    Args:
        ids: Array of valid scene indices.
        batch_size: Number of pairs.
        rng: numpy RandomState.

    Returns:
        ia, ib: Arrays of scene indices for scenes A and B.
    """
    ia = rng.choice(ids, size=batch_size, replace=True)
    ib = rng.choice(ids, size=batch_size, replace=True)
    return ia, ib


def split_views(data, n_agents, frames_per_agent):
    """Split temporal features into per-agent views.

    Args:
        data: (batch, total_frames, dim) tensor.
        n_agents: Number of agents.
        frames_per_agent: Frames per agent.

    Returns:
        List of (batch, frames_per_agent, dim) tensors.
    """
    return [data[:, i * frames_per_agent:(i + 1) * frames_per_agent, :]
            for i in range(n_agents)]

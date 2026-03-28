"""Compositionality and communication metrics."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute discrete mutual information between two integer arrays.

    Args:
        x: 1D integer array.
        y: 1D integer array of the same length.

    Returns:
        MI value in nats.
    """
    n = len(x)
    mi = 0.0
    for xv in np.unique(x):
        for yv in np.unique(y):
            p_xy = np.sum((x == xv) & (y == yv)) / n
            p_x = np.sum(x == xv) / n
            p_y = np.sum(y == yv) / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    return float(mi)


def compute_mi_matrix(tokens: np.ndarray, attributes: np.ndarray
                      ) -> np.ndarray:
    """Compute the full MI matrix between message positions and attributes.

    Args:
        tokens: (N, n_positions) integer token array.
        attributes: (N, n_attributes) integer attribute array.

    Returns:
        mi_matrix: (n_positions, n_attributes) MI values.
    """
    n_pos = tokens.shape[1]
    n_attr = attributes.shape[1]
    mi_matrix = np.zeros((n_pos, n_attr))
    for p in range(n_pos):
        for a in range(n_attr):
            mi_matrix[p, a] = mutual_information(tokens[:, p], attributes[:, a])
    return mi_matrix


def compute_posdis(tokens: np.ndarray, attributes: np.ndarray,
                   vocab_size: int = 3
                   ) -> Tuple[float, np.ndarray, List[float]]:
    """Compute Positional Disentanglement (PosDis).

    For each position, measures the gap between its highest and
    second-highest MI with attributes. High PosDis means each position
    specializes for one property.

    Args:
        tokens: (N, n_positions) integer token array.
        attributes: (N, n_attributes) integer attribute array.
        vocab_size: Vocabulary size per position.

    Returns:
        posdis: Scalar in [0, 1].
        mi_matrix: (n_positions, n_attributes) MI values.
        entropies: Normalized entropy per position.
    """
    n_pos = tokens.shape[1]

    entropies: List[float] = []
    for p in range(n_pos):
        counts = np.bincount(tokens[:, p], minlength=vocab_size)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(vocab_size)))

    mi_matrix = compute_mi_matrix(tokens, attributes)

    if n_pos >= 2:
        posdis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                posdis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        posdis /= n_pos
    else:
        posdis = 0.0

    return float(posdis), mi_matrix, entropies


def compute_topsim(tokens: np.ndarray, prop1_bins: np.ndarray,
                   prop2_bins: np.ndarray, n_pairs: int = 5000,
                   seed: int = 42) -> float:
    """Compute Topographic Similarity (TopSim).

    Spearman correlation between meaning distances (Manhattan in property
    space) and message distances (Hamming in symbol space).

    Args:
        tokens: (N, n_positions) integer token array.
        prop1_bins: (N,) integer property labels.
        prop2_bins: (N,) integer property labels.
        n_pairs: Number of random pairs to sample.
        seed: Random seed.

    Returns:
        topsim: Scalar in [-1, 1].
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    meaning_dists: List[float] = []
    message_dists: List[float] = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        meaning_dists.append(
            abs(int(prop1_bins[i]) - int(prop1_bins[j])) +
            abs(int(prop2_bins[i]) - int(prop2_bins[j])))
        message_dists.append(int((tokens[i] != tokens[j]).sum()))

    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    return 0.0 if np.isnan(topsim) else float(topsim)


def compute_bosdis(tokens: np.ndarray, attributes: np.ndarray,
                   vocab_size: int = 3) -> float:
    """Compute Bag-of-Symbols Disentanglement (BosDis).

    Position-independent metric: measures whether specific symbols
    (regardless of position) uniquely encode specific attributes.

    Args:
        tokens: (N, n_positions) integer token array.
        attributes: (N, n_attributes) integer attribute array.
        vocab_size: Vocabulary size per position.

    Returns:
        bosdis: Scalar in [0, 1].
    """
    n_samples = len(tokens)
    n_attr = attributes.shape[1]
    bosdis = 0.0
    n_active = 0

    for s in range(vocab_size):
        contains_s = np.any(tokens == s, axis=1).astype(int)
        if contains_s.sum() == 0 or contains_s.sum() == n_samples:
            continue
        mis = [mutual_information(contains_s, attributes[:, a])
               for a in range(n_attr)]
        sorted_mi = sorted(mis, reverse=True)
        if sorted_mi[0] > 1e-10:
            if len(sorted_mi) > 1 and sorted_mi[1] > 1e-10:
                bosdis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
            else:
                bosdis += 1.0
            n_active += 1

    return float(bosdis / max(n_active, 1))


def make_attributes(mass_values: np.ndarray, obj_names: List[str]
                    ) -> np.ndarray:
    """Create standard attribute bins from mass values and object names.

    Args:
        mass_values: (N,) continuous mass values.
        obj_names: List of N object name strings.

    Returns:
        attributes: (N, 2) integer array [mass_bins, obj_bins].
    """
    mass_bins = np.digitize(
        mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    obj_indices = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins = np.digitize(
        obj_indices, np.quantile(obj_indices, [0.2, 0.4, 0.6, 0.8]))
    return np.stack([mass_bins, obj_bins], axis=1)

"""
Compositionality and communication metrics.

Implements PosDis (positional disentanglement), TopSim (topographic similarity),
mutual information, and evaluation utilities.
"""

import numpy as np
from scipy import stats


def mutual_information(x, y):
    """Compute discrete mutual information between two arrays.

    Args:
        x, y: 1D integer arrays of the same length.
    Returns:
        MI value in nats.
    """
    x_vals, y_vals = np.unique(x), np.unique(y)
    n = len(x)
    mi = 0.0
    for xv in x_vals:
        for yv in y_vals:
            p_xy = np.sum((x == xv) & (y == yv)) / n
            p_x = np.sum(x == xv) / n
            p_y = np.sum(y == yv) / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi


def positional_disentanglement(tokens, attributes, vocab_size=5):
    """Compute PosDis: how well each message position specializes for one attribute.

    For each position, measures the gap between its highest and second-highest
    MI with attributes. High PosDis means each position maps to a single attribute.

    Args:
        tokens: (N, n_positions) integer token array.
        attributes: (N, n_attributes) integer attribute array.
        vocab_size: Vocabulary size per position.

    Returns:
        pos_dis: Scalar in [0, 1]. Higher = more compositional.
        mi_matrix: (n_positions, n_attributes) MI values.
        entropies: List of normalized entropies per position.
    """
    n_pos = tokens.shape[1]
    n_attr = attributes.shape[1]

    # Per-position entropy
    entropies = []
    for p in range(n_pos):
        counts = np.bincount(tokens[:, p], minlength=vocab_size)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(vocab_size)))

    # MI matrix
    mi_matrix = np.zeros((n_pos, n_attr))
    for p in range(n_pos):
        for a in range(n_attr):
            mi_matrix[p, a] = mutual_information(tokens[:, p], attributes[:, a])

    # PosDis
    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    return float(pos_dis), mi_matrix, entropies


def topographic_similarity(tokens, prop1_bins, prop2_bins, n_pairs=5000, seed=42):
    """Compute TopSim: Spearman correlation between meaning and message distances.

    Args:
        tokens: (N, n_positions) integer token array.
        prop1_bins, prop2_bins: (N,) integer property labels.
        n_pairs: Number of random pairs to sample.
        seed: Random seed.

    Returns:
        topsim: Scalar in [-1, 1]. Higher = more topographic.
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        meaning_dists.append(
            abs(int(prop1_bins[i]) - int(prop1_bins[j])) +
            abs(int(prop2_bins[i]) - int(prop2_bins[j]))
        )
        message_dists.append(int((tokens[i] != tokens[j]).sum()))

    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0
    return float(topsim)


def compute_compositionality(sender, data, prop1_bins, prop2_bins, device,
                             batch_size=32, vocab_size=5):
    """Full compositionality analysis for a sender.

    Args:
        sender: CompositionalSender or MultiAgentSender.
        data: (N, T, D) feature tensor.
        prop1_bins, prop2_bins: (N,) integer property labels.
        device: torch device.
        batch_size: Batch size for inference.
        vocab_size: Vocabulary size per position.

    Returns:
        Dict with pos_dis, topsim, entropies, mi_matrix.
    """
    import torch

    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            # Handle MultiAgentSender vs CompositionalSender
            if hasattr(sender, 'senders'):
                n_agents = len(sender.senders)
                frames_per = batch.shape[1] // n_agents
                views = [batch[:, j * frames_per:(j + 1) * frames_per, :]
                         for j in range(n_agents)]
                _, logits = sender(views)
            else:
                _, logits = sender(batch)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)
    attributes = np.stack([prop1_bins, prop2_bins], axis=1)

    pos_dis, mi_matrix, entropies = positional_disentanglement(
        all_tokens, attributes, vocab_size)
    topsim = topographic_similarity(all_tokens, prop1_bins, prop2_bins)

    return {
        'pos_dis': pos_dis,
        'topsim': topsim,
        'entropies': entropies,
        'mi_matrix': mi_matrix.tolist(),
    }


def compute_compositionality_multiagent(multi_sender, data, prop1_bins, prop2_bins,
                                        device, n_agents, frames_per_agent, n_heads=2,
                                        batch_size=32, vocab_size=5):
    """Compositionality analysis for multi-agent sender with per-agent PosDis.

    Returns:
        Dict with pos_dis_global, best_agent_posdis, per_agent_posdis, topsim, etc.
    """
    import torch

    multi_sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            views = [batch[:, j * frames_per_agent:(j + 1) * frames_per_agent, :]
                     for j in range(n_agents)]
            _, logits = multi_sender(views)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)
    n_pos = all_tokens.shape[1]
    attributes = np.stack([prop1_bins, prop2_bins], axis=1)

    # Global PosDis
    pos_dis_global, mi_matrix, entropies = positional_disentanglement(
        all_tokens, attributes, vocab_size)

    # Per-agent PosDis
    mi_matrix_np = np.array(mi_matrix) if isinstance(mi_matrix, list) else mi_matrix
    per_agent_posdis = []
    for agent_idx in range(n_agents):
        start = agent_idx * n_heads
        agent_mi = mi_matrix_np[start:start + n_heads]
        agent_pd = 0.0
        for p in range(n_heads):
            sorted_mi = np.sort(agent_mi[p])[::-1]
            if sorted_mi[0] > 1e-10:
                agent_pd += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        agent_pd /= n_heads
        per_agent_posdis.append(float(agent_pd))

    topsim = topographic_similarity(all_tokens, prop1_bins, prop2_bins)

    return {
        'pos_dis_global': float(pos_dis_global),
        'best_agent_posdis': float(max(per_agent_posdis)),
        'per_agent_posdis': per_agent_posdis,
        'topsim': topsim,
        'entropies': entropies,
        'mi_matrix': mi_matrix_np.tolist() if hasattr(mi_matrix_np, 'tolist') else mi_matrix,
    }

"""Onboard a new encoder into an existing protocol."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Callable

from wmcp.protocol import Protocol, AgentSender


def onboard_new_encoder(
    protocol: Protocol,
    agent_slot: int,
    new_features: torch.Tensor,
    all_agent_views: List[torch.Tensor],
    mass_values: np.ndarray,
    obj_names: List[str],
    new_input_dim: int,
    n_steps: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    checkpoint_interval: int = 10,
    callback: Optional[Callable] = None,
) -> Dict:
    """Onboard a new encoder by fine-tuning only its projection layer.

    All existing agents and receivers are frozen. Only the new agent's
    sender is trained.

    Args:
        protocol: Trained Protocol instance.
        agent_slot: Index of the agent to replace.
        new_features: (N, T, D_new) features from the new encoder.
        all_agent_views: List of (N, T, D) per agent (with slot replaced).
        mass_values: (N,) mass values.
        obj_names: List of N object names.
        new_input_dim: Feature dimension of the new encoder.
        n_steps: Training steps.
        lr: Learning rate.
        device: Torch device.
        checkpoint_interval: Steps between accuracy evaluations.
        callback: Optional function called with (step, accuracy) at checkpoints.

    Returns:
        Dict with training curve and final metrics.
    """
    protocol = protocol.to(device)

    # Replace sender at agent_slot
    n_frames = new_features.shape[1]
    new_sender = AgentSender(
        new_input_dim, hidden_dim=128,
        vocab_size=protocol.vocab_size,
        n_heads=protocol.n_heads,
        n_frames=n_frames,
    ).to(device)
    protocol.senders[agent_slot] = new_sender

    # Freeze everything except new sender
    for name, param in protocol.named_parameters():
        if f"senders.{agent_slot}" not in name:
            param.requires_grad = False
    for r in protocol.receivers:
        for param in r.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(new_sender.parameters(), lr=lr)
    recv = protocol.receivers[0]

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)
    rng = np.random.RandomState(42)
    holdout = set(rng.choice(unique_objs, n_holdout, replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout])
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(device)

    curve = []

    for step in range(n_steps):
        protocol.train()
        ia = rng.choice(train_ids, 32)
        ib = rng.choice(train_ids, 32)
        s = ia == ib
        while s.any():
            ib[s] = rng.choice(train_ids, s.sum())
            s = ia == ib
        md = np.abs(mass_values[ia] - mass_values[ib])
        keep = md > 0.5
        if keep.sum() < 4:
            continue
        ia, ib = ia[keep], ib[keep]

        va = [v[ia].to(device) for v in all_agent_views]
        vb = [v[ib].to(device) for v in all_agent_views]
        label = (mass_dev[ia] > mass_dev[ib]).float()
        pred = protocol.communicate(va, vb)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_sender.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % checkpoint_interval == 0:
            protocol.eval()
            with torch.no_grad():
                c = t = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    ib_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    kh = mdh > 0.5
                    if kh.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    va_h = [v[ia_h].to(device) for v in all_agent_views]
                    vb_h = [v[ib_h].to(device) for v in all_agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    p = protocol.communicate(va_h, vb_h) > 0
                    c += (p == la_h).sum().item()
                    t += len(la_h)
                acc = c / max(t, 1)
            curve.append({"step": step + 1, "accuracy": float(acc)})
            if callback:
                callback(step + 1, acc)

    return {
        "curve": curve,
        "final_accuracy": curve[-1]["accuracy"] if curve else 0,
        "n_steps": n_steps,
    }

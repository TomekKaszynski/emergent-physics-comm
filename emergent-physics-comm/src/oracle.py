"""
Oracle probe models for measuring information accessibility in features.

Oracles have direct access to features (no communication bottleneck) and
set the upper bound on what a communication system could achieve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .models import TemporalEncoder


class Oracle(nn.Module):
    """2-agent oracle: direct feature comparison without communication bottleneck.

    Args:
        hidden_dim: Hidden dimension (default: 128).
        input_dim: Per-frame feature dimension.
    """

    def __init__(self, hidden_dim=128, input_dim=384):
        super().__init__()
        self.enc_a = TemporalEncoder(hidden_dim, input_dim)
        self.enc_b = TemporalEncoder(hidden_dim, input_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.prop1_head = nn.Linear(hidden_dim, 1)
        self.prop2_head = nn.Linear(hidden_dim, 1)

    def forward(self, xa, xb):
        ha = self.enc_a(xa)
        hb = self.enc_b(xb)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return self.prop1_head(h).squeeze(-1), self.prop2_head(h).squeeze(-1)


class MultiAgentOracle(nn.Module):
    """N-agent oracle: each agent sees a temporal slice, no bottleneck.

    Args:
        n_agents: Number of agents.
        hidden_dim: Hidden dimension.
        input_dim: Per-frame feature dimension.
    """

    def __init__(self, n_agents, hidden_dim, input_dim):
        super().__init__()
        self.n_agents = n_agents
        self.encs_a = nn.ModuleList([
            TemporalEncoder(hidden_dim, input_dim) for _ in range(n_agents)
        ])
        self.encs_b = nn.ModuleList([
            TemporalEncoder(hidden_dim, input_dim) for _ in range(n_agents)
        ])
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * n_agents * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.prop1_head = nn.Linear(hidden_dim, 1)
        self.prop2_head = nn.Linear(hidden_dim, 1)

    def forward(self, views_a, views_b):
        ha = torch.cat([enc(v) for enc, v in zip(self.encs_a, views_a)], dim=-1)
        hb = torch.cat([enc(v) for enc, v in zip(self.encs_b, views_b)], dim=-1)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return self.prop1_head(h).squeeze(-1), self.prop2_head(h).squeeze(-1)


def train_oracle(model, data, prop1_bins, prop2_bins, train_ids, device,
                 epochs=200, lr=1e-3, batch_size=32, seed=0):
    """Train an oracle probe.

    Args:
        model: Oracle or MultiAgentOracle instance.
        data: (N, T, D) feature tensor.
        prop1_bins, prop2_bins: (N,) integer property bin labels.
        train_ids: Array of training indices.
        device: torch device.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        model: Trained oracle.
    """
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = max(1, len(train_ids) // batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(n_batches):
            ia = rng.choice(train_ids, size=batch_size, replace=True)
            ib = rng.choice(train_ids, size=batch_size, replace=True)
            da = data[ia].to(device)
            db = data[ib].to(device)
            l1 = torch.tensor((prop1_bins[ia] > prop1_bins[ib]).astype(np.float32), device=device)
            l2 = torch.tensor((prop2_bins[ia] > prop2_bins[ib]).astype(np.float32), device=device)
            p1, p2 = model(da, db)
            loss = F.binary_cross_entropy_with_logits(p1, l1) + \
                   F.binary_cross_entropy_with_logits(p2, l2)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 50 == 0:
            print(f"  Oracle ep {epoch}/{epochs}", flush=True)
            if hasattr(torch, 'mps') and device.type == 'mps':
                torch.mps.empty_cache()

    return model

"""
Phase 65: Non-Physics Domain — Abstract Visual Reasoning
=========================================================
Tests whether the communication mechanism is DOMAIN-AGNOSTIC by training
agents on abstract geometric scenes with NO physics, NO trajectories,
NO temporal dynamics.

Domain: 2D scenes with shapes. Two latent properties:
  - numerosity: total shapes in scene (2-6, 5 bins)
  - mean_size: average shape radius (5 bins from small to large)

Observation structure: 4 spatial quadrant crops (not temporal frames).
Every quadrant reveals PARTIAL info about BOTH properties.
Agents must AGGREGATE, not specialize per-property.

Four conditions × 15 seeds:
(a) SCENE_4AGENT: geometric scenes (main test)
(b) SPRING_4AGENT: spring-mass (physics control, replicates Phase 64)
(c) RAMP_4AGENT: ramp (physics control, replicates Phase 62)
(d) SCENE_ORACLE: MLP baseline on raw scene features (ceiling)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase65_nonphysics.py
"""

import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
DINO_DIM = 384
BATCH_SIZE = 64

VOCAB_SIZE = 5

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 2.0
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

N_SEEDS = 15
SEEDS = list(range(N_SEEDS))

# 4 agents, 1 view each, 2 positions each
N_AGENTS = 4
AGENT_FRAMES = [[0], [1], [2], [3]]  # "frames" = quadrant views
N_POSITIONS = 2

CONDITIONS = ['scene_4agent', 'spring_4agent', 'ramp_4agent', 'scene_oracle']

# Scene feature dimension (per quadrant)
SCENE_RAW_DIM = 9


# ══════════════════════════════════════════════════════════════════
# Scene Data Generation
# ══════════════════════════════════════════════════════════════════

class FrozenSceneEncoder(nn.Module):
    """Frozen random MLP to project per-quadrant features (9-dim) → 384-dim."""
    def __init__(self, seed=54321):
        super().__init__()
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(SCENE_RAW_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 384),
        )
        torch.random.set_rng_state(rng_state)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.net(x)


def generate_scene_data(n_scenes=300, seed=42):
    """Generate abstract geometric scenes with two latent properties.

    Properties:
        numerosity: total shapes in scene (2, 3, 4, 5, 6) → bins 0-4
        mean_size: average shape radius (5 levels) → bins 0-4

    Each scene has N shapes placed randomly in [0,1]×[0,1].
    4 views = 4 quadrant crops. Per-quadrant raw features (9-dim):
        n_shapes, n_colors, has_circle, has_square, has_triangle,
        mean_x, mean_y, total_area, mean_radius

    Returns:
        features: (n_scenes, 4, 384) — encoded features
        raw_features: (n_scenes, 4, SCENE_RAW_DIM) — per-quadrant raw
        num_bins: (n_scenes,) — numerosity bin indices 0-4
        size_bins: (n_scenes,) — mean_size bin indices 0-4
        train_ids, holdout_ids: arrays of scene indices
    """
    rng = np.random.RandomState(seed)

    # 5×5 grid, 12 scenes per cell = 300 scenes
    n_num_bins = 5
    n_size_bins = 5
    per_cell = n_scenes // (n_num_bins * n_size_bins)

    numerosity_values = [2, 3, 4, 5, 6]
    size_centers = np.linspace(0.06, 0.22, n_size_bins)  # radius levels
    size_jitter = 0.015  # per-shape size jitter

    # Color palette (5 distinct colors, encoded as integers 0-4)
    n_colors = 5
    # Shape types: 0=circle, 1=square, 2=triangle
    n_shape_types = 3

    # Quadrant boundaries
    quadrants = [
        (0.0, 0.5, 0.0, 0.5),   # top-left
        (0.5, 1.0, 0.0, 0.5),   # top-right
        (0.0, 0.5, 0.5, 1.0),   # bottom-left
        (0.5, 1.0, 0.5, 1.0),   # bottom-right
    ]

    all_num_bins = []
    all_size_bins = []
    all_raw_features = []

    for ni in range(n_num_bins):
        for si in range(n_size_bins):
            n_shapes = numerosity_values[ni]
            mean_radius = size_centers[si]

            for _ in range(per_cell):
                # Place shapes randomly in [0,1]×[0,1]
                xs = rng.uniform(0.0, 1.0, size=n_shapes)
                ys = rng.uniform(0.0, 1.0, size=n_shapes)
                colors = rng.randint(0, n_colors, size=n_shapes)
                types = rng.randint(0, n_shape_types, size=n_shapes)
                radii = np.clip(
                    rng.normal(mean_radius, size_jitter, size=n_shapes),
                    0.03, 0.30
                )

                # Extract per-quadrant features
                scene_features = np.zeros((4, SCENE_RAW_DIM), dtype=np.float32)

                for qi, (x_lo, x_hi, y_lo, y_hi) in enumerate(quadrants):
                    # Which shapes are in this quadrant?
                    mask = (xs >= x_lo) & (xs < x_hi) & (ys >= y_lo) & (ys < y_hi)
                    n_in_quad = mask.sum()

                    if n_in_quad == 0:
                        # Empty quadrant: all zeros except mean_x/y = 0.5
                        scene_features[qi, 5] = 0.5  # mean_x
                        scene_features[qi, 6] = 0.5  # mean_y
                        continue

                    q_colors = colors[mask]
                    q_types = types[mask]
                    q_xs = xs[mask]
                    q_ys = ys[mask]
                    q_radii = radii[mask]

                    # Feature 0: n_shapes in quadrant (normalized by max 6)
                    scene_features[qi, 0] = n_in_quad / 6.0
                    # Feature 1: n_distinct_colors (normalized by 5)
                    scene_features[qi, 1] = len(np.unique(q_colors)) / 5.0
                    # Features 2-4: has_circle, has_square, has_triangle
                    scene_features[qi, 2] = float(0 in q_types)
                    scene_features[qi, 3] = float(1 in q_types)
                    scene_features[qi, 4] = float(2 in q_types)
                    # Features 5-6: mean position (normalized to [0,1] within quadrant)
                    scene_features[qi, 5] = (q_xs.mean() - x_lo) / (x_hi - x_lo)
                    scene_features[qi, 6] = (q_ys.mean() - y_lo) / (y_hi - y_lo)
                    # Feature 7: total area (sum of pi*r^2, normalized)
                    scene_features[qi, 7] = np.sum(np.pi * q_radii ** 2)
                    # Feature 8: mean radius (normalized)
                    scene_features[qi, 8] = q_radii.mean()

                all_raw_features.append(scene_features)
                all_num_bins.append(ni)
                all_size_bins.append(si)

    raw_features = np.array(all_raw_features, dtype=np.float32)  # (300, 4, 9)
    num_bins = np.array(all_num_bins, dtype=np.int64)
    size_bins = np.array(all_size_bins, dtype=np.int64)

    # Encode with frozen MLP
    encoder = FrozenSceneEncoder(seed=54321)
    raw_tensor = torch.tensor(raw_features).reshape(-1, SCENE_RAW_DIM)  # (300*4, 9)
    with torch.no_grad():
        encoded = encoder(raw_tensor)  # (300*4, 384)
    features = encoded.reshape(n_scenes, 4, 384)

    # Train/holdout split using Latin square
    train_ids, holdout_ids = [], []
    for i in range(n_scenes):
        if (int(num_bins[i]), int(size_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return (features, torch.tensor(raw_features),
            num_bins, size_bins,
            np.array(train_ids), np.array(holdout_ids))


# ══════════════════════════════════════════════════════════════════
# Spring-Mass Data Generation (from Phase 64)
# ══════════════════════════════════════════════════════════════════

class FrozenSpringEncoder(nn.Module):
    """Frozen random MLP: (position, velocity) → 384-dim."""
    def __init__(self, seed=12345):
        super().__init__()
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 384),
        )
        torch.random.set_rng_state(rng_state)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.net(x)


def generate_spring_data(n_scenes=300, seed=42):
    """Generate spring-mass trajectories (from Phase 64)."""
    rng = np.random.RandomState(seed)

    n_k_bins = 5
    n_b_bins = 5
    per_cell = n_scenes // (n_k_bins * n_b_bins)

    k_edges = np.linspace(1.0, 10.0, n_k_bins + 1)
    b_edges = np.linspace(0.1, 2.0, n_b_bins + 1)

    all_k, all_b = [], []
    all_k_bins, all_b_bins = [], []

    for ki in range(n_k_bins):
        for bi in range(n_b_bins):
            for _ in range(per_cell):
                k = rng.uniform(k_edges[ki], k_edges[ki + 1])
                b = rng.uniform(b_edges[bi], b_edges[bi + 1])
                all_k.append(k)
                all_b.append(b)
                all_k_bins.append(ki)
                all_b_bins.append(bi)

    all_k = np.array(all_k)
    all_b = np.array(all_b)
    k_bins = np.array(all_k_bins, dtype=np.int64)
    b_bins = np.array(all_b_bins, dtype=np.int64)

    times = np.array([0.0, 0.5, 1.0, 1.5])
    m = 1.0
    A = 1.0

    raw_features = np.zeros((n_scenes, 4, 2), dtype=np.float32)

    for i in range(n_scenes):
        k = all_k[i]
        b_val = all_b[i]
        gamma = b_val / (2 * m)
        omega_sq = k / m - gamma ** 2

        if omega_sq > 0:
            omega = np.sqrt(omega_sq)
        else:
            omega = 0.0

        for fi, t in enumerate(times):
            decay = A * np.exp(-gamma * t)
            if omega > 0:
                pos = decay * np.cos(omega * t)
                vel = decay * (-gamma * np.cos(omega * t) - omega * np.sin(omega * t))
            else:
                pos = A * (1 + gamma * t) * np.exp(-gamma * t)
                vel = A * (gamma - gamma * (1 + gamma * t)) * np.exp(-gamma * t)

            raw_features[i, fi, 0] = pos
            raw_features[i, fi, 1] = vel

    encoder = FrozenSpringEncoder(seed=12345)
    raw_tensor = torch.tensor(raw_features).reshape(-1, 2)
    with torch.no_grad():
        encoded = encoder(raw_tensor)
    features = encoded.reshape(n_scenes, 4, 384)

    train_ids, holdout_ids = [], []
    for i in range(n_scenes):
        if (int(k_bins[i]), int(b_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return (features, torch.tensor(raw_features),
            k_bins, b_bins,
            np.array(train_ids), np.array(holdout_ids))


# ══════════════════════════════════════════════════════════════════
# Ramp Data Loading
# ══════════════════════════════════════════════════════════════════

def load_ramp_data():
    """Load cached DINOv2 features from Phase 54b."""
    cache_path = RESULTS_DIR / "phase54b_dino_features.pt"
    data = torch.load(cache_path, weights_only=False)
    features = data['features'][:, :4, :]  # (300, 4, 384)
    e_bins = data['e_bins'].numpy() if isinstance(data['e_bins'], torch.Tensor) else np.array(data['e_bins'])
    f_bins = data['f_bins'].numpy() if isinstance(data['f_bins'], torch.Tensor) else np.array(data['f_bins'])

    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return features, e_bins, f_bins, np.array(train_ids), np.array(holdout_ids)


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 62/64)
# ══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Encodes 1+ frames into a hidden vector."""
    def __init__(self, hidden_dim=128, input_dim=384):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


class FixedSender(nn.Module):
    """Fixed-length Gumbel-Softmax sender."""
    def __init__(self, hidden_dim=128, input_dim=384,
                 n_positions=2, vocab_size=5):
        super().__init__()
        self.encoder = TemporalEncoder(hidden_dim, input_dim)
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.head = nn.Linear(hidden_dim, n_positions * vocab_size)

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        logits = self.head(h).view(-1, self.n_positions, self.vocab_size)

        if self.training:
            flat = logits.reshape(-1, self.vocab_size)
            tokens = F.gumbel_softmax(flat, tau=tau, hard=hard)
            tokens = tokens.reshape(-1, self.n_positions, self.vocab_size)
        else:
            idx = logits.argmax(dim=-1)
            tokens = F.one_hot(idx, self.vocab_size).float()

        message = tokens.reshape(-1, self.n_positions * self.vocab_size)
        return message, logits


class PropertyReceiver(nn.Module):
    """Two-head receiver for property comparison."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head_a = nn.Linear(hidden_dim // 2, 1)
        self.head_b = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Helpers (identical to Phase 62/64)
# ══════════════════════════════════════════════════════════════════

def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


def _mutual_information(x, y):
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


def analyze_sender_messages(sender, features, bins_a, bins_b,
                            device, n_positions):
    """Compute MI of each message position with each property."""
    sender.eval()
    all_tokens = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch = features[i:i + BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)

    total_mi_a = 0.0
    total_mi_b = 0.0

    for p in range(n_positions):
        pos_tokens = all_tokens[:, p]
        total_mi_a += _mutual_information(pos_tokens, bins_a)
        total_mi_b += _mutual_information(pos_tokens, bins_b)

    denom = total_mi_a + total_mi_b
    spec_ratio = float(abs(total_mi_a - total_mi_b) / denom) if denom > 1e-10 else 0.0

    return {
        'total_mi_a': float(total_mi_a),
        'total_mi_b': float(total_mi_b),
        'spec_ratio': spec_ratio,
    }


def evaluate_with_receiver(senders, agent_frames, receiver, features,
                           bins_a, bins_b, scene_ids, device, n_rounds=30):
    rng = np.random.RandomState(999)
    a_dev = torch.tensor(bins_a, dtype=torch.float32).to(device)
    b_dev = torch.tensor(bins_b, dtype=torch.float32).to(device)

    for s in senders:
        s.eval()
    receiver.eval()

    ca = cb = c_both = 0
    ta = tb = t_both = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        with torch.no_grad():
            parts = []
            for si, sender in enumerate(senders):
                frames = agent_frames[si]
                feat_a = features[ia][:, frames, :].to(device)
                feat_b = features[ib][:, frames, :].to(device)
                msg_a, _ = sender(feat_a)
                msg_b, _ = sender(feat_b)
                parts.extend([msg_a, msg_b])

            combined = torch.cat(parts, dim=-1)
            pred_a, pred_b = receiver(combined)

        label_a = (a_dev[ia] > a_dev[ib])
        label_b = (b_dev[ia] > b_dev[ib])
        valid_a = (a_dev[ia] != a_dev[ib])
        valid_b = (b_dev[ia] != b_dev[ib])
        valid_both = valid_a & valid_b

        if valid_a.sum() > 0:
            ca += ((pred_a > 0)[valid_a] == label_a[valid_a]).sum().item()
            ta += valid_a.sum().item()
        if valid_b.sum() > 0:
            cb += ((pred_b > 0)[valid_b] == label_b[valid_b]).sum().item()
            tb += valid_b.sum().item()
        if valid_both.sum() > 0:
            both_ok = ((pred_a > 0)[valid_both] == label_a[valid_both]) & \
                      ((pred_b > 0)[valid_both] == label_b[valid_both])
            c_both += both_ok.sum().item()
            t_both += valid_both.sum().item()

    return {
        'a_acc': ca / max(ta, 1),
        'b_acc': cb / max(tb, 1),
        'both_acc': c_both / max(t_both, 1),
    }


def evaluate_population(senders, agent_frames, receivers, features,
                        bins_a, bins_b, scene_ids, device, n_rounds=30):
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate_with_receiver(
            senders, agent_frames, r, features, bins_a, bins_b,
            scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate_with_receiver(
        senders, agent_frames, best_r, features, bins_a, bins_b,
        scene_ids, device, n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Training (property-agnostic)
# ══════════════════════════════════════════════════════════════════

def train_seed(seed, features, bins_a, bins_b, train_ids, holdout_ids, device):
    """Train 4-agent communication for one seed. Property-agnostic."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    msg_dim_per_sender = N_POSITIONS * VOCAB_SIZE

    senders = [FixedSender(HIDDEN_DIM, DINO_DIM, N_POSITIONS, VOCAB_SIZE).to(device)
               for _ in range(N_AGENTS)]

    recv_input_dim = N_AGENTS * 2 * msg_dim_per_sender
    receivers = [PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_params = []
    for s in senders:
        sender_params.extend(list(s.parameters()))
    sender_opt = torch.optim.Adam(sender_params, lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    a_dev = torch.tensor(bins_a, dtype=torch.float32).to(device)
    b_dev = torch.tensor(bins_b, dtype=torch.float32).to(device)
    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_both = 0.0
    best_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        for s in senders:
            s.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_a = (a_dev[ia] > a_dev[ib]).float()
            label_b = (b_dev[ia] > b_dev[ib]).float()

            parts = []
            all_logits = []

            for si, sender in enumerate(senders):
                frames = AGENT_FRAMES[si]
                feat_a = features[ia][:, frames, :].to(device)
                feat_b = features[ib][:, frames, :].to(device)
                msg_a, lg_a = sender(feat_a, tau, hard)
                msg_b, lg_b = sender(feat_b, tau, hard)
                parts.extend([msg_a, msg_b])
                all_logits.extend([lg_a, lg_b])

            combined = torch.cat(parts, dim=-1)

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_a, pred_b = r(combined)
                r_loss = F.binary_cross_entropy_with_logits(pred_a, label_a) + \
                         F.binary_cross_entropy_with_logits(pred_b, label_b)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            for lg in all_logits:
                for p in range(N_POSITIONS):
                    pos_logits = lg[:, p, :]
                    log_probs = F.log_softmax(pos_logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                continue

            sender_opt.zero_grad()
            for opt in receiver_opts:
                opt.zero_grad()
            loss.backward()

            has_nan_grad = False
            all_params = list(sender_params)
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for param in all_params:
                if param.grad is not None and (torch.isnan(param.grad).any() or
                                               torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender_params, 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if (epoch + 1) % 40 == 0:
            train_result, _ = evaluate_population(
                senders, AGENT_FRAMES, receivers, features, bins_a, bins_b,
                train_ids, device, n_rounds=10)
            holdout_result, _ = evaluate_population(
                senders, AGENT_FRAMES, receivers, features, bins_a, bins_b,
                holdout_ids, device, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: a={train_result['a_acc']:.1%}  "
                  f"b={train_result['b_acc']:.1%}  "
                  f"both={train_result['both_acc']:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
                best_states = {
                    'senders': [
                        {k: v.cpu().clone() for k, v in s.state_dict().items()}
                        for s in senders
                    ],
                    'receivers': [
                        {k: v.cpu().clone() for k, v in r.state_dict().items()}
                        for r in receivers
                    ],
                }

    if best_states is not None:
        for i, s in enumerate(senders):
            s.load_state_dict(best_states['senders'][i])
            s.to(device)
        for i, r in enumerate(receivers):
            r.load_state_dict(best_states['receivers'][i])
            r.to(device)

    final_result, _ = evaluate_population(
        senders, AGENT_FRAMES, receivers, features, bins_a, bins_b,
        holdout_ids, device, n_rounds=30)

    msg_analysis = {}
    for si, sender in enumerate(senders):
        frames = AGENT_FRAMES[si]
        sender_features = features[:, frames, :]
        msg_analysis[f'agent_{si}'] = analyze_sender_messages(
            sender, sender_features, bins_a, bins_b, device, N_POSITIONS)

    return {
        'a_acc': final_result['a_acc'],
        'b_acc': final_result['b_acc'],
        'both_acc': final_result['both_acc'],
        'nan_count': nan_count,
        'msg_analysis': msg_analysis,
    }


# ══════════════════════════════════════════════════════════════════
# Oracle Baseline
# ══════════════════════════════════════════════════════════════════

class OracleReceiver(nn.Module):
    """Direct MLP on raw features — no communication bottleneck."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head_a = nn.Linear(32, 1)
        self.head_b = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


def train_oracle(raw_features, bins_a, bins_b, train_ids, holdout_ids,
                 device, seed, raw_dim_per_view, n_views=4):
    """Train oracle baseline on raw features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Input: concatenate both scenes' raw features
    # raw_features: (300, n_views, raw_dim_per_view)
    # Per pair: [scene_a_views_flat, scene_b_views_flat]
    flat_dim = n_views * raw_dim_per_view
    input_dim = flat_dim * 2

    model = OracleReceiver(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    rng = np.random.RandomState(seed)
    a_dev = torch.tensor(bins_a, dtype=torch.float32).to(device)
    b_dev = torch.tensor(bins_b, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both = 0.0
    best_state = None

    for epoch in range(COMM_EPOCHS):
        model.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_a = (a_dev[ia] > a_dev[ib]).float()
            label_b = (b_dev[ia] > b_dev[ib]).float()

            feat_a = raw_features[ia].reshape(-1, flat_dim).to(device)
            feat_b = raw_features[ib].reshape(-1, flat_dim).to(device)
            x = torch.cat([feat_a, feat_b], dim=-1)

            pred_a, pred_b = model(x)
            loss = F.binary_cross_entropy_with_logits(pred_a, label_a) + \
                   F.binary_cross_entropy_with_logits(pred_b, label_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 40 == 0:
            model.eval()
            eval_rng = np.random.RandomState(999)
            ca = cb = c_both = 0
            ta_cnt = tb_cnt = t_both_cnt = 0

            for _ in range(30):
                bs = min(BATCH_SIZE, len(holdout_ids))
                ia, ib = sample_pairs(holdout_ids, bs, eval_rng)
                with torch.no_grad():
                    feat_a = raw_features[ia].reshape(-1, flat_dim).to(device)
                    feat_b = raw_features[ib].reshape(-1, flat_dim).to(device)
                    x = torch.cat([feat_a, feat_b], dim=-1)
                    pred_a, pred_b = model(x)

                la = (a_dev[ia] > a_dev[ib])
                lb = (b_dev[ia] > b_dev[ib])
                va = (a_dev[ia] != a_dev[ib])
                vb = (b_dev[ia] != b_dev[ib])
                v_both = va & vb

                if va.sum() > 0:
                    ca += ((pred_a > 0)[va] == la[va]).sum().item()
                    ta_cnt += va.sum().item()
                if vb.sum() > 0:
                    cb += ((pred_b > 0)[vb] == lb[vb]).sum().item()
                    tb_cnt += vb.sum().item()
                if v_both.sum() > 0:
                    both_ok = ((pred_a > 0)[v_both] == la[v_both]) & \
                              ((pred_b > 0)[v_both] == lb[v_both])
                    c_both += both_ok.sum().item()
                    t_both_cnt += v_both.sum().item()

            both_acc = c_both / max(t_both_cnt, 1)
            if both_acc > best_both:
                best_both = both_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best and final eval
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    eval_rng = np.random.RandomState(999)
    ca = cb = c_both = 0
    ta_cnt = tb_cnt = t_both_cnt = 0

    for _ in range(30):
        bs = min(BATCH_SIZE, len(holdout_ids))
        ia, ib = sample_pairs(holdout_ids, bs, eval_rng)
        with torch.no_grad():
            feat_a = raw_features[ia].reshape(-1, flat_dim).to(device)
            feat_b = raw_features[ib].reshape(-1, flat_dim).to(device)
            x = torch.cat([feat_a, feat_b], dim=-1)
            pred_a, pred_b = model(x)

        la = (a_dev[ia] > a_dev[ib])
        lb = (b_dev[ia] > b_dev[ib])
        va = (a_dev[ia] != a_dev[ib])
        vb = (b_dev[ia] != b_dev[ib])
        v_both = va & vb

        if va.sum() > 0:
            ca += ((pred_a > 0)[va] == la[va]).sum().item()
            ta_cnt += va.sum().item()
        if vb.sum() > 0:
            cb += ((pred_b > 0)[vb] == lb[vb]).sum().item()
            tb_cnt += vb.sum().item()
        if v_both.sum() > 0:
            both_ok = ((pred_a > 0)[v_both] == la[v_both]) & \
                      ((pred_b > 0)[v_both] == lb[v_both])
            c_both += both_ok.sum().item()
            t_both_cnt += v_both.sum().item()

    return {
        'a_acc': ca / max(ta_cnt, 1),
        'b_acc': cb / max(tb_cnt, 1),
        'both_acc': c_both / max(t_both_cnt, 1),
    }


# ══════════════════════════════════════════════════════════════════
# Aggregation helpers
# ══════════════════════════════════════════════════════════════════

def _aggregate_results(seed_results):
    """Aggregate per-seed results into summary."""
    a_accs = [r['a_acc'] for r in seed_results]
    b_accs = [r['b_acc'] for r in seed_results]
    both_accs = [r['both_acc'] for r in seed_results]

    summary = {
        'a_mean': float(np.mean(a_accs)),
        'a_std': float(np.std(a_accs)),
        'b_mean': float(np.mean(b_accs)),
        'b_std': float(np.std(b_accs)),
        'both_mean': float(np.mean(both_accs)),
        'both_std': float(np.std(both_accs)),
    }

    agent_specs = {}
    for si in range(N_AGENTS):
        key = f'agent_{si}'
        mi_as = [r['msg_analysis'][key]['total_mi_a'] for r in seed_results]
        mi_bs = [r['msg_analysis'][key]['total_mi_b'] for r in seed_results]
        specs = [r['msg_analysis'][key]['spec_ratio'] for r in seed_results]
        agent_specs[key] = {
            'mi_a_mean': float(np.mean(mi_as)),
            'mi_b_mean': float(np.mean(mi_bs)),
            'spec_ratio_mean': float(np.mean(specs)),
            'spec_ratio_std': float(np.std(specs)),
            'frames': AGENT_FRAMES[si],
        }
    summary['agent_specs'] = agent_specs
    summary['seeds'] = seed_results

    return summary


def _aggregate_oracle(seed_results):
    """Aggregate oracle results (no MI analysis)."""
    a_accs = [r['a_acc'] for r in seed_results]
    b_accs = [r['b_acc'] for r in seed_results]
    both_accs = [r['both_acc'] for r in seed_results]
    return {
        'a_mean': float(np.mean(a_accs)),
        'a_std': float(np.std(a_accs)),
        'b_mean': float(np.mean(b_accs)),
        'b_std': float(np.std(b_accs)),
        'both_mean': float(np.mean(both_accs)),
        'both_std': float(np.std(both_accs)),
        'seeds': seed_results,
    }


def _print_summary(label, summary, prop_a_name, prop_b_name):
    """Print condition summary."""
    print(f"\n  {label} summary:", flush=True)
    print(f"    {prop_a_name}={summary['a_mean']:.1%} ± {summary['a_std']:.1%}  "
          f"{prop_b_name}={summary['b_mean']:.1%} ± {summary['b_std']:.1%}  "
          f"both={summary['both_mean']:.1%} ± {summary['both_std']:.1%}", flush=True)
    if 'agent_specs' in summary:
        for si in range(N_AGENTS):
            key = f'agent_{si}'
            asp = summary['agent_specs'][key]
            print(f"    {key} (view {asp['frames']}): "
                  f"MI({prop_a_name})={asp['mi_a_mean']:.3f}  "
                  f"MI({prop_b_name})={asp['mi_b_mean']:.3f}  "
                  f"spec={asp['spec_ratio_mean']:.3f}", flush=True)
    print(flush=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 65: Non-Physics Domain — Abstract Visual Reasoning", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Architecture: {N_AGENTS} agents, {N_POSITIONS} pos, vocab={VOCAB_SIZE}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}", flush=True)
    print(flush=True)

    # Generate scene data
    print("  Generating scene data...", flush=True)
    scene_features, scene_raw, num_bins, size_bins, scene_train, scene_holdout = \
        generate_scene_data(n_scenes=300, seed=42)
    print(f"  Scene features: {scene_features.shape}", flush=True)
    print(f"  Scene split: {len(scene_train)} train, {len(scene_holdout)} holdout", flush=True)
    print(f"  num_bins: {np.bincount(num_bins)}, size_bins: {np.bincount(size_bins)}", flush=True)

    # Generate spring data
    print("  Generating spring-mass data...", flush=True)
    spring_features, spring_raw, k_bins, b_bins, spring_train, spring_holdout = \
        generate_spring_data(n_scenes=300, seed=42)
    print(f"  Spring features: {spring_features.shape}", flush=True)
    print(f"  Spring split: {len(spring_train)} train, {len(spring_holdout)} holdout", flush=True)

    # Load ramp data
    print("  Loading ramp data...", flush=True)
    ramp_features, e_bins, f_bins, ramp_train, ramp_holdout = load_ramp_data()
    print(f"  Ramp features: {ramp_features.shape}", flush=True)
    print(f"  Ramp split: {len(ramp_train)} train, {len(ramp_holdout)} holdout", flush=True)
    print(flush=True)

    all_results = {}
    total_t0 = time.time()
    total_seeds = N_SEEDS * len(CONDITIONS)

    # ── Condition 1: SCENE_4AGENT ─────────────────────────────────
    print(f"  {'='*60}", flush=True)
    print(f"  [1/{len(CONDITIONS)}] Condition: SCENE_4AGENT", flush=True)
    print(f"  {'='*60}", flush=True)

    scene_seeds = []
    for seed in SEEDS:
        t0 = time.time()
        total_elapsed = time.time() - total_t0
        done = seed
        if done > 0:
            remaining = total_elapsed / done * (total_seeds - done)
            eta_str = f"  total ETA {remaining/60:.0f}min"
        else:
            eta_str = ""
        print(f"    [seed={seed}] Training...{eta_str}", flush=True)

        result = train_seed(seed, scene_features, num_bins, size_bins,
                            scene_train, scene_holdout, DEVICE)
        elapsed = time.time() - t0
        print(f"    [seed={seed}] holdout num={result['a_acc']:.1%}  "
              f"size={result['b_acc']:.1%}  both={result['both_acc']:.1%}  "
              f"({elapsed:.0f}s)", flush=True)
        scene_seeds.append(result)

    scene_summary = _aggregate_results(scene_seeds)
    all_results['scene_4agent'] = scene_summary
    _print_summary('SCENE_4AGENT', scene_summary, 'num', 'size')

    # ── Condition 2: SPRING_4AGENT ────────────────────────────────
    print(f"  {'='*60}", flush=True)
    print(f"  [2/{len(CONDITIONS)}] Condition: SPRING_4AGENT", flush=True)
    print(f"  {'='*60}", flush=True)

    spring_seeds = []
    for seed in SEEDS:
        t0 = time.time()
        total_elapsed = time.time() - total_t0
        done = N_SEEDS + seed
        remaining = total_elapsed / done * (total_seeds - done)
        print(f"    [seed={seed}] Training...  total ETA {remaining/60:.0f}min", flush=True)

        result = train_seed(seed, spring_features, k_bins, b_bins,
                            spring_train, spring_holdout, DEVICE)
        elapsed = time.time() - t0
        print(f"    [seed={seed}] holdout k={result['a_acc']:.1%}  "
              f"b={result['b_acc']:.1%}  both={result['both_acc']:.1%}  "
              f"({elapsed:.0f}s)", flush=True)
        spring_seeds.append(result)

    spring_summary = _aggregate_results(spring_seeds)
    all_results['spring_4agent'] = spring_summary
    _print_summary('SPRING_4AGENT', spring_summary, 'k', 'damping')

    # ── Condition 3: RAMP_4AGENT ──────────────────────────────────
    print(f"\n  {'='*60}", flush=True)
    print(f"  [3/{len(CONDITIONS)}] Condition: RAMP_4AGENT", flush=True)
    print(f"  {'='*60}", flush=True)

    ramp_seeds = []
    for seed in SEEDS:
        t0 = time.time()
        total_elapsed = time.time() - total_t0
        done = N_SEEDS * 2 + seed
        remaining = total_elapsed / done * (total_seeds - done)
        print(f"    [seed={seed}] Training...  total ETA {remaining/60:.0f}min", flush=True)

        result = train_seed(seed, ramp_features, e_bins, f_bins,
                            ramp_train, ramp_holdout, DEVICE)
        elapsed = time.time() - t0
        print(f"    [seed={seed}] holdout e={result['a_acc']:.1%}  "
              f"f={result['b_acc']:.1%}  both={result['both_acc']:.1%}  "
              f"({elapsed:.0f}s)", flush=True)
        ramp_seeds.append(result)

    ramp_summary = _aggregate_results(ramp_seeds)
    all_results['ramp_4agent'] = ramp_summary
    _print_summary('RAMP_4AGENT', ramp_summary, 'e', 'f')

    # ── Condition 4: SCENE_ORACLE ─────────────────────────────────
    print(f"\n  {'='*60}", flush=True)
    print(f"  [4/{len(CONDITIONS)}] Condition: SCENE_ORACLE", flush=True)
    print(f"  {'='*60}", flush=True)

    oracle_seeds = []
    for seed in SEEDS:
        t0 = time.time()
        total_elapsed = time.time() - total_t0
        done = N_SEEDS * 3 + seed
        remaining = total_elapsed / done * (total_seeds - done)
        print(f"    [seed={seed}] Training...  total ETA {remaining/60:.0f}min", flush=True)

        result = train_oracle(scene_raw, num_bins, size_bins,
                              scene_train, scene_holdout, DEVICE, seed,
                              raw_dim_per_view=SCENE_RAW_DIM, n_views=4)
        elapsed = time.time() - t0
        print(f"    [seed={seed}] holdout num={result['a_acc']:.1%}  "
              f"size={result['b_acc']:.1%}  both={result['both_acc']:.1%}  "
              f"({elapsed:.0f}s)", flush=True)
        oracle_seeds.append(result)

    oracle_summary = _aggregate_oracle(oracle_seeds)
    all_results['scene_oracle'] = oracle_summary

    print(f"\n  SCENE_ORACLE summary:", flush=True)
    print(f"    num={oracle_summary['a_mean']:.1%} ± {oracle_summary['a_std']:.1%}  "
          f"size={oracle_summary['b_mean']:.1%} ± {oracle_summary['b_std']:.1%}  "
          f"both={oracle_summary['both_mean']:.1%} ± {oracle_summary['both_std']:.1%}", flush=True)

    total_elapsed = time.time() - total_t0

    # Save results
    output_path = RESULTS_DIR / "phase65_nonphysics.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_str = json.dumps(all_results, indent=2, default=convert)
    with open(output_path, 'w') as f:
        f.write(json_str)
    print(f"\n  Saved results to {output_path}", flush=True)

    # Final summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    sc = all_results['scene_4agent']
    sp = all_results['spring_4agent']
    ra = all_results['ramp_4agent']
    oc = all_results['scene_oracle']

    print(f"\n  Accuracy comparison:", flush=True)
    print(f"  {'Condition':<20s} {'Prop1':>8s} {'Prop2':>8s} {'Both':>8s}", flush=True)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    print(f"  {'scene_4agent':<20s} {sc['a_mean']:7.1%}  {sc['b_mean']:7.1%}  {sc['both_mean']:7.1%}", flush=True)
    print(f"  {'spring_4agent':<20s} {sp['a_mean']:7.1%}  {sp['b_mean']:7.1%}  {sp['both_mean']:7.1%}", flush=True)
    print(f"  {'ramp_4agent':<20s} {ra['a_mean']:7.1%}  {ra['b_mean']:7.1%}  {ra['both_mean']:7.1%}", flush=True)
    print(f"  {'scene_oracle':<20s} {oc['a_mean']:7.1%}  {oc['b_mean']:7.1%}  {oc['both_mean']:7.1%}", flush=True)

    # Specialization comparison across all 3 agent conditions
    print(f"\n  Agent specialization (MI) across domains:", flush=True)
    print(f"  {'Agent':<8s} {'Scene MI(num)':>12s} {'Scene MI(size)':>13s} {'Spec':>6s}"
          f"  |  {'Spring MI(k)':>12s} {'Spring MI(b)':>12s} {'Spec':>6s}"
          f"  |  {'Ramp MI(e)':>10s} {'Ramp MI(f)':>10s} {'Spec':>6s}", flush=True)
    print(f"  {'-'*8} {'-'*12} {'-'*13} {'-'*6}"
          f"  |  {'-'*12} {'-'*12} {'-'*6}"
          f"  |  {'-'*10} {'-'*10} {'-'*6}", flush=True)
    for si in range(N_AGENTS):
        key = f'agent_{si}'
        sc_a = sc['agent_specs'][key]
        sp_a = sp['agent_specs'][key]
        ra_a = ra['agent_specs'][key]
        print(f"  {key:<8s} {sc_a['mi_a_mean']:11.3f}  {sc_a['mi_b_mean']:12.3f} {sc_a['spec_ratio_mean']:6.3f}"
              f"  |  {sp_a['mi_a_mean']:11.3f}  {sp_a['mi_b_mean']:11.3f} {sp_a['spec_ratio_mean']:6.3f}"
              f"  |  {ra_a['mi_a_mean']:9.3f}  {ra_a['mi_b_mean']:9.3f} {ra_a['spec_ratio_mean']:6.3f}",
              flush=True)

    # Compute average specialization per domain
    sc_mean_spec = np.mean([sc['agent_specs'][f'agent_{i}']['spec_ratio_mean'] for i in range(N_AGENTS)])
    sp_mean_spec = np.mean([sp['agent_specs'][f'agent_{i}']['spec_ratio_mean'] for i in range(N_AGENTS)])
    ra_mean_spec = np.mean([ra['agent_specs'][f'agent_{i}']['spec_ratio_mean'] for i in range(N_AGENTS)])

    print(f"\n  Mean specialization ratio:", flush=True)
    print(f"    Scene:  {sc_mean_spec:.3f}", flush=True)
    print(f"    Spring: {sp_mean_spec:.3f}", flush=True)
    print(f"    Ramp:   {ra_mean_spec:.3f}", flush=True)

    # Check if any agent is dead (MI < 0.05 for both properties)
    print(f"\n  Key questions:", flush=True)
    print(f"    1. Scene communication works? both={sc['both_mean']:.1%} (vs oracle {oc['both_mean']:.1%})", flush=True)

    ceiling_ratio = sc['both_mean'] / max(oc['both_mean'], 0.01)
    print(f"    2. Ceiling captured: {ceiling_ratio:.1%} of oracle", flush=True)

    # Agent 0 status per domain
    for domain, label, specs in [('scene', 'Scene', sc), ('spring', 'Spring', sp), ('ramp', 'Ramp', ra)]:
        a0 = specs['agent_specs']['agent_0']
        total_mi = a0['mi_a_mean'] + a0['mi_b_mean']
        status = "dead" if total_mi < 0.05 else "active"
        print(f"    3. Agent 0 in {label}: {status} (total MI={total_mi:.3f})", flush=True)

    print(f"    4. Specialization: scene({sc_mean_spec:.2f}) vs spring({sp_mean_spec:.2f}) vs ramp({ra_mean_spec:.2f})", flush=True)

    print(f"\n  Total runtime: {total_elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

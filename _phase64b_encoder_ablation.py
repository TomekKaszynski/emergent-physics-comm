"""
Phase 64b: Encoder Ablation on Spring-Mass
============================================
Tests whether specialization patterns depend on encoder quality.

Phase 64 used a frozen random MLP to map (pos, vel) -> 384-dim features.
This ablation replaces it with a deterministic encoding:
  normalize (pos, vel) to [0,1], tile/repeat to 384-dim.

Same config: 4 agents, 2 positions, vocab=5, 400 epochs, population IL, 15 seeds.

Compare against Phase 64 spring_4agent results:
  - Accuracy
  - Specialization ratio (does Agent 0 still specialize for damping?)
  - MI patterns

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase64b_encoder_ablation.py
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
# Configuration (identical to Phase 64)
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

N_AGENTS = 4
AGENT_FRAMES = [[0], [1], [2], [3]]
N_POSITIONS = 2


# ══════════════════════════════════════════════════════════════════
# Deterministic Tiled Encoder (replaces FrozenEncoder)
# ══════════════════════════════════════════════════════════════════

def deterministic_encode(raw_features, n_scenes=300):
    """Normalize (pos, vel) to [0,1] and tile to 384-dim.

    raw_features: (n_scenes, 4, 2) — raw (position, velocity)
    Returns: (n_scenes, 4, 384) tensor
    """
    # Normalize each dimension to [0,1] across all scenes and frames
    flat = raw_features.reshape(-1, 2)  # (n_scenes*4, 2)

    # Per-dimension min/max normalization
    mins = flat.min(dim=0).values  # (2,)
    maxs = flat.max(dim=0).values  # (2,)
    ranges = (maxs - mins).clamp(min=1e-8)
    normed = (flat - mins) / ranges  # (n_scenes*4, 2) in [0,1]

    # Tile the 2-dim vector to 384-dim: repeat 192 times
    tiled = normed.repeat(1, DINO_DIM // 2)  # (n_scenes*4, 384)

    return tiled.reshape(n_scenes, 4, DINO_DIM)


# ══════════════════════════════════════════════════════════════════
# Spring-Mass Data Generation (same physics, different encoder)
# ══════════════════════════════════════════════════════════════════

def generate_spring_data(n_scenes=300, seed=42):
    """Generate spring-mass trajectories and encode with deterministic tiling.

    Returns:
        features: (n_scenes, 4, 384) — tiled normalized features
        raw_features: (n_scenes, 4, 2) — raw (position, velocity)
        k_bins: (n_scenes,) — spring constant bin indices 0-4
        b_bins: (n_scenes,) — damping bin indices 0-4
        train_ids, holdout_ids: arrays of scene indices
    """
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
        b = all_b[i]
        gamma = b / (2 * m)
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

    raw_tensor = torch.tensor(raw_features)
    features = deterministic_encode(raw_tensor, n_scenes)

    train_ids, holdout_ids = [], []
    for i in range(n_scenes):
        if (int(k_bins[i]), int(b_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return (features, raw_tensor,
            k_bins, b_bins,
            np.array(train_ids), np.array(holdout_ids))


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 64)
# ══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim // 2, 1)
        self.friction_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Helpers (identical to Phase 64)
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
# Training (identical to Phase 64)
# ══════════════════════════════════════════════════════════════════

def train_seed(seed, features, bins_a, bins_b, train_ids, holdout_ids, device):
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
# Aggregation
# ══════════════════════════════════════════════════════════════════

def _aggregate_results(seed_results):
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


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 64b: Encoder Ablation on Spring-Mass", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Encoder: deterministic normalize+tile (NOT frozen random MLP)", flush=True)
    print(f"  Architecture: {N_AGENTS} agents, {N_POSITIONS} pos, vocab={VOCAB_SIZE}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}, Seeds: {N_SEEDS}", flush=True)
    print(flush=True)

    # Generate spring-mass data with deterministic encoding
    print("  Generating spring-mass data (deterministic tiled encoding)...", flush=True)
    features, raw_feats, k_bins, b_bins, train_ids, holdout_ids = \
        generate_spring_data(n_scenes=300, seed=42)
    print(f"  Features: {features.shape}", flush=True)
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]", flush=True)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)
    print(flush=True)

    # Load Phase 64 results for comparison
    phase64_path = RESULTS_DIR / "phase64_transfer.json"
    with open(phase64_path) as f:
        phase64_results = json.load(f)
    p64_spring = phase64_results['spring_4agent']
    print(f"  Phase 64 reference (frozen random MLP):", flush=True)
    print(f"    both={p64_spring['both_mean']:.1%}  "
          f"k={p64_spring['a_mean']:.1%}  "
          f"damping={p64_spring['b_mean']:.1%}", flush=True)
    print(flush=True)

    # Train with deterministic encoding
    print(f"  {'='*60}", flush=True)
    print(f"  Training with deterministic tiled encoder", flush=True)
    print(f"  {'='*60}", flush=True)

    seed_results = []
    total_t0 = time.time()

    for seed in SEEDS:
        t0 = time.time()
        total_elapsed = time.time() - total_t0
        if seed > 0:
            remaining = total_elapsed / seed * (N_SEEDS - seed)
            eta_str = f"  ETA {remaining/60:.0f}min"
        else:
            eta_str = ""
        print(f"    [seed={seed}] Training...{eta_str}", flush=True)

        result = train_seed(seed, features, k_bins, b_bins,
                            train_ids, holdout_ids, DEVICE)
        elapsed = time.time() - t0
        print(f"    [seed={seed}] holdout k={result['a_acc']:.1%}  "
              f"b={result['b_acc']:.1%}  both={result['both_acc']:.1%}  "
              f"({elapsed:.0f}s)", flush=True)
        seed_results.append(result)

    tiled_summary = _aggregate_results(seed_results)
    total_elapsed = time.time() - total_t0

    # Build comparison results
    all_results = {
        'tiled_encoder': tiled_summary,
        'phase64_frozen_mlp': {
            'a_mean': p64_spring['a_mean'],
            'a_std': p64_spring['a_std'],
            'b_mean': p64_spring['b_mean'],
            'b_std': p64_spring['b_std'],
            'both_mean': p64_spring['both_mean'],
            'both_std': p64_spring['both_std'],
            'agent_specs': p64_spring['agent_specs'],
        },
        'comparison': {},
    }

    # Compute comparison metrics
    comp = all_results['comparison']
    comp['both_acc_diff'] = tiled_summary['both_mean'] - p64_spring['both_mean']
    comp['a_acc_diff'] = tiled_summary['a_mean'] - p64_spring['a_mean']
    comp['b_acc_diff'] = tiled_summary['b_mean'] - p64_spring['b_mean']

    # Per-agent specialization comparison
    for si in range(N_AGENTS):
        key = f'agent_{si}'
        t_spec = tiled_summary['agent_specs'][key]
        p_spec = p64_spring['agent_specs'][key]
        comp[f'{key}_spec_ratio_diff'] = t_spec['spec_ratio_mean'] - p_spec['spec_ratio_mean']
        comp[f'{key}_mi_a_diff'] = t_spec['mi_a_mean'] - p_spec['mi_a_mean']
        comp[f'{key}_mi_b_diff'] = t_spec['mi_b_mean'] - p_spec['mi_b_mean']

    # Agent 0 damping specialization check
    t_a0 = tiled_summary['agent_specs']['agent_0']
    p_a0 = p64_spring['agent_specs']['agent_0']
    comp['agent0_specializes_damping_tiled'] = t_a0['mi_b_mean'] > t_a0['mi_a_mean']
    comp['agent0_specializes_damping_frozen'] = p_a0['mi_b_mean'] > p_a0['mi_a_mean']

    # Save results
    output_path = RESULTS_DIR / "phase64b_encoder_ablation.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return obj
        return obj

    json_str = json.dumps(all_results, indent=2, default=convert)
    with open(output_path, 'w') as f:
        f.write(json_str)
    print(f"\n  Saved results to {output_path}", flush=True)

    # Final summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("COMPARISON: Tiled Encoder vs Frozen Random MLP", flush=True)
    print("=" * 70, flush=True)

    ts = tiled_summary
    ps = p64_spring

    print(f"\n  Accuracy:", flush=True)
    print(f"  {'Encoder':<25s} {'k':>8s} {'damping':>8s} {'Both':>8s}", flush=True)
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    print(f"  {'Deterministic tiled':<25s} {ts['a_mean']:7.1%}  {ts['b_mean']:7.1%}  {ts['both_mean']:7.1%}", flush=True)
    print(f"  {'Frozen random MLP (P64)':<25s} {ps['a_mean']:7.1%}  {ps['b_mean']:7.1%}  {ps['both_mean']:7.1%}", flush=True)
    print(f"  {'Difference':<25s} {comp['a_acc_diff']:+7.1%}  {comp['b_acc_diff']:+7.1%}  {comp['both_acc_diff']:+7.1%}", flush=True)

    print(f"\n  Agent Specialization (MI):", flush=True)
    print(f"  {'Agent':<12s} {'Tiled MI(k)':>12s} {'Tiled MI(b)':>12s}  |  {'MLP MI(k)':>12s} {'MLP MI(b)':>12s}", flush=True)
    print(f"  {'-'*12} {'-'*12} {'-'*12}  |  {'-'*12} {'-'*12}", flush=True)
    for si in range(N_AGENTS):
        key = f'agent_{si}'
        t_a = ts['agent_specs'][key]
        p_a = ps['agent_specs'][key]
        print(f"  {key:<12s} {t_a['mi_a_mean']:11.3f}  {t_a['mi_b_mean']:11.3f}  "
              f"|  {p_a['mi_a_mean']:11.3f}  {p_a['mi_b_mean']:11.3f}", flush=True)

    print(f"\n  Specialization Ratios:", flush=True)
    for si in range(N_AGENTS):
        key = f'agent_{si}'
        t_a = ts['agent_specs'][key]
        p_a = ps['agent_specs'][key]
        print(f"  {key}: tiled={t_a['spec_ratio_mean']:.3f}  "
              f"MLP={p_a['spec_ratio_mean']:.3f}  "
              f"diff={comp[f'{key}_spec_ratio_diff']:+.3f}", flush=True)

    print(f"\n  Key findings:", flush=True)
    print(f"    Agent 0 specializes for damping? "
          f"tiled={comp['agent0_specializes_damping_tiled']}  "
          f"MLP={comp['agent0_specializes_damping_frozen']}", flush=True)
    print(f"    Accuracy delta: {comp['both_acc_diff']:+.1%}", flush=True)
    same_pattern = comp['agent0_specializes_damping_tiled'] == comp['agent0_specializes_damping_frozen']
    print(f"    Same specialization pattern? {same_pattern}", flush=True)

    print(f"\n  Total runtime: {total_elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

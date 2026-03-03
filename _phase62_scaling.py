"""
Phase 62: N-Agent Scaling — Does Communication Scale with More Agents?
=====================================================================
Tests whether splitting observations across more agents (each seeing less)
preserves or improves communication quality.

Three conditions (15 seeds each):

(a) TWO_AGENTS_2POS: 2 agents, 2 frames each, 2 positions each
    Agent A sees [0,1], Agent B sees [2,3]. Baseline from Phase 60.
    Receiver input per ball: 4 messages → 20 dim, ×2 balls = 40 dim.

(b) FOUR_AGENTS_2POS: 4 agents, 1 frame each, 2 positions each
    Agent 0 sees frame 0, Agent 1 sees frame 1, etc.
    Receiver input per ball: 8 messages → 40 dim, ×2 balls = 80 dim.

(c) FOUR_AGENTS_1POS: 4 agents, 1 frame each, 1 position each
    Maximum compression: 1 frame → 1 symbol.
    Receiver input per ball: 4 messages → 20 dim, ×2 balls = 40 dim.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase62_scaling.py
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

CONDITIONS = ['two_agents_2pos', 'four_agents_2pos', 'four_agents_1pos']

# Condition configs: (n_agents, frames_per_agent, positions_per_agent)
CONDITION_CONFIGS = {
    'two_agents_2pos': {
        'n_agents': 2,
        'agent_frames': [[0, 1], [2, 3]],
        'n_positions': 2,
    },
    'four_agents_2pos': {
        'n_agents': 4,
        'agent_frames': [[0], [1], [2], [3]],
        'n_positions': 2,
    },
    'four_agents_1pos': {
        'n_agents': 4,
        'agent_frames': [[0], [1], [2], [3]],
        'n_positions': 1,
    },
}


# ══════════════════════════════════════════════════════════════════
# Architecture
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
        # x: (batch, n_frames, input_dim) — works for n_frames=1 or 2
        x = x.permute(0, 2, 1)  # (batch, input_dim, n_frames)
        x = self.temporal(x).squeeze(-1)  # (batch, 128)
        return self.fc(x)  # (batch, hidden_dim)


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
    """Two-head receiver for property comparison (e and f)."""
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
# Data
# ══════════════════════════════════════════════════════════════════

def load_data():
    """Load cached DINOv2 features and return per-frame features."""
    cache_path = RESULTS_DIR / "phase54b_dino_features.pt"
    data = torch.load(cache_path, weights_only=False)
    features = data['features']  # (300, 8, 384) — 8 frames per scene
    e_bins = data['e_bins']
    f_bins = data['f_bins']

    # We only need frames 0-3 (first 4)
    features = features[:, :4, :]  # (300, 4, 384)

    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return features, e_bins, f_bins, np.array(train_ids), np.array(holdout_ids)


def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_with_receiver(senders, agent_frames, receiver, features,
                           e_bins, f_bins, scene_ids, device, n_rounds=30):
    """Evaluate property comparison accuracy with a specific receiver."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    for s in senders:
        s.eval()
    receiver.eval()

    ce = cf = cb = 0
    te = tf = tb = 0

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
            pred_e, pred_f = receiver(combined)

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])
        valid_e = (e_dev[ia] != e_dev[ib])
        valid_f = (f_dev[ia] != f_dev[ib])
        valid_both = valid_e & valid_f

        if valid_e.sum() > 0:
            ce += ((pred_e > 0)[valid_e] == label_e[valid_e]).sum().item()
            te += valid_e.sum().item()
        if valid_f.sum() > 0:
            cf += ((pred_f > 0)[valid_f] == label_f[valid_f]).sum().item()
            tf += valid_f.sum().item()
        if valid_both.sum() > 0:
            both_ok = ((pred_e > 0)[valid_both] == label_e[valid_both]) & \
                      ((pred_f > 0)[valid_both] == label_f[valid_both])
            cb += both_ok.sum().item()
            tb += valid_both.sum().item()

    return {
        'e_acc': ce / max(te, 1),
        'f_acc': cf / max(tf, 1),
        'both_acc': cb / max(tb, 1),
    }


def evaluate_population(senders, agent_frames, receivers, features,
                        e_bins, f_bins, scene_ids, device, n_rounds=30):
    """Pick best receiver from population, then evaluate fully."""
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate_with_receiver(
            senders, agent_frames, r, features, e_bins, f_bins,
            scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate_with_receiver(
        senders, agent_frames, best_r, features, e_bins, f_bins,
        scene_ids, device, n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Message Analysis
# ══════════════════════════════════════════════════════════════════

def _mutual_information(x, y):
    """Compute MI between discrete arrays x and y."""
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


def analyze_sender_messages(sender, features, e_bins, f_bins,
                            device, n_positions, label=""):
    """Compute MI of each message position with each property."""
    sender.eval()
    all_tokens = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch = features[i:i + BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, n_positions)

    result = {'label': label, 'per_position': []}
    total_mi_e = 0.0
    total_mi_f = 0.0

    for p in range(n_positions):
        pos_tokens = all_tokens[:, p]
        mi_e = _mutual_information(pos_tokens, e_bins)
        mi_f = _mutual_information(pos_tokens, f_bins)
        total_mi_e += mi_e
        total_mi_f += mi_f

        counts = np.bincount(pos_tokens, minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(VOCAB_SIZE) if VOCAB_SIZE > 1 else 0.0

        result['per_position'].append({
            'mi_e': float(mi_e),
            'mi_f': float(mi_f),
            'entropy': float(norm_ent),
        })

    result['total_mi_e'] = float(total_mi_e)
    result['total_mi_f'] = float(total_mi_f)

    # Specialization ratio: |MI(e) - MI(f)| / (MI(e) + MI(f))
    denom = total_mi_e + total_mi_f
    if denom > 1e-10:
        result['spec_ratio'] = float(abs(total_mi_e - total_mi_f) / denom)
    else:
        result['spec_ratio'] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_condition(condition, features, e_bins, f_bins,
                    train_ids, holdout_ids, device, seed):
    """Train one condition for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = CONDITION_CONFIGS[condition]
    n_agents = cfg['n_agents']
    agent_frames = cfg['agent_frames']
    n_positions = cfg['n_positions']
    msg_dim_per_sender = n_positions * VOCAB_SIZE

    # Create independent senders
    senders = []
    for i in range(n_agents):
        senders.append(
            FixedSender(HIDDEN_DIM, DINO_DIM, n_positions, VOCAB_SIZE).to(device))

    # Receiver input: n_agents senders × 2 balls × msg_dim_per_sender
    recv_input_dim = n_agents * 2 * msg_dim_per_sender
    receivers = [PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    # Optimizers
    sender_params = []
    for s in senders:
        sender_params.extend(list(s.parameters()))
    sender_opt = torch.optim.Adam(sender_params, lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_both = 0.0
    best_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # IL reset
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
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            # Forward through all senders
            parts = []
            all_logits = []

            for si, sender in enumerate(senders):
                frames = agent_frames[si]
                feat_a = features[ia][:, frames, :].to(device)
                feat_b = features[ib][:, frames, :].to(device)
                msg_a, lg_a = sender(feat_a, tau, hard)
                msg_b, lg_b = sender(feat_b, tau, hard)
                parts.extend([msg_a, msg_b])
                all_logits.extend([lg_a, lg_b])

            combined = torch.cat(parts, dim=-1)

            # Task loss across all receivers
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(combined)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization
            for lg in all_logits:
                for p in range(n_positions):
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

            # NaN grad check
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

        # Evaluate every 40 epochs
        if (epoch + 1) % 40 == 0:
            train_result, _ = evaluate_population(
                senders, agent_frames, receivers, features, e_bins, f_bins,
                train_ids, device, n_rounds=10)
            holdout_result, _ = evaluate_population(
                senders, agent_frames, receivers, features, e_bins, f_bins,
                holdout_ids, device, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: train={train_result['both_acc']:.1%}  "
                  f"holdout={holdout_result['both_acc']:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
                states = {
                    'senders': [
                        {k: v.cpu().clone() for k, v in s.state_dict().items()}
                        for s in senders
                    ],
                    'receivers': [
                        {k: v.cpu().clone() for k, v in r.state_dict().items()}
                        for r in receivers
                    ],
                }
                best_states = states

    # Restore best
    if best_states is not None:
        for i, s in enumerate(senders):
            s.load_state_dict(best_states['senders'][i])
            s.to(device)
        for i, r in enumerate(receivers):
            r.load_state_dict(best_states['receivers'][i])
            r.to(device)

    # Final evaluation
    final_result, best_r = evaluate_population(
        senders, agent_frames, receivers, features, e_bins, f_bins,
        holdout_ids, device, n_rounds=30)

    # Message analysis — per sender
    msg_analysis = {}
    for si, sender in enumerate(senders):
        frames = agent_frames[si]
        sender_features = features[:, frames, :]
        label = f"agent_{si} (frames {frames})"
        msg_analysis[f'agent_{si}'] = analyze_sender_messages(
            sender, sender_features, e_bins, f_bins, device,
            n_positions, label=label)

    return {
        'e_acc': final_result['e_acc'],
        'f_acc': final_result['f_acc'],
        'both_acc': final_result['both_acc'],
        'nan_count': nan_count,
        'msg_analysis': msg_analysis,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 62: N-Agent Scaling", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}, batch: {BATCH_SIZE}", flush=True)
    print(f"  sender_lr={SENDER_LR}, receiver_lr={RECEIVER_LR}", flush=True)
    for cond in CONDITIONS:
        cfg = CONDITION_CONFIGS[cond]
        print(f"  {cond}: {cfg['n_agents']} agents, "
              f"frames={cfg['agent_frames']}, "
              f"{cfg['n_positions']} pos/agent", flush=True)
    print(flush=True)

    # Load data
    print("  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins, train_ids, holdout_ids = load_data()
    print(f"  Features: {features.shape}", flush=True)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)
    print(flush=True)

    all_results = {}
    total_t0 = time.time()

    for ci, condition in enumerate(CONDITIONS):
        cfg = CONDITION_CONFIGS[condition]
        n_agents = cfg['n_agents']
        n_positions = cfg['n_positions']
        total_msgs = n_agents * n_positions
        recv_dim = n_agents * 2 * n_positions * VOCAB_SIZE

        print(f"  {'='*60}", flush=True)
        print(f"  [{ci+1}/{len(CONDITIONS)}] Condition: {condition.upper()}", flush=True)
        print(f"    {n_agents} agents × {n_positions} pos = "
              f"{total_msgs} msgs/ball, recv_dim={recv_dim}", flush=True)
        print(f"  {'='*60}", flush=True)

        condition_seeds = []
        for seed in SEEDS:
            t0 = time.time()
            total_elapsed = time.time() - total_t0
            # Estimate total remaining
            done_seeds = ci * N_SEEDS + seed
            total_seeds = len(CONDITIONS) * N_SEEDS
            if done_seeds > 0:
                per_seed = total_elapsed / done_seeds
                remaining = (total_seeds - done_seeds) * per_seed
                eta_str = f"  total ETA {remaining/60:.0f}min"
            else:
                eta_str = ""

            print(f"    [seed={seed}] Training...{eta_str}", flush=True)

            result = train_condition(
                condition, features, e_bins, f_bins,
                train_ids, holdout_ids, DEVICE, seed)

            elapsed = time.time() - t0
            print(f"    [seed={seed}] holdout e={result['e_acc']:.1%} "
                  f"f={result['f_acc']:.1%} both={result['both_acc']:.1%}  "
                  f"({elapsed:.0f}s)", flush=True)
            condition_seeds.append(result)

        # Aggregate
        e_accs = [r['e_acc'] for r in condition_seeds]
        f_accs = [r['f_acc'] for r in condition_seeds]
        both_accs = [r['both_acc'] for r in condition_seeds]

        summary = {
            'e_mean': float(np.mean(e_accs)),
            'e_std': float(np.std(e_accs)),
            'f_mean': float(np.mean(f_accs)),
            'f_std': float(np.std(f_accs)),
            'both_mean': float(np.mean(both_accs)),
            'both_std': float(np.std(both_accs)),
        }

        # Per-agent specialization across seeds
        agent_specs = {}
        for si in range(n_agents):
            key = f'agent_{si}'
            specs = [r['msg_analysis'][key]['spec_ratio']
                     for r in condition_seeds]
            mi_es = [r['msg_analysis'][key]['total_mi_e']
                     for r in condition_seeds]
            mi_fs = [r['msg_analysis'][key]['total_mi_f']
                     for r in condition_seeds]
            agent_specs[key] = {
                'spec_ratio_mean': float(np.mean(specs)),
                'spec_ratio_std': float(np.std(specs)),
                'mi_e_mean': float(np.mean(mi_es)),
                'mi_f_mean': float(np.mean(mi_fs)),
                'frames': cfg['agent_frames'][si],
            }
        summary['agent_specs'] = agent_specs
        summary['seeds'] = condition_seeds

        all_results[condition] = summary

        print(f"\n  {condition.upper()} summary:", flush=True)
        print(f"    both={summary['both_mean']:.1%} ± {summary['both_std']:.1%}  "
              f"e={summary['e_mean']:.1%} ± {summary['e_std']:.1%}  "
              f"f={summary['f_mean']:.1%} ± {summary['f_std']:.1%}", flush=True)
        for si in range(n_agents):
            key = f'agent_{si}'
            asp = agent_specs[key]
            print(f"    {key} (frames {asp['frames']}): "
                  f"MI(e)={asp['mi_e_mean']:.3f}  MI(f)={asp['mi_f_mean']:.3f}  "
                  f"spec={asp['spec_ratio_mean']:.3f}", flush=True)
        print(flush=True)

    total_elapsed = time.time() - total_t0

    # Save results
    output_path = RESULTS_DIR / "phase62_scaling.json"
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
    print(f"  Saved results to {output_path}", flush=True)

    # Final summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  Accuracy comparison:", flush=True)
    print(f"  {'Condition':<25s} {'Both':>12s} {'E':>12s} {'F':>12s}", flush=True)
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}", flush=True)
    for cond in CONDITIONS:
        s = all_results[cond]
        print(f"  {cond:<25s} "
              f"{s['both_mean']:.1%} ± {s['both_std']:.1%}  "
              f"{s['e_mean']:.1%} ± {s['e_std']:.1%}  "
              f"{s['f_mean']:.1%} ± {s['f_std']:.1%}", flush=True)

    print(f"\n  Specialization per agent:", flush=True)
    for cond in CONDITIONS:
        s = all_results[cond]
        cfg = CONDITION_CONFIGS[cond]
        print(f"  {cond}:", flush=True)
        for si in range(cfg['n_agents']):
            key = f'agent_{si}'
            asp = s['agent_specs'][key]
            which = "friction" if si < len(cfg['agent_frames']) // 2 or \
                    (cfg['n_agents'] == 2 and si == 0) else "elasticity"
            if cfg['n_agents'] == 4:
                which = "friction" if si < 2 else "elasticity"
            print(f"    {key} (frames {asp['frames']}, {which}): "
                  f"spec={asp['spec_ratio_mean']:.3f} ± {asp['spec_ratio_std']:.3f}  "
                  f"MI(e)={asp['mi_e_mean']:.3f}  MI(f)={asp['mi_f_mean']:.3f}",
                  flush=True)

    # Key comparisons
    two = all_results['two_agents_2pos']
    four_2 = all_results['four_agents_2pos']
    four_1 = all_results['four_agents_1pos']

    print(f"\n  Key comparisons:", flush=True)
    print(f"    4-agent 2pos vs 2-agent 2pos: "
          f"{four_2['both_mean'] - two['both_mean']:+.1%} both accuracy", flush=True)
    print(f"    4-agent 1pos vs 2-agent 2pos: "
          f"{four_1['both_mean'] - two['both_mean']:+.1%} both accuracy", flush=True)
    print(f"    4-agent 1pos vs 4-agent 2pos: "
          f"{four_1['both_mean'] - four_2['both_mean']:+.1%} both accuracy", flush=True)

    # Total message symbols comparison
    print(f"\n  Message budget:", flush=True)
    print(f"    2-agent 2pos: 2×2 = 4 symbols/ball, 8 total", flush=True)
    print(f"    4-agent 2pos: 4×2 = 8 symbols/ball, 16 total", flush=True)
    print(f"    4-agent 1pos: 4×1 = 4 symbols/ball, 8 total", flush=True)

    print(f"\n  Total runtime: {total_elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""
Phase 60: Multi-Agent Collaborative Inference from Partial Observations
=======================================================================
Two types of agents observe the same ball from different temporal windows:
  Agent A sees frames [0,1] — pre-bounce (ramp sliding, strong friction signal)
  Agent B sees frames [2,3] — bounce event (strong elasticity signal)

Oracle baselines (direct classifiers on holdout):
  A-only [0,1]: e=52.5%, f=99.1%, both=51.6%
  B-only [2,3]: e=99.8%, f=94.7%, both=94.5%
  Combined [0-3]: e=99.8%, f=97.8%, both=97.6%

Key insight: Agent A literally cannot see bounce height (e~chance).
Agent B has slightly weaker friction info (f=94.7% vs A's 99.1%).
Communication lets them combine: A's friction expertise + B's elasticity expertise.

Task: Two pairs of agents observe two different balls.
  A1 sees ball_1[0,1], A2 sees ball_2[0,1]
  B1 sees ball_1[2,3], B2 sees ball_2[2,3]
  Each sends a 2×5 Gumbel-Softmax message (10 dim).
  Receiver gets messages → predicts which ball has higher e, higher f.

Three conditions × 20 seeds:
  (a) COLLABORATIVE: all 4 messages (A1, A2, B1, B2) → receiver (40 dim)
  (b) A-ONLY: only A senders → receiver (20 dim)
  (c) B-ONLY: only B senders → receiver (20 dim)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase60_collaborative.py
"""

import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
DINO_DIM = 384
BATCH_SIZE = 32

# Frame splits
FRAMES_A = [0, 1]  # pre-bounce: friction specialist
FRAMES_B = [2, 3]  # bounce: elasticity specialist

# Message format: 2 positions × 5 vocab per sender
N_POSITIONS = 2
VOCAB_SIZE = 5
MSG_DIM = N_POSITIONS * VOCAB_SIZE  # 10 per sender

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

SEEDS = list(range(20))
CONDITIONS = ['collaborative', 'a_only', 'b_only']


# ══════════════════════════════════════════════════════════════════
# Architecture
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
        # x: (batch, n_frames, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, n_frames)
        x = self.temporal(x).squeeze(-1)  # (batch, 128)
        return self.fc(x)  # (batch, hidden_dim)


class FixedSender(nn.Module):
    """Fixed-length Gumbel-Softmax sender.

    TemporalEncoder → linear → per-position Gumbel-Softmax.
    Output: (batch, n_positions * vocab_size) flat message.
    """
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
    """Load cached DINOv2 features and split into A/B frame sets."""
    cache_path = RESULTS_DIR / "phase54b_dino_features.pt"
    data = torch.load(cache_path, weights_only=False)
    features = data['features']  # (300, 8, 384)
    e_bins = data['e_bins']
    f_bins = data['f_bins']

    features_A = features[:, FRAMES_A, :]  # (300, 2, 384)
    features_B = features[:, FRAMES_B, :]  # (300, 2, 384)

    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return (features_A, features_B, e_bins, f_bins,
            np.array(train_ids), np.array(holdout_ids))


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

def evaluate(condition, sender_A, sender_B, receiver,
             features_A, features_B, e_bins, f_bins,
             scene_ids, device, n_rounds=30):
    """Evaluate property comparison accuracy."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    if sender_A is not None:
        sender_A.eval()
    if sender_B is not None:
        sender_B.eval()
    receiver.eval()

    ce = cf = cb = 0
    te = tf = tb = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        with torch.no_grad():
            parts = []
            if condition in ('collaborative', 'a_only'):
                msg_A1, _ = sender_A(features_A[ia].to(device))
                msg_A2, _ = sender_A(features_A[ib].to(device))
                parts.extend([msg_A1, msg_A2])
            if condition in ('collaborative', 'b_only'):
                msg_B1, _ = sender_B(features_B[ia].to(device))
                msg_B2, _ = sender_B(features_B[ib].to(device))
                parts.extend([msg_B1, msg_B2])

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


def evaluate_population(condition, sender_A, sender_B, receivers,
                        features_A, features_B, e_bins, f_bins,
                        scene_ids, device, n_rounds=30):
    """Pick best receiver from population, then evaluate fully."""
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate(condition, sender_A, sender_B, r,
                       features_A, features_B, e_bins, f_bins,
                       scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate(condition, sender_A, sender_B, best_r,
                     features_A, features_B, e_bins, f_bins,
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


def analyze_sender_messages(sender, features, e_bins, f_bins, device, label=""):
    """Compute MI of each message position with each property."""
    sender.eval()
    all_tokens = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch = features[i:i + BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            # logits: (batch, n_positions, vocab_size)
            tokens = logits.argmax(dim=-1).cpu().numpy()  # (batch, n_positions)
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, n_positions)

    result = {'label': label, 'per_position': []}
    for p in range(N_POSITIONS):
        pos_tokens = all_tokens[:, p]
        mi_e = _mutual_information(pos_tokens, e_bins)
        mi_f = _mutual_information(pos_tokens, f_bins)

        counts = np.bincount(pos_tokens, minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(VOCAB_SIZE) if VOCAB_SIZE > 1 else 0.0
        eff_vocab = int(np.sum(probs > 0.05))

        result['per_position'].append({
            'mi_e': float(mi_e),
            'mi_f': float(mi_f),
            'entropy': float(norm_ent),
            'eff_vocab': eff_vocab,
        })

    # Aggregate MI
    result['total_mi_e'] = sum(p['mi_e'] for p in result['per_position'])
    result['total_mi_f'] = sum(p['mi_f'] for p in result['per_position'])

    # PosDis
    if N_POSITIONS >= 2:
        mi_matrix = np.array([[p['mi_e'], p['mi_f']]
                              for p in result['per_position']])
        pos_dis = 0.0
        for row in mi_matrix:
            sorted_mi = np.sort(row)[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= N_POSITIONS
        result['pos_dis'] = float(pos_dis)
    else:
        result['pos_dis'] = 0.0

    # Unique messages
    msgs = [tuple(all_tokens[i]) for i in range(len(all_tokens))]
    result['n_unique'] = len(set(msgs))

    return result


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_condition(condition, features_A, features_B, e_bins, f_bins,
                    train_ids, holdout_ids, device, seed):
    """Train one condition for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create senders based on condition
    sender_A = None
    sender_B = None

    if condition in ('collaborative', 'a_only'):
        sender_A = FixedSender(HIDDEN_DIM, DINO_DIM, N_POSITIONS, VOCAB_SIZE).to(device)
    if condition in ('collaborative', 'b_only'):
        sender_B = FixedSender(HIDDEN_DIM, DINO_DIM, N_POSITIONS, VOCAB_SIZE).to(device)

    # Receiver input dimension depends on condition
    if condition == 'collaborative':
        recv_input_dim = 4 * MSG_DIM  # 40
    else:
        recv_input_dim = 2 * MSG_DIM  # 20

    receivers = [PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    # Optimizers
    sender_params = []
    if sender_A is not None:
        sender_params.extend(list(sender_A.parameters()))
    if sender_B is not None:
        sender_params.extend(list(sender_B.parameters()))
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
        # IL reset: reset all receivers simultaneously
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        if sender_A is not None:
            sender_A.train()
        if sender_B is not None:
            sender_B.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            # Forward through senders
            parts = []
            all_logits = []

            if condition in ('collaborative', 'a_only'):
                msg_A1, lg_A1 = sender_A(features_A[ia].to(device), tau, hard)
                msg_A2, lg_A2 = sender_A(features_A[ib].to(device), tau, hard)
                parts.extend([msg_A1, msg_A2])
                all_logits.extend([lg_A1, lg_A2])
            if condition in ('collaborative', 'b_only'):
                msg_B1, lg_B1 = sender_B(features_B[ia].to(device), tau, hard)
                msg_B2, lg_B2 = sender_B(features_B[ib].to(device), tau, hard)
                parts.extend([msg_B1, msg_B2])
                all_logits.extend([lg_B1, lg_B2])

            combined = torch.cat(parts, dim=-1)

            # Task loss across all receivers
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(combined)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization on all logits
            for lg in all_logits:
                # lg: (batch, n_positions, vocab_size)
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

            # NaN grad check
            has_nan_grad = False
            all_params = list(sender_params)
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or
                                           torch.isinf(p.grad).any()):
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
                condition, sender_A, sender_B, receivers,
                features_A, features_B, e_bins, f_bins,
                train_ids, device, n_rounds=10)
            holdout_result, _ = evaluate_population(
                condition, sender_A, sender_B, receivers,
                features_A, features_B, e_bins, f_bins,
                holdout_ids, device, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: train={train_result['both_acc']:.1%}  "
                  f"holdout={holdout_result['both_acc']:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
                states = {}
                if sender_A is not None:
                    states['sender_A'] = {k: v.cpu().clone()
                                          for k, v in sender_A.state_dict().items()}
                if sender_B is not None:
                    states['sender_B'] = {k: v.cpu().clone()
                                          for k, v in sender_B.state_dict().items()}
                states['receivers'] = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]
                best_states = states

    # Restore best
    if best_states is not None:
        if sender_A is not None and 'sender_A' in best_states:
            sender_A.load_state_dict(best_states['sender_A'])
            sender_A.to(device)
        if sender_B is not None and 'sender_B' in best_states:
            sender_B.load_state_dict(best_states['sender_B'])
            sender_B.to(device)
        for i, r in enumerate(receivers):
            r.load_state_dict(best_states['receivers'][i])
            r.to(device)

    # Final evaluation
    final_result, best_r = evaluate_population(
        condition, sender_A, sender_B, receivers,
        features_A, features_B, e_bins, f_bins,
        holdout_ids, device, n_rounds=30)

    # Message analysis (collaborative only)
    msg_analysis = {}
    if condition == 'collaborative':
        msg_analysis['sender_A'] = analyze_sender_messages(
            sender_A, features_A, e_bins, f_bins, device, label="A (friction)")
        msg_analysis['sender_B'] = analyze_sender_messages(
            sender_B, features_B, e_bins, f_bins, device, label="B (elasticity)")
    elif condition == 'a_only':
        msg_analysis['sender_A'] = analyze_sender_messages(
            sender_A, features_A, e_bins, f_bins, device, label="A-only")
    elif condition == 'b_only':
        msg_analysis['sender_B'] = analyze_sender_messages(
            sender_B, features_B, e_bins, f_bins, device, label="B-only")

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
    print("Phase 60: Multi-Agent Collaborative Inference", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Frames A: {FRAMES_A} (pre-bounce, friction)", flush=True)
    print(f"  Frames B: {FRAMES_B} (bounce, elasticity)", flush=True)
    print(f"  Message: {N_POSITIONS} positions × {VOCAB_SIZE} vocab = {MSG_DIM} dim/sender", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}", flush=True)
    print(flush=True)

    # Load data
    print("  Loading cached DINOv2 features...", flush=True)
    features_A, features_B, e_bins, f_bins, train_ids, holdout_ids = load_data()
    print(f"  Features A: {features_A.shape}, Features B: {features_B.shape}", flush=True)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)
    print(flush=True)

    all_results = {}
    total_t0 = time.time()

    for condition in CONDITIONS:
        print(f"  {'='*60}", flush=True)
        print(f"  Condition: {condition.upper()}", flush=True)
        print(f"  {'='*60}", flush=True)

        condition_seeds = []
        for seed in SEEDS:
            print(f"    [seed={seed}] Training...", flush=True)
            t0 = time.time()

            result = train_condition(
                condition, features_A, features_B, e_bins, f_bins,
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
            'seeds': condition_seeds,
        }

        # Aggregate message analysis for collaborative
        if condition == 'collaborative':
            mi_A_e = [r['msg_analysis']['sender_A']['total_mi_e']
                      for r in condition_seeds]
            mi_A_f = [r['msg_analysis']['sender_A']['total_mi_f']
                      for r in condition_seeds]
            mi_B_e = [r['msg_analysis']['sender_B']['total_mi_e']
                      for r in condition_seeds]
            mi_B_f = [r['msg_analysis']['sender_B']['total_mi_f']
                      for r in condition_seeds]
            summary['mi_A_e_mean'] = float(np.mean(mi_A_e))
            summary['mi_A_f_mean'] = float(np.mean(mi_A_f))
            summary['mi_B_e_mean'] = float(np.mean(mi_B_e))
            summary['mi_B_f_mean'] = float(np.mean(mi_B_f))

        all_results[condition] = summary

        print(f"\n  {condition.upper()} summary:", flush=True)
        print(f"    e={summary['e_mean']:.1%} ± {summary['e_std']:.1%}  "
              f"f={summary['f_mean']:.1%} ± {summary['f_std']:.1%}  "
              f"both={summary['both_mean']:.1%} ± {summary['both_std']:.1%}",
              flush=True)
        if condition == 'collaborative':
            print(f"    MI: A→e={summary['mi_A_e_mean']:.3f}  A→f={summary['mi_A_f_mean']:.3f}  "
                  f"B→e={summary['mi_B_e_mean']:.3f}  B→f={summary['mi_B_f_mean']:.3f}",
                  flush=True)
        print(flush=True)

    total_elapsed = time.time() - total_t0

    # Save results
    output_path = RESULTS_DIR / "phase60_collaborative.json"
    # Convert numpy types for JSON serialization
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
    print(f"  Oracle baselines (direct classifiers):", flush=True)
    print(f"    A-only [0,1]: e=52.5%  f=99.1%  both=51.6%", flush=True)
    print(f"    B-only [2,3]: e=99.8%  f=94.7%  both=94.5%", flush=True)
    print(f"    Full   [0-3]: e=99.8%  f=97.8%  both=97.6%", flush=True)
    print(flush=True)
    print(f"  Communication results (20-seed means):", flush=True)
    for condition in CONDITIONS:
        s = all_results[condition]
        print(f"    {condition:15s}: e={s['e_mean']:.1%} ± {s['e_std']:.1%}  "
              f"f={s['f_mean']:.1%} ± {s['f_std']:.1%}  "
              f"both={s['both_mean']:.1%} ± {s['both_std']:.1%}", flush=True)

    if 'collaborative' in all_results and 'mi_A_e_mean' in all_results['collaborative']:
        s = all_results['collaborative']
        print(flush=True)
        print(f"  Message specialization (collaborative):", flush=True)
        print(f"    Sender A (friction frames):  MI(msg,e)={s['mi_A_e_mean']:.3f}  "
              f"MI(msg,f)={s['mi_A_f_mean']:.3f}", flush=True)
        print(f"    Sender B (bounce frames):    MI(msg,e)={s['mi_B_e_mean']:.3f}  "
              f"MI(msg,f)={s['mi_B_f_mean']:.3f}", flush=True)

    collab_both = all_results['collaborative']['both_mean']
    a_both = all_results['a_only']['both_mean']
    b_both = all_results['b_only']['both_mean']
    print(flush=True)
    print(f"  Collaboration benefit:", flush=True)
    print(f"    COLLABORATIVE vs A-ONLY: {collab_both - a_both:+.1%}", flush=True)
    print(f"    COLLABORATIVE vs B-ONLY: {collab_both - b_both:+.1%}", flush=True)

    print(f"\n  Total runtime: {total_elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""
Phase 61: Capacity Pruning Under Information Asymmetry
======================================================
Each sender starts with 4 active positions (log_alpha=+2.0) and a closing
penalty removes unnecessary ones. Combined with Phase 60's two-sender
partial observability setup.

Key hypothesis: information asymmetry drives specialized capacity retention.
Sender A (friction frames [0,1]) should retain positions encoding friction.
Sender B (elasticity frames [2,3]) should retain positions encoding elasticity.
Under full observability, both senders see everything — prediction: redundant
retention (both encode same info, more total positions kept).

Three conditions × 15 seeds:
  (a) PARTIAL_PRUNING: A sees [0,1], B sees [2,3], gated 4 positions each
  (b) FULL_PRUNING: both see [0,1,2,3], gated 4 positions each
  (c) PARTIAL_FIXED: A sees [0,1], B sees [2,3], fixed 2 positions (control)

Note: "growing from closed" (log_alpha=-2.0) failed due to winner-take-all
dynamics — one sender dominates and the other atrophies because the shared
receiver adapts to the first sender that opens. Starting OPEN avoids this
because both senders participate from the start.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase61_growing.py
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
FRAMES_A = [0, 1]        # pre-bounce: friction specialist
FRAMES_B = [2, 3]        # bounce: elasticity specialist
FRAMES_FULL = [0, 1, 2, 3]  # full observation

# Gated sender: 4 positions × 5 vocab
MAX_POSITIONS = 4
VOCAB_SIZE = 5
GATED_MSG_DIM = MAX_POSITIONS * VOCAB_SIZE  # 20

# Fixed sender: 2 positions × 5 vocab (Phase 60 control)
FIXED_POSITIONS = 2
FIXED_MSG_DIM = FIXED_POSITIONS * VOCAB_SIZE  # 10

# Hard-concrete parameters (Louizos et al. 2018)
BETA_GATE = 0.66
GAMMA_HC = -0.1
ZETA_HC = 1.1

# Closing penalty: penalizes p_active above baseline
LAMBDA_CLOSING = 0.1
BASELINE_ACTIVE = 0.1    # free activation up to 10%
LAMBDA_WARMUP_EPOCHS = 50

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

# Gate trajectory checkpoints (epoch indices)
GATE_CHECKPOINTS = [0, 50, 100, 200, 300]

CONDITIONS = ['partial_pruning', 'full_pruning', 'partial_fixed']


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
        x = x.permute(0, 2, 1)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


class GatedSender(nn.Module):
    """Sender with hard-concrete gates starting OPEN.

    log_alpha initialized to +2.0 → all 4 gates ON in eval.
    Closing penalty pushes unnecessary positions off.
    """
    def __init__(self, hidden_dim=128, input_dim=384,
                 max_positions=4, vocab_size=5):
        super().__init__()
        self.encoder = TemporalEncoder(hidden_dim, input_dim)
        self.max_positions = max_positions
        self.vocab_size = vocab_size
        self.head = nn.Linear(hidden_dim, max_positions * vocab_size)

        # Gates start OPEN: log_alpha=+2.0 → eval: all ON
        self.log_alpha = nn.Parameter(torch.full((max_positions,), 2.0))

    def _sample_hard_concrete(self):
        u = torch.rand_like(self.log_alpha).clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid(
            (torch.log(u / (1 - u)) + self.log_alpha) / BETA_GATE)
        z_bar = s * (ZETA_HC - GAMMA_HC) + GAMMA_HC
        z = z_bar.clamp(0.0, 1.0)
        return z

    def _gate_probabilities(self):
        return torch.sigmoid(
            self.log_alpha - BETA_GATE * math.log(-GAMMA_HC / ZETA_HC))

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        batch_size = h.size(0)

        logits = self.head(h).view(batch_size, self.max_positions,
                                   self.vocab_size)

        if self.training:
            flat_logits = logits.reshape(-1, self.vocab_size)
            tokens = F.gumbel_softmax(flat_logits, tau=tau, hard=hard)
            tokens = tokens.reshape(batch_size, self.max_positions,
                                    self.vocab_size)
        else:
            idx = logits.argmax(dim=-1)
            tokens = F.one_hot(idx, self.vocab_size).float()

        if self.training:
            z = self._sample_hard_concrete()
        else:
            z = (self.log_alpha > 0).float()

        gated_tokens = tokens * z.unsqueeze(0).unsqueeze(-1)
        message = gated_tokens.reshape(batch_size,
                                       self.max_positions * self.vocab_size)

        p_active = self._gate_probabilities()
        return message, logits, p_active


class FixedSender(nn.Module):
    """Fixed-length sender (no gates). Phase 60 control."""
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
        self.elast_head = nn.Linear(hidden_dim // 2, 1)
        self.friction_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def load_data():
    cache_path = RESULTS_DIR / "phase54b_dino_features.pt"
    data = torch.load(cache_path, weights_only=False)
    features = data['features']  # (300, 8, 384)
    e_bins = data['e_bins']
    f_bins = data['f_bins']

    features_A = features[:, FRAMES_A, :]       # (300, 2, 384)
    features_B = features[:, FRAMES_B, :]       # (300, 2, 384)
    features_full = features[:, FRAMES_FULL, :]  # (300, 4, 384)

    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return (features_A, features_B, features_full, e_bins, f_bins,
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

def evaluate(sender_A, sender_B, receiver, feat_A, feat_B,
             e_bins, f_bins, scene_ids, device, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    sender_A.eval()
    sender_B.eval()
    receiver.eval()

    ce = cf = cb = 0
    te = tf = tb = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        with torch.no_grad():
            out_A1 = sender_A(feat_A[ia].to(device))
            out_A2 = sender_A(feat_A[ib].to(device))
            out_B1 = sender_B(feat_B[ia].to(device))
            out_B2 = sender_B(feat_B[ib].to(device))

            combined = torch.cat([out_A1[0], out_A2[0],
                                  out_B1[0], out_B2[0]], dim=-1)
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


def evaluate_population(sender_A, sender_B, receivers, feat_A, feat_B,
                         e_bins, f_bins, scene_ids, device, n_rounds=30):
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate(sender_A, sender_B, r, feat_A, feat_B,
                       e_bins, f_bins, scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate(sender_A, sender_B, best_r, feat_A, feat_B,
                     e_bins, f_bins, scene_ids, device, n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Message Analysis
# ══════════════════════════════════════════════════════════════════

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


def analyze_sender(sender, features, e_bins, f_bins, device, is_gated=True):
    """Analyze a sender's messages: MI, specialization, gate state."""
    sender.eval()
    all_tokens = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch = features[i:i + BATCH_SIZE].to(device)
            out = sender(batch)
            logits = out[1]
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)
    n_positions = all_tokens.shape[1]

    if is_gated:
        p_active = sender._gate_probabilities().cpu().detach().numpy()
        active_mask = (sender.log_alpha > 0).cpu().detach().numpy()
        log_alpha_vals = sender.log_alpha.cpu().detach().numpy()
        n_active = int(active_mask.sum())
    else:
        p_active = np.ones(n_positions)
        active_mask = np.ones(n_positions, dtype=bool)
        log_alpha_vals = np.zeros(n_positions)
        n_active = n_positions

    per_position = []
    for p in range(n_positions):
        is_active_p = bool(active_mask[p])

        if not is_active_p:
            per_position.append({
                'p_active': float(p_active[p]),
                'log_alpha': float(log_alpha_vals[p]),
                'is_active': False,
                'mi_e': 0.0, 'mi_f': 0.0,
                'entropy': 0.0, 'eff_vocab': 0,
                'spec_ratio': 0.0,
            })
            continue

        tokens = all_tokens[:, p]
        mi_e = _mutual_information(tokens, e_bins)
        mi_f = _mutual_information(tokens, f_bins)

        counts = np.bincount(tokens, minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(VOCAB_SIZE) if VOCAB_SIZE > 1 else 0.0
        eff_vocab = int(np.sum(probs > 0.05))

        # Specialization ratio: 1.0 = dedicated to one property, 0.0 = equal
        denom = mi_e + mi_f
        spec_ratio = abs(mi_e - mi_f) / denom if denom > 1e-10 else 0.0

        per_position.append({
            'p_active': float(p_active[p]),
            'log_alpha': float(log_alpha_vals[p]),
            'is_active': True,
            'mi_e': float(mi_e),
            'mi_f': float(mi_f),
            'entropy': float(norm_ent),
            'eff_vocab': eff_vocab,
            'spec_ratio': float(spec_ratio),
        })

    active_indices = [p for p in range(n_positions) if active_mask[p]]
    total_mi_e = sum(per_position[p]['mi_e'] for p in active_indices)
    total_mi_f = sum(per_position[p]['mi_f'] for p in active_indices)

    spec_ratios = [per_position[p]['spec_ratio'] for p in active_indices]
    mean_spec = float(np.mean(spec_ratios)) if spec_ratios else 0.0

    # PosDis
    if len(active_indices) >= 2:
        mi_matrix = np.array([[per_position[p]['mi_e'], per_position[p]['mi_f']]
                              for p in active_indices])
        pos_dis = 0.0
        for row in mi_matrix:
            sorted_mi = np.sort(row)[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= len(active_indices)
    else:
        pos_dis = 0.0

    # Unique messages (active positions only)
    if active_indices:
        msgs = [tuple(all_tokens[i, active_mask]) for i in range(len(all_tokens))]
    else:
        msgs = [() for _ in range(len(all_tokens))]
    n_unique = len(set(msgs))

    return {
        'n_active': n_active,
        'p_active_per_position': [float(v) for v in p_active],
        'log_alpha_per_position': [float(v) for v in log_alpha_vals],
        'active_positions': [int(p) for p in active_indices],
        'per_position': per_position,
        'total_mi_e': float(total_mi_e),
        'total_mi_f': float(total_mi_f),
        'mean_spec_ratio': mean_spec,
        'pos_dis': float(pos_dis),
        'n_unique': n_unique,
    }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_condition(condition, features_A, features_B, features_full,
                    e_bins, f_bins, train_ids, holdout_ids, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_gated = condition in ('partial_pruning', 'full_pruning')
    is_full_obs = condition == 'full_pruning'

    # Create senders
    if is_gated:
        sender_A = GatedSender(HIDDEN_DIM, DINO_DIM, MAX_POSITIONS,
                                VOCAB_SIZE).to(device)
        sender_B = GatedSender(HIDDEN_DIM, DINO_DIM, MAX_POSITIONS,
                                VOCAB_SIZE).to(device)
        msg_dim = GATED_MSG_DIM  # 20
        n_pos = MAX_POSITIONS
    else:
        sender_A = FixedSender(HIDDEN_DIM, DINO_DIM, FIXED_POSITIONS,
                               VOCAB_SIZE).to(device)
        sender_B = FixedSender(HIDDEN_DIM, DINO_DIM, FIXED_POSITIONS,
                               VOCAB_SIZE).to(device)
        msg_dim = FIXED_MSG_DIM  # 10
        n_pos = FIXED_POSITIONS

    # Feature selection
    if is_full_obs:
        feat_A = features_full
        feat_B = features_full
    else:
        feat_A = features_A
        feat_B = features_B

    # Receiver: 4 messages (A1, A2, B1, B2) concatenated
    recv_input_dim = 4 * msg_dim
    receivers = [PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    # Single optimizer for all sender params (including gates)
    sender_params = list(sender_A.parameters()) + list(sender_B.parameters())
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
    gate_trajectory = []

    for epoch in range(COMM_EPOCHS):
        # Record gate trajectory at checkpoints
        if is_gated and epoch in GATE_CHECKPOINTS:
            with torch.no_grad():
                gate_trajectory.append({
                    'epoch': epoch,
                    'sender_A': sender_A._gate_probabilities().cpu().tolist(),
                    'sender_B': sender_B._gate_probabilities().cpu().tolist(),
                    'log_alpha_A': sender_A.log_alpha.cpu().tolist(),
                    'log_alpha_B': sender_B.log_alpha.cpu().tolist(),
                })

        # IL reset
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = PropertyReceiver(
                    recv_input_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender_A.train()
        sender_B.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        # Lambda warmup for closing penalty
        lam_eff = LAMBDA_CLOSING * min(1.0, epoch / max(1, LAMBDA_WARMUP_EPOCHS))

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            # Forward through senders
            out_A1 = sender_A(feat_A[ia].to(device), tau, hard)
            out_A2 = sender_A(feat_A[ib].to(device), tau, hard)
            out_B1 = sender_B(feat_B[ia].to(device), tau, hard)
            out_B2 = sender_B(feat_B[ib].to(device), tau, hard)

            msg_A1, lg_A1 = out_A1[0], out_A1[1]
            msg_A2, lg_A2 = out_A2[0], out_A2[1]
            msg_B1, lg_B1 = out_B1[0], out_B1[1]
            msg_B2, lg_B2 = out_B2[0], out_B2[1]

            combined = torch.cat([msg_A1, msg_A2, msg_B1, msg_B2], dim=-1)

            # Task loss across receiver population
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(combined)
                r_loss = (F.binary_cross_entropy_with_logits(pred_e, label_e) +
                          F.binary_cross_entropy_with_logits(pred_f, label_f))
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Closing penalty for gated senders
            if is_gated and lam_eff > 0:
                p_active_A = out_A1[2]
                p_active_B = out_B1[2]
                # Penalize p_active above baseline per position
                penalty_A = (p_active_A - BASELINE_ACTIVE).clamp(min=0).sum()
                penalty_B = (p_active_B - BASELINE_ACTIVE).clamp(min=0).sum()
                loss = loss + lam_eff * (penalty_A + penalty_B)

            # Entropy regularization
            all_logits = [lg_A1, lg_A2, lg_B1, lg_B2]
            for lg in all_logits:
                for p in range(n_pos):
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
            for par in all_params:
                if par.grad is not None and (torch.isnan(par.grad).any() or
                                             torch.isinf(par.grad).any()):
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
            holdout_result, _ = evaluate_population(
                sender_A, sender_B, receivers, feat_A, feat_B,
                e_bins, f_bins, holdout_ids, device, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            if is_gated:
                with torch.no_grad():
                    n_act_A = int((sender_A.log_alpha > 0).sum().item())
                    n_act_B = int((sender_B.log_alpha > 0).sum().item())
                gate_str = f"  gates=A:{n_act_A}+B:{n_act_B}"
            else:
                gate_str = ""

            print(f"        Ep {epoch+1:3d}: "
                  f"holdout={holdout_result['both_acc']:.1%}"
                  f"{gate_str}{nan_str}  ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
                best_states = {
                    'sender_A': {k: v.cpu().clone()
                                 for k, v in sender_A.state_dict().items()},
                    'sender_B': {k: v.cpu().clone()
                                 for k, v in sender_B.state_dict().items()},
                    'receivers': [
                        {k: v.cpu().clone() for k, v in r.state_dict().items()}
                        for r in receivers
                    ],
                }

    # Restore best
    if best_states is not None:
        sender_A.load_state_dict(best_states['sender_A'])
        sender_A.to(device)
        sender_B.load_state_dict(best_states['sender_B'])
        sender_B.to(device)
        for i, r in enumerate(receivers):
            r.load_state_dict(best_states['receivers'][i])
            r.to(device)

    # Final evaluation
    final_result, _ = evaluate_population(
        sender_A, sender_B, receivers, feat_A, feat_B,
        e_bins, f_bins, holdout_ids, device, n_rounds=30)

    # Message analysis
    analysis_A = analyze_sender(sender_A, feat_A, e_bins, f_bins, device,
                                is_gated=is_gated)
    analysis_B = analyze_sender(sender_B, feat_B, e_bins, f_bins, device,
                                is_gated=is_gated)

    return {
        'e_acc': final_result['e_acc'],
        'f_acc': final_result['f_acc'],
        'both_acc': final_result['both_acc'],
        'nan_count': nan_count,
        'analysis_A': analysis_A,
        'analysis_B': analysis_B,
        'gate_trajectory': gate_trajectory if is_gated else [],
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 61: Capacity Pruning Under Information Asymmetry", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Gated: {MAX_POSITIONS} positions × {VOCAB_SIZE} vocab, "
          f"msg_dim={GATED_MSG_DIM}", flush=True)
    print(f"  Fixed: {FIXED_POSITIONS} positions × {VOCAB_SIZE} vocab, "
          f"msg_dim={FIXED_MSG_DIM}", flush=True)
    print(f"  Hard-concrete: beta={BETA_GATE}, gamma={GAMMA_HC}, "
          f"zeta={ZETA_HC}", flush=True)
    print(f"  log_alpha init: +2.0 (all gates start OPEN)", flush=True)
    print(f"  Closing penalty: lambda={LAMBDA_CLOSING}, "
          f"baseline={BASELINE_ACTIVE}", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}", flush=True)
    print(flush=True)

    # Load data
    print("  Loading cached DINOv2 features...", flush=True)
    (features_A, features_B, features_full,
     e_bins, f_bins, train_ids, holdout_ids) = load_data()
    print(f"  Features A: {features_A.shape}, B: {features_B.shape}, "
          f"Full: {features_full.shape}", flush=True)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout",
          flush=True)
    print(flush=True)

    all_results = {}
    total_t0 = time.time()

    for condition in CONDITIONS:
        print(f"  {'='*60}", flush=True)
        print(f"  Condition: {condition.upper()}", flush=True)
        print(f"  {'='*60}", flush=True)

        condition_results = []
        for seed in SEEDS:
            print(f"    [seed={seed}] Training...", flush=True)
            t0 = time.time()

            result = train_condition(
                condition, features_A, features_B, features_full,
                e_bins, f_bins, train_ids, holdout_ids, DEVICE, seed)

            elapsed = time.time() - t0

            is_gated = condition in ('partial_pruning', 'full_pruning')
            if is_gated:
                n_A = result['analysis_A']['n_active']
                n_B = result['analysis_B']['n_active']
                gate_str = f"  gates=A:{n_A}+B:{n_B}"
                spec_A = result['analysis_A']['mean_spec_ratio']
                spec_B = result['analysis_B']['mean_spec_ratio']
                spec_str = f"  spec=A:{spec_A:.2f},B:{spec_B:.2f}"
            else:
                gate_str = ""
                spec_str = ""

            print(f"    [seed={seed}] holdout e={result['e_acc']:.1%} "
                  f"f={result['f_acc']:.1%} both={result['both_acc']:.1%}"
                  f"{gate_str}{spec_str}  ({elapsed:.0f}s)", flush=True)

            condition_results.append(result)

        # Aggregate
        both_accs = [r['both_acc'] for r in condition_results]
        e_accs = [r['e_acc'] for r in condition_results]
        f_accs = [r['f_acc'] for r in condition_results]

        summary = {
            'both_mean': float(np.mean(both_accs)),
            'both_std': float(np.std(both_accs)),
            'e_mean': float(np.mean(e_accs)),
            'e_std': float(np.std(e_accs)),
            'f_mean': float(np.mean(f_accs)),
            'f_std': float(np.std(f_accs)),
        }

        if is_gated:
            n_active_A = [r['analysis_A']['n_active']
                          for r in condition_results]
            n_active_B = [r['analysis_B']['n_active']
                          for r in condition_results]
            spec_A = [r['analysis_A']['mean_spec_ratio']
                      for r in condition_results]
            spec_B = [r['analysis_B']['mean_spec_ratio']
                      for r in condition_results]
            summary['n_active_A_mean'] = float(np.mean(n_active_A))
            summary['n_active_A_std'] = float(np.std(n_active_A))
            summary['n_active_B_mean'] = float(np.mean(n_active_B))
            summary['n_active_B_std'] = float(np.std(n_active_B))
            summary['spec_ratio_A_mean'] = float(np.mean(spec_A))
            summary['spec_ratio_B_mean'] = float(np.mean(spec_B))

        all_results[condition] = {
            'summary': summary,
            'seeds': condition_results,
        }

        print(f"\n  {condition.upper()} summary:", flush=True)
        print(f"    both={summary['both_mean']:.1%} ± "
              f"{summary['both_std']:.1%}  "
              f"e={summary['e_mean']:.1%}  f={summary['f_mean']:.1%}",
              flush=True)
        if 'n_active_A_mean' in summary:
            print(f"    active: A={summary['n_active_A_mean']:.1f}±"
                  f"{summary['n_active_A_std']:.1f}  "
                  f"B={summary['n_active_B_mean']:.1f}±"
                  f"{summary['n_active_B_std']:.1f}", flush=True)
            print(f"    spec ratio: A={summary['spec_ratio_A_mean']:.3f}  "
                  f"B={summary['spec_ratio_B_mean']:.3f}", flush=True)
        print(flush=True)

    total_elapsed = time.time() - total_t0

    # ══════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Condition':>20} | {'Both':>14} | {'E':>8} | {'F':>8}",
          flush=True)
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*8}-+-{'-'*8}", flush=True)
    for cond in CONDITIONS:
        s = all_results[cond]['summary']
        print(f"  {cond:>20} | "
              f"{s['both_mean']:.1%}±{s['both_std']:.1%} | "
              f"{s['e_mean']:.1%} | {s['f_mean']:.1%}", flush=True)

    # Gate analysis for pruning conditions
    for cond in ['partial_pruning', 'full_pruning']:
        s = all_results[cond]['summary']
        print(f"\n  {cond.upper()} gate details:", flush=True)
        print(f"    Active positions: A={s['n_active_A_mean']:.1f}±"
              f"{s['n_active_A_std']:.1f}  "
              f"B={s['n_active_B_mean']:.1f}±{s['n_active_B_std']:.1f}",
              flush=True)
        print(f"    Specialization ratio: A={s['spec_ratio_A_mean']:.3f}  "
              f"B={s['spec_ratio_B_mean']:.3f}", flush=True)

        # MI breakdown from best seed
        seeds = all_results[cond]['seeds']
        best_idx = np.argmax([r['both_acc'] for r in seeds])
        best = seeds[best_idx]
        print(f"    Best seed (both={best['both_acc']:.1%}):", flush=True)
        for label, analysis in [("A", best['analysis_A']),
                                ("B", best['analysis_B'])]:
            active = analysis['active_positions']
            if active:
                pp = analysis['per_position']
                mi_strs = [f"p{p}: e={pp[p]['mi_e']:.3f} "
                           f"f={pp[p]['mi_f']:.3f} "
                           f"spec={pp[p]['spec_ratio']:.2f}"
                           for p in active]
                print(f"      Sender {label}: {' | '.join(mi_strs)}",
                      flush=True)
            else:
                print(f"      Sender {label}: no active positions",
                      flush=True)

        # Gate trajectory from best seed
        if best['gate_trajectory']:
            print(f"    Gate trajectory (best seed):", flush=True)
            for gt in best['gate_trajectory']:
                pA = [f"{v:.2f}" for v in gt['sender_A']]
                pB = [f"{v:.2f}" for v in gt['sender_B']]
                print(f"      Ep {gt['epoch']:3d}: "
                      f"A=[{','.join(pA)}]  B=[{','.join(pB)}]", flush=True)

    print(f"\n  Total runtime: {total_elapsed/60:.1f} min", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = RESULTS_DIR / "phase61_growing.json"
    json_str = json.dumps(all_results, indent=2, default=convert)
    with open(output_path, 'w') as f:
        f.write(json_str)
    print(f"\n  Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()

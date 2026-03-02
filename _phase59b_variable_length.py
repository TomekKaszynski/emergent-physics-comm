"""
Phase 59b: Variable-Length Emergent Messages
=============================================
Sender generates tokens autoregressively via GRU with a STOP token.
Per-token cost (lambda) creates pressure for brevity.
The sender CHOOSES its message length — if 2 tokens suffice, it should stop at 2.

Architecture:
- TemporalEncoder → 128-dim scene representation (= GRU initial hidden state)
- GRU cell: input=prev_token_embedding, hidden=scene_rep → logits over [sym0..sym4, STOP]
- Gumbel-Softmax over 6 options per step
- Message = concatenation of one-hots at each step (6 × 6 = 36 dim), zeros after STOP

Three cost levels × 20 seeds:
  lambda=0.00 (no cost)
  lambda=0.01 (mild)
  lambda=0.05 (strong)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase59b_variable_length.py
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

VOCAB_SIZE = 5
MAX_LENGTH = 6
TOTAL_VOCAB = VOCAB_SIZE + 1  # 5 symbols + STOP
MSG_DIM = MAX_LENGTH * TOTAL_VOCAB  # 6 * 6 = 36

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

LAMBDA_WARMUP_EPOCHS = 50  # ramp length cost from 0 to target over first 50 epochs

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

SEEDS = list(range(20))
LAMBDAS = [0.0, 0.01, 0.05]


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


class GRUCellManual(nn.Module):
    """Manual GRU cell using only linear layers — avoids MPS GRUCell crash."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        return (1 - z) * h + z * h_tilde


class AutoregressiveSender(nn.Module):
    """Generates variable-length messages autoregressively via GRU.

    At each step, outputs logits over [sym0..sym4, STOP].
    Message = concatenation of masked one-hots. After STOP, remaining = zeros.
    """
    def __init__(self, encoder, hidden_dim, vocab_size=5, max_length=6):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.total_vocab = vocab_size + 1  # +1 for STOP

        # Token embedding for GRU input
        self.token_embed = nn.Linear(self.total_vocab, hidden_dim)

        # Manual GRU cell (avoids MPS nn.GRUCell crash)
        self.gru = GRUCellManual(hidden_dim, hidden_dim)

        # Output logits
        self.output_head = nn.Linear(hidden_dim, self.total_vocab)

        # Learned start token embedding
        self.start_embed = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, x, tau=1.0, hard=True):
        """
        Returns:
            message: (batch, max_length * total_vocab) = (batch, 36)
            all_logits: list of max_length tensors, each (batch, total_vocab)
            n_tokens: (batch,) differentiable count of steps taken
        """
        h = self.encoder(x)  # (batch, hidden_dim) — initial GRU hidden
        batch_size = h.size(0)
        device = h.device

        prev_embed = self.start_embed.unsqueeze(0).expand(batch_size, -1)

        all_outputs = []
        all_logits = []
        # running_continue = probability of NOT having emitted STOP before this step
        rc = torch.ones(batch_size, 1, device=device)
        n_tokens = torch.zeros(batch_size, device=device)

        for t in range(self.max_length):
            # This step counts if we haven't stopped yet
            n_tokens = n_tokens + rc.squeeze(-1)

            # GRU step
            h = self.gru(prev_embed, h)

            # Output logits over [sym0..sym4, STOP]
            logits = self.output_head(h)
            all_logits.append(logits)

            # Sample token
            if self.training:
                token = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                idx = logits.argmax(dim=-1)
                token = F.one_hot(idx, self.total_vocab).float()

            # Mask by running continue (zeros after STOP)
            masked = token * rc
            all_outputs.append(masked)

            # Update running continue: STOP is the last symbol (index = total_vocab - 1)
            p_stop = token[:, -1:]  # (batch, 1)
            rc = rc * (1 - p_stop)

            # Embed token for next GRU input
            prev_embed = self.token_embed(token)

        # Concatenate all step outputs → flat message vector
        message = torch.cat(all_outputs, dim=-1)  # (batch, max_length * total_vocab)
        return message, all_logits, n_tokens


class PropertyReceiver(nn.Module):
    """Two-head receiver for property comparison (e and f)."""
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim // 2, 1)
        self.friction_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def load_cached_features(cache_path):
    data = torch.load(cache_path, weights_only=False)
    return data['features'], data['e_bins'], data['f_bins']


def create_splits(e_bins, f_bins, holdout_cells):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in holdout_cells:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


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

def evaluate_property(sender, receiver, data_t, e_bins, f_bins,
                      scene_ids, device, n_rounds=30):
    """Evaluate property comparison: e, f, both accuracy."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    ce = cf = cb = 0
    te = tf = tb = 0

    sender.eval()
    receiver.eval()

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)

        with torch.no_grad():
            msg_a, _, _ = sender(da)
            msg_b, _, _ = sender(db)
            pred_e, pred_f = receiver(msg_a, msg_b)

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


def evaluate_property_population(sender, receivers, data_t, e_bins, f_bins,
                                 scene_ids, device, n_rounds=30):
    """Pick best receiver from population."""
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate_property(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate_property(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Variable-length message analysis
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


def analyze_variable_messages(sender, data_t, e_bins, f_bins, device):
    """Analyze variable-length messages: length distribution, per-position MI, etc."""
    sender.eval()
    all_lengths = []
    all_tokens = []  # (N, max_length) with -1 for positions after STOP

    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i + BATCH_SIZE].to(device)
            msg, logits_list, n_tok = sender(batch)

            for j in range(batch.size(0)):
                tokens = []
                length = 0
                for t, lgt in enumerate(logits_list):
                    sym = lgt[j].argmax().item()
                    if sym == TOTAL_VOCAB - 1:  # STOP
                        break
                    tokens.append(sym)
                    length += 1
                # If never stopped, length = max_length
                if length == 0 and len(tokens) == 0:
                    # Immediate STOP
                    pass
                while len(tokens) < MAX_LENGTH:
                    tokens.append(-1)
                all_tokens.append(tokens)
                all_lengths.append(length)

    all_tokens = np.array(all_tokens)  # (N, max_length)
    all_lengths = np.array(all_lengths)

    # Length statistics
    mean_length = float(all_lengths.mean())
    std_length = float(all_lengths.std())
    length_dist = np.bincount(all_lengths, minlength=MAX_LENGTH + 1).tolist()

    # Per-position analysis (only over scenes that reach that position)
    per_position = []
    for p in range(MAX_LENGTH):
        active = all_tokens[:, p] >= 0
        n_active = int(active.sum())
        active_frac = n_active / len(all_tokens)

        if n_active < 10:
            per_position.append({
                'active_frac': float(active_frac),
                'mi_e': 0.0, 'mi_f': 0.0,
                'entropy': 0.0, 'eff_vocab': 0,
            })
            continue

        active_tokens = all_tokens[active, p]
        active_e = e_bins[active]
        active_f = f_bins[active]

        mi_e = _mutual_information(active_tokens, active_e)
        mi_f = _mutual_information(active_tokens, active_f)

        counts = np.bincount(active_tokens, minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(VOCAB_SIZE) if VOCAB_SIZE > 1 else 0.0
        eff_vocab = int(np.sum(probs > 0.05))

        per_position.append({
            'active_frac': float(active_frac),
            'mi_e': float(mi_e),
            'mi_f': float(mi_f),
            'entropy': float(norm_ent),
            'eff_vocab': eff_vocab,
        })

    # PosDis over positions active in >50% of messages
    active_pos_indices = [p for p in range(MAX_LENGTH)
                          if per_position[p]['active_frac'] > 0.5]
    n_active_positions = len(active_pos_indices)

    if n_active_positions >= 2:
        mi_matrix = np.array([[per_position[p]['mi_e'], per_position[p]['mi_f']]
                              for p in active_pos_indices])
        pos_dis = 0.0
        for row in mi_matrix:
            sorted_mi = np.sort(row)[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_active_positions
    else:
        pos_dis = 0.0

    # Unique messages (content tokens only, before STOP)
    msgs = []
    for i in range(len(all_tokens)):
        msg = tuple(all_tokens[i, :all_lengths[i]])
        msgs.append(msg)
    n_unique = len(set(msgs))

    # TopSim with variable-length Hamming distance
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(data_t), size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        li, lj = all_lengths[i], all_lengths[j]
        dist = abs(li - lj)  # length difference
        for p in range(min(li, lj)):
            if all_tokens[i, p] != all_tokens[j, p]:
                dist += 1
        message_dists.append(dist)
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'mean_length': mean_length,
        'std_length': std_length,
        'length_dist': length_dist,
        'per_position': per_position,
        'pos_dis': float(pos_dis),
        'n_active_positions': n_active_positions,
        'n_unique_messages': n_unique,
        'topsim': float(topsim),
    }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_variable_length(lam, data_t, e_bins, f_bins, train_ids, holdout_ids,
                          device, seed):
    """Train autoregressive sender + receiver population with per-token cost.

    Returns (sender, receivers, nan_count).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    sender = AutoregressiveSender(encoder, HIDDEN_DIM, VOCAB_SIZE, MAX_LENGTH).to(device)
    receivers = [PropertyReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    max_entropy = math.log(TOTAL_VOCAB)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_both = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # Simultaneous IL: reset all receivers together
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = PropertyReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        # Lambda warmup: ramp from 0 to target lambda over first LAMBDA_WARMUP_EPOCHS
        lam_eff = lam * min(1.0, epoch / max(1, LAMBDA_WARMUP_EPOCHS))

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a, ntok_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b, ntok_b = sender(db, tau=tau, hard=hard)

            # Task loss
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Per-token cost
            if lam_eff > 0:
                length_cost = lam_eff * (ntok_a.mean() + ntok_b.mean()) / 2
                loss = loss + length_cost

            # Entropy regularization on all step logits
            for logits in logits_a + logits_b:
                log_probs = F.log_softmax(logits, dim=-1)
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
            all_params = list(sender.parameters())
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

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        # Evaluate every 40 epochs
        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                train_result, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=10)
                holdout_result, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=10)
                # Quick length estimate
                sample = data_t[:BATCH_SIZE].to(device)
                _, _, ntok = sender(sample)
                avg_len = ntok.mean().item()

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: train={train_result['both_acc']:.1%}  "
                  f"holdout={holdout_result['both_acc']:.1%}  "
                  f"len={avg_len:.1f}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}
                best_receiver_states = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]

    # Restore best
    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return sender, receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    print("=" * 70, flush=True)
    print("Phase 59b: Variable-Length Emergent Messages", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Task: 2-property comparison (e + f)", flush=True)
    print(f"  Max length: {MAX_LENGTH}, Vocab: {VOCAB_SIZE} + STOP", flush=True)
    print(f"  Lambdas: {LAMBDAS}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Comm epochs: {COMM_EPOCHS}, lambda warmup: {LAMBDA_WARMUP_EPOCHS}", flush=True)

    # Load features
    print("\n  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins = load_cached_features(
        RESULTS_DIR / "phase54b_dino_features.pt")
    data_t = features.clone()
    print(f"  Features: {data_t.shape}", flush=True)

    # Splits
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Run all seeds × lambdas
    all_results = []

    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n  {'='*60}", flush=True)
        print(f"  Seed {seed}", flush=True)
        print(f"  {'='*60}", flush=True)

        seed_result = {'seed': seed, 'conditions': {}}

        for lam in LAMBDAS:
            lam_name = f"lam={lam:.2f}"
            t_cond = time.time()
            print(f"    [{lam_name}] Training...", flush=True)

            sender, receivers, nan_count = train_variable_length(
                lam, data_t, e_bins, f_bins,
                train_ids, holdout_ids, DEVICE, seed)

            sender.eval()
            for r in receivers:
                r.eval()

            with torch.no_grad():
                train_eval, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, DEVICE, n_rounds=50)
                holdout_eval, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, DEVICE, n_rounds=50)

                analysis = analyze_variable_messages(
                    sender, data_t, e_bins, f_bins, DEVICE)

            dt_cond = time.time() - t_cond
            print(f"    {lam_name}: holdout e={holdout_eval['e_acc']:.1%} "
                  f"f={holdout_eval['f_acc']:.1%} "
                  f"both={holdout_eval['both_acc']:.1%}  "
                  f"len={analysis['mean_length']:.1f}±{analysis['std_length']:.1f}  "
                  f"uniq={analysis['n_unique_messages']}  "
                  f"PD={analysis['pos_dis']:.3f}  ({dt_cond:.0f}s)", flush=True)

            seed_result['conditions'][lam_name] = {
                'lambda': lam,
                'train': train_eval,
                'holdout': holdout_eval,
                'analysis': analysis,
                'nan_count': nan_count,
                'time_sec': dt_cond,
            }

            torch.mps.empty_cache()

        dt_seed = time.time() - t_seed
        print(f"    Seed {seed} total: {dt_seed:.0f}s", flush=True)
        all_results.append(seed_result)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY: Phase 59b Variable-Length Messages", flush=True)
    print("=" * 70, flush=True)

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        e_accs = [r['conditions'][lam_name]['holdout']['e_acc']
                  for r in all_results]
        f_accs = [r['conditions'][lam_name]['holdout']['f_acc']
                  for r in all_results]
        both_accs = [r['conditions'][lam_name]['holdout']['both_acc']
                     for r in all_results]
        lengths = [r['conditions'][lam_name]['analysis']['mean_length']
                   for r in all_results]
        pd_vals = [r['conditions'][lam_name]['analysis']['pos_dis']
                   for r in all_results]
        uniq_vals = [r['conditions'][lam_name]['analysis']['n_unique_messages']
                     for r in all_results]
        ts_vals = [r['conditions'][lam_name]['analysis']['topsim']
                   for r in all_results]
        act_vals = [r['conditions'][lam_name]['analysis']['n_active_positions']
                    for r in all_results]

        print(f"\n  --- {lam_name} ---", flush=True)
        print(f"    Holdout: e={np.mean(e_accs):.1%}±{np.std(e_accs):.1%}  "
              f"f={np.mean(f_accs):.1%}±{np.std(f_accs):.1%}  "
              f"both={np.mean(both_accs):.1%}±{np.std(both_accs):.1%}", flush=True)
        print(f"    Length: {np.mean(lengths):.2f}±{np.std(lengths):.2f}  "
              f"Active pos: {np.mean(act_vals):.1f}  "
              f"Unique: {np.mean(uniq_vals):.1f}", flush=True)
        print(f"    PosDis={np.mean(pd_vals):.3f}±{np.std(pd_vals):.3f}  "
              f"TopSim={np.mean(ts_vals):.3f}±{np.std(ts_vals):.3f}", flush=True)

        # Length distribution (averaged over seeds)
        all_ldists = [r['conditions'][lam_name]['analysis']['length_dist']
                      for r in all_results]
        max_len_vals = max(len(ld) for ld in all_ldists)
        avg_ldist = np.zeros(max_len_vals)
        for ld in all_ldists:
            for i, v in enumerate(ld):
                avg_ldist[i] += v
        avg_ldist /= len(all_ldists)
        total = avg_ldist.sum()
        ldist_pcts = [f"L{i}={avg_ldist[i]/total:.0%}" for i in range(max_len_vals)
                      if avg_ldist[i] > 0]
        print(f"    Length dist: {' '.join(ldist_pcts)}", flush=True)

        # Per-position summary (only positions active in >50% of seeds)
        for p in range(MAX_LENGTH):
            act_fracs = [r['conditions'][lam_name]['analysis']['per_position'][p]['active_frac']
                         for r in all_results]
            if np.mean(act_fracs) < 0.3:
                continue
            mis_e = [r['conditions'][lam_name]['analysis']['per_position'][p]['mi_e']
                     for r in all_results]
            mis_f = [r['conditions'][lam_name]['analysis']['per_position'][p]['mi_f']
                     for r in all_results]
            ents = [r['conditions'][lam_name]['analysis']['per_position'][p]['entropy']
                    for r in all_results]
            print(f"    pos{p}: active={np.mean(act_fracs):.0%}  "
                  f"H={np.mean(ents):.2f}  "
                  f"MI(e)={np.mean(mis_e):.3f}  MI(f)={np.mean(mis_f):.3f}",
                  flush=True)

    # Comparison table
    print(f"\n  === Comparison ===", flush=True)
    print(f"  {'Lambda':>8} | {'Both':>12} | {'Length':>10} | "
          f"{'PosDis':>10} | {'Unique':>7} | {'TopSim':>8}", flush=True)
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*7}-+-{'-'*8}", flush=True)

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        both = [r['conditions'][lam_name]['holdout']['both_acc']
                for r in all_results]
        lengths = [r['conditions'][lam_name]['analysis']['mean_length']
                   for r in all_results]
        pd = [r['conditions'][lam_name]['analysis']['pos_dis']
              for r in all_results]
        uniq = [r['conditions'][lam_name]['analysis']['n_unique_messages']
                for r in all_results]
        ts = [r['conditions'][lam_name]['analysis']['topsim']
              for r in all_results]
        print(f"  {lam:8.2f} | "
              f"{np.mean(both):.1%}±{np.std(both):.1%} | "
              f"{np.mean(lengths):.2f}±{np.std(lengths):.2f} | "
              f"{np.mean(pd):.3f}±{np.std(pd):.3f} | "
              f"{np.mean(uniq):5.1f} | "
              f"{np.mean(ts):.3f}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════

    output = {
        'config': {
            'task': 'property_comparison',
            'max_length': MAX_LENGTH,
            'vocab_size': VOCAB_SIZE,
            'total_vocab': TOTAL_VOCAB,
            'msg_dim': MSG_DIM,
            'lambdas': LAMBDAS,
            'lambda_warmup_epochs': LAMBDA_WARMUP_EPOCHS,
            'comm_epochs': COMM_EPOCHS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'entropy_threshold': ENTROPY_THRESHOLD,
            'entropy_coef': ENTROPY_COEF,
            'n_seeds': len(SEEDS),
        },
        'per_seed': all_results,
        'summary': {},
    }

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        both = [r['conditions'][lam_name]['holdout']['both_acc']
                for r in all_results]
        e = [r['conditions'][lam_name]['holdout']['e_acc'] for r in all_results]
        f = [r['conditions'][lam_name]['holdout']['f_acc'] for r in all_results]
        lengths = [r['conditions'][lam_name]['analysis']['mean_length']
                   for r in all_results]
        pd = [r['conditions'][lam_name]['analysis']['pos_dis']
              for r in all_results]
        uniq = [r['conditions'][lam_name]['analysis']['n_unique_messages']
                for r in all_results]
        ts = [r['conditions'][lam_name]['analysis']['topsim']
              for r in all_results]
        act = [r['conditions'][lam_name]['analysis']['n_active_positions']
               for r in all_results]

        output['summary'][lam_name] = {
            'e_holdout_mean': float(np.mean(e)),
            'e_holdout_std': float(np.std(e)),
            'f_holdout_mean': float(np.mean(f)),
            'f_holdout_std': float(np.std(f)),
            'both_holdout_mean': float(np.mean(both)),
            'both_holdout_std': float(np.std(both)),
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            'pos_dis_mean': float(np.mean(pd)),
            'pos_dis_std': float(np.std(pd)),
            'n_active_mean': float(np.mean(act)),
            'n_unique_mean': float(np.mean(uniq)),
            'topsim_mean': float(np.mean(ts)),
            'topsim_std': float(np.std(ts)),
        }

    out_path = RESULTS_DIR / "phase59b_variable_length.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

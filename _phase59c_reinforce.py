"""
Phase 59c: REINFORCE Variable-Length Communication
===================================================
Sender generates tokens autoregressively with REINFORCE.
At each step: sample from categorical over [sym0..sym4, STOP].
Log the log-probability. Message = token sequence until STOP (max 6).
Pad remaining positions with zeros.

Reward = accuracy on batch (both correct) - lambda * message_length.
REINFORCE with baseline (EMA of reward, alpha=0.99).
Sender loss = -log_prob * (reward - baseline).
Receiver trained with standard BCE backprop.

Three lambda values x 20 seeds:
  lambda=0.00 (no cost)
  lambda=0.005 (mild)
  lambda=0.02 (moderate)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase59c_reinforce.py
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

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

EMA_ALPHA = 0.99  # for REINFORCE baseline

SEEDS = list(range(20))
LAMBDAS = [0.0, 0.005, 0.02]


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


class ReinforceSender(nn.Module):
    """Autoregressive sender using REINFORCE (categorical sampling).

    At each step, samples a token from categorical distribution.
    Records log_probs for policy gradient.
    Message = concatenation of one-hot tokens, zeros after STOP.
    """
    def __init__(self, encoder, hidden_dim, vocab_size=5, max_length=6):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.total_vocab = vocab_size + 1  # +1 for STOP

        self.token_embed = nn.Linear(self.total_vocab, hidden_dim)
        self.gru = GRUCellManual(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, self.total_vocab)
        self.start_embed = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, x):
        """
        Returns:
            message: (batch, max_length * total_vocab) = (batch, 36)
                     one-hot at each position, zeros after STOP
            log_probs: (batch,) sum of log-probs of sampled tokens
            lengths: (batch,) number of tokens before STOP (int, non-differentiable)
        """
        h = self.encoder(x)  # (batch, hidden_dim)
        batch_size = h.size(0)
        device = h.device

        prev_embed = self.start_embed.unsqueeze(0).expand(batch_size, -1)

        all_onehots = []
        total_log_prob = torch.zeros(batch_size, device=device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for t in range(self.max_length):
            h = self.gru(prev_embed, h)
            logits = self.output_head(h)  # (batch, total_vocab)

            # Categorical distribution
            dist = torch.distributions.Categorical(logits=logits)

            if self.training:
                token_idx = dist.sample()  # (batch,)
            else:
                token_idx = logits.argmax(dim=-1)

            # Log prob (only for non-stopped samples)
            lp = dist.log_prob(token_idx)  # (batch,)
            lp = lp * (~stopped).float()  # zero out already-stopped
            total_log_prob = total_log_prob + lp

            # One-hot encoding
            onehot = F.one_hot(token_idx, self.total_vocab).float()  # (batch, total_vocab)

            # Zero out positions after STOP
            onehot = onehot * (~stopped).float().unsqueeze(-1)
            all_onehots.append(onehot)

            # Update lengths for samples that haven't stopped and didn't emit STOP
            is_stop = (token_idx == self.total_vocab - 1)
            not_yet_stopped = ~stopped
            lengths = lengths + (not_yet_stopped & ~is_stop).long()

            # Mark newly stopped
            stopped = stopped | is_stop

            # Embed for next step (use actual sampled token)
            prev_embed = self.token_embed(onehot.detach() if self.training else onehot)

            # Early exit if all stopped
            if stopped.all():
                # Pad remaining positions with zeros
                for _ in range(t + 1, self.max_length):
                    all_onehots.append(torch.zeros(batch_size, self.total_vocab, device=device))
                break

        message = torch.cat(all_onehots, dim=-1)  # (batch, 36)
        return message, total_log_prob, lengths


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
    sender.eval()
    all_lengths = []
    all_tokens = []  # (N, max_length) with -1 for positions after STOP

    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i + BATCH_SIZE].to(device)
            msg, _, lengths = sender(batch)

            # Decode tokens from one-hot message
            for j in range(batch.size(0)):
                length = lengths[j].item()
                tokens = []
                for t in range(MAX_LENGTH):
                    start = t * TOTAL_VOCAB
                    end = start + TOTAL_VOCAB
                    oh = msg[j, start:end]
                    if oh.sum() < 0.5:  # zero = after STOP
                        break
                    tokens.append(oh.argmax().item())
                while len(tokens) < MAX_LENGTH:
                    tokens.append(-1)
                all_tokens.append(tokens)
                all_lengths.append(length)

    all_tokens = np.array(all_tokens)
    all_lengths = np.array(all_lengths)

    mean_length = float(all_lengths.mean())
    std_length = float(all_lengths.std())
    length_dist = np.bincount(all_lengths, minlength=MAX_LENGTH + 1).tolist()

    # Per-position analysis
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

    # Unique messages
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
        dist = abs(li - lj)
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


def quick_length_stats(sender, data_t, device):
    """Quick length estimate from first batch."""
    sender.eval()
    with torch.no_grad():
        batch = data_t[:BATCH_SIZE].to(device)
        _, _, lengths = sender(batch)
    return lengths.float().mean().item()


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_reinforce(lam, data_t, e_bins, f_bins, train_ids, holdout_ids,
                    device, seed):
    """Train REINFORCE sender + receiver population.

    Sender: REINFORCE with EMA baseline.
    Receiver: standard BCE backprop (detached messages).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    sender = ReinforceSender(encoder, HIDDEN_DIM, VOCAB_SIZE, MAX_LENGTH).to(device)
    receivers = [PropertyReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    baseline = 0.0  # EMA baseline for REINFORCE
    best_holdout_both = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0
    t_start = time.time()

    # Track length trajectory
    length_trajectory = []

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

        epoch_rewards = []

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            # --- Sender forward (REINFORCE: samples tokens) ---
            msg_a, log_prob_a, len_a = sender(da)
            msg_b, log_prob_b, len_b = sender(db)

            # Messages are detached for receiver (receiver doesn't backprop through sender)
            msg_a_det = msg_a.detach()
            msg_b_det = msg_b.detach()

            # --- Receiver forward + loss ---
            receiver_loss = torch.tensor(0.0, device=device)
            all_pred_e = []
            all_pred_f = []
            for r in receivers:
                pred_e, pred_f = r(msg_a_det, msg_b_det)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                receiver_loss = receiver_loss + r_loss
                all_pred_e.append(pred_e.detach())
                all_pred_f.append(pred_f.detach())
            receiver_loss = receiver_loss / len(receivers)

            # Update receivers
            for opt in receiver_opts:
                opt.zero_grad()
            receiver_loss.backward()
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            for opt in receiver_opts:
                opt.step()

            # --- Compute reward for sender ---
            # Average predictions across receivers for reward signal
            avg_pred_e = torch.stack(all_pred_e).mean(dim=0)
            avg_pred_f = torch.stack(all_pred_f).mean(dim=0)

            valid_e = (e_dev[ia] != e_dev[ib])
            valid_f = (f_dev[ia] != f_dev[ib])
            valid_both = valid_e & valid_f

            # Per-sample reward: 1 if both correct, 0 otherwise (on valid pairs)
            correct_e = ((avg_pred_e > 0) == (label_e > 0.5))
            correct_f = ((avg_pred_f > 0) == (label_f > 0.5))
            both_correct = (correct_e & correct_f).float()

            # For invalid pairs, give reward 0.5 (neutral)
            reward = torch.where(valid_both, both_correct, torch.tensor(0.5, device=device))

            # Subtract length cost
            avg_length = (len_a.float() + len_b.float()) / 2.0
            reward = reward - lam * avg_length

            # REINFORCE: sender loss = -log_prob * (reward - baseline)
            advantage = reward - baseline
            sender_loss = -(log_prob_a + log_prob_b) * advantage.detach()
            sender_loss = sender_loss.mean()

            if torch.isnan(sender_loss) or torch.isinf(sender_loss):
                sender_opt.zero_grad()
                nan_count += 1
                continue

            # Update sender
            sender_opt.zero_grad()
            sender_loss.backward()

            has_nan_grad = False
            for p in sender.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or
                                           torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            sender_opt.step()

            # Update baseline (EMA)
            batch_reward = reward.mean().item()
            baseline = EMA_ALPHA * baseline + (1 - EMA_ALPHA) * batch_reward
            epoch_rewards.append(batch_reward)

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
                avg_len = quick_length_stats(sender, data_t, device)

            # Track length trajectory
            length_trajectory.append({
                'epoch': epoch + 1,
                'mean_length': avg_len,
                'train_both': train_result['both_acc'],
                'holdout_both': holdout_result['both_acc'],
            })

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            avg_rew = np.mean(epoch_rewards) if epoch_rewards else 0
            print(f"        Ep {epoch+1:3d}: train={train_result['both_acc']:.1%}  "
                  f"holdout={holdout_result['both_acc']:.1%}  "
                  f"len={avg_len:.1f}  rew={avg_rew:.3f}{nan_str}  "
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

    return sender, receivers, nan_count, length_trajectory


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    print("=" * 70, flush=True)
    print("Phase 59c: REINFORCE Variable-Length Communication", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Task: 2-property comparison (e + f)", flush=True)
    print(f"  Max length: {MAX_LENGTH}, Vocab: {VOCAB_SIZE} + STOP", flush=True)
    print(f"  Lambdas: {LAMBDAS}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Comm epochs: {COMM_EPOCHS}", flush=True)
    print(f"  REINFORCE baseline: EMA alpha={EMA_ALPHA}", flush=True)
    print(f"  Sender LR: {SENDER_LR}, Receiver LR: {RECEIVER_LR}", flush=True)

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
            lam_name = f"lam={lam:.3f}"
            t_cond = time.time()
            print(f"    [{lam_name}] Training...", flush=True)

            sender, receivers, nan_count, length_traj = train_reinforce(
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
                'length_trajectory': length_traj,
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
    print("RESULTS SUMMARY: Phase 59c REINFORCE Variable-Length", flush=True)
    print("=" * 70, flush=True)

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.3f}"
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

        # Length trajectory averaged over seeds
        all_trajs = [r['conditions'][lam_name]['length_trajectory']
                     for r in all_results]
        if all_trajs and all_trajs[0]:
            n_checkpoints = len(all_trajs[0])
            print(f"    Length trajectory:", flush=True)
            for cp in range(n_checkpoints):
                epoch = all_trajs[0][cp]['epoch']
                avg_len = np.mean([t[cp]['mean_length'] for t in all_trajs
                                   if cp < len(t)])
                avg_both = np.mean([t[cp]['holdout_both'] for t in all_trajs
                                    if cp < len(t)])
                print(f"      Ep {epoch:3d}: len={avg_len:.2f}  "
                      f"both={avg_both:.1%}", flush=True)

        # Per-position summary
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
        lam_name = f"lam={lam:.3f}"
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
        print(f"  {lam:8.3f} | "
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
            'comm_epochs': COMM_EPOCHS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'ema_alpha': EMA_ALPHA,
            'sender_lr': SENDER_LR,
            'receiver_lr': RECEIVER_LR,
            'n_seeds': len(SEEDS),
            'method': 'REINFORCE with EMA baseline',
        },
        'per_seed': all_results,
        'summary': {},
    }

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.3f}"
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

    out_path = RESULTS_DIR / "phase59c_reinforce.json"
    with open(out_path, 'w') as f_out:
        json.dump(output, f_out, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

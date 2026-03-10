"""
Phase 76: Cross-seed zero-shot coordination
=============================================
Retrain seeds 0-19 (Phase 69b config), save sender+best_receiver per seed.
Then test every cross-seed (sender_i, receiver_j) pair on holdout.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase76_cross_seed.py
"""

import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration (identical to Phase 69b)
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

ORACLE_EPOCHS = 100
ORACLE_LR = 1e-3
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

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
COMP_THRESHOLD = 0.4


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 69b)
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


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)
        ])

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        messages = []
        all_logits = []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                idx = logits.argmax(dim=-1)
                msg = F.one_hot(idx, self.vocab_size).float()
            messages.append(msg)
            all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
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


class Oracle(nn.Module):
    def __init__(self, encoder_cls, encoder_kwargs, hidden_dim):
        super().__init__()
        self.enc_a = encoder_cls(**encoder_kwargs)
        self.enc_b = encoder_cls(**encoder_kwargs)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim, 1)
        self.friction_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_a, x_b):
        ha = self.enc_a(x_a)
        hb = self.enc_b(x_b)
        h = self.shared(torch.cat([ha, hb], dim=-1))
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

def evaluate_accuracy(sender, receiver, data_t, e_bins, f_bins,
                      scene_ids, device, oracle_model=None, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])

        if oracle_model is not None:
            pred_e, pred_f = oracle_model(da, db)
        else:
            msg_a, _ = sender(da)
            msg_b, _ = sender(db)
            pred_e, pred_f = receiver(msg_a, msg_b)

        pred_e_bin = pred_e > 0
        pred_f_bin = pred_f > 0

        e_diff = torch.tensor(e_bins[ia] != e_bins[ib], device=device)
        f_diff = torch.tensor(f_bins[ia] != f_bins[ib], device=device)

        if e_diff.sum() > 0:
            correct_e += (pred_e_bin[e_diff] == label_e[e_diff]).sum().item()
            total_e += e_diff.sum().item()
        if f_diff.sum() > 0:
            correct_f += (pred_f_bin[f_diff] == label_f[f_diff]).sum().item()
            total_f += f_diff.sum().item()
        both_diff = e_diff & f_diff
        if both_diff.sum() > 0:
            both_ok = (pred_e_bin[both_diff] == label_e[both_diff]) & \
                      (pred_f_bin[both_diff] == label_f[both_diff])
            correct_both += both_ok.sum().item()
            total_both += both_diff.sum().item()

    return (correct_e / max(total_e, 1),
            correct_f / max(total_f, 1),
            correct_both / max(total_both, 1))


def evaluate_population(sender, receivers, data_t, e_bins, f_bins,
                        scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    best_idx = 0
    for ri, r in enumerate(receivers):
        _, _, both = evaluate_accuracy(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
            best_idx = ri
    return evaluate_accuracy(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r, best_idx


# ══════════════════════════════════════════════════════════════════
# Compositionality metrics
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


def compute_compositionality(sender, data_t, e_bins, f_bins, device):
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i+BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)
    n_pos = all_tokens.shape[1]

    attributes = np.stack([e_bins, f_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    return float(pos_dis), mi_matrix, all_tokens


# ══════════════════════════════════════════════════════════════════
# Training (identical to Phase 69b)
# ══════════════════════════════════════════════════════════════════

def train_oracle(data_t, e_bins, f_bins, train_ids, device, seed):
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    best_acc = 0.0
    best_state = None
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                _, _, acc_both = evaluate_accuracy(
                    None, None, data_t, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone()
                              for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def train_population(sender, receivers, data_t, e_bins, f_bins,
                     train_ids, holdout_ids, device, msg_dim, seed):
    s_lr = SENDER_LR
    r_lr = RECEIVER_LR
    sender_opt = torch.optim.Adam(sender.parameters(), lr=s_lr)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=r_lr) for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0

    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=r_lr)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b = sender(db, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
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
                epoch_nan += 1
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
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            break

        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (te, tf, tb), best_r, _ = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            print(f"        Ep {epoch+1:3d}: train={tb:.1%}  ETA {eta/60:.0f}min",
                  flush=True)

            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}
                best_receiver_states = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# Phase 1: Train all seeds and save models
# ══════════════════════════════════════════════════════════════════

def train_and_save_seed(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    # Oracle
    oracle, oracle_acc = train_oracle(data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Sender
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Population of receivers
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    receivers, nan_count = train_population(
        sender, receivers, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval — find best receiver
    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r, best_r_idx = evaluate_population(
            sender, receivers, data_t, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy(
            sender, best_r, data_t, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    # Compositionality
    with torch.no_grad():
        pos_dis, mi_matrix, all_tokens = compute_compositionality(
            sender, data_t, e_bins, f_bins, device)

    dt = time.time() - t0
    is_comp = pos_dis > COMP_THRESHOLD
    tag = "COMP" if is_comp else "hol"
    print(f"    -> holdout={hb:.1%}  PosDis={pos_dis:.3f}  [{tag}]  ({dt:.0f}s)",
          flush=True)

    # Determine position mapping: which position encodes which attribute?
    # mi_matrix is (n_pos, 2) where columns are [elasticity, friction]
    pos0_attr = 'e' if mi_matrix[0, 0] > mi_matrix[0, 1] else 'f'
    pos1_attr = 'e' if mi_matrix[1, 0] > mi_matrix[1, 1] else 'f'

    return {
        'seed': seed,
        'sender_state': {k: v.cpu() for k, v in sender.state_dict().items()},
        'receiver_state': {k: v.cpu() for k, v in best_r.state_dict().items()},
        'holdout_both': float(hb),
        'train_both': float(tb),
        'pos_dis': float(pos_dis),
        'is_comp': is_comp,
        'mi_matrix': mi_matrix.tolist(),
        'pos_mapping': [pos0_attr, pos1_attr],
        'all_tokens': all_tokens,  # (N_scenes, 2) token assignments
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Phase 2: Cross-seed evaluation
# ══════════════════════════════════════════════════════════════════

def cross_seed_eval(seed_data, data_t, e_bins, f_bins, holdout_ids, device):
    """Evaluate all sender_i × receiver_j pairs on holdout."""
    n = len(seed_data)
    msg_dim = VOCAB_SIZE * N_HEADS
    matrix = np.zeros((n, n))

    print(f"\n  Cross-seed evaluation ({n}x{n} = {n*n} pairs)...", flush=True)
    t0 = time.time()

    for i, sd_i in enumerate(seed_data):
        # Reconstruct sender_i
        encoder_i = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        sender_i = CompositionalSender(encoder_i, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
        sender_i.load_state_dict({k: v.to(device) for k, v in sd_i['sender_state'].items()})
        sender_i.eval()

        for j, sd_j in enumerate(seed_data):
            # Reconstruct receiver_j
            receiver_j = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
            receiver_j.load_state_dict({k: v.to(device) for k, v in sd_j['receiver_state'].items()})
            receiver_j.eval()

            with torch.no_grad():
                _, _, hb = evaluate_accuracy(
                    sender_i, receiver_j, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=50)
            matrix[i, j] = hb

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"    Row {i+1}/{n} done, ETA {eta/60:.1f}min", flush=True)

        torch.mps.empty_cache()

    return matrix


# ══════════════════════════════════════════════════════════════════
# Phase 3: Analysis
# ══════════════════════════════════════════════════════════════════

def analyze(seed_data, matrix):
    n = len(seed_data)
    comp_idx = [i for i in range(n) if seed_data[i]['is_comp']]
    hol_idx = [i for i in range(n) if not seed_data[i]['is_comp']]

    print(f"\n{'='*70}", flush=True)
    print(f"CROSS-SEED ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Compositional seeds: {len(comp_idx)} ({[seed_data[i]['seed'] for i in comp_idx]})",
          flush=True)
    print(f"  Holistic seeds:      {len(hol_idx)} ({[seed_data[i]['seed'] for i in hol_idx]})",
          flush=True)

    # Diagonal = same-seed (matched) pairs
    diag = np.array([matrix[i, i] for i in range(n)])
    print(f"\n  Same-seed (diagonal): {diag.mean():.1%} ± {diag.std():.1%}", flush=True)

    # 4 conditions (off-diagonal only)
    cc_vals, ch_vals, hc_vals, hh_vals = [], [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si_comp = seed_data[i]['is_comp']
            rj_comp = seed_data[j]['is_comp']
            if si_comp and rj_comp:
                cc_vals.append(matrix[i, j])
            elif si_comp and not rj_comp:
                ch_vals.append(matrix[i, j])
            elif not si_comp and rj_comp:
                hc_vals.append(matrix[i, j])
            else:
                hh_vals.append(matrix[i, j])

    cc = np.array(cc_vals) if cc_vals else np.array([0.0])
    ch = np.array(ch_vals) if ch_vals else np.array([0.0])
    hc = np.array(hc_vals) if hc_vals else np.array([0.0])
    hh = np.array(hh_vals) if hh_vals else np.array([0.0])

    print(f"\n  Cross-seed conditions (off-diagonal):", flush=True)
    print(f"  {'Condition':<30} | {'Mean':>7} | {'Std':>6} | {'N':>4}", flush=True)
    print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}", flush=True)
    print(f"  {'Comp sender + Comp receiver':<30} | {cc.mean():>6.1%} | {cc.std():>5.1%} | {len(cc_vals):>4}",
          flush=True)
    print(f"  {'Comp sender + Hol receiver':<30} | {ch.mean():>6.1%} | {ch.std():>5.1%} | {len(ch_vals):>4}",
          flush=True)
    print(f"  {'Hol sender + Comp receiver':<30} | {hc.mean():>6.1%} | {hc.std():>5.1%} | {len(hc_vals):>4}",
          flush=True)
    print(f"  {'Hol sender + Hol receiver':<30} | {hh.mean():>6.1%} | {hh.std():>5.1%} | {len(hh_vals):>4}",
          flush=True)

    # Statistical tests
    print(f"\n  Statistical tests:", flush=True)
    if len(cc_vals) > 1:
        t_cc_chance, p_cc_chance = stats.ttest_1samp(cc, 0.25)  # chance = 25% for both correct
        print(f"    Comp×Comp vs chance (25%): t={t_cc_chance:.2f}, p={p_cc_chance:.4f}",
              flush=True)
    if len(cc_vals) > 1 and len(hh_vals) > 1:
        t_cc_hh, p_cc_hh = stats.ttest_ind(cc, hh)
        print(f"    Comp×Comp vs Hol×Hol: t={t_cc_hh:.2f}, p={p_cc_hh:.4f}", flush=True)

    # Position mapping analysis
    print(f"\n  Position mapping (compositional seeds only):", flush=True)
    print(f"  {'Seed':>4} | {'Pos0':>4} | {'Pos1':>4} | {'MI(0,e)':>7} | {'MI(0,f)':>7} | {'MI(1,e)':>7} | {'MI(1,f)':>7}",
          flush=True)
    pos_mappings = {}
    for i in comp_idx:
        sd = seed_data[i]
        mi = sd['mi_matrix']
        mapping = tuple(sd['pos_mapping'])
        pos_mappings.setdefault(mapping, []).append(sd['seed'])
        print(f"  {sd['seed']:>4} | {sd['pos_mapping'][0]:>4} | {sd['pos_mapping'][1]:>4} | "
              f"{mi[0][0]:>7.3f} | {mi[0][1]:>7.3f} | {mi[1][0]:>7.3f} | {mi[1][1]:>7.3f}",
              flush=True)

    print(f"\n  Position mapping groups:", flush=True)
    for mapping, seeds in sorted(pos_mappings.items()):
        print(f"    {mapping}: {len(seeds)} seeds — {seeds}", flush=True)

    # Symbol alignment analysis (for same-mapping compositional seeds)
    print(f"\n  Symbol alignment (same-mapping comp seeds):", flush=True)
    for mapping, seed_list in sorted(pos_mappings.items()):
        if len(seed_list) < 2:
            continue
        print(f"\n    Mapping {mapping} ({len(seed_list)} seeds):", flush=True)
        # For each pair of seeds, check token agreement
        mapping_indices = [i for i in comp_idx if tuple(seed_data[i]['pos_mapping']) == mapping]
        agreements = []
        for a in range(len(mapping_indices)):
            for b in range(a + 1, len(mapping_indices)):
                ia, ib = mapping_indices[a], mapping_indices[b]
                tokens_a = seed_data[ia]['all_tokens']  # (N, 2)
                tokens_b = seed_data[ib]['all_tokens']  # (N, 2)
                # Per-position token agreement rate
                agree_p0 = np.mean(tokens_a[:, 0] == tokens_b[:, 0])
                agree_p1 = np.mean(tokens_a[:, 1] == tokens_b[:, 1])
                agreements.append((seed_data[ia]['seed'], seed_data[ib]['seed'],
                                   agree_p0, agree_p1))
        if agreements:
            avg_p0 = np.mean([a[2] for a in agreements])
            avg_p1 = np.mean([a[3] for a in agreements])
            print(f"      Avg token agreement: pos0={avg_p0:.1%}, pos1={avg_p1:.1%}",
                  flush=True)
            print(f"      (Chance agreement for vocab={VOCAB_SIZE}: {1/VOCAB_SIZE:.1%})",
                  flush=True)
            # Show a few examples
            for sa, sb, ap0, ap1 in agreements[:5]:
                print(f"      Seeds {sa}-{sb}: pos0={ap0:.1%}, pos1={ap1:.1%}", flush=True)

    # Cross-seed accuracy within same mapping vs different mapping
    print(f"\n  Cross-seed accuracy by position mapping:", flush=True)
    same_map_vals = []
    diff_map_vals = []
    for i in comp_idx:
        for j in comp_idx:
            if i == j:
                continue
            if tuple(seed_data[i]['pos_mapping']) == tuple(seed_data[j]['pos_mapping']):
                same_map_vals.append(matrix[i, j])
            else:
                diff_map_vals.append(matrix[i, j])
    if same_map_vals:
        print(f"    Same mapping: {np.mean(same_map_vals):.1%} ± {np.std(same_map_vals):.1%} (n={len(same_map_vals)})",
              flush=True)
    if diff_map_vals:
        print(f"    Diff mapping: {np.mean(diff_map_vals):.1%} ± {np.std(diff_map_vals):.1%} (n={len(diff_map_vals)})",
              flush=True)
    if same_map_vals and diff_map_vals and len(same_map_vals) > 1 and len(diff_map_vals) > 1:
        t_map, p_map = stats.ttest_ind(same_map_vals, diff_map_vals)
        print(f"    t={t_map:.2f}, p={p_map:.4f}", flush=True)

    return {
        'same_seed_mean': float(diag.mean()),
        'same_seed_std': float(diag.std()),
        'cc_mean': float(cc.mean()), 'cc_std': float(cc.std()), 'cc_n': len(cc_vals),
        'ch_mean': float(ch.mean()), 'ch_std': float(ch.std()), 'ch_n': len(ch_vals),
        'hc_mean': float(hc.mean()), 'hc_std': float(hc.std()), 'hc_n': len(hc_vals),
        'hh_mean': float(hh.mean()), 'hh_std': float(hh.std()), 'hh_n': len(hh_vals),
        'pos_mappings': {str(k): v for k, v in pos_mappings.items()},
    }


# ══════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════

def plot_heatmap(seed_data, matrix):
    n = len(seed_data)

    # Sort by PosDis
    sort_idx = np.argsort([sd['pos_dis'] for sd in seed_data])[::-1]
    sorted_matrix = matrix[sort_idx][:, sort_idx]
    sorted_seeds = [seed_data[i] for i in sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 0.4]})

    # Heatmap
    im = ax1.imshow(sorted_matrix, cmap='RdYlGn', vmin=0.2, vmax=1.0, aspect='equal')
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Holdout accuracy', fontsize=10)

    # Labels
    labels = []
    for sd in sorted_seeds:
        tag = '*' if sd['is_comp'] else ''
        labels.append(f"s{sd['seed']}{tag}")

    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, fontsize=7, rotation=45)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel('Receiver seed', fontsize=11)
    ax1.set_ylabel('Sender seed', fontsize=11)
    ax1.set_title('Cross-seed holdout accuracy\n(sorted by PosDis, * = compositional)', fontsize=12)

    # Draw box around comp×comp region
    n_comp = sum(1 for sd in sorted_seeds if sd['is_comp'])
    rect = plt.Rectangle((-0.5, -0.5), n_comp, n_comp,
                          linewidth=2, edgecolor='blue', facecolor='none',
                          linestyle='--')
    ax1.add_patch(rect)

    # Bar chart of conditions
    comp_idx = [i for i in range(n) if seed_data[i]['is_comp']]
    hol_idx = [i for i in range(n) if not seed_data[i]['is_comp']]

    conditions = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si = seed_data[i]['is_comp']
            rj = seed_data[j]['is_comp']
            key = ('C' if si else 'H') + '→' + ('C' if rj else 'H')
            conditions.setdefault(key, []).append(matrix[i, j])

    labels_bar = ['C→C', 'C→H', 'H→C', 'H→H', 'Same\nseed']
    means = [np.mean(conditions.get(k, [0])) for k in ['C→C', 'C→H', 'H→C', 'H→H']]
    means.append(np.mean([matrix[i, i] for i in range(n)]))
    stds = [np.std(conditions.get(k, [0])) for k in ['C→C', 'C→H', 'H→C', 'H→H']]
    stds.append(np.std([matrix[i, i] for i in range(n)]))
    colors = ['#2ca02c', '#98df8a', '#aec7e8', '#888888', '#ff7f0e']

    bars = ax2.bar(range(len(labels_bar)), means, yerr=stds, capsize=4,
                   color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(labels_bar)))
    ax2.set_xticklabels(labels_bar, fontsize=10)
    ax2.set_ylabel('Holdout accuracy', fontsize=11)
    ax2.set_title('Cross-seed conditions', fontsize=12)
    ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=9)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{mean:.0%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    fig_path = FIGURES_DIR / "fig_cross_seed_heatmap.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {fig_path}", flush=True)

    png_path = RESULTS_DIR / "phase76_cross_seed_heatmap.png"
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved PNG to {png_path}", flush=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 76: Cross-seed zero-shot coordination", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Seeds: {N_SEEDS}, Epochs: {COMM_EPOCHS}", flush=True)

    t_total = time.time()

    # Load data
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Step 1: Train all seeds
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"STEP 1: Training {N_SEEDS} seeds", flush=True)
    print(f"{'='*70}", flush=True)

    seed_data = []
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---", flush=True)
        result = train_and_save_seed(seed, data_t, e_bins, f_bins,
                                      train_ids, holdout_ids, DEVICE)
        seed_data.append(result)
        torch.mps.empty_cache()

        elapsed = time.time() - t_total
        per_seed = elapsed / (seed + 1)
        remaining = per_seed * (N_SEEDS - seed - 1)
        print(f"    [{seed+1}/{N_SEEDS} done, ETA {remaining/60:.0f}min]", flush=True)

    n_comp = sum(1 for sd in seed_data if sd['is_comp'])
    print(f"\n  Trained: {n_comp}/{N_SEEDS} compositional", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Step 2: Cross-seed evaluation
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"STEP 2: Cross-seed evaluation", flush=True)
    print(f"{'='*70}", flush=True)

    matrix = cross_seed_eval(seed_data, data_t, e_bins, f_bins,
                              holdout_ids, DEVICE)

    # ═══════════════════════════════════════════════════════════
    # Step 3: Analysis
    # ═══════════════════════════════════════════════════════════
    analysis = analyze(seed_data, matrix)

    # ═══════════════════════════════════════════════════════════
    # Step 4: Plot
    # ═══════════════════════════════════════════════════════════
    plot_heatmap(seed_data, matrix)

    # ═══════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'n_seeds': N_SEEDS,
            'comm_epochs': COMM_EPOCHS,
            'comp_threshold': COMP_THRESHOLD,
        },
        'per_seed': [
            {
                'seed': sd['seed'],
                'holdout_both': sd['holdout_both'],
                'train_both': sd['train_both'],
                'pos_dis': sd['pos_dis'],
                'is_comp': sd['is_comp'],
                'mi_matrix': sd['mi_matrix'],
                'pos_mapping': sd['pos_mapping'],
            }
            for sd in seed_data
        ],
        'cross_seed_matrix': matrix.tolist(),
        'analysis': analysis,
    }

    save_path = RESULTS_DIR / "phase76_cross_seed.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved data to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

"""
Phase 54c Multi-seed: Reproducibility test across 5 random seeds.
Same iterated learning setup, different initializations.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase54c_multiseed.py
"""

import time
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration (identical to Phase 54c)
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 8
N_HEADS = 2
CONTROL_VOCAB = 64
N_FRAMES = 8
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

ORACLE_EPOCHS = 100
ORACLE_LR = 1e-3
COMM_EPOCHS = 200
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40

SEEDS = [42, 123, 456, 789, 1337]


# ══════════════════════════════════════════════════════════════════
# Architecture (copied from Phase 54c)
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
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_cached_features(cache_path):
    data = torch.load(cache_path, weights_only=False)
    return data['features'], data['e_bins'], data['f_bins']


def create_splits(e_bins, f_bins, holdout_cells):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        cell = (int(e_bins[i]), int(f_bins[i]))
        if cell in holdout_cells:
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
        da = data_t[ia].to(device)
        db = data_t[ib].to(device)

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

    # Entropy
    entropies = []
    for p in range(n_pos):
        v = VOCAB_SIZE if n_pos > 1 else CONTROL_VOCAB
        counts = np.bincount(all_tokens[:, p], minlength=v)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(v)))

    # MI matrix
    attributes = np.stack([e_bins, f_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    # PosDis
    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    # TopSim
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(data_t), size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'pos_dis': float(pos_dis),
        'topsim': float(topsim),
        'entropies': entropies,
        'mi_matrix': mi_matrix,
    }


# ══════════════════════════════════════════════════════════════════
# Training (oracle + communication with IL)
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


def train_communication(sender, receiver, data_t, e_bins, f_bins,
                        train_ids, holdout_ids, device, msg_dim, seed,
                        tag="2x8"):
    s_lr = SENDER_LR * (0.5 if tag == "1x64" else 1.0)
    r_lr = RECEIVER_LR * (0.5 if tag == "1x64" else 1.0)
    sender_opt = torch.optim.Adam(sender.parameters(), lr=s_lr)
    receiver_opt = torch.optim.Adam(receiver.parameters(), lr=r_lr)

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    vocab = VOCAB_SIZE if tag == "2x8" else CONTROL_VOCAB
    max_entropy = math.log(vocab)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_state = None
    nan_count = 0

    for epoch in range(COMM_EPOCHS):
        # Iterated learning: reset receiver
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            receiver = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
            receiver_opt = torch.optim.Adam(receiver.parameters(), lr=r_lr)

        sender.train()
        receiver.train()

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
            pred_e, pred_f = receiver(msg_a, msg_b)

            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)

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
                receiver_opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            sender_opt.zero_grad()
            receiver_opt.zero_grad()
            loss.backward()

            has_nan_grad = False
            for p in list(sender.parameters()) + list(receiver.parameters()):
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                receiver_opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(receiver.parameters(), 1.0)
            sender_opt.step()
            receiver_opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            print(f"    WARNING: NaN divergence at epoch {epoch+1}", flush=True)
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            sender.eval()
            receiver.eval()
            with torch.no_grad():
                _, _, acc_both = evaluate_accuracy(
                    sender, receiver, data_t, e_bins, f_bins, train_ids, device)
            if acc_both > best_both_acc:
                best_both_acc = acc_both
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}
                best_receiver_state = {k: v.cpu().clone()
                                       for k, v in receiver.state_dict().items()}

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)
    if best_receiver_state is not None:
        receiver.load_state_dict(best_receiver_state)
    return receiver, nan_count


# ══════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════

def run_single_seed(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
    """Run full pipeline for one seed. Returns dict of metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()

    # Oracle
    oracle, oracle_acc = train_oracle(data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()

    # 2x8 + IL
    encoder_2x8 = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder_2x8.load_state_dict(oracle_enc_state)
    sender_2x8 = CompositionalSender(encoder_2x8, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim_2x8 = VOCAB_SIZE * N_HEADS
    receiver_2x8 = CompositionalReceiver(msg_dim_2x8, HIDDEN_DIM).to(device)

    receiver_2x8, nan_2x8 = train_communication(
        sender_2x8, receiver_2x8, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim=msg_dim_2x8,
        seed=seed + 1000, tag="2x8")

    # Final eval (2x8)
    sender_2x8.eval()
    receiver_2x8.eval()
    with torch.no_grad():
        te, tf, tb = evaluate_accuracy(
            sender_2x8, receiver_2x8, data_t, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy(
            sender_2x8, receiver_2x8, data_t, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    # Compositionality
    with torch.no_grad():
        comp = compute_compositionality(sender_2x8, data_t, e_bins, f_bins, device)

    # Best MI per property (position-agnostic)
    mi = comp['mi_matrix']  # (2, 2): rows=positions, cols=[e, f]
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())

    dt = time.time() - t0

    return {
        'seed': seed,
        'oracle_both': oracle_acc,
        'train_e': te, 'train_f': tf, 'train_both': tb,
        'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
        'pos_dis': comp['pos_dis'],
        'topsim': comp['topsim'],
        'entropies': comp['entropies'],
        'mi_matrix': mi.tolist(),
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'nan_count': nan_2x8,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 54c Multi-seed: 5 seeds × Iterated Learning", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Receiver reset interval: {RECEIVER_RESET_INTERVAL}", flush=True)

    t_total = time.time()

    # Load cached features (shared across all seeds)
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout\n", flush=True)

    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"{'─'*70}", flush=True)
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)})", flush=True)
        print(f"{'─'*70}", flush=True)

        result = run_single_seed(seed, data_t, e_bins, f_bins,
                                 train_ids, holdout_ids, DEVICE)
        all_results.append(result)

        print(f"  → holdout_both={result['holdout_both']:.1%}  "
              f"PosDis={result['pos_dis']:.3f}  "
              f"MI→e={result['best_mi_e']:.3f}  MI→f={result['best_mi_f']:.3f}  "
              f"NaN={result['nan_count']}  "
              f"({result['time_sec']:.0f}s)\n", flush=True)

    # ════════════════════════════════════════════════════════════
    # Summary table
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY TABLE", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"{'Seed':>6} | {'Holdout Both':>12} | {'PosDis':>7} | "
              f"{'TopSim':>7} | {'MI→e':>7} | {'MI→f':>7} | {'NaN':>4}")
    print(header, flush=True)
    print(f"{'─'*6}-+-{'─'*12}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*4}",
          flush=True)

    holdout_boths = []
    pos_diss = []
    topsims = []
    mi_es = []
    mi_fs = []

    for r in all_results:
        print(f"{r['seed']:>6} | {r['holdout_both']:>11.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['topsim']:>7.3f} | "
              f"{r['best_mi_e']:>7.3f} | {r['best_mi_f']:>7.3f} | "
              f"{r['nan_count']:>4}", flush=True)
        holdout_boths.append(r['holdout_both'])
        pos_diss.append(r['pos_dis'])
        topsims.append(r['topsim'])
        mi_es.append(r['best_mi_e'])
        mi_fs.append(r['best_mi_f'])

    print(f"{'─'*6}-+-{'─'*12}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*4}",
          flush=True)

    def fmt_mean_std(vals, pct=False):
        m, s = np.mean(vals), np.std(vals)
        if pct:
            return f"{m:.1%}±{s:.1%}"
        return f"{m:.3f}±{s:.3f}"

    print(f"{'Mean':>6} | {fmt_mean_std(holdout_boths, pct=True):>12} | "
          f"{fmt_mean_std(pos_diss):>7} | {fmt_mean_std(topsims):>7} | "
          f"{fmt_mean_std(mi_es):>7} | {fmt_mean_std(mi_fs):>7} |",
          flush=True)

    # Additional stats
    print(f"\n  Holdout both: {np.mean(holdout_boths):.1%} ± {np.std(holdout_boths):.1%}"
          f"  (min={np.min(holdout_boths):.1%}, max={np.max(holdout_boths):.1%})", flush=True)
    print(f"  PosDis:       {np.mean(pos_diss):.3f} ± {np.std(pos_diss):.3f}"
          f"  (min={np.min(pos_diss):.3f}, max={np.max(pos_diss):.3f})", flush=True)
    print(f"  Best MI→e:    {np.mean(mi_es):.3f} ± {np.std(mi_es):.3f}", flush=True)
    print(f"  Best MI→f:    {np.mean(mi_fs):.3f} ± {np.std(mi_fs):.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════
    save_data = {
        'seeds': SEEDS,
        'per_seed': all_results,
        'summary': {
            'holdout_both_mean': float(np.mean(holdout_boths)),
            'holdout_both_std': float(np.std(holdout_boths)),
            'pos_dis_mean': float(np.mean(pos_diss)),
            'pos_dis_std': float(np.std(pos_diss)),
            'topsim_mean': float(np.mean(topsims)),
            'topsim_std': float(np.std(topsims)),
            'best_mi_e_mean': float(np.mean(mi_es)),
            'best_mi_e_std': float(np.std(mi_es)),
            'best_mi_f_mean': float(np.mean(mi_fs)),
            'best_mi_f_std': float(np.std(mi_fs)),
        },
        'config': {
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'comm_epochs': COMM_EPOCHS,
            'oracle_epochs': ORACLE_EPOCHS,
        },
    }

    results_path = RESULTS_DIR / "phase54c_multiseed.json"
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {results_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/len(SEEDS):.0f}s per seed)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

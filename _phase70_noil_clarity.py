"""
Phase 70: No-IL baseline with train+holdout for ablation clarity
================================================================
Same architecture as Phase 54f but:
- No receiver resets (no iterated learning)
- Single receiver (not population of 3)
- 400 epochs (same as compositional for fair comparison)
- 20 seeds
- Reports BOTH train and holdout accuracy per seed

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase70_noil_clarity.py
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
VOCAB_SIZE = 5
N_HEADS = 2
CONTROL_VOCAB = 25
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

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 54f)
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
        'mi_matrix': mi_matrix,
    }


# ══════════════════════════════════════════════════════════════════
# Training
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


def train_noil(sender, receiver, data_t, e_bins, f_bins,
               train_ids, holdout_ids, device, seed):
    """Train sender + single receiver, NO resets (no iterated learning)."""
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opt = torch.optim.Adam(receiver.parameters(), lr=RECEIVER_LR)

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_state = None
    nan_count = 0

    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
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

        if (epoch + 1) % 40 == 0:
            sender.eval()
            receiver.eval()
            with torch.no_grad():
                te, tf, tb = evaluate_accuracy(
                    sender, receiver, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
                he, hf, hb = evaluate_accuracy(
                    sender, receiver, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            print(f"        Ep {epoch+1:3d}: "
                  f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                  f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}
                best_receiver_state = {k: v.cpu().clone()
                                       for k, v in receiver.state_dict().items()}

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)
    if best_receiver_state is not None:
        receiver.load_state_dict(best_receiver_state)

    return nan_count


# ══════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════

def run_single_seed(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle(data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Sender (vocab=5, 2 heads -> msg_dim=10)
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Single receiver, no IL
    receiver = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)

    print(f"    Training sender (2x{VOCAB_SIZE}) vs 1 receiver (NO IL)...", flush=True)

    nan_count = train_noil(
        sender, receiver, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, seed)

    # Final eval
    sender.eval()
    receiver.eval()
    with torch.no_grad():
        te, tf, tb = evaluate_accuracy(
            sender, receiver, data_t, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy(
            sender, receiver, data_t, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    # Compositionality
    with torch.no_grad():
        comp = compute_compositionality(sender, data_t, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())

    dt = time.time() - t0

    gap = tb - hb
    print(f"    -> train={tb:.1%}  holdout={hb:.1%}  gap={gap:.1%}  "
          f"PosDis={comp['pos_dis']:.3f}  NaN={nan_count}  ({dt:.0f}s)", flush=True)

    return {
        'seed': seed,
        'oracle_both': oracle_acc,
        'train_e': te, 'train_f': tf, 'train_both': tb,
        'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
        'train_holdout_gap': gap,
        'pos_dis': comp['pos_dis'],
        'topsim': comp['topsim'],
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'nan_count': nan_count,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 70: No-IL baseline (single receiver, no resets)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Config: same arch as Phase 54f, NO iterated learning", flush=True)
    print(f"  Vocab: 2x{VOCAB_SIZE}, single receiver, no resets", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}, Seeds: {N_SEEDS}", flush=True)

    t_total = time.time()

    # Load cached features
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    all_results = []
    for seed in SEEDS:
        total_elapsed = time.time() - t_total
        done = seed
        if done > 0:
            remaining = total_elapsed / done * (N_SEEDS - done)
            print(f"\n  [Progress: {done}/{N_SEEDS}, "
                  f"ETA {remaining/60:.0f}min]", flush=True)

        result = run_single_seed(seed, data_t, e_bins, f_bins,
                                 train_ids, holdout_ids, DEVICE)
        all_results.append(result)

    # ════════════════════════════════════════════════════════════
    # Analysis
    # ════════════════════════════════════════════════════════════
    tb_all = np.array([r['train_both'] for r in all_results])
    hb_all = np.array([r['holdout_both'] for r in all_results])
    gap_all = np.array([r['train_holdout_gap'] for r in all_results])
    pd_all = np.array([r['pos_dis'] for r in all_results])
    ts_all = np.array([r['topsim'] for r in all_results])

    # ════════════════════════════════════════════════════════════
    # Full table
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"FULL TABLE: Phase 70 — No-IL baseline ({N_SEEDS} seeds)", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"  {'Seed':>4} | {'Train':>7} | {'Holdout':>8} | "
              f"{'Gap':>6} | {'PosDis':>7} | {'TopSim':>7}")
    print(header, flush=True)
    print(f"  {'----':>4}-+-{'-------':>7}-+-{'--------':>8}-+-"
          f"{'------':>6}-+-{'-------':>7}-+-{'-------':>7}",
          flush=True)

    for r in all_results:
        print(f"  {r['seed']:>4} | {r['train_both']:>6.1%} | "
              f"{r['holdout_both']:>7.1%} | {r['train_holdout_gap']:>5.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['topsim']:>7.3f}", flush=True)

    print(f"  {'----':>4}-+-{'-------':>7}-+-{'--------':>8}-+-"
          f"{'------':>6}-+-{'-------':>7}-+-{'-------':>7}",
          flush=True)
    print(f"  {'Mean':>4} | {tb_all.mean():>6.1%} | "
          f"{hb_all.mean():>7.1%} | {gap_all.mean():>5.1%} | "
          f"{pd_all.mean():>7.3f} | {ts_all.mean():>7.3f}", flush=True)
    print(f"  {'Std':>4} | {tb_all.std():>6.1%} | "
          f"{hb_all.std():>7.1%} | {gap_all.std():>5.1%} | "
          f"{pd_all.std():>7.3f} | {ts_all.std():>7.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Key insight: train-holdout gap
    # ════════════════════════════════════════════════════════════
    print(f"\n  KEY INSIGHT: Train vs Holdout", flush=True)
    print(f"    Train accuracy:   {tb_all.mean():.1%} +/- {tb_all.std():.1%}", flush=True)
    print(f"    Holdout accuracy: {hb_all.mean():.1%} +/- {hb_all.std():.1%}", flush=True)
    print(f"    Gap (overfit):    {gap_all.mean():.1%} +/- {gap_all.std():.1%}", flush=True)

    # Paired t-test: train vs holdout
    t_paired, p_paired = stats.ttest_rel(tb_all, hb_all)
    print(f"    Paired t-test (train > holdout): t={t_paired:.2f}, p={p_paired:.4f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Compositionality
    # ════════════════════════════════════════════════════════════
    compositional = [r for r in all_results if r['pos_dis'] > 0.4]
    non_comp = [r for r in all_results if r['pos_dis'] <= 0.4]
    print(f"\n  COMPOSITIONALITY: {len(compositional)}/{N_SEEDS} = "
          f"{len(compositional)/N_SEEDS:.0%} (PosDis > 0.4)", flush=True)
    print(f"  PosDis: mean={pd_all.mean():.3f} std={pd_all.std():.3f} "
          f"min={pd_all.min():.3f} max={pd_all.max():.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Correlation
    # ════════════════════════════════════════════════════════════
    r_corr, p_corr = stats.pearsonr(pd_all, hb_all)
    rho_corr, p_rho = stats.spearmanr(pd_all, hb_all)
    print(f"\n  PosDis-holdout correlation:", flush=True)
    print(f"    Pearson: r={r_corr:.3f}, p={p_corr:.4f}", flush=True)
    print(f"    Spearman: rho={rho_corr:.3f}, p={p_rho:.4f}", flush=True)

    r_gap, p_gap = stats.pearsonr(pd_all, gap_all)
    print(f"\n  PosDis-gap correlation:", flush=True)
    print(f"    Pearson: r={r_gap:.3f}, p={p_gap:.4f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'n_heads': N_HEADS,
            'n_receivers': 1,
            'receiver_reset_interval': None,
            'iterated_learning': False,
            'comm_epochs': COMM_EPOCHS,
            'n_seeds': N_SEEDS,
        },
        'per_seed': all_results,
        'summary': {
            'train_both_mean': float(tb_all.mean()),
            'train_both_std': float(tb_all.std()),
            'holdout_both_mean': float(hb_all.mean()),
            'holdout_both_std': float(hb_all.std()),
            'gap_mean': float(gap_all.mean()),
            'gap_std': float(gap_all.std()),
            'pos_dis_mean': float(pd_all.mean()),
            'pos_dis_std': float(pd_all.std()),
            'topsim_mean': float(ts_all.mean()),
            'topsim_std': float(ts_all.std()),
        },
        'statistics': {
            'paired_ttest_t': float(t_paired),
            'paired_ttest_p': float(p_paired),
            'posdis_holdout_pearson_r': float(r_corr),
            'posdis_holdout_pearson_p': float(p_corr),
            'posdis_holdout_spearman_rho': float(rho_corr),
            'posdis_holdout_spearman_p': float(p_rho),
            'posdis_gap_pearson_r': float(r_gap),
            'posdis_gap_pearson_p': float(p_gap),
            'compositional_count': len(compositional),
            'compositional_rate': len(compositional) / N_SEEDS,
        },
    }

    save_path = RESULTS_DIR / "phase70_noil_traintest.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/len(SEEDS):.0f}s per seed)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

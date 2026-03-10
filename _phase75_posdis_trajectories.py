"""
Phase 75: PosDis trajectory analysis — when does the comp/holistic split happen?
================================================================================
Same config as Phase 69b (2-agent, IL+population, DINOv2, 2x5, Latin square holdout)
but logs PosDis at checkpoint epochs [0, 40, 80, 120, 200, 300, 400].

Seeds 0-19 only. We know from Phase 69b which are comp vs holistic.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase75_posdis_trajectories.py
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

# Epochs at which to log PosDis
CHECKPOINT_EPOCHS = [0, 40, 80, 120, 200, 300, 400]


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
    for r in receivers:
        _, _, both = evaluate_accuracy(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


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


def compute_posdis(sender, data_t, e_bins, f_bins, device):
    """Compute PosDis only (lighter than full compositionality metrics)."""
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

    return float(pos_dis)


# ══════════════════════════════════════════════════════════════════
# Training with PosDis checkpointing
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


def train_with_checkpoints(sender, receivers, data_t, e_bins, f_bins,
                           train_ids, holdout_ids, device, msg_dim, seed):
    """Train population with PosDis logging at checkpoint epochs."""
    s_lr = SENDER_LR
    r_lr = RECEIVER_LR
    sender_opt = torch.optim.Adam(sender.parameters(), lr=s_lr)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=r_lr) for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    nan_count = 0
    t_start = time.time()

    # Log PosDis at epoch 0 (before any training)
    posdis_trajectory = {}
    if 0 in CHECKPOINT_EPOCHS:
        sender.eval()
        with torch.no_grad():
            pd = compute_posdis(sender, data_t, e_bins, f_bins, device)
        posdis_trajectory[0] = pd

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
            print(f"    WARNING: NaN divergence at epoch {epoch+1}", flush=True)
            # Fill remaining checkpoints with last known value
            last_pd = posdis_trajectory.get(max(posdis_trajectory.keys()), 0.0)
            for cp in CHECKPOINT_EPOCHS:
                if cp not in posdis_trajectory:
                    posdis_trajectory[cp] = last_pd
            break

        # Check if this is a checkpoint epoch (epoch+1 because we log after training)
        completed_epoch = epoch + 1
        if completed_epoch in CHECKPOINT_EPOCHS:
            sender.eval()
            with torch.no_grad():
                pd = compute_posdis(sender, data_t, e_bins, f_bins, device)
            posdis_trajectory[completed_epoch] = pd

        # Print progress
        if (epoch + 1) % 40 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            pd_str = f"  PD={posdis_trajectory.get(completed_epoch, '?')}"
            if completed_epoch in CHECKPOINT_EPOCHS:
                pd_str = f"  PD={posdis_trajectory[completed_epoch]:.3f}"
            print(f"        Ep {epoch+1:3d}: ETA {eta/60:.0f}min{pd_str}",
                  flush=True)

    return receivers, nan_count, posdis_trajectory


# ══════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════

def run_single_seed(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
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

    # Train with PosDis checkpointing
    receivers, nan_count, posdis_traj = train_with_checkpoints(
        sender, receivers, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval
    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population(
            sender, receivers, data_t, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy(
            sender, best_r, data_t, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    dt = time.time() - t0
    final_pd = posdis_traj.get(COMM_EPOCHS, posdis_traj.get(max(posdis_traj.keys()), 0.0))

    print(f"    -> holdout={hb:.1%}  PosDis={final_pd:.3f}  ({dt:.0f}s)", flush=True)
    traj_str = "  ".join(f"ep{k}={v:.3f}" for k, v in sorted(posdis_traj.items()))
    print(f"       trajectory: {traj_str}", flush=True)

    return {
        'seed': seed,
        'holdout_both': float(hb),
        'train_both': float(tb),
        'final_posdis': float(final_pd),
        'posdis_trajectory': {str(k): float(v) for k, v in sorted(posdis_traj.items())},
        'nan_count': nan_count,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Analysis & Plotting
# ══════════════════════════════════════════════════════════════════

def analyze_and_plot(results):
    """Analyze trajectories and create figure."""
    comp_threshold = 0.4

    # Separate comp vs holistic based on final PosDis
    comp_seeds = [r for r in results if r['final_posdis'] > comp_threshold]
    hol_seeds = [r for r in results if r['final_posdis'] <= comp_threshold]

    print(f"\n{'='*70}", flush=True)
    print(f"ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Compositional: {len(comp_seeds)}/{len(results)}", flush=True)
    print(f"  Holistic:      {len(hol_seeds)}/{len(results)}", flush=True)

    # 1. At what epoch does the split become visible?
    print(f"\n  Mean PosDis by epoch:", flush=True)
    print(f"  {'Epoch':>6} | {'Comp':>8} | {'Holistic':>8} | {'Gap':>8} | {'t-stat':>7} | {'p-val':>7}",
          flush=True)
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}", flush=True)

    split_epoch = None
    for ep in CHECKPOINT_EPOCHS:
        ep_str = str(ep)
        comp_vals = [r['posdis_trajectory'][ep_str] for r in comp_seeds
                     if ep_str in r['posdis_trajectory']]
        hol_vals = [r['posdis_trajectory'][ep_str] for r in hol_seeds
                    if ep_str in r['posdis_trajectory']]
        if comp_vals and hol_vals:
            cm, hm = np.mean(comp_vals), np.mean(hol_vals)
            gap = cm - hm
            if len(comp_vals) > 1 and len(hol_vals) > 1:
                t, p = stats.ttest_ind(comp_vals, hol_vals)
            else:
                t, p = 0.0, 1.0
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"  {ep:>6} | {cm:>8.3f} | {hm:>8.3f} | {gap:>+8.3f} | {t:>7.2f} | {p:>7.4f}{sig}",
                  flush=True)
            if split_epoch is None and p < 0.05:
                split_epoch = ep

    print(f"\n  Split becomes significant (p<0.05) at epoch: {split_epoch}", flush=True)

    # 2. Prediction from epoch 40
    ep40_vals = []
    final_labels = []
    for r in results:
        if '40' in r['posdis_trajectory']:
            ep40_vals.append(r['posdis_trajectory']['40'])
            final_labels.append(1 if r['final_posdis'] > comp_threshold else 0)

    if ep40_vals:
        # Try different thresholds for epoch-40 classification
        best_acc = 0
        best_thresh = 0
        for thresh in np.arange(0.0, 1.0, 0.01):
            preds = [1 if v > thresh else 0 for v in ep40_vals]
            acc = sum(1 for p, l in zip(preds, final_labels) if p == l) / len(preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        print(f"\n  Epoch-40 prediction of final compositionality:", flush=True)
        print(f"    Best threshold: {best_thresh:.2f}", flush=True)
        print(f"    Classification accuracy: {best_acc:.1%} ({sum(1 for p, l in zip([1 if v > best_thresh else 0 for v in ep40_vals], final_labels) if p == l)}/{len(final_labels)})",
              flush=True)

        # Also try epoch 80
        ep80_vals = [r['posdis_trajectory'].get('80', None) for r in results]
        if all(v is not None for v in ep80_vals):
            best_acc_80 = 0
            best_thresh_80 = 0
            for thresh in np.arange(0.0, 1.0, 0.01):
                preds = [1 if v > thresh else 0 for v in ep80_vals]
                acc = sum(1 for p, l in zip(preds, final_labels) if p == l) / len(preds)
                if acc > best_acc_80:
                    best_acc_80 = acc
                    best_thresh_80 = thresh
            print(f"    Epoch-80 best accuracy: {best_acc_80:.1%} (thresh={best_thresh_80:.2f})",
                  flush=True)

    # 3. Do any seeds transition between regimes?
    print(f"\n  Regime transitions:", flush=True)
    transitions = 0
    for r in results:
        traj = r['posdis_trajectory']
        epochs_sorted = sorted(int(k) for k in traj.keys())
        regimes = [(ep, 'C' if traj[str(ep)] > comp_threshold else 'H')
                   for ep in epochs_sorted]
        changes = []
        for i in range(1, len(regimes)):
            if regimes[i][1] != regimes[i-1][1]:
                changes.append(f"{regimes[i-1][1]}→{regimes[i][1]} at ep{regimes[i][0]}")
        if changes:
            transitions += 1
            print(f"    Seed {r['seed']}: {', '.join(changes)}", flush=True)
    if transitions == 0:
        print(f"    None — all seeds stay in their initial regime", flush=True)
    print(f"    Total seeds with transitions: {transitions}/{len(results)}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Figure
    # ════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot individual trajectories
    for r in results:
        traj = r['posdis_trajectory']
        epochs = sorted(int(k) for k in traj.keys())
        vals = [traj[str(ep)] for ep in epochs]
        is_comp = r['final_posdis'] > comp_threshold
        color = '#2ca02c' if is_comp else '#888888'
        alpha = 0.5 if is_comp else 0.3
        lw = 1.2 if is_comp else 0.8
        ax.plot(epochs, vals, color=color, alpha=alpha, linewidth=lw)

    # Plot means
    comp_means = []
    hol_means = []
    for ep in CHECKPOINT_EPOCHS:
        ep_str = str(ep)
        cv = [r['posdis_trajectory'][ep_str] for r in comp_seeds
              if ep_str in r['posdis_trajectory']]
        hv = [r['posdis_trajectory'][ep_str] for r in hol_seeds
              if ep_str in r['posdis_trajectory']]
        comp_means.append(np.mean(cv) if cv else np.nan)
        hol_means.append(np.mean(hv) if hv else np.nan)

    ax.plot(CHECKPOINT_EPOCHS, comp_means, color='#2ca02c', linewidth=3,
            label=f'Compositional mean (n={len(comp_seeds)})', zorder=10)
    ax.plot(CHECKPOINT_EPOCHS, hol_means, color='#555555', linewidth=3,
            linestyle='--', label=f'Holistic mean (n={len(hol_seeds)})', zorder=10)

    # Receiver reset lines
    reset_epochs = list(range(RECEIVER_RESET_INTERVAL, COMM_EPOCHS + 1,
                              RECEIVER_RESET_INTERVAL))
    for re in reset_epochs:
        ax.axvline(x=re, color='#cccccc', linestyle=':', linewidth=0.7, zorder=0)

    # Threshold line
    ax.axhline(y=comp_threshold, color='red', linestyle='--', linewidth=1,
               alpha=0.5, label=f'Comp threshold ({comp_threshold})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PosDis', fontsize=12)
    ax.set_title('PosDis Trajectories: Compositional vs Holistic Seeds', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(-10, COMM_EPOCHS + 10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(CHECKPOINT_EPOCHS)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_posdis_trajectory.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {fig_path}", flush=True)

    # Also save PNG for quick viewing
    png_path = RESULTS_DIR / "phase75_posdis_trajectory.png"
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved PNG to {png_path}", flush=True)
    plt.close()

    return {
        'split_epoch': split_epoch,
        'ep40_best_accuracy': best_acc if ep40_vals else None,
        'ep40_best_threshold': best_thresh if ep40_vals else None,
        'n_transitions': transitions,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 75: PosDis trajectory analysis", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Seeds: {N_SEEDS}, Epochs: {COMM_EPOCHS}", flush=True)
    print(f"  Checkpoints: {CHECKPOINT_EPOCHS}", flush=True)

    t_total = time.time()

    # Load data
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Run all seeds
    results = []
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---", flush=True)
        result = run_single_seed(seed, data_t, e_bins, f_bins,
                                  train_ids, holdout_ids, DEVICE)
        results.append(result)
        torch.mps.empty_cache()

        # Progress
        elapsed = time.time() - t_total
        per_seed = elapsed / (seed + 1)
        remaining = per_seed * (N_SEEDS - seed - 1)
        print(f"    [{seed+1}/{N_SEEDS} done, ETA {remaining/60:.0f}min]", flush=True)

    # Analysis and plotting
    analysis = analyze_and_plot(results)

    # Save data
    save_data = {
        'config': {
            'n_seeds': N_SEEDS,
            'comm_epochs': COMM_EPOCHS,
            'checkpoint_epochs': CHECKPOINT_EPOCHS,
            'comp_threshold': 0.4,
        },
        'per_seed': results,
        'analysis': analysis,
    }

    save_path = RESULTS_DIR / "phase75_trajectories.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved data to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

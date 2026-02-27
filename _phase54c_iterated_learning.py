"""
Phase 54c: Iterated Learning for Reliable Compositionality
==========================================================
Same as Phase 54b (DINOv2 encoder) but with periodic receiver resets.
Every 40 epochs the receiver is reinitialized from scratch, forcing the
sender to maintain a learnable (compositional) language structure.

4 "generations" of receivers at epochs 40, 80, 120, 160.
Sender keeps its weights throughout — the language persists, the listener changes.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase54c_iterated_learning.py
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
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

DATASET_DIR = Path("kubric/output/ramp_dataset")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
DINO_DIM = 384          # DINOv2 ViT-S CLS token dimension
VOCAB_SIZE = 8           # per head (compositional)
N_HEADS = 2              # 2 heads x 8 vocab each
CONTROL_VOCAB = 64       # 1 head x 64 vocab (same capacity: 8*8=64)
N_FRAMES = 8             # subsampled from 24 rendered frames
BATCH_SIZE = 32

# Holdout: Latin square — 5 cells, one per row AND one per column
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

# Iterated learning
RECEIVER_RESET_INTERVAL = 40  # Reset receiver every N epochs


# ══════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Temporal pooling over pre-extracted DINOv2 features.
    Input: (B, T, 384) DINOv2 CLS tokens → (B, hidden_dim)."""
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
        # x: (B, T, 384)
        x = x.permute(0, 2, 1)  # (B, 384, T)
        x = self.temporal(x).squeeze(-1)  # (B, 128)
        return self.fc(x)  # (B, hidden_dim)


class CompositionalSender(nn.Module):
    """Sender with N Gumbel-Softmax heads (one per message position)."""
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


class ControlSender(nn.Module):
    """Control: single Gumbel-Softmax head with large vocab."""
    def __init__(self, encoder, hidden_dim, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = 1
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        logits = self.head(h).clamp(-15, 15)
        if self.training:
            msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            idx = logits.argmax(dim=-1)
            msg = F.one_hot(idx, self.vocab_size).float()
        return msg, [logits]


class CompositionalReceiver(nn.Module):
    """Receiver with two output heads (elasticity, friction comparison)."""
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
    """Direct comparison oracle (no communication bottleneck)."""
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

RESTITUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
FRICTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
SCENES_PER_CELL = 12
TOTAL_SCENES = 300


def get_grid_cell(scene_id):
    """Map scene_id -> (restitution, friction, e_bin, f_bin). Deterministic."""
    cell_idx = scene_id // SCENES_PER_CELL
    e_bin = cell_idx // len(FRICTION_LEVELS)
    f_bin = cell_idx % len(FRICTION_LEVELS)
    return (RESTITUTION_LEVELS[e_bin], FRICTION_LEVELS[f_bin], e_bin, f_bin)


def load_cached_features(cache_path):
    """Load pre-extracted DINOv2 features from Phase 54b cache."""
    print(f"  Loading cached DINOv2 features from {cache_path}", flush=True)
    data = torch.load(cache_path, weights_only=False)
    print(f"  Features shape: {data['features'].shape}", flush=True)
    return data['features'], data['e_bins'], data['f_bins']


def create_splits(e_bins, f_bins, holdout_cells):
    """Split scenes into train and holdout based on grid cells."""
    train_ids = []
    holdout_ids = []
    for i in range(len(e_bins)):
        cell = (int(e_bins[i]), int(f_bins[i]))
        if cell in holdout_cells:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


def sample_pairs(scene_ids, batch_size, rng):
    """Sample pairs of different scenes."""
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
    """Evaluate accuracy on given scene IDs."""
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

    acc_e = correct_e / max(total_e, 1)
    acc_f = correct_f / max(total_f, 1)
    acc_both = correct_both / max(total_both, 1)
    return acc_e, acc_f, acc_both


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_oracle(data_t, e_bins, f_bins, train_ids, device):
    """Pretrain oracle (no communication bottleneck)."""
    print(f"\n{'='*60}", flush=True)
    print(f"Oracle Pretrain ({ORACLE_EPOCHS} epochs)", flush=True)
    print(f"{'='*60}", flush=True)

    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)

    n_params = sum(p.numel() for p in oracle.parameters())
    print(f"  Oracle trainable params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(42)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    best_acc = 0.0
    best_state = None
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        epoch_loss = 0.0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da = data_t[ia].to(device)
            db = data_t[ib].to(device)

            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            pred_e, pred_f = oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                acc_e, acc_f, acc_both = evaluate_accuracy(
                    None, None, data_t, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f}  "
                  f"e_acc={acc_e:.1%}  f_acc={acc_f:.1%}  both={acc_both:.1%}",
                  flush=True)

            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone()
                              for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    print(f"  Best oracle: both_acc={best_acc:.1%}", flush=True)
    return oracle


def train_communication(sender, receiver, data_t, e_bins, f_bins,
                        train_ids, holdout_ids, device, msg_dim, tag="2x8"):
    """Train sender-receiver with iterated learning (periodic receiver resets)."""
    print(f"\n{'='*60}", flush=True)
    print(f"Communication Training [{tag}] ({COMM_EPOCHS} epochs, "
          f"receiver reset every {RECEIVER_RESET_INTERVAL})", flush=True)
    print(f"{'='*60}", flush=True)

    s_lr = SENDER_LR * (0.5 if tag == "1x64" else 1.0)
    r_lr = RECEIVER_LR * (0.5 if tag == "1x64" else 1.0)
    sender_opt = torch.optim.Adam(sender.parameters(), lr=s_lr)
    receiver_opt = torch.optim.Adam(receiver.parameters(), lr=r_lr)

    rng = np.random.RandomState(42)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    vocab = VOCAB_SIZE if tag == "2x8" else CONTROL_VOCAB
    max_entropy = math.log(vocab)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_state = None

    history = {
        'loss': [], 'acc_e': [], 'acc_f': [], 'acc_both': [],
        'holdout_e': [], 'holdout_f': [], 'holdout_both': [],
        'tau': [], 'entropy': [],
    }

    t_start = time.time()
    nan_count = 0
    generation = 0

    for epoch in range(COMM_EPOCHS):
        # Iterated learning: reset receiver at interval boundaries
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            generation += 1
            receiver = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
            receiver_opt = torch.optim.Adam(receiver.parameters(), lr=r_lr)
            print(f"  ** Receiver reset at epoch {epoch+1} "
                  f"(generation {generation}) **", flush=True)

        sender.train()
        receiver.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da = data_t[ia].to(device)
            db = data_t[ib].to(device)

            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b = sender(db, tau=tau, hard=hard)
            pred_e, pred_f = receiver(msg_a, msg_b)

            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)

            ent_sum = 0.0
            n_heads = 0
            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    ent_sum += ent.item()
                    n_heads += 1
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

            # NaN-safe gradient step: skip if loss is NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                sender_opt.zero_grad()
                receiver_opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            sender_opt.zero_grad()
            receiver_opt.zero_grad()
            loss.backward()

            # Check for NaN gradients
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

            epoch_loss += loss.item()
            epoch_entropy += ent_sum / max(n_heads, 1)

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        # If too many NaN steps in this epoch, model has diverged — stop early
        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            print(f"  WARNING: {epoch_nan}/{n_batches} NaN steps in epoch {epoch+1}, "
                  f"stopping early", flush=True)
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            sender.eval()
            receiver.eval()
            with torch.no_grad():
                acc_e, acc_f, acc_both = evaluate_accuracy(
                    sender, receiver, data_t, e_bins, f_bins, train_ids, device)
                ho_e, ho_f, ho_both = evaluate_accuracy(
                    sender, receiver, data_t, e_bins, f_bins, holdout_ids, device)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)

            history['loss'].append(epoch_loss / n_batches)
            history['acc_e'].append(acc_e)
            history['acc_f'].append(acc_f)
            history['acc_both'].append(acc_both)
            history['holdout_e'].append(ho_e)
            history['holdout_f'].append(ho_f)
            history['holdout_both'].append(ho_both)
            history['tau'].append(tau)
            history['entropy'].append(epoch_entropy / n_batches)

            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            gen_str = f"  gen={generation}" if generation > 0 else ""
            print(f"  Ep {epoch+1:3d}: loss={epoch_loss/n_batches:.4f}  tau={tau:.2f}  "
                  f"train[e={acc_e:.1%} f={acc_f:.1%} both={acc_both:.1%}]  "
                  f"holdout[e={ho_e:.1%} f={ho_f:.1%} both={ho_both:.1%}]  "
                  f"ent={epoch_entropy/n_batches:.2f}{nan_str}{gen_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

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
    print(f"  Best [{tag}]: both_acc={best_both_acc:.1%} "
          f"({generation+1} generations)", flush=True)
    return history, receiver


# ══════════════════════════════════════════════════════════════════
# Compositionality metrics
# ══════════════════════════════════════════════════════════════════

def _mutual_information(x, y):
    """Compute mutual information between discrete variables."""
    x_vals = np.unique(x)
    y_vals = np.unique(y)
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
    """Compute PosDis, TopSim, symbol entropy, CBM translation."""
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

    # Symbol entropy per position (normalized)
    entropies = []
    for p in range(n_pos):
        v = VOCAB_SIZE if n_pos > 1 else CONTROL_VOCAB
        counts = np.bincount(all_tokens[:, p], minlength=v)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        max_ent = np.log(v)
        entropies.append(float(ent / max_ent))

    # Token usage counts
    token_usage = {}
    for p in range(n_pos):
        v = VOCAB_SIZE if n_pos > 1 else CONTROL_VOCAB
        token_usage[p] = np.bincount(all_tokens[:, p], minlength=v)

    # PosDis
    attributes = np.stack([e_bins, f_bins], axis=1)
    n_attrs = attributes.shape[1]

    mi_matrix = np.zeros((n_pos, n_attrs))
    for p in range(n_pos):
        for a in range(n_attrs):
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

    # TopSim
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)

    meaning_dists = []
    message_dists = []

    for _ in range(n_pairs):
        i, j = rng.choice(len(data_t), size=2, replace=False)
        m_dist = abs(int(e_bins[i]) - int(e_bins[j])) + \
                 abs(int(f_bins[i]) - int(f_bins[j]))
        msg_dist = int((all_tokens[i] != all_tokens[j]).sum())
        meaning_dists.append(m_dist)
        message_dists.append(msg_dist)

    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    # CBM translation table
    cbm = {}
    for p in range(n_pos):
        v = VOCAB_SIZE if n_pos > 1 else CONTROL_VOCAB
        cbm[p] = {}
        for t in range(v):
            mask = all_tokens[:, p] == t
            if mask.sum() > 0:
                cbm[p][t] = {
                    'count': int(mask.sum()),
                    'modal_e': int(np.median(e_bins[mask])),
                    'modal_f': int(np.median(f_bins[mask])),
                    'mean_e': float(e_bins[mask].mean()),
                    'mean_f': float(f_bins[mask].mean()),
                }

    return {
        'pos_dis': float(pos_dis),
        'topsim': float(topsim),
        'entropies': entropies,
        'token_usage': token_usage,
        'mi_matrix': mi_matrix,
        'cbm': cbm,
        'all_tokens': all_tokens,
    }


# ══════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════

def plot_results(history_2x8, history_1x64, comp_2x8, comp_1x64,
                 final_results, e_bins, f_bins, save_path):
    """Create 6-panel visualization."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Training curves (2x8) — with vertical lines at receiver resets
    ax1 = fig.add_subplot(gs[0, 0])
    n_pts = len(history_2x8['acc_e'])
    epochs = [1] + list(range(10, 10 * n_pts, 10))
    epochs = epochs[:n_pts]
    ax1.plot(epochs, history_2x8['acc_e'], 'b-', label='train elast')
    ax1.plot(epochs, history_2x8['acc_f'], 'r-', label='train frict')
    ax1.plot(epochs, history_2x8['holdout_e'], 'b--', label='holdout elast')
    ax1.plot(epochs, history_2x8['holdout_f'], 'r--', label='holdout frict')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    # Mark receiver resets
    for reset_ep in range(RECEIVER_RESET_INTERVAL, COMM_EPOCHS,
                          RECEIVER_RESET_INTERVAL):
        ax1.axvline(x=reset_ep, color='green', linestyle='--', alpha=0.4,
                     linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('2x8 + Iterated Learning (DINOv2)')
    ax1.legend(fontsize=7)
    ax1.set_ylim(0.3, 1.05)

    # Panel 2: Token usage heatmap (2x8)
    ax2 = fig.add_subplot(gs[0, 1])
    if comp_2x8 is not None and len(comp_2x8['token_usage']) == 2:
        usage = np.zeros((2, VOCAB_SIZE))
        for p in range(2):
            counts = comp_2x8['token_usage'][p][:VOCAB_SIZE]
            usage[p] = counts / max(counts.sum(), 1)
        im = ax2.imshow(usage, aspect='auto', cmap='YlOrRd')
        ax2.set_xlabel('Token value')
        ax2.set_ylabel('Position')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Pos 0', 'Pos 1'])
        ax2.set_xticks(range(VOCAB_SIZE))
        ax2.set_title('Token Usage (2x8)')
        plt.colorbar(im, ax=ax2)

    # Panel 3: Token scatter (color=elasticity, marker=friction)
    ax3 = fig.add_subplot(gs[0, 2])
    if comp_2x8 is not None and comp_2x8['all_tokens'].shape[1] == 2:
        tokens = comp_2x8['all_tokens']
        jitter = np.random.RandomState(0).randn(len(tokens), 2) * 0.15
        scatter = ax3.scatter(
            tokens[:, 0] + jitter[:, 0],
            tokens[:, 1] + jitter[:, 1],
            c=e_bins, cmap='viridis', alpha=0.5, s=15)
        ax3.set_xlabel('Token 0')
        ax3.set_ylabel('Token 1')
        ax3.set_title('Token Space (color=e_bin)')
        ax3.set_xticks(range(VOCAB_SIZE))
        ax3.set_yticks(range(VOCAB_SIZE))
        plt.colorbar(scatter, ax=ax3)

    # Panel 4: Holdout comparison bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    labels = ['Elast', 'Frict', 'Both']
    train_accs = [final_results['2x8_train_e'],
                  final_results['2x8_train_f'],
                  final_results['2x8_train_both']]
    holdout_accs = [final_results['2x8_holdout_e'],
                    final_results['2x8_holdout_f'],
                    final_results['2x8_holdout_both']]
    ctrl_train = [final_results['1x64_train_e'],
                  final_results['1x64_train_f'],
                  final_results['1x64_train_both']]
    ctrl_holdout = [final_results['1x64_holdout_e'],
                    final_results['1x64_holdout_f'],
                    final_results['1x64_holdout_both']]

    x = np.arange(len(labels))
    w = 0.2
    ax4.bar(x - 1.5*w, train_accs, w, label='2x8 train', color='steelblue')
    ax4.bar(x - 0.5*w, holdout_accs, w, label='2x8 holdout', color='lightblue')
    ax4.bar(x + 0.5*w, ctrl_train, w, label='1x64 train', color='coral')
    ax4.bar(x + 1.5*w, ctrl_holdout, w, label='1x64 holdout', color='lightsalmon')
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.set_ylabel('Accuracy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_title('Train vs Holdout (IL)')
    ax4.legend(fontsize=7)
    ax4.set_ylim(0.3, 1.05)

    # Panel 5: MI matrix (position x attribute)
    ax5 = fig.add_subplot(gs[1, 1])
    if comp_2x8 is not None:
        mi = comp_2x8['mi_matrix']
        im = ax5.imshow(mi, aspect='auto', cmap='Blues', vmin=0)
        ax5.set_xticks([0, 1])
        ax5.set_xticklabels(['Elasticity', 'Friction'])
        ax5.set_yticks(range(mi.shape[0]))
        ax5.set_yticklabels([f'Pos {i}' for i in range(mi.shape[0])])
        ax5.set_title('Mutual Information\n(Position x Attribute)')
        for i in range(mi.shape[0]):
            for j in range(mi.shape[1]):
                ax5.text(j, i, f'{mi[i,j]:.3f}', ha='center',
                         va='center', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax5)

    # Panel 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_lines = [
        "Phase 54c: Iterated Learning (DINOv2)",
        "=" * 44,
        f"Dataset: 300 ramp scenes (5x5 grid)",
        f"Encoder: DINOv2 ViT-S/14 (frozen)",
        f"Receiver resets: every {RECEIVER_RESET_INTERVAL} epochs",
        f"Holdout: 5 cells (Latin square)",
        "",
        "--- 2x8 Compositional ---",
        f"  Train: e={final_results['2x8_train_e']:.1%}"
        f" f={final_results['2x8_train_f']:.1%}"
        f" both={final_results['2x8_train_both']:.1%}",
        f"  Holdout: e={final_results['2x8_holdout_e']:.1%}"
        f" f={final_results['2x8_holdout_f']:.1%}"
        f" both={final_results['2x8_holdout_both']:.1%}",
    ]
    if comp_2x8 is not None:
        summary_lines += [
            f"  PosDis: {comp_2x8['pos_dis']:.3f}",
            f"  TopSim: {comp_2x8['topsim']:.3f}",
            f"  Entropy: [{', '.join(f'{e:.2f}' for e in comp_2x8['entropies'])}]",
        ]
    summary_lines += [
        "",
        "--- 1x64 Control ---",
        f"  Train: e={final_results['1x64_train_e']:.1%}"
        f" f={final_results['1x64_train_f']:.1%}"
        f" both={final_results['1x64_train_both']:.1%}",
        f"  Holdout: e={final_results['1x64_holdout_e']:.1%}"
        f" f={final_results['1x64_holdout_f']:.1%}"
        f" both={final_results['1x64_holdout_both']:.1%}",
        "",
        f"Holdout gap (both):",
        f"  2x8:  {final_results['2x8_train_both'] - final_results['2x8_holdout_both']:+.1%}",
        f"  1x64: {final_results['1x64_train_both'] - final_results['1x64_holdout_both']:+.1%}",
    ]
    ax6.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax6.transAxes,
             fontsize=8, va='top', fontfamily='monospace')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved visualization to {save_path}", flush=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60, flush=True)
    print("Phase 54c: Iterated Learning (DINOv2)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Receiver reset interval: {RECEIVER_RESET_INTERVAL} epochs", flush=True)

    t_total = time.time()

    # ================================================================
    # STAGE 0: Load cached DINOv2 features (from Phase 54b)
    # ================================================================
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)

    print(f"  Data shape: {data_t.shape}", flush=True)
    print(f"  Loaded {len(data_t)}/{TOTAL_SCENES} scenes", flush=True)

    # --- Create splits ---
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)
    print(f"  Holdout cells: {HOLDOUT_CELLS}", flush=True)

    print(f"\n  Grid distribution:", flush=True)
    for e in range(5):
        row_t = []
        row_h = []
        for f in range(5):
            mask_t = (e_bins[train_ids] == e) & (f_bins[train_ids] == f)
            mask_h = (e_bins[holdout_ids] == e) & (f_bins[holdout_ids] == f)
            row_t.append(f"{mask_t.sum():2d}")
            row_h.append(f"{mask_h.sum():2d}")
        held = [f"{'*' if (e,f) in HOLDOUT_CELLS else ' '}" for f in range(5)]
        print(f"    e={e}: train=[{', '.join(row_t)}]  "
              f"holdout=[{', '.join(row_h)}]  held={held}", flush=True)

    # ================================================================
    # STAGE 1: Oracle pretrain
    # ================================================================
    oracle = train_oracle(data_t, e_bins, f_bins, train_ids, DEVICE)

    # Extract encoder weights for sender initialization
    oracle_enc_state = oracle.enc_a.state_dict()

    # ================================================================
    # STAGE 2a: Communication training — 2x8 compositional + IL
    # ================================================================
    encoder_2x8 = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder_2x8.load_state_dict(oracle_enc_state)

    sender_2x8 = CompositionalSender(
        encoder_2x8, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(DEVICE)
    msg_dim_2x8 = VOCAB_SIZE * N_HEADS
    receiver_2x8 = CompositionalReceiver(msg_dim_2x8, HIDDEN_DIM).to(DEVICE)

    history_2x8, receiver_2x8 = train_communication(
        sender_2x8, receiver_2x8, data_t, e_bins, f_bins,
        train_ids, holdout_ids, DEVICE, msg_dim=msg_dim_2x8, tag="2x8")

    # ================================================================
    # STAGE 2b: Communication training — 1x64 control + IL
    # ================================================================
    encoder_1x64 = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder_1x64.load_state_dict(oracle_enc_state)

    sender_1x64 = ControlSender(
        encoder_1x64, HIDDEN_DIM, CONTROL_VOCAB).to(DEVICE)
    receiver_1x64 = CompositionalReceiver(CONTROL_VOCAB, HIDDEN_DIM).to(DEVICE)

    history_1x64, receiver_1x64 = train_communication(
        sender_1x64, receiver_1x64, data_t, e_bins, f_bins,
        train_ids, holdout_ids, DEVICE, msg_dim=CONTROL_VOCAB, tag="1x64")

    # ================================================================
    # STAGE 3: Final evaluation
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Final Evaluation", flush=True)
    print(f"{'='*60}", flush=True)

    sender_2x8.eval()
    receiver_2x8.eval()
    sender_1x64.eval()
    receiver_1x64.eval()

    with torch.no_grad():
        te, tf, tb = evaluate_accuracy(
            sender_2x8, receiver_2x8, data_t, e_bins, f_bins,
            train_ids, DEVICE, n_rounds=50)
        he, hf, hb = evaluate_accuracy(
            sender_2x8, receiver_2x8, data_t, e_bins, f_bins,
            holdout_ids, DEVICE, n_rounds=50)
        cte, ctf, ctb = evaluate_accuracy(
            sender_1x64, receiver_1x64, data_t, e_bins, f_bins,
            train_ids, DEVICE, n_rounds=50)
        che, chf, chb = evaluate_accuracy(
            sender_1x64, receiver_1x64, data_t, e_bins, f_bins,
            holdout_ids, DEVICE, n_rounds=50)

    final_results = {
        '2x8_train_e': te, '2x8_train_f': tf, '2x8_train_both': tb,
        '2x8_holdout_e': he, '2x8_holdout_f': hf, '2x8_holdout_both': hb,
        '1x64_train_e': cte, '1x64_train_f': ctf, '1x64_train_both': ctb,
        '1x64_holdout_e': che, '1x64_holdout_f': chf, '1x64_holdout_both': chb,
    }

    print(f"\n  2x8 Compositional + IL:", flush=True)
    print(f"    Train:   e={te:.1%}  f={tf:.1%}  both={tb:.1%}", flush=True)
    print(f"    Holdout: e={he:.1%}  f={hf:.1%}  both={hb:.1%}", flush=True)
    print(f"    Gap:     e={te-he:+.1%}  f={tf-hf:+.1%}  both={tb-hb:+.1%}",
          flush=True)

    print(f"\n  1x64 Control + IL:", flush=True)
    print(f"    Train:   e={cte:.1%}  f={ctf:.1%}  both={ctb:.1%}", flush=True)
    print(f"    Holdout: e={che:.1%}  f={chf:.1%}  both={chb:.1%}", flush=True)
    print(f"    Gap:     e={cte-che:+.1%}  f={ctf-chf:+.1%}  both={ctb-chb:+.1%}",
          flush=True)

    # ================================================================
    # STAGE 4: Compositionality metrics
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Compositionality Metrics", flush=True)
    print(f"{'='*60}", flush=True)

    with torch.no_grad():
        comp_2x8 = compute_compositionality(
            sender_2x8, data_t, e_bins, f_bins, DEVICE)
        comp_1x64 = compute_compositionality(
            sender_1x64, data_t, e_bins, f_bins, DEVICE)

    print(f"\n  2x8 Compositional + IL:", flush=True)
    print(f"    PosDis: {comp_2x8['pos_dis']:.3f}", flush=True)
    print(f"    TopSim: {comp_2x8['topsim']:.3f}", flush=True)
    print(f"    Entropy: [{', '.join(f'{e:.3f}' for e in comp_2x8['entropies'])}]",
          flush=True)
    print(f"    MI matrix (rows=positions, cols=[elast, frict]):", flush=True)
    for p in range(comp_2x8['mi_matrix'].shape[0]):
        mi_row = comp_2x8['mi_matrix'][p]
        print(f"      Pos {p}: [{mi_row[0]:.4f}, {mi_row[1]:.4f}]", flush=True)

    print(f"\n  1x64 Control + IL:", flush=True)
    print(f"    TopSim: {comp_1x64['topsim']:.3f}", flush=True)
    print(f"    Entropy: [{', '.join(f'{e:.3f}' for e in comp_1x64['entropies'])}]",
          flush=True)

    # CBM translation
    print(f"\n  CBM Translation (2x8):", flush=True)
    for p in range(2):
        print(f"    Position {p}:", flush=True)
        for t in range(VOCAB_SIZE):
            if t in comp_2x8['cbm'][p]:
                info = comp_2x8['cbm'][p][t]
                print(f"      token={t}: n={info['count']:3d}  "
                      f"e={info['mean_e']:.1f}  f={info['mean_f']:.1f}",
                      flush=True)

    # Save results
    final_results['2x8_posdis'] = comp_2x8['pos_dis']
    final_results['2x8_topsim'] = comp_2x8['topsim']
    final_results['2x8_entropy'] = comp_2x8['entropies']
    final_results['1x64_topsim'] = comp_1x64['topsim']
    final_results['1x64_entropy'] = comp_1x64['entropies']
    final_results['mode'] = 'dino_iterated_learning'
    final_results['receiver_reset_interval'] = RECEIVER_RESET_INTERVAL

    results_path = RESULTS_DIR / "phase54c_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  Saved results to {results_path}", flush=True)

    # Save models
    model_path = RESULTS_DIR / "phase54c_model.pt"
    torch.save({
        'sender_2x8': sender_2x8.state_dict(),
        'receiver_2x8': receiver_2x8.state_dict(),
        'sender_1x64': sender_1x64.state_dict(),
        'receiver_1x64': receiver_1x64.state_dict(),
    }, model_path)
    print(f"  Saved model to {model_path}", flush=True)

    # Visualization
    plot_path = RESULTS_DIR / "phase54c_iterated_learning.png"
    plot_results(history_2x8, history_1x64, comp_2x8, comp_1x64,
                 final_results, e_bins, f_bins, plot_path)

    dt = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 54c complete! Total time: {dt/60:.1f} min", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

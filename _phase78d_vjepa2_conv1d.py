"""
Phase 78d: V-JEPA 2 spatial-mean → Conv1D (controlled comparison vs DINOv2)
============================================================================
Load full V-JEPA 2 tokens (300, 2048, 1024), reshape to (300, 8, 256, 1024),
mean-pool across 256 spatial tokens → (300, 8, 1024).

Then run EXACTLY Phase 54f architecture but with input_dim=1024:
  TemporalEncoder: Conv1D(1024,256,k=3) → ReLU → Conv1D(256,128,k=3) → pool → 128-dim
  2×5 vocab, population IL, 400 epochs, 20 seeds.

Perfectly controlled: DINOv2 (8×384) vs V-JEPA 2 (8×1024), same Conv1D backbone.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase78d_vjepa2_conv1d.py
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
# Configuration — identical to Phase 54f
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
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

SEEDS = list(range(20))


# ══════════════════════════════════════════════════════════════════
# Architecture — EXACTLY Phase 54f but input_dim=1024
# ══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024):
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
        # x: (B, 8, 1024)
        x = x.permute(0, 2, 1)  # (B, 1024, 8)
        x = self.temporal(x).squeeze(-1)  # (B, 128)
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

def load_vjepa_spatial_mean(path):
    """Load V-JEPA 2 full tokens, reshape, spatial mean → (300, 8, 1024)."""
    data = torch.load(path, weights_only=False)
    features = data['features'].float()  # (300, 2048, 1024) float16→float32
    index = data['index']
    e_bins = np.array([entry['elasticity_bin'] for entry in index])
    f_bins = np.array([entry['friction_bin'] for entry in index])

    # Reshape: (300, 2048, 1024) → (300, 8, 256, 1024) → mean over spatial → (300, 8, 1024)
    N = features.shape[0]
    features = features.view(N, 8, 256, -1)
    features = features.mean(dim=2)  # (300, 8, 1024)

    return features, e_bins, f_bins


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

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(VOCAB_SIZE)))

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
# Training
# ══════════════════════════════════════════════════════════════════

def train_oracle(data_t, e_bins, f_bins, train_ids, device, seed):
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': VJEPA_DIM}
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
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

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
            gen = epoch // RECEIVER_RESET_INTERVAL
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)
            print(f"    ** Reset ALL {len(receivers)} receivers at epoch {epoch+1} "
                  f"(gen {gen}) **", flush=True)

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
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (te, tf, tb), best_r = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
                (he, hf, hb), _ = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                  f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                  f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

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
# Single seed
# ══════════════════════════════════════════════════════════════════

def run_single_seed(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print(f"\n  --- Seed {seed} ---", flush=True)

    oracle, oracle_acc = train_oracle(data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    encoder = TemporalEncoder(HIDDEN_DIM, VJEPA_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    print(f"    Training sender (2x{VOCAB_SIZE}) vs {N_RECEIVERS} receivers "
          f"(IL={RECEIVER_RESET_INTERVAL}, simultaneous)...", flush=True)

    receivers, nan_count = train_population(
        sender, receivers, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

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

    with torch.no_grad():
        comp = compute_compositionality(sender, data_t, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())

    dt = time.time() - t0

    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"MI->e={best_mi_e:.3f}  MI->f={best_mi_f:.3f}  "
          f"NaN={nan_count}  ({dt:.0f}s)", flush=True)

    torch.mps.empty_cache()

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
        'nan_count': nan_count,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 78d: V-JEPA 2 spatial-mean → Conv1D (controlled comparison)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Backbone: V-JEPA 2 ViT-L → spatial mean → (8, 1024)", flush=True)
    print(f"  Encoder: Conv1D(1024→256→128) + pool — SAME as DINOv2 Phase 54f", flush=True)
    print(f"  Vocab: 2x{VOCAB_SIZE}", flush=True)
    print(f"  Receivers: {N_RECEIVERS} (simultaneous IL every {RECEIVER_RESET_INTERVAL})", flush=True)
    print(f"  Seeds: {len(SEEDS)}", flush=True)

    t_total = time.time()

    # Load and preprocess
    vjepa_path = str(RESULTS_DIR / "vjepa2_features_full.pt")
    print(f"  Loading {vjepa_path}...", flush=True)
    data_t, e_bins, f_bins = load_vjepa_spatial_mean(vjepa_path)
    print(f"  Features: {data_t.shape} (dtype={data_t.dtype})", flush=True)
    print(f"  (spatial mean of 256 tokens per temporal position)", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, data_t, e_bins, f_bins,
                                 train_ids, holdout_ids, DEVICE)
        all_results.append(result)

        elapsed = time.time() - t_total
        avg_per_seed = elapsed / len(all_results)
        remaining = avg_per_seed * (len(SEEDS) - len(all_results))
        print(f"\n  [Progress: {len(all_results)}/{len(SEEDS)}, "
              f"ETA {remaining/60:.0f}min]\n", flush=True)

    # ════════════════════════════════════════════════════════════
    # Full table
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"FULL TABLE: Phase 78d V-JEPA 2 Conv1D (400 epochs, 20 seeds)", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"  {'Seed':>4} | {'Holdout':>8} | {'PosDis':>7} | "
              f"{'TopSim':>7} | {'MI->e':>7} | {'MI->f':>7} | {'NaN':>3}")
    print(header, flush=True)
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-{'---':>3}",
          flush=True)

    hb, pd, ts, me, mf = [], [], [], [], []
    for r in all_results:
        tag = ""
        if r['pos_dis'] > 0.4:
            tag = " *"
        print(f"  {r['seed']:>4} | {r['holdout_both']:>7.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['topsim']:>7.3f} | "
              f"{r['best_mi_e']:>7.3f} | {r['best_mi_f']:>7.3f} | "
              f"{r['nan_count']:>3}{tag}", flush=True)
        hb.append(r['holdout_both'])
        pd.append(r['pos_dis'])
        ts.append(r['topsim'])
        me.append(r['best_mi_e'])
        mf.append(r['best_mi_f'])

    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-{'---':>3}",
          flush=True)
    print(f"  {'Mean':>4} | {np.mean(hb):>7.1%} | "
          f"{np.mean(pd):>7.3f} | {np.mean(ts):>7.3f} | "
          f"{np.mean(me):>7.3f} | {np.mean(mf):>7.3f} |", flush=True)
    print(f"  {'Std':>4} | {np.std(hb):>7.1%} | "
          f"{np.std(pd):>7.3f} | {np.std(ts):>7.3f} | "
          f"{np.std(me):>7.3f} | {np.std(mf):>7.3f} |", flush=True)

    # Group analysis
    compositional = [r for r in all_results if r['pos_dis'] > 0.4]
    holistic = [r for r in all_results if r['pos_dis'] < 0.15]
    intermediate = [r for r in all_results if 0.15 <= r['pos_dis'] <= 0.4]
    n = len(all_results)

    print(f"\n  GROUP ANALYSIS:", flush=True)
    print(f"  Compositional (PosDis > 0.4):  {len(compositional):>2}/{n} "
          f"({len(compositional)/n:.0%})", flush=True)
    if compositional:
        c_hb = [r['holdout_both'] for r in compositional]
        c_pd = [r['pos_dis'] for r in compositional]
        print(f"    Mean holdout: {np.mean(c_hb):.1%} +/- {np.std(c_hb):.1%}", flush=True)
        print(f"    Mean PosDis:  {np.mean(c_pd):.3f} +/- {np.std(c_pd):.3f}", flush=True)
    print(f"  Holistic (PosDis < 0.15):      {len(holistic):>2}/{n} "
          f"({len(holistic)/n:.0%})", flush=True)
    print(f"  Intermediate (0.15-0.4):       {len(intermediate):>2}/{n} "
          f"({len(intermediate)/n:.0%})", flush=True)

    # ════════════════════════════════════════════════════════════
    # Comparison vs DINOv2 Phase 54f
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"COMPARISON: V-JEPA 2 Conv1D vs DINOv2 Conv1D (Phase 54f)", flush=True)
    print(f"{'='*70}", flush=True)

    dino_path = RESULTS_DIR / "phase54f_extended.json"
    with open(dino_path) as f:
        dino_data = json.load(f)

    dino_hb = [r['holdout_both'] for r in dino_data['per_seed']]
    dino_pd = [r['pos_dis'] for r in dino_data['per_seed']]
    dino_ts = [r['topsim'] for r in dino_data['per_seed']]
    dino_comp = dino_data['groups']['compositional_count']

    # Also load V-JEPA 2 mean-pool (Phase 78)
    vpool_path = RESULTS_DIR / "phase78_vjepa2.json"
    with open(vpool_path) as f:
        vpool_data = json.load(f)
    vpool_hb = [r['holdout_both'] for r in vpool_data['per_seed']]
    vpool_pd = [r['pos_dis'] for r in vpool_data['per_seed']]
    vpool_comp = vpool_data['groups']['compositional_count']

    print(f"\n  {'Condition':<30} | {'Holdout':<16} | {'PosDis':<16} | {'Comp':>5}",
          flush=True)
    print(f"  {'-'*30}-+-{'-'*16}-+-{'-'*16}-+-{'-'*5}", flush=True)
    print(f"  {'DINOv2 Conv1D (54f)':<30} | {np.mean(dino_hb):.1%} +/- {np.std(dino_hb):.1%}"
          f"{'':>3} | {np.mean(dino_pd):.3f} +/- {np.std(dino_pd):.3f}"
          f"  | {dino_comp:>2}/20", flush=True)
    print(f"  {'V-JEPA2 mean-pool MLP (78)':<30} | {np.mean(vpool_hb):.1%} +/- {np.std(vpool_hb):.1%}"
          f"{'':>3} | {np.mean(vpool_pd):.3f} +/- {np.std(vpool_pd):.3f}"
          f"  | {vpool_comp:>2}/20", flush=True)
    print(f"  {'V-JEPA2 Conv1D (78d)':<30} | {np.mean(hb):.1%} +/- {np.std(hb):.1%}"
          f"{'':>3} | {np.mean(pd):.3f} +/- {np.std(pd):.3f}"
          f"  | {len(compositional):>2}/20", flush=True)

    # Statistical tests
    print(f"\n  STATISTICAL TESTS:", flush=True)

    # 78d vs DINOv2
    t_hb, p_hb = stats.ttest_ind(hb, dino_hb)
    t_pd, p_pd = stats.ttest_ind(pd, dino_pd)
    d_hb = (np.mean(hb) - np.mean(dino_hb)) / np.sqrt(
        (np.std(hb)**2 + np.std(dino_hb)**2) / 2)
    d_pd = (np.mean(pd) - np.mean(dino_pd)) / np.sqrt(
        (np.std(pd)**2 + np.std(dino_pd)**2) / 2)

    print(f"\n  78d vs DINOv2:", flush=True)
    print(f"    Holdout: t={t_hb:.2f}, p={p_hb:.4f}, Cohen's d={d_hb:.2f}", flush=True)
    print(f"    PosDis:  t={t_pd:.2f}, p={p_pd:.4f}, Cohen's d={d_pd:.2f}", flush=True)

    # 78d vs 78 (mean-pool MLP)
    t_hb2, p_hb2 = stats.ttest_ind(hb, vpool_hb)
    d_hb2 = (np.mean(hb) - np.mean(vpool_hb)) / np.sqrt(
        (np.std(hb)**2 + np.std(vpool_hb)**2) / 2)
    print(f"\n  78d vs V-JEPA2 mean-pool:", flush=True)
    print(f"    Holdout: t={t_hb2:.2f}, p={p_hb2:.4f}, Cohen's d={d_hb2:.2f}", flush=True)

    # 95% CIs
    ci_78d = stats.t.interval(0.95, len(hb)-1,
                              loc=np.mean(hb), scale=stats.sem(hb))
    ci_dino = stats.t.interval(0.95, len(dino_hb)-1,
                               loc=np.mean(dino_hb), scale=stats.sem(dino_hb))
    ci_vpool = stats.t.interval(0.95, len(vpool_hb)-1,
                                loc=np.mean(vpool_hb), scale=stats.sem(vpool_hb))
    print(f"\n  95% CIs (holdout):", flush=True)
    print(f"    DINOv2:         [{ci_dino[0]:.1%}, {ci_dino[1]:.1%}]", flush=True)
    print(f"    V-JEPA2 pool:   [{ci_vpool[0]:.1%}, {ci_vpool[1]:.1%}]", flush=True)
    print(f"    V-JEPA2 Conv1D: [{ci_78d[0]:.1%}, {ci_78d[1]:.1%}]", flush=True)

    # MI for best compositional seed
    if compositional:
        best_comp = max(compositional, key=lambda r: r['pos_dis'])
        print(f"\n  Best compositional seed: seed {best_comp['seed']}", flush=True)
        print(f"    PosDis={best_comp['pos_dis']:.3f}, holdout={best_comp['holdout_both']:.1%}", flush=True)
        mi = np.array(best_comp['mi_matrix'])
        print(f"    MI matrix:", flush=True)
        print(f"      {'':>6} {'elast':>7} {'frict':>7}", flush=True)
        for p in range(mi.shape[0]):
            print(f"      pos{p}  {mi[p,0]:>7.3f} {mi[p,1]:>7.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'backbone': 'V-JEPA 2 ViT-L (spatial mean → 8×1024)',
            'feature_dim': VJEPA_DIM,
            'temporal_frames': 8,
            'encoder': 'Conv1D(1024→256→128) + pool — SAME as Phase 54f',
            'vocab_size': VOCAB_SIZE,
            'n_heads': N_HEADS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'reset_mode': 'simultaneous',
            'comm_epochs': COMM_EPOCHS,
            'oracle_epochs': ORACLE_EPOCHS,
            'sender_lr': SENDER_LR,
            'receiver_lr': RECEIVER_LR,
            'tau_start': TAU_START,
            'tau_end': TAU_END,
        },
        'seeds': SEEDS,
        'per_seed': all_results,
        'summary': {
            'holdout_both_mean': float(np.mean(hb)),
            'holdout_both_std': float(np.std(hb)),
            'pos_dis_mean': float(np.mean(pd)),
            'pos_dis_std': float(np.std(pd)),
            'topsim_mean': float(np.mean(ts)),
            'topsim_std': float(np.std(ts)),
            'best_mi_e_mean': float(np.mean(me)),
            'best_mi_e_std': float(np.std(me)),
            'best_mi_f_mean': float(np.mean(mf)),
            'best_mi_f_std': float(np.std(mf)),
        },
        'groups': {
            'compositional_count': len(compositional),
            'compositional_seeds': [r['seed'] for r in compositional],
            'holistic_count': len(holistic),
            'holistic_seeds': [r['seed'] for r in holistic],
            'intermediate_count': len(intermediate),
        },
        'comparison_vs_dino': {
            'holdout_t': float(t_hb),
            'holdout_p': float(p_hb),
            'holdout_d': float(d_hb),
            'posdis_t': float(t_pd),
            'posdis_p': float(p_pd),
            'posdis_d': float(d_pd),
            'ci_78d': [float(ci_78d[0]), float(ci_78d[1])],
            'ci_dino': [float(ci_dino[0]), float(ci_dino[1])],
        },
    }

    save_path = RESULTS_DIR / "phase78d_vjepa2_conv1d.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/len(SEEDS):.0f}s per seed)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

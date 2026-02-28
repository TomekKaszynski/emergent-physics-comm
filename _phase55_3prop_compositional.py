"""
Phase 55: 3-Property Compositional Communication
==================================================
Tests whether compositionality scales from 2 properties (Phase 54f) to 3.
Uses restitution + friction + linearDamping with 3×5 Gumbel-Softmax messages.

Includes:
- DINOv2 feature extraction (Stage 2)
- Latin cube holdout (Stage 3)
- 3×5 compositional training with population + simultaneous IL (Stages 4-5)
- Compositionality evaluation: 3×3 MI matrix, PosDis, TopSim (Stage 6)
- Visualization (Stage 7)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase55_3prop_compositional.py
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
DATASET_DIR = Path("kubric/output/ramp_3prop_dataset")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 3          # 3 positions for 3 properties
N_PROPERTIES = 3
BATCH_SIZE = 64      # larger dataset, larger batch

# Latin cube holdout: (e_bin + f_bin + d_bin) % 5 == 0
# 25 held-out triples out of 125 (20%)
def is_holdout(e_bin, f_bin, d_bin):
    return (e_bin + f_bin + d_bin) % 5 == 0

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
# Stage 2: DINOv2 Feature Extraction
# ══════════════════════════════════════════════════════════════════

def extract_dino_features(dataset_dir, cache_path, n_frames=8):
    """Extract DINOv2 ViT-S/14 CLS tokens from rendered scenes."""
    if Path(cache_path).exists():
        print(f"  Loading cached features from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=False)
        return data['features'], data['e_bins'], data['f_bins'], data['d_bins']

    import torchvision.transforms as T
    from PIL import Image

    print(f"  Extracting DINOv2 features from {dataset_dir}...", flush=True)

    # Load DINOv2
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino = dino.to(DEVICE)
    dino.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load index
    index_path = dataset_dir / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    n_scenes = len(index)
    # Subsample frames: 8 evenly spaced from 24
    frame_indices = np.linspace(0, 23, n_frames, dtype=int)

    all_features = []
    e_bins = []
    f_bins = []
    d_bins = []

    t0 = time.time()
    for i, meta in enumerate(index):
        scene_dir = dataset_dir / f"scene_{meta['scene_id']:04d}"
        frames = []
        for fi in frame_indices:
            img_path = scene_dir / f"rgba_{fi:05d}.png"
            img = Image.open(img_path).convert("RGB")
            frames.append(transform(img))

        batch = torch.stack(frames).to(DEVICE)  # (T, 3, 224, 224)
        with torch.no_grad():
            features = dino(batch)  # (T, 384)
        all_features.append(features.cpu())

        e_bins.append(meta['elasticity_bin'])
        f_bins.append(meta['friction_bin'])
        d_bins.append(meta['damping_bin'])

        if (i + 1) % 50 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (n_scenes - i - 1)
            print(f"    [{i+1}/{n_scenes}] {dt:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

        if (i + 1) % 100 == 0:
            torch.mps.empty_cache()

    features_t = torch.stack(all_features)  # (N, T, 384)
    e_bins = np.array(e_bins)
    f_bins = np.array(f_bins)
    d_bins = np.array(d_bins)

    # Save cache
    torch.save({
        'features': features_t,
        'e_bins': e_bins,
        'f_bins': f_bins,
        'd_bins': d_bins,
    }, cache_path)

    print(f"  Saved {features_t.shape} features to {cache_path}", flush=True)
    return features_t, e_bins, f_bins, d_bins


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
    """Receiver with 3 output heads (elasticity, friction, damping)."""
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
        self.damping_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return (self.elast_head(h).squeeze(-1),
                self.friction_head(h).squeeze(-1),
                self.damping_head(h).squeeze(-1))


class Oracle(nn.Module):
    """Oracle with 3 output heads."""
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
        self.damping_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_a, x_b):
        ha = self.enc_a(x_a)
        hb = self.enc_b(x_b)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return (self.elast_head(h).squeeze(-1),
                self.friction_head(h).squeeze(-1),
                self.damping_head(h).squeeze(-1))


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def create_splits(e_bins, f_bins, d_bins):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if is_holdout(int(e_bins[i]), int(f_bins[i]), int(d_bins[i])):
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

def evaluate_accuracy(sender, receiver, data_t, e_bins, f_bins, d_bins,
                      scene_ids, device, oracle_model=None, n_rounds=30):
    """Evaluate 3-property accuracy."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    d_dev = torch.tensor(d_bins, dtype=torch.float32).to(device)

    correct_e = correct_f = correct_d = correct_all = 0
    total_e = total_f = total_d = total_all = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])
        label_d = (d_dev[ia] > d_dev[ib])

        if oracle_model is not None:
            pred_e, pred_f, pred_d = oracle_model(da, db)
        else:
            msg_a, _ = sender(da)
            msg_b, _ = sender(db)
            pred_e, pred_f, pred_d = receiver(msg_a, msg_b)

        pred_e_bin = pred_e > 0
        pred_f_bin = pred_f > 0
        pred_d_bin = pred_d > 0

        e_diff = torch.tensor(e_bins[ia] != e_bins[ib], device=device)
        f_diff = torch.tensor(f_bins[ia] != f_bins[ib], device=device)
        d_diff = torch.tensor(d_bins[ia] != d_bins[ib], device=device)

        if e_diff.sum() > 0:
            correct_e += (pred_e_bin[e_diff] == label_e[e_diff]).sum().item()
            total_e += e_diff.sum().item()
        if f_diff.sum() > 0:
            correct_f += (pred_f_bin[f_diff] == label_f[f_diff]).sum().item()
            total_f += f_diff.sum().item()
        if d_diff.sum() > 0:
            correct_d += (pred_d_bin[d_diff] == label_d[d_diff]).sum().item()
            total_d += d_diff.sum().item()

        all_diff = e_diff & f_diff & d_diff
        if all_diff.sum() > 0:
            all_ok = ((pred_e_bin[all_diff] == label_e[all_diff]) &
                      (pred_f_bin[all_diff] == label_f[all_diff]) &
                      (pred_d_bin[all_diff] == label_d[all_diff]))
            correct_all += all_ok.sum().item()
            total_all += all_diff.sum().item()

    return {
        'e': correct_e / max(total_e, 1),
        'f': correct_f / max(total_f, 1),
        'd': correct_d / max(total_d, 1),
        'all': correct_all / max(total_all, 1),
    }


def evaluate_population(sender, receivers, data_t, e_bins, f_bins, d_bins,
                        scene_ids, device, n_rounds=30):
    """Evaluate with best receiver (highest all-correct)."""
    best_all = 0
    best_r = None
    for r in receivers:
        acc = evaluate_accuracy(
            sender, r, data_t, e_bins, f_bins, d_bins,
            scene_ids, device, n_rounds=10)
        if acc['all'] > best_all:
            best_all = acc['all']
            best_r = r
    return evaluate_accuracy(
        sender, best_r, data_t, e_bins, f_bins, d_bins,
        scene_ids, device, n_rounds=n_rounds), best_r


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


def compute_compositionality(sender, data_t, e_bins, f_bins, d_bins, device,
                             vocab_size=VOCAB_SIZE):
    """Compute 3×3 MI matrix, PosDis, TopSim for 3-property case."""
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

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, n_pos)
    n_pos = all_tokens.shape[1]

    # Entropy per position
    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=vocab_size)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(vocab_size)))

    # 3×3 MI matrix: positions × attributes
    attributes = np.stack([e_bins, f_bins, d_bins], axis=1)
    mi_matrix = np.zeros((n_pos, N_PROPERTIES))
    for p in range(n_pos):
        for a in range(N_PROPERTIES):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    # PosDis (extended to 3 positions)
    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    # TopSim (3 attributes)
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(data_t), size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])) +
                             abs(int(d_bins[i]) - int(d_bins[j])))
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

def train_oracle(data_t, e_bins, f_bins, d_bins, train_ids, device, seed):
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    d_dev = torch.tensor(d_bins, dtype=torch.float32).to(device)

    best_acc = {}
    best_state = None
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            label_d = (d_dev[ia] > d_dev[ib]).float()
            pred_e, pred_f, pred_d = oracle(da, db)
            loss = (F.binary_cross_entropy_with_logits(pred_e, label_e) +
                    F.binary_cross_entropy_with_logits(pred_f, label_f) +
                    F.binary_cross_entropy_with_logits(pred_d, label_d))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                acc = evaluate_accuracy(
                    None, None, data_t, e_bins, f_bins, d_bins,
                    train_ids, device, oracle_model=oracle)
            if not best_acc or acc['all'] > best_acc.get('all', 0):
                best_acc = acc
                best_state = {k: v.cpu().clone()
                              for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def train_population(sender, receivers, data_t, e_bins, f_bins, d_bins,
                     train_ids, holdout_ids, device, msg_dim, seed):
    """Train sender against population of receivers with simultaneous IL."""
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    d_dev = torch.tensor(d_bins, dtype=torch.float32).to(device)

    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_all_acc = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0

    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # Simultaneous IL: reset ALL receivers at once
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
            label_d = (d_dev[ia] > d_dev[ib]).float()

            msg_a, logits_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b = sender(db, tau=tau, hard=hard)

            # Average loss across all receivers
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f, pred_d = r(msg_a, msg_b)
                r_loss = (F.binary_cross_entropy_with_logits(pred_e, label_e) +
                          F.binary_cross_entropy_with_logits(pred_f, label_f) +
                          F.binary_cross_entropy_with_logits(pred_d, label_d))
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization on sender logits
            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

            # NaN-safe gradient step
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
                train_acc, best_r = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins, d_bins,
                    train_ids, device, n_rounds=20)
                holdout_acc, _ = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins, d_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                  f"train[e={train_acc['e']:.1%} f={train_acc['f']:.1%} "
                  f"d={train_acc['d']:.1%} all={train_acc['all']:.1%}]  "
                  f"hold[e={holdout_acc['e']:.1%} f={holdout_acc['f']:.1%} "
                  f"d={holdout_acc['d']:.1%} all={holdout_acc['all']:.1%}]{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if train_acc['all'] > best_all_acc:
                best_all_acc = train_acc['all']
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
# Single seed run
# ══════════════════════════════════════════════════════════════════

def run_single_seed(seed, data_t, e_bins, f_bins, d_bins,
                    train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle(
        data_t, e_bins, f_bins, d_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: e={oracle_acc['e']:.1%} f={oracle_acc['f']:.1%} "
          f"d={oracle_acc['d']:.1%} all={oracle_acc['all']:.1%}", flush=True)

    # Check damping signal
    if oracle_acc['d'] < 0.70:
        print(f"    WARNING: damping oracle <70% ({oracle_acc['d']:.1%}). "
              f"Signal may be weak.", flush=True)

    # Sender (vocab=5, 3 heads -> msg_dim=15)
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS  # 15

    # Population of receivers
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    print(f"    Training sender (3x{VOCAB_SIZE}) vs {N_RECEIVERS} receivers "
          f"(IL={RECEIVER_RESET_INTERVAL}, simultaneous)...", flush=True)

    receivers, nan_count = train_population(
        sender, receivers, data_t, e_bins, f_bins, d_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval with best receiver
    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        train_acc, best_r = evaluate_population(
            sender, receivers, data_t, e_bins, f_bins, d_bins,
            train_ids, device, n_rounds=50)
        holdout_acc = evaluate_accuracy(
            sender, best_r, data_t, e_bins, f_bins, d_bins,
            holdout_ids, device, n_rounds=50)

    # Compositionality
    with torch.no_grad():
        comp = compute_compositionality(
            sender, data_t, e_bins, f_bins, d_bins, device)

    mi = comp['mi_matrix']  # (3, 3)
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())
    best_mi_d = float(mi[:, 2].max())

    dt = time.time() - t0

    print(f"    -> holdout_all={holdout_acc['all']:.1%}  "
          f"PosDis={comp['pos_dis']:.3f}  TopSim={comp['topsim']:.3f}  "
          f"MI->e={best_mi_e:.3f}  MI->f={best_mi_f:.3f}  MI->d={best_mi_d:.3f}  "
          f"NaN={nan_count}  ({dt:.0f}s)", flush=True)

    return {
        'seed': seed,
        'oracle': oracle_acc,
        'train': train_acc,
        'holdout': holdout_acc,
        'pos_dis': comp['pos_dis'],
        'topsim': comp['topsim'],
        'entropies': comp['entropies'],
        'mi_matrix': mi.tolist(),
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'best_mi_d': best_mi_d,
        'nan_count': nan_count,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 55: 3-Property Compositional Communication", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Vocab: 3x{VOCAB_SIZE} (msg_dim={VOCAB_SIZE * N_HEADS})", flush=True)
    print(f"  Properties: restitution, friction, linearDamping", flush=True)
    print(f"  Receivers: {N_RECEIVERS}", flush=True)
    print(f"  Simultaneous IL: reset ALL every {RECEIVER_RESET_INTERVAL} epochs",
          flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)

    t_total = time.time()

    # Stage 2: Extract/load features
    cache_path = str(RESULTS_DIR / "phase55_dino_features.pt")
    data_t, e_bins, f_bins, d_bins = extract_dino_features(
        DATASET_DIR, cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    # Stage 3: Splits
    train_ids, holdout_ids = create_splits(e_bins, f_bins, d_bins)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Verify holdout coverage
    holdout_triples = set()
    for i in holdout_ids:
        holdout_triples.add((int(e_bins[i]), int(f_bins[i]), int(d_bins[i])))
    print(f"  Holdout triples: {len(holdout_triples)} (expect 25)", flush=True)

    # Run all seeds
    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, data_t, e_bins, f_bins, d_bins,
                                 train_ids, holdout_ids, DEVICE)
        all_results.append(result)

    # ════════════════════════════════════════════════════════════
    # Full table
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}", flush=True)
    print(f"FULL TABLE: Phase 55 (3x5, 400 epochs, 20 seeds)", flush=True)
    print(f"{'='*80}", flush=True)

    header = (f"  {'Seed':>4} | {'Hold_all':>8} | {'Hold_e':>7} | "
              f"{'Hold_f':>7} | {'Hold_d':>7} | {'PosDis':>7} | "
              f"{'TopSim':>7} | {'MI->e':>6} | {'MI->f':>6} | {'MI->d':>6}")
    print(header, flush=True)
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'------':>6}-+-{'------':>6}-+-{'------':>6}",
          flush=True)

    hb, pd_list, ts, me, mf, md = [], [], [], [], [], []
    he_list, hf_list, hd_list = [], [], []
    for r in all_results:
        tag = " *" if r['pos_dis'] > 0.4 else ""
        print(f"  {r['seed']:>4} | {r['holdout']['all']:>7.1%} | "
              f"{r['holdout']['e']:>6.1%} | {r['holdout']['f']:>6.1%} | "
              f"{r['holdout']['d']:>6.1%} | {r['pos_dis']:>7.3f} | "
              f"{r['topsim']:>7.3f} | {r['best_mi_e']:>6.3f} | "
              f"{r['best_mi_f']:>6.3f} | {r['best_mi_d']:>6.3f}{tag}", flush=True)
        hb.append(r['holdout']['all'])
        he_list.append(r['holdout']['e'])
        hf_list.append(r['holdout']['f'])
        hd_list.append(r['holdout']['d'])
        pd_list.append(r['pos_dis'])
        ts.append(r['topsim'])
        me.append(r['best_mi_e'])
        mf.append(r['best_mi_f'])
        md.append(r['best_mi_d'])

    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'------':>6}-+-{'------':>6}-+-{'------':>6}",
          flush=True)
    print(f"  {'Mean':>4} | {np.mean(hb):>7.1%} | "
          f"{np.mean(he_list):>6.1%} | {np.mean(hf_list):>6.1%} | "
          f"{np.mean(hd_list):>6.1%} | {np.mean(pd_list):>7.3f} | "
          f"{np.mean(ts):>7.3f} | {np.mean(me):>6.3f} | "
          f"{np.mean(mf):>6.3f} | {np.mean(md):>6.3f}", flush=True)
    print(f"  {'Std':>4} | {np.std(hb):>7.1%} | "
          f"{np.std(he_list):>6.1%} | {np.std(hf_list):>6.1%} | "
          f"{np.std(hd_list):>6.1%} | {np.std(pd_list):>7.3f} | "
          f"{np.std(ts):>7.3f} | {np.std(me):>6.3f} | "
          f"{np.std(mf):>6.3f} | {np.std(md):>6.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Oracle summary
    # ════════════════════════════════════════════════════════════
    print(f"\n  ORACLE ACCURACY (sanity check):", flush=True)
    oracle_e = [r['oracle']['e'] for r in all_results]
    oracle_f = [r['oracle']['f'] for r in all_results]
    oracle_d = [r['oracle']['d'] for r in all_results]
    oracle_all = [r['oracle']['all'] for r in all_results]
    print(f"    e:   {np.mean(oracle_e):.1%} +/- {np.std(oracle_e):.1%}", flush=True)
    print(f"    f:   {np.mean(oracle_f):.1%} +/- {np.std(oracle_f):.1%}", flush=True)
    print(f"    d:   {np.mean(oracle_d):.1%} +/- {np.std(oracle_d):.1%}", flush=True)
    print(f"    all: {np.mean(oracle_all):.1%} +/- {np.std(oracle_all):.1%}", flush=True)

    if np.mean(oracle_d) < 0.70:
        print(f"    ** WARNING: damping oracle below 70% — weak signal! **", flush=True)

    # ════════════════════════════════════════════════════════════
    # Group analysis
    # ════════════════════════════════════════════════════════════
    compositional = [r for r in all_results if r['pos_dis'] > 0.4]
    holistic = [r for r in all_results if r['pos_dis'] < 0.15]
    intermediate = [r for r in all_results if 0.15 <= r['pos_dis'] <= 0.4]

    n = len(all_results)
    print(f"\n  GROUP ANALYSIS:", flush=True)
    print(f"  Compositional (PosDis > 0.4):  {len(compositional):>2}/{n} "
          f"({len(compositional)/n:.0%})", flush=True)
    if compositional:
        c_hb = [r['holdout']['all'] for r in compositional]
        c_pd = [r['pos_dis'] for r in compositional]
        print(f"    Mean holdout_all: {np.mean(c_hb):.1%} +/- {np.std(c_hb):.1%}", flush=True)
        print(f"    Mean PosDis:      {np.mean(c_pd):.3f} +/- {np.std(c_pd):.3f}", flush=True)
        print(f"    Seeds: {[r['seed'] for r in compositional]}", flush=True)

    print(f"  Holistic (PosDis < 0.15):      {len(holistic):>2}/{n} "
          f"({len(holistic)/n:.0%})", flush=True)
    if holistic:
        h_hb = [r['holdout']['all'] for r in holistic]
        print(f"    Mean holdout_all: {np.mean(h_hb):.1%} +/- {np.std(h_hb):.1%}", flush=True)
        print(f"    Seeds: {[r['seed'] for r in holistic]}", flush=True)

    print(f"  Intermediate (0.15-0.4):       {len(intermediate):>2}/{n} "
          f"({len(intermediate)/n:.0%})", flush=True)

    # ════════════════════════════════════════════════════════════
    # PosDis histogram
    # ════════════════════════════════════════════════════════════
    print(f"\n  PosDis HISTOGRAM:", flush=True)
    bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    bin_labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4",
                  "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7+   "]
    for i in range(len(bin_edges) - 1):
        count = sum(1 for p in pd_list if bin_edges[i] <= p < bin_edges[i + 1])
        bar = "#" * (count * 3)
        print(f"  {bin_labels[i]} | {bar} {count}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Average MI matrix (headline figure)
    # ════════════════════════════════════════════════════════════
    all_mi = np.array([r['mi_matrix'] for r in all_results])  # (20, 3, 3)
    mean_mi = all_mi.mean(axis=0)
    print(f"\n  AVERAGE MI MATRIX (positions × properties):", flush=True)
    print(f"  {'':>6} | {'elast':>8} | {'frict':>8} | {'damp':>8}", flush=True)
    print(f"  {'------':>6}-+-{'--------':>8}-+-{'--------':>8}-+-{'--------':>8}",
          flush=True)
    for p in range(mean_mi.shape[0]):
        print(f"  pos_{p} | {mean_mi[p,0]:>8.3f} | {mean_mi[p,1]:>8.3f} | "
              f"{mean_mi[p,2]:>8.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'n_heads': N_HEADS,
            'msg_dim': VOCAB_SIZE * N_HEADS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'reset_mode': 'simultaneous',
            'comm_epochs': COMM_EPOCHS,
            'n_properties': N_PROPERTIES,
            'condition': 'compositional_3x5',
        },
        'seeds': SEEDS,
        'per_seed': all_results,
        'summary': {
            'holdout_all_mean': float(np.mean(hb)),
            'holdout_all_std': float(np.std(hb)),
            'holdout_e_mean': float(np.mean(he_list)),
            'holdout_f_mean': float(np.mean(hf_list)),
            'holdout_d_mean': float(np.mean(hd_list)),
            'pos_dis_mean': float(np.mean(pd_list)),
            'pos_dis_std': float(np.std(pd_list)),
            'topsim_mean': float(np.mean(ts)),
            'topsim_std': float(np.std(ts)),
            'best_mi_e_mean': float(np.mean(me)),
            'best_mi_f_mean': float(np.mean(mf)),
            'best_mi_d_mean': float(np.mean(md)),
            'mean_mi_matrix': mean_mi.tolist(),
            'oracle_e_mean': float(np.mean(oracle_e)),
            'oracle_f_mean': float(np.mean(oracle_f)),
            'oracle_d_mean': float(np.mean(oracle_d)),
            'oracle_all_mean': float(np.mean(oracle_all)),
        },
        'groups': {
            'compositional_count': len(compositional),
            'compositional_seeds': [r['seed'] for r in compositional],
            'holistic_count': len(holistic),
            'intermediate_count': len(intermediate),
        },
    }

    save_path = RESULTS_DIR / "phase55_results.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/len(SEEDS):.0f}s per seed)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

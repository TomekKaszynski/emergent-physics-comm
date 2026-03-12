"""
Phase 77: E2E perception scaled to 20 seeds
=============================================
Run e2e seeds 5-19 (15 new seeds), merge with Phase 74 seeds 0-4.
Config identical to Phase 74 e2e condition. 20 e2e vs 20 frozen.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase77_e2e_15seeds.py
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
from PIL import Image
from sklearn.linear_model import Ridge

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration (identical to Phase 74)
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
DATASET_DIR = Path("kubric/output/ramp_dataset")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
N_FRAMES = 8

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
ENCODER_LR = 1e-5
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

MSG_DIM = VOCAB_SIZE * N_HEADS  # 10
UNFREEZE_BLOCKS = [10, 11]

# Seeds 5-19 are new; 0-4 come from Phase 74
NEW_SEEDS = list(range(5, 20))
TOTAL_SCENES = 300


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 74)
# ══════════════════════════════════════════════════════════════════

class DINOEncoder(nn.Module):
    def __init__(self, dino_model, hidden_dim=128, unfreeze_blocks=None):
        super().__init__()
        self.dino = dino_model
        self.unfreeze_blocks = unfreeze_blocks or []
        for p in self.dino.parameters():
            p.requires_grad = False
        if self.unfreeze_blocks:
            for idx in self.unfreeze_blocks:
                for p in self.dino.blocks[idx].parameters():
                    p.requires_grad = True
            for p in self.dino.norm.parameters():
                p.requires_grad = True

        n_total = sum(p.numel() for p in self.dino.parameters())
        n_trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        print(f"    DINOv2: {n_total:,} total, {n_trainable:,} trainable "
              f"({n_trainable/n_total:.1%})", flush=True)

        self.temporal = nn.Sequential(
            nn.Conv1d(DINO_DIM, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def extract_features(self, images):
        return self.dino(images)

    def forward(self, images_bt, n_frames=N_FRAMES):
        bt = images_bt.shape[0]
        b = bt // n_frames
        cls_tokens = self.extract_features(images_bt)
        cls_tokens = cls_tokens.reshape(b, n_frames, DINO_DIM)
        x = cls_tokens.permute(0, 2, 1)
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
# Data loading
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


RESTITUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
FRICTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
SCENES_PER_CELL = 12


def get_grid_cell(scene_id):
    cell_idx = scene_id // SCENES_PER_CELL
    e_bin = cell_idx // len(FRICTION_LEVELS)
    f_bin = cell_idx % len(FRICTION_LEVELS)
    return (RESTITUTION_LEVELS[e_bin], FRICTION_LEVELS[f_bin], e_bin, f_bin)


def load_raw_images(dataset_dir, device):
    """Load all scene images as tensors. Returns (N, T, 3, 224, 224), e_bins, f_bins."""
    dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
    frame_indices = np.linspace(0, 23, N_FRAMES, dtype=int)

    all_images = []
    e_bins = []
    f_bins = []

    for sid in range(TOTAL_SCENES):
        scene_dir = dataset_dir / f"scene_{sid:04d}"
        _, _, e_bin, f_bin = get_grid_cell(sid)

        frames = []
        for fi in frame_indices:
            fpath = scene_dir / f"rgba_{fi:05d}.png"
            img = Image.open(fpath).convert('RGB')
            img = img.resize((224, 224), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            frames.append(img_np)

        scene_t = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
        all_images.append(scene_t)
        e_bins.append(e_bin)
        f_bins.append(f_bin)

        if (sid + 1) % 100 == 0:
            print(f"    Loaded {sid+1}/{TOTAL_SCENES} scenes", flush=True)

    images = torch.stack(all_images)
    images = (images - dino_mean) / dino_std
    e_bins = np.array(e_bins, dtype=int)
    f_bins = np.array(f_bins, dtype=int)
    print(f"  Images: {images.shape} ({images.numel() * 4 / 1e9:.2f} GB)", flush=True)
    return images, e_bins, f_bins


# ══════════════════════════════════════════════════════════════════
# Evaluation functions (e2e — operate on raw images)
# ══════════════════════════════════════════════════════════════════

def evaluate_accuracy_e2e(sender, receiver, images, e_bins, f_bins,
                          scene_ids, device, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        imgs_a = images[ia].view(-1, 3, 224, 224).to(device)
        imgs_b = images[ib].view(-1, 3, 224, 224).to(device)
        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])

        msg_a, _ = sender(imgs_a)
        msg_b, _ = sender(imgs_b)
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


def evaluate_population_e2e(sender, receivers, images, e_bins, f_bins,
                             scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_e2e(
            sender, r, images, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_e2e(
        sender, best_r, images, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


def compute_compositionality_e2e(sender, images, e_bins, f_bins, device):
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE]
            imgs_flat = batch.view(-1, 3, 224, 224).to(device)
            msg, logits = sender(imgs_flat)
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

    topsim = 0.0
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(images) * (len(images) - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(images), size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
    corr, _ = stats.spearmanr(meaning_dists, message_dists)
    if not np.isnan(corr):
        topsim = float(corr)

    return {
        'pos_dis': float(pos_dis),
        'topsim': topsim,
        'mi_matrix': mi_matrix.tolist(),
    }


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


# ══════════════════════════════════════════════════════════════════
# Linear probe & feature extraction
# ══════════════════════════════════════════════════════════════════

def extract_dino_features_from_model(dino_encoder, images, device):
    dino_encoder.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE]
            bt = batch.view(-1, 3, 224, 224).to(device)
            cls_tokens = dino_encoder.extract_features(bt)
            cls_tokens = cls_tokens.reshape(batch.shape[0], -1)
            all_feats.append(cls_tokens.cpu())
    return torch.cat(all_feats, dim=0)


def linear_probe(features, e_bins, f_bins, train_ids, holdout_ids):
    X_train = features[train_ids].numpy()
    X_test = features[holdout_ids].numpy()
    y_e_train, y_e_test = e_bins[train_ids], e_bins[holdout_ids]
    y_f_train, y_f_test = f_bins[train_ids], f_bins[holdout_ids]

    ridge_e = Ridge(alpha=1.0).fit(X_train, y_e_train)
    ridge_f = Ridge(alpha=1.0).fit(X_train, y_f_train)
    r2_e = ridge_e.score(X_test, y_e_test)
    r2_f = ridge_f.score(X_test, y_f_test)
    return float(r2_e), float(r2_f)


# ══════════════════════════════════════════════════════════════════
# E2E training (identical to Phase 74)
# ══════════════════════════════════════════════════════════════════

def train_population_e2e(sender, receivers, images, e_bins, f_bins,
                         train_ids, holdout_ids, device, seed):
    encoder_params = list(sender.encoder.dino.parameters())
    encoder_params += list(sender.encoder.temporal.parameters())
    encoder_params += list(sender.encoder.fc.parameters())
    head_params = list(sender.heads.parameters())

    sender_opt = torch.optim.Adam([
        {'params': [p for p in encoder_params if p.requires_grad], 'lr': ENCODER_LR},
        {'params': head_params, 'lr': SENDER_LR},
    ])
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
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            imgs_a = images[ia].view(-1, 3, 224, 224).to(device)
            imgs_b = images[ib].view(-1, 3, 224, 224).to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a = sender(imgs_a, tau=tau, hard=hard)
            msg_b, logits_b = sender(imgs_b, tau=tau, hard=hard)

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
                (te, tf, tb), _ = evaluate_population_e2e(
                    sender, receivers, images, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            print(f"        Ep {epoch+1:3d}: train={tb:.1%}  "
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
# Single e2e seed
# ══════════════════════════════════════════════════════════════════

def run_e2e_seed(seed, images, e_bins, f_bins, train_ids, holdout_ids,
                 device, frozen_features):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                          pretrained=True)
    dino_encoder = DINOEncoder(dino, HIDDEN_DIM, UNFREEZE_BLOCKS).to(device)

    sender = CompositionalSender(dino_encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    receivers = [CompositionalReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    print(f"    Training e2e sender (DINOv2 blocks {UNFREEZE_BLOCKS} unfrozen)...",
          flush=True)

    receivers, nan_count = train_population_e2e(
        sender, receivers, images, e_bins, f_bins,
        train_ids, holdout_ids, device, seed)

    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population_e2e(
            sender, receivers, images, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy_e2e(
            sender, best_r, images, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)
        comp = compute_compositionality_e2e(sender, images, e_bins, f_bins, device)

    with torch.no_grad():
        e2e_features = extract_dino_features_from_model(
            dino_encoder, images, device)
    r2_e, r2_f = linear_probe(e2e_features, e_bins, f_bins, train_ids, holdout_ids)

    with torch.no_grad():
        cos_sim = F.cosine_similarity(
            e2e_features.float(), frozen_features.float(), dim=-1).mean().item()

    dt = time.time() - t0
    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"probe R²(e)={r2_e:.3f} R²(f)={r2_f:.3f}  "
          f"cos_sim={cos_sim:.4f}  ({dt:.0f}s)", flush=True)

    del dino, dino_encoder
    torch.mps.empty_cache()

    return {
        'seed': seed,
        'train_both': float(tb), 'holdout_both': float(hb),
        'holdout_e': float(he), 'holdout_f': float(hf),
        'pos_dis': float(comp['pos_dis']),
        'topsim': float(comp['topsim']),
        'mi_matrix': comp['mi_matrix'],
        'probe_r2_e': float(r2_e),
        'probe_r2_f': float(r2_f),
        'feature_cos_sim': float(cos_sim),
        'nan_count': nan_count,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 77: E2E perception — 20 seeds", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  New seeds: {NEW_SEEDS} (reusing Phase 74 seeds 0-4)", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}", flush=True)

    t_total = time.time()

    # Load cached features for splits
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Load raw images
    print(f"\n  Loading raw images...", flush=True)
    images, _, _ = load_raw_images(DATASET_DIR, DEVICE)

    # Pre-compute frozen DINOv2 features for divergence measurement
    print(f"  Pre-computing frozen DINOv2 features...", flush=True)
    frozen_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                  pretrained=True)
    frozen_dino_enc = DINOEncoder(frozen_dino, HIDDEN_DIM, unfreeze_blocks=[])
    frozen_dino_enc.eval().to(DEVICE)
    with torch.no_grad():
        frozen_features = extract_dino_features_from_model(
            frozen_dino_enc, images, DEVICE)
    del frozen_dino, frozen_dino_enc
    torch.mps.empty_cache()
    print(f"  Frozen features: {frozen_features.shape}", flush=True)

    # Frozen linear probe (same for all seeds)
    frozen_flat = data_t.view(len(data_t), -1)
    frozen_r2_e, frozen_r2_f = linear_probe(frozen_flat, e_bins, f_bins,
                                             train_ids, holdout_ids)
    print(f"  Frozen linear probe: R²(e)={frozen_r2_e:.3f}, R²(f)={frozen_r2_f:.3f}",
          flush=True)

    # ═══════════════════════════════════════════════════════════
    # Run new e2e seeds (5-14)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"Running {len(NEW_SEEDS)} new e2e seeds", flush=True)
    print(f"{'='*70}", flush=True)

    new_results = []
    t_e2e_start = time.time()
    for i, seed in enumerate(NEW_SEEDS):
        if i > 0:
            elapsed = time.time() - t_e2e_start
            per_seed = elapsed / i
            remaining = per_seed * (len(NEW_SEEDS) - i)
            print(f"\n  [Progress: {i}/{len(NEW_SEEDS)}, "
                  f"ETA {remaining/60:.0f}min]", flush=True)

        print(f"\n  --- E2E seed {seed} ---", flush=True)
        result = run_e2e_seed(seed, images, e_bins, f_bins,
                               train_ids, holdout_ids, DEVICE, frozen_features)
        new_results.append(result)
        torch.mps.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Merge with Phase 74 seeds 0-4
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"Merging with Phase 74 seeds 0-4", flush=True)
    print(f"{'='*70}", flush=True)

    with open(RESULTS_DIR / "phase74_e2e_perception.json") as f:
        p74 = json.load(f)
    old_results = p74['e2e']['per_seed']
    print(f"  Phase 74 seeds: {[r['seed'] for r in old_results]}", flush=True)
    print(f"  New seeds: {[r['seed'] for r in new_results]}", flush=True)

    all_e2e = old_results + new_results
    all_e2e.sort(key=lambda r: r['seed'])
    print(f"  Total e2e: {len(all_e2e)} seeds", flush=True)

    # Also load frozen results from Phase 74
    frozen_results = p74['frozen']['per_seed']
    print(f"  Frozen seeds: {len(frozen_results)}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Full comparison
    # ═══════════════════════════════════════════════════════════
    e2e_hb = np.array([r['holdout_both'] for r in all_e2e])
    e2e_pd = np.array([r['pos_dis'] for r in all_e2e])
    e2e_r2e = np.array([r['probe_r2_e'] for r in all_e2e])
    e2e_r2f = np.array([r['probe_r2_f'] for r in all_e2e])
    e2e_cos = np.array([r['feature_cos_sim'] for r in all_e2e])
    e2e_comp = sum(1 for r in all_e2e if r['pos_dis'] > 0.4)

    frozen_hb = np.array([r['holdout_both'] for r in frozen_results])
    frozen_pd = np.array([r['pos_dis'] for r in frozen_results])
    frozen_comp = sum(1 for r in frozen_results if r['pos_dis'] > 0.4)

    print(f"\n{'='*70}", flush=True)
    print(f"COMPARISON: {len(all_e2e)} E2E seeds vs {len(frozen_results)} Frozen seeds", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"  {'Metric':<30} | {f'E2E ({len(all_e2e)})':>16} | {f'Frozen ({len(frozen_results)})':>16}", flush=True)
    print(f"  {'-'*30}-+-{'-'*16}-+-{'-'*16}", flush=True)
    print(f"  {'Holdout both':<30} | "
          f"{e2e_hb.mean():>5.1%}±{e2e_hb.std():>4.1%}    | "
          f"{frozen_hb.mean():>5.1%}±{frozen_hb.std():>4.1%}   ", flush=True)
    print(f"  {'PosDis':<30} | "
          f"{e2e_pd.mean():>5.3f}±{e2e_pd.std():>4.3f}    | "
          f"{frozen_pd.mean():>5.3f}±{frozen_pd.std():>4.3f}   ", flush=True)
    print(f"  {'Comp rate (PosDis>0.4)':<30} | "
          f"{e2e_comp:>5}/{len(all_e2e):<10} | "
          f"{frozen_comp:>5}/{len(frozen_results):<10}", flush=True)
    print(f"  {'Linear probe R²(elast)':<30} | "
          f"{e2e_r2e.mean():>5.3f}±{e2e_r2e.std():>4.3f}    | "
          f"{frozen_r2_e:>16.3f}", flush=True)
    print(f"  {'Linear probe R²(frict)':<30} | "
          f"{e2e_r2f.mean():>5.3f}±{e2e_r2f.std():>4.3f}    | "
          f"{frozen_r2_f:>16.3f}", flush=True)
    print(f"  {'Feature cos sim':<30} | "
          f"{e2e_cos.mean():>5.4f}±{e2e_cos.std():>.4f}  | "
          f"{'1.0000':>16}", flush=True)

    # Statistical tests
    t_acc, p_acc = stats.ttest_ind(e2e_hb, frozen_hb)
    cohens_d = (frozen_hb.mean() - e2e_hb.mean()) / np.sqrt(
        ((len(frozen_hb) - 1) * frozen_hb.std()**2 +
         (len(e2e_hb) - 1) * e2e_hb.std()**2) /
        (len(frozen_hb) + len(e2e_hb) - 2))
    print(f"\n  Holdout accuracy:", flush=True)
    print(f"    t-test: t={t_acc:.2f}, p={p_acc:.4f}", flush=True)
    print(f"    Cohen's d: {cohens_d:.2f} (frozen advantage)", flush=True)

    t_pd, p_pd = stats.ttest_ind(e2e_pd, frozen_pd)
    print(f"  PosDis:", flush=True)
    print(f"    t-test: t={t_pd:.2f}, p={p_pd:.4f}", flush=True)

    # 95% CIs
    def ci95(arr):
        m, s, n = arr.mean(), arr.std(), len(arr)
        se = s / np.sqrt(n)
        return m - 1.96 * se, m + 1.96 * se
    e2e_ci = ci95(e2e_hb)
    frz_ci = ci95(frozen_hb)
    print(f"\n  95% CIs:", flush=True)
    print(f"    E2E holdout:    [{e2e_ci[0]:.1%}, {e2e_ci[1]:.1%}]", flush=True)
    print(f"    Frozen holdout: [{frz_ci[0]:.1%}, {frz_ci[1]:.1%}]", flush=True)

    e2e_r2e_ci = ci95(e2e_r2e)
    e2e_r2f_ci = ci95(e2e_r2f)
    print(f"    E2E R²(e):      [{e2e_r2e_ci[0]:.3f}, {e2e_r2e_ci[1]:.3f}]", flush=True)
    print(f"    E2E R²(f):      [{e2e_r2f_ci[0]:.3f}, {e2e_r2f_ci[1]:.3f}]", flush=True)

    # Per-seed breakdown
    print(f"\n  E2E PER-SEED:", flush=True)
    print(f"  {'Seed':>4} | {'Holdout':>8} | {'PosDis':>7} | "
          f"{'R²(e)':>7} | {'R²(f)':>7} | {'CosSim':>7} | {'Source':>7}", flush=True)
    for r in all_e2e:
        tag = " *" if r['pos_dis'] > 0.4 else ""
        src = "P74" if r['seed'] < 5 else "NEW"
        print(f"  {r['seed']:>4} | {r['holdout_both']:>7.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['probe_r2_e']:>7.3f} | "
              f"{r['probe_r2_f']:>7.3f} | {r['feature_cos_sim']:>7.4f} | {src}{tag}",
              flush=True)

    # ═══════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'unfreeze_blocks': UNFREEZE_BLOCKS,
            'encoder_lr': ENCODER_LR,
            'sender_lr': SENDER_LR,
            'receiver_lr': RECEIVER_LR,
            'comm_epochs': COMM_EPOCHS,
            'n_seeds_e2e': len(all_e2e),
            'n_seeds_frozen': len(frozen_results),
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'seeds_from_phase74': list(range(5)),
            'seeds_new': list(NEW_SEEDS),
        },
        'e2e': {
            'per_seed': all_e2e,
            'holdout_mean': float(e2e_hb.mean()),
            'holdout_std': float(e2e_hb.std()),
            'pos_dis_mean': float(e2e_pd.mean()),
            'pos_dis_std': float(e2e_pd.std()),
            'comp_count': e2e_comp,
            'probe_r2_e_mean': float(e2e_r2e.mean()),
            'probe_r2_e_std': float(e2e_r2e.std()),
            'probe_r2_f_mean': float(e2e_r2f.mean()),
            'probe_r2_f_std': float(e2e_r2f.std()),
            'cos_sim_mean': float(e2e_cos.mean()),
            'cos_sim_std': float(e2e_cos.std()),
        },
        'frozen': {
            'holdout_mean': float(frozen_hb.mean()),
            'holdout_std': float(frozen_hb.std()),
            'pos_dis_mean': float(frozen_pd.mean()),
            'pos_dis_std': float(frozen_pd.std()),
            'comp_count': frozen_comp,
            'probe_r2_e': frozen_r2_e,
            'probe_r2_f': frozen_r2_f,
        },
        'stats': {
            'holdout_ttest_t': float(t_acc),
            'holdout_ttest_p': float(p_acc),
            'cohens_d': float(cohens_d),
            'posdis_ttest_t': float(t_pd),
            'posdis_ttest_p': float(p_pd),
        },
    }

    save_path = RESULTS_DIR / "phase77_e2e_15seeds.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/3600:.1f} hours)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

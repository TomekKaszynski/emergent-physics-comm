"""
Phase 79: Collision Dynamics — Full Pipeline
=============================================
STEP 1: Sanity checks on rendered dataset
STEP 2: DINOv2 feature extraction (24 frames from 48)
STEP 3: Oracle probes (5 seeds, 200 epochs)
STEP 4: DINOv2 2-agent communication (20 seeds, 400 epochs)
STEP 5: DINOv2 4-agent communication (20 seeds, 400 epochs)
STEP 6: Summary table

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase79_collision_pipeline.py
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
DATASET_DIR = Path("kubric/output/collision_dataset")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32

# 5x5 grid: mass_ratio_bin x restitution_bin
HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

ORACLE_EPOCHS = 200
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

# DINOv2 extraction
N_FRAMES_EXTRACT = 24  # evenly spaced from 48 rendered frames

# 4-agent config
N_AGENTS = 4
FRAMES_PER_AGENT = N_FRAMES_EXTRACT // N_AGENTS  # 6


# ══════════════════════════════════════════════════════════════════
# STEP 1: Sanity Checks
# ══════════════════════════════════════════════════════════════════

def step1_sanity_checks():
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: Sanity Checks", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    index_path = DATASET_DIR / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    n_scenes = len(index)
    print(f"  Total scenes: {n_scenes}", flush=True)
    assert n_scenes == 600, f"Expected 600, got {n_scenes}"

    # Check frame counts
    missing = 0
    for entry in index[:10]:  # spot check
        scene_dir = DATASET_DIR / f"scene_{entry['scene_id']:04d}"
        frames = list(scene_dir.glob("rgba_*.png"))
        if len(frames) != 48:
            missing += 1
            print(f"  WARNING: scene_{entry['scene_id']:04d} has {len(frames)} frames", flush=True)

    # Check image resolution
    sample_img = Image.open(DATASET_DIR / "scene_0000" / "rgba_00000.png")
    w, h = sample_img.size
    print(f"  Resolution: {w}x{h}", flush=True)
    assert w == 256 and h == 256

    # Extract mass/restitution bins
    mass_bins = np.array([entry['mass_ratio_bin'] for entry in index])
    rest_bins = np.array([entry['restitution_bin'] for entry in index])
    mass_ratios = np.array([entry['sphere_b_mass'] for entry in index])
    restitutions = np.array([entry['restitution'] for entry in index])

    # Grid distribution
    print(f"\n  Grid distribution (scenes per cell):", flush=True)
    MASS_RATIOS = [1.0, 2.0, 3.0, 4.0, 5.0]
    RESTITUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
    for mi, m_val in enumerate(MASS_RATIOS):
        row = []
        for ri, r_val in enumerate(RESTITUTION_LEVELS):
            count = int(((mass_bins == mi) & (rest_bins == ri)).sum())
            row.append(f"{count:3d}")
        print(f"    m_b={m_val:.1f}: [{', '.join(row)}]", flush=True)

    # Velocity statistics by mass_ratio
    print(f"\n  Post-collision Sphere B velocity by mass_ratio:", flush=True)
    for mi, m_val in enumerate(MASS_RATIOS):
        mask = mass_bins == mi
        vels = np.array([index[i]['post_collision_vel_b'] for i in range(n_scenes) if mask[i]])
        print(f"    m_b={m_val:.1f}: {vels.mean():+.3f} ± {vels.std():.3f}  "
              f"(n={len(vels)})", flush=True)

    # Velocity statistics by restitution
    print(f"\n  Post-collision Sphere B velocity by restitution:", flush=True)
    for ri, r_val in enumerate(RESTITUTION_LEVELS):
        mask = rest_bins == ri
        vels = np.array([index[i]['post_collision_vel_b'] for i in range(n_scenes) if mask[i]])
        print(f"    e={r_val:.1f}: {vels.mean():+.3f} ± {vels.std():.3f}  "
              f"(n={len(vels)})", flush=True)

    # Visual ambiguity check: compare RGB from different mass ratios at same frame
    print(f"\n  Visual ambiguity check (pre-collision frame 5):", flush=True)
    for mi in [0, 2, 4]:
        scene_idx = mi * 5 * 24  # first scene in that mass bin
        scene_dir = DATASET_DIR / f"scene_{scene_idx:04d}"
        img = np.array(Image.open(scene_dir / "rgba_00005.png"))[:, :, :3]
        print(f"    m_b={MASS_RATIOS[mi]:.1f} scene_{scene_idx:04d}: "
              f"mean_rgb=({img.mean(axis=(0,1))[0]:.1f}, "
              f"{img.mean(axis=(0,1))[1]:.1f}, "
              f"{img.mean(axis=(0,1))[2]:.1f})", flush=True)

    dt = time.time() - t0
    print(f"\n  Step 1 complete: {dt:.1f}s", flush=True)
    return index, mass_bins, rest_bins


# ══════════════════════════════════════════════════════════════════
# STEP 2: DINOv2 Feature Extraction
# ══════════════════════════════════════════════════════════════════

def step2_extract_dinov2(index):
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: DINOv2 Feature Extraction", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    cache_path = RESULTS_DIR / "collision_dinov2_features.pt"
    if cache_path.exists():
        print(f"  Loading cached features from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=False)
        features = data['features']
        print(f"  Shape: {features.shape}", flush=True)
        return features, data['mass_bins'], data['rest_bins']

    from transformers import AutoModel, AutoImageProcessor

    print(f"  Loading DINOv2 ViT-S/14...", flush=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model = model.to(DEVICE)
    model.eval()

    n_scenes = len(index)
    # 24 evenly spaced from 48: indices 0, 2, 4, ..., 46
    frame_indices = list(range(0, 48, 48 // N_FRAMES_EXTRACT))[:N_FRAMES_EXTRACT]
    print(f"  Extracting {N_FRAMES_EXTRACT} frames per scene: {frame_indices[:5]}...{frame_indices[-2:]}", flush=True)

    all_features = torch.zeros(n_scenes, N_FRAMES_EXTRACT, DINO_DIM)
    mass_bins = np.array([entry['mass_ratio_bin'] for entry in index])
    rest_bins = np.array([entry['restitution_bin'] for entry in index])

    for si in range(n_scenes):
        scene_dir = DATASET_DIR / f"scene_{index[si]['scene_id']:04d}"
        frames = []
        for fi in frame_indices:
            img = Image.open(scene_dir / f"rgba_{fi:05d}.png").convert("RGB")
            frames.append(img)

        # Process in batches of 24
        inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # (24, 384)
            all_features[si] = cls_tokens.cpu()

        if (si + 1) % 50 == 0 or si == 0:
            dt = time.time() - t0
            eta = dt / (si + 1) * (n_scenes - si - 1)
            print(f"  [{si+1:4d}/{n_scenes}] {dt:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

        if si % 100 == 0:
            torch.mps.empty_cache()

    # Save
    torch.save({
        'features': all_features,
        'mass_bins': mass_bins,
        'rest_bins': rest_bins,
        'index': index,
        'frame_indices': frame_indices,
        'model': 'facebook/dinov2-small',
        'shape_info': f'({n_scenes}, {N_FRAMES_EXTRACT}, {DINO_DIM})',
    }, cache_path)

    dt = time.time() - t0
    print(f"\n  Saved {cache_path}: {all_features.shape}", flush=True)
    print(f"  Step 2 complete: {dt:.0f}s", flush=True)

    del model, processor
    torch.mps.empty_cache()

    return all_features, mass_bins, rest_bins


# ══════════════════════════════════════════════════════════════════
# Architecture (shared by Steps 3-5)
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


class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)

    def forward(self, views, tau=1.0, hard=True):
        messages = []
        all_logits = []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
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


class MultiAgentOracle(nn.Module):
    def __init__(self, n_agents, hidden_dim, input_dim):
        super().__init__()
        self.n_agents = n_agents
        self.encs_a = nn.ModuleList([
            TemporalEncoder(hidden_dim, input_dim) for _ in range(n_agents)
        ])
        self.encs_b = nn.ModuleList([
            TemporalEncoder(hidden_dim, input_dim) for _ in range(n_agents)
        ])
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * n_agents * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim, 1)
        self.friction_head = nn.Linear(hidden_dim, 1)

    def forward(self, views_a, views_b):
        ha = torch.cat([enc(v) for enc, v in zip(self.encs_a, views_a)], dim=-1)
        hb = torch.cat([enc(v) for enc, v in zip(self.encs_b, views_b)], dim=-1)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════

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


def split_views(data_t, n_agents, frames_per_agent):
    return [data_t[:, i*frames_per_agent:(i+1)*frames_per_agent, :]
            for i in range(n_agents)]


# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_accuracy_2agent(sender, receiver, data_t, e_bins, f_bins,
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


def evaluate_population_2agent(sender, receivers, data_t, e_bins, f_bins,
                               scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_2agent(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_2agent(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


def evaluate_accuracy_4agent(multi_sender, receiver, agent_views, e_bins, f_bins,
                             scene_ids, device, oracle_model=None, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        views_a = [v[ia].to(device) for v in agent_views]
        views_b = [v[ib].to(device) for v in agent_views]

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])

        if oracle_model is not None:
            pred_e, pred_f = oracle_model(views_a, views_b)
        else:
            msg_a, _ = multi_sender(views_a)
            msg_b, _ = multi_sender(views_b)
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


def evaluate_population_4agent(multi_sender, receivers, agent_views, e_bins, f_bins,
                               scene_ids, device, msg_dim, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_4agent(
            multi_sender, r, agent_views, e_bins, f_bins,
            scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_4agent(
        multi_sender, best_r, agent_views, e_bins, f_bins,
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


def compute_compositionality_2agent(sender, data_t, e_bins, f_bins, device):
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


def compute_compositionality_4agent(multi_sender, agent_views, e_bins, f_bins, device):
    multi_sender.eval()
    all_tokens = []
    n_scenes = len(agent_views[0])

    with torch.no_grad():
        for i in range(0, n_scenes, BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(device) for v in agent_views]
            msg, logits = multi_sender(views)
            tokens_batch = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)  # [N, 8]
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

    # Per-agent PosDis
    per_agent_posdis = []
    for agent_idx in range(N_AGENTS):
        start = agent_idx * N_HEADS
        agent_mi = mi_matrix[start:start + N_HEADS]
        agent_pd = 0.0
        for p in range(N_HEADS):
            sorted_mi = np.sort(agent_mi[p])[::-1]
            if sorted_mi[0] > 1e-10:
                agent_pd += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        agent_pd /= N_HEADS
        per_agent_posdis.append(float(agent_pd))

    # Global PosDis
    pos_dis_global = 0.0
    for p in range(n_pos):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if sorted_mi[0] > 1e-10:
            pos_dis_global += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
    pos_dis_global /= n_pos

    best_agent_posdis = max(per_agent_posdis)

    rng = np.random.RandomState(42)
    n_pairs = min(5000, n_scenes * (n_scenes - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(n_scenes, size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'pos_dis': float(best_agent_posdis),
        'pos_dis_global': float(pos_dis_global),
        'pos_dis_per_agent': per_agent_posdis,
        'topsim': float(topsim),
        'entropies': entropies,
        'mi_matrix': mi_matrix,
    }


# ══════════════════════════════════════════════════════════════════
# Training functions
# ══════════════════════════════════════════════════════════════════

def train_oracle_2agent(data_t, e_bins, f_bins, train_ids, device, seed):
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
                _, _, acc_both = evaluate_accuracy_2agent(
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


def train_population_2agent(sender, receivers, data_t, e_bins, f_bins,
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
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
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
                (te, tf, tb), best_r = evaluate_population_2agent(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
                (he, hf, hb), _ = evaluate_population_2agent(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            if (epoch + 1) % 50 == 0 or epoch == 0:
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


def train_population_4agent(multi_sender, receivers, agent_views, e_bins, f_bins,
                            train_ids, holdout_ids, device, msg_dim, seed):
    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)
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
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        multi_sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            views_a = [v[ia].to(device) for v in agent_views]
            views_b = [v[ib].to(device) for v in agent_views]
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a = multi_sender(views_a, tau=tau, hard=hard)
            msg_b, logits_b = multi_sender(views_b, tau=tau, hard=hard)

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
            all_params = list(multi_sender.parameters())
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

            torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), 1.0)
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
            multi_sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (te, tf, tb), best_r = evaluate_population_4agent(
                    multi_sender, receivers, agent_views, e_bins, f_bins,
                    train_ids, device, msg_dim, n_rounds=20)
                (he, hf, hb), _ = evaluate_population_4agent(
                    multi_sender, receivers, agent_views, e_bins, f_bins,
                    holdout_ids, device, msg_dim, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                      f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                      f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                      f"ETA {eta/60:.0f}min", flush=True)

            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in multi_sender.state_dict().items()}
                best_receiver_states = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]

    if best_sender_state is not None:
        multi_sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# STEP 3: Oracle Probes
# ══════════════════════════════════════════════════════════════════

def step3_oracle_probes(features, mass_bins, rest_bins):
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Oracle Probes (5 seeds, 200 epochs)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    train_ids, holdout_ids = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    print(f"  Train: {len(train_ids)}, Holdout: {len(holdout_ids)}", flush=True)

    data_t = features.clone()
    oracle_results = []

    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        oracle, train_acc = train_oracle_2agent(
            data_t, mass_bins, rest_bins, train_ids, DEVICE, seed)

        oracle.eval()
        with torch.no_grad():
            he, hf, hb = evaluate_accuracy_2agent(
                None, None, data_t, mass_bins, rest_bins, holdout_ids, DEVICE,
                oracle_model=oracle, n_rounds=50)

        print(f"  Seed {seed}: train={train_acc:.1%}  holdout[m={he:.1%} e={hf:.1%} both={hb:.1%}]", flush=True)
        oracle_results.append({
            'seed': seed,
            'train_both': train_acc,
            'holdout_mass': he,
            'holdout_rest': hf,
            'holdout_both': hb,
        })

    mean_holdout = np.mean([r['holdout_both'] for r in oracle_results])
    std_holdout = np.std([r['holdout_both'] for r in oracle_results])

    result = {
        'config': {
            'encoder': 'TemporalEncoder Conv1D',
            'input_dim': DINO_DIM,
            'n_frames': N_FRAMES_EXTRACT,
            'oracle_epochs': ORACLE_EPOCHS,
            'n_seeds': 5,
        },
        'per_seed': oracle_results,
        'summary': {
            'holdout_both_mean': float(mean_holdout),
            'holdout_both_std': float(std_holdout),
        }
    }

    save_path = RESULTS_DIR / "phase79_dinov2_oracle_probe.json"
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)

    dt = time.time() - t0
    print(f"\n  Oracle holdout: {mean_holdout:.1%} ± {std_holdout:.1%}", flush=True)
    print(f"  Saved {save_path}", flush=True)
    print(f"  Step 3 complete: {dt:.0f}s", flush=True)

    return result


# ══════════════════════════════════════════════════════════════════
# STEP 4: DINOv2 2-agent Communication
# ══════════════════════════════════════════════════════════════════

def run_single_seed_2agent(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle_2agent(
        data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Sender
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Population
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    receivers, nan_count = train_population_2agent(
        sender, receivers, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval
    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population_2agent(
            sender, receivers, data_t, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy_2agent(
            sender, best_r, data_t, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    with torch.no_grad():
        comp = compute_compositionality_2agent(sender, data_t, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())

    dt = time.time() - t0
    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"MI->m={best_mi_e:.3f}  MI->e={best_mi_f:.3f}  "
          f"NaN={nan_count}  ({dt:.0f}s)", flush=True)

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


def step4_2agent_comm(features, mass_bins, rest_bins):
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: DINOv2 2-agent Communication (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    train_ids, holdout_ids = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    data_t = features.clone()

    all_results = []
    for seed in range(20):
        result = run_single_seed_2agent(
            seed, data_t, mass_bins, rest_bins, train_ids, holdout_ids, DEVICE)
        all_results.append(result)

    # Summary
    holdouts = [r['holdout_both'] for r in all_results]
    posdis = [r['pos_dis'] for r in all_results]
    compositional = [s for s, p in enumerate(posdis) if p >= 0.4]

    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis)),
        'pos_dis_std': float(np.std(posdis)),
        'compositional_count': len(compositional),
        'compositional_seeds': compositional,
    }

    output = {
        'config': {
            'encoder': 'TemporalEncoder Conv1D',
            'dataset': 'collision_dynamics',
            'n_scenes': 600,
            'n_frames': N_FRAMES_EXTRACT,
            'comm_epochs': COMM_EPOCHS,
            'n_seeds': 20,
            'vocab': f'{N_HEADS}x{VOCAB_SIZE}',
            'IL_interval': RECEIVER_RESET_INTERVAL,
        },
        'per_seed': all_results,
        'summary': summary,
    }

    save_path = RESULTS_DIR / "phase79_dinov2_collision.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    dt = time.time() - t0
    print(f"\n  2-agent summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']:.1%} ± {summary['holdout_both_std']:.1%}", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {summary['compositional_count']}/20", flush=True)
    print(f"  Saved {save_path}", flush=True)
    print(f"  Step 4 complete: {dt/60:.1f}min", flush=True)

    return output


# ══════════════════════════════════════════════════════════════════
# STEP 5: DINOv2 4-agent Communication
# ══════════════════════════════════════════════════════════════════

def train_oracle_4agent(agent_views, e_bins, f_bins, train_ids, device, seed):
    oracle = MultiAgentOracle(N_AGENTS, HIDDEN_DIM, DINO_DIM).to(device)
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
            va = [v[ia].to(device) for v in agent_views]
            vb = [v[ib].to(device) for v in agent_views]
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = oracle(va, vb)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                _, _, acc_both = evaluate_accuracy_4agent(
                    None, None, agent_views, e_bins, f_bins, train_ids, device,
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


def run_single_seed_4agent(seed, agent_views, e_bins, f_bins, train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    msg_dim = N_AGENTS * VOCAB_SIZE * N_HEADS  # 4*5*2 = 40

    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle_4agent(
        agent_views, e_bins, f_bins, train_ids, device, seed)
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Multi-agent sender: each agent gets 6 frames
    senders = []
    for agent_idx in range(N_AGENTS):
        enc = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        # Initialize from oracle
        enc.load_state_dict(oracle.encs_a[agent_idx].state_dict())
        s = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
        senders.append(s)
    multi_sender = MultiAgentSender(senders).to(device)

    # Population
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    receivers, nan_count = train_population_4agent(
        multi_sender, receivers, agent_views, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval
    multi_sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population_4agent(
            multi_sender, receivers, agent_views, e_bins, f_bins,
            train_ids, device, msg_dim, n_rounds=50)
        he, hf, hb = evaluate_accuracy_4agent(
            multi_sender, best_r, agent_views, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    with torch.no_grad():
        comp = compute_compositionality_4agent(
            multi_sender, agent_views, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())

    dt = time.time() - t0
    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"MI->m={best_mi_e:.3f}  MI->e={best_mi_f:.3f}  "
          f"NaN={nan_count}  ({dt:.0f}s)", flush=True)

    return {
        'seed': seed,
        'oracle_both': oracle_acc,
        'train_e': te, 'train_f': tf, 'train_both': tb,
        'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
        'pos_dis': comp['pos_dis'],
        'pos_dis_global': comp['pos_dis_global'],
        'pos_dis_per_agent': comp['pos_dis_per_agent'],
        'topsim': comp['topsim'],
        'entropies': comp['entropies'],
        'mi_matrix': mi.tolist(),
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'nan_count': nan_count,
        'time_sec': dt,
    }


def step5_4agent_comm(features, mass_bins, rest_bins):
    print("\n" + "=" * 70, flush=True)
    print("STEP 5: DINOv2 4-agent Communication (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Agent temporal splits:", flush=True)
    for i in range(N_AGENTS):
        start = i * FRAMES_PER_AGENT
        end = start + FRAMES_PER_AGENT
        label = ["pre-collision", "collision moment",
                 "early aftermath", "late aftermath"][i]
        print(f"    Agent {i}: frames {start}-{end-1} ({label})", flush=True)
    t0 = time.time()

    train_ids, holdout_ids = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)

    all_results = []
    for seed in range(20):
        result = run_single_seed_4agent(
            seed, agent_views, mass_bins, rest_bins,
            train_ids, holdout_ids, DEVICE)
        all_results.append(result)

    # Summary
    holdouts = [r['holdout_both'] for r in all_results]
    posdis = [r['pos_dis'] for r in all_results]
    compositional = [s for s, p in enumerate(posdis) if p >= 0.4]

    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis)),
        'pos_dis_std': float(np.std(posdis)),
        'compositional_count': len(compositional),
        'compositional_seeds': compositional,
    }

    output = {
        'config': {
            'encoder': 'TemporalEncoder Conv1D per agent',
            'dataset': 'collision_dynamics',
            'n_scenes': 600,
            'n_frames': N_FRAMES_EXTRACT,
            'frames_per_agent': FRAMES_PER_AGENT,
            'n_agents': N_AGENTS,
            'comm_epochs': COMM_EPOCHS,
            'n_seeds': 20,
            'vocab': f'{N_AGENTS}x{N_HEADS}x{VOCAB_SIZE}',
            'IL_interval': RECEIVER_RESET_INTERVAL,
        },
        'per_seed': all_results,
        'summary': summary,
    }

    save_path = RESULTS_DIR / "phase79_dinov2_4agent_collision.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    dt = time.time() - t0
    print(f"\n  4-agent summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']:.1%} ± {summary['holdout_both_std']:.1%}", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {summary['compositional_count']}/20", flush=True)
    print(f"  Saved {save_path}", flush=True)
    print(f"  Step 5 complete: {dt/60:.1f}min", flush=True)

    return output


# ══════════════════════════════════════════════════════════════════
# STEP 6: Summary Table
# ══════════════════════════════════════════════════════════════════

def step6_summary(oracle_result, twoa_result, foura_result):
    print("\n" + "=" * 70, flush=True)
    print("STEP 6: Summary Comparison", flush=True)
    print("=" * 70, flush=True)

    oracle_h = oracle_result['summary']['holdout_both_mean']

    # 2-agent collision
    c2 = twoa_result['summary']
    # 4-agent collision
    c4 = foura_result['summary']

    print(f"\n  {'Condition':<35} {'Oracle':>8} {'Holdout':>10} {'PosDis':>8} {'Comp%':>7}", flush=True)
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*7}", flush=True)

    print(f"  {'DINOv2 2-agent (collision)':<35} "
          f"{oracle_h:>7.1%} "
          f"{c2['holdout_both_mean']:>7.1%}±{c2['holdout_both_std']:.1%} "
          f"{c2['pos_dis_mean']:>7.3f} "
          f"{c2['compositional_count']:>4}/20", flush=True)

    print(f"  {'DINOv2 4-agent (collision)':<35} "
          f"{'—':>8} "
          f"{c4['holdout_both_mean']:>7.1%}±{c4['holdout_both_std']:.1%} "
          f"{c4['pos_dis_mean']:>7.3f} "
          f"{c4['compositional_count']:>4}/20", flush=True)

    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*7}", flush=True)
    print(f"  {'DINOv2 2-agent (ramp ref)':<35} "
          f"{'~90%':>8} "
          f"{'76.7%±6.5%':>10} "
          f"{'0.486':>8} "
          f"{'16/20':>7}", flush=True)
    print(f"  {'DINOv2 4-agent (ramp ref)':<35} "
          f"{'—':>8} "
          f"{'98.3%±1.6%':>10} "
          f"{'0.999':>8} "
          f"{'80/80':>7}", flush=True)

    # T-test: collision 2-agent vs ramp 2-agent
    # Load ramp reference if available
    ramp_path = RESULTS_DIR / "phase54f_extended.json"
    if ramp_path.exists():
        with open(ramp_path) as f:
            ramp_data = json.load(f)
        ramp_holdouts = [r['holdout_both'] for r in ramp_data['per_seed']]
        coll_holdouts = [r['holdout_both'] for r in twoa_result['per_seed']]
        t, p = stats.ttest_ind(coll_holdouts, ramp_holdouts)
        d = (np.mean(coll_holdouts) - np.mean(ramp_holdouts)) / np.sqrt(
            (np.std(coll_holdouts)**2 + np.std(ramp_holdouts)**2) / 2)
        print(f"\n  t-test (collision 2a vs ramp 2a): t={t:.2f}, p={p:.4f}, d={d:.2f}", flush=True)

    print(flush=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()

    print("Phase 79: Collision Dynamics — Full Pipeline", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Step 1
    index, mass_bins, rest_bins = step1_sanity_checks()

    # Step 2
    features, mass_bins, rest_bins = step2_extract_dinov2(index)

    # Step 3
    oracle_result = step3_oracle_probes(features, mass_bins, rest_bins)

    # Step 4
    twoa_result = step4_2agent_comm(features, mass_bins, rest_bins)

    # Step 5
    foura_result = step5_4agent_comm(features, mass_bins, rest_bins)

    # Step 6
    step6_summary(oracle_result, twoa_result, foura_result)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total pipeline time: {dt/60:.1f} min ({dt/3600:.1f} hours)", flush=True)
    print(f"{'='*70}", flush=True)

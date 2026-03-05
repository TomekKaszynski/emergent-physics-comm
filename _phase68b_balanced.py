"""
Phase 68b: Inverse Loss Weighting — Balanced Specialization
============================================================
Phase 68 showed distributed MI (no clean diagonal) because agents concentrate
bandwidth on easy properties. Test whether inverse loss weighting forces clean
position-to-property specialization.

Single change from Phase 68: weight each property's BCE loss inversely to
oracle accuracy from Phase 68 results.

Three conditions × 10 seeds:
  (a) COMP_6POS_BALANCED: 6 positions × vocab 5, inverse loss weighting
  (b) HOLISTIC_BALANCED:  1 position  × vocab 100, inverse loss weighting
  (c) ORACLE_BALANCED:    raw features, inverse loss weighting

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase68b_balanced.py
"""

import os
import time
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
CIFAR_ROOT = "./cifar100_data"
DINO_CACHE = RESULTS_DIR / "phase66_dino_cifar100.pt"

HIDDEN_DIM = 128
DINO_DIM = 384
BATCH_SIZE = 64
N_BINS = 5
N_PROPERTIES = 6

PROPERTY_NAMES = ['brightness', 'saturation', 'hue_conc',
                  'edge_density', 'spatial_freq', 'color_diversity']

COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 2.0
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

N_RECEIVERS = 3
RECEIVER_RESET_INTERVAL = 40
N_BATCHES_PER_EPOCH = 32

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

CONDITIONS = ['comp_6pos_bal', 'holistic_bal', 'oracle_bal']

CONDITION_CONFIGS = {
    'comp_6pos_bal': {'n_positions': 6, 'vocab_size': 5},
    'holistic_bal':  {'n_positions': 1, 'vocab_size': 100},
}

# Inverse loss weights from Phase 68 oracle per-property accuracy
# Oracle: sat=87.1%, bright=86.4%, spat=85.6%, hue=82.2%, col_div=79.2%, edge=71.1%
ORACLE_ACCS = {
    'brightness': 0.864, 'saturation': 0.871, 'hue_conc': 0.822,
    'edge_density': 0.711, 'spatial_freq': 0.856, 'color_diversity': 0.792,
}
_raw_weights = [1.0 / ORACLE_ACCS[p] for p in PROPERTY_NAMES]
_total = sum(_raw_weights)
LOSS_WEIGHTS = [w * N_PROPERTIES / _total for w in _raw_weights]
# Result: easy props get weight < 1, hard props get weight > 1

MIN_DIFFERING_PROPS = 3  # require ≥3 of 6 properties to differ in bin


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_cifar100_images(root=CIFAR_ROOT):
    """Load raw CIFAR-100 images (60K, uint8, CHW)."""
    import torchvision.datasets
    print("  Downloading CIFAR-100 (if needed)...", flush=True)
    torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=root, train=False, download=True)

    base = os.path.join(root, 'cifar-100-python')

    with open(os.path.join(base, 'train'), 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
    with open(os.path.join(base, 'test'), 'rb') as f:
        test_data = pickle.load(f, encoding='latin1')

    images = np.concatenate([
        train_data['data'].reshape(-1, 3, 32, 32),
        test_data['data'].reshape(-1, 3, 32, 32),
    ], axis=0)  # (60000, 3, 32, 32) uint8

    print(f"  CIFAR-100: {len(images)} images, shape {images.shape}", flush=True)
    return images


def load_dino_features(cache_path=DINO_CACHE):
    """Load cached DINOv2 features from Phase 66."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"DINOv2 cache not found at {cache_path}. "
            "Run Phase 66 first to generate cached features.")
    print(f"  Loading cached DINOv2 features from {cache_path}", flush=True)
    data = torch.load(cache_path, weights_only=False)
    features = data['features']
    print(f"  Features shape: {features.shape}", flush=True)
    return features


# ══════════════════════════════════════════════════════════════════
# Property extraction
# ══════════════════════════════════════════════════════════════════

def extract_all_properties(images):
    """Extract 6 continuous properties from raw CIFAR-100 images.

    Args:
        images: (N, 3, 32, 32) uint8

    Returns:
        dict mapping property name → (N,) float32 array
    """
    N = len(images)
    imgs = images.astype(np.float32) / 255.0  # (N, 3, 32, 32)

    # 1. Brightness: grayscale = 0.299*R + 0.587*G + 0.114*B, mean over pixels
    gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]  # (N, 32, 32)
    brightness = gray.mean(axis=(1, 2))  # (N,)

    # 2. Saturation: S = (max_rgb - min_rgb) / (max_rgb + 1e-8), mean over pixels
    max_rgb = imgs.max(axis=1)  # (N, 32, 32)
    min_rgb = imgs.min(axis=1)  # (N, 32, 32)
    sat_per_pixel = (max_rgb - min_rgb) / (max_rgb + 1e-8)  # (N, 32, 32)
    saturation = sat_per_pixel.mean(axis=(1, 2))  # (N,)

    # 3. Hue concentration: circular mean resultant length
    # Convert to HSV: H from RGB
    hue_conc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        r, g, b = imgs[i, 0], imgs[i, 1], imgs[i, 2]  # each (32, 32)
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        diff = mx - mn

        # Compute hue [0, 1]
        h = np.zeros_like(mx)
        mask_diff = diff > 1e-8
        mask_r = (mx == r) & mask_diff
        mask_g = (mx == g) & mask_diff
        mask_b = (mx == b) & mask_diff
        h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
        h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2.0
        h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4.0
        h = h / 6.0  # normalize to [0, 1]

        # Filter: only pixels with S > 0.1
        s_pixel = diff / (mx + 1e-8)
        sat_mask = s_pixel > 0.1

        if sat_mask.sum() > 5:
            h_valid = h[sat_mask]
            angles = 2 * np.pi * h_valid
            R_val = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
            hue_conc[i] = float(R_val)
        else:
            hue_conc[i] = 0.0

        if (i + 1) % 10000 == 0:
            print(f"    Hue concentration: {i+1}/{N}", flush=True)

    # 4. Edge density: Sobel magnitude, fraction above threshold
    # Sobel kernels (horizontal and vertical)
    sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    edge_density = np.zeros(N, dtype=np.float32)
    for i in range(N):
        g = gray[i]  # (32, 32)
        # Manual convolution (pad with zeros)
        padded = np.pad(g, 1, mode='constant')  # (34, 34)
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        for dy in range(3):
            for dx in range(3):
                gx += sobel_h[dy, dx] * padded[dy:dy+32, dx:dx+32]
                gy += sobel_v[dy, dx] * padded[dy:dy+32, dx:dx+32]
        mag = np.sqrt(gx**2 + gy**2)
        threshold = mag.mean() + mag.std()
        edge_density[i] = float((mag > threshold).mean())

        if (i + 1) % 10000 == 0:
            print(f"    Edge density: {i+1}/{N}", flush=True)

    # 5. Spatial frequency: 2D FFT, high-freq energy ratio
    spatial_freq = np.zeros(N, dtype=np.float32)
    for i in range(N):
        g = gray[i]  # (32, 32)
        fft = np.fft.fft2(g)
        fft_shift = np.fft.fftshift(fft)
        energy = np.abs(fft_shift)**2

        h_dim, w_dim = g.shape
        cy, cx = h_dim // 2, w_dim // 2
        Y, X = np.ogrid[:h_dim, :w_dim]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)
        inner_radius = 0.25 * max_dist

        inner_mask = dist <= inner_radius
        total_energy = energy.sum()
        inner_energy = energy[inner_mask].sum()
        outer_energy = total_energy - inner_energy
        spatial_freq[i] = float(outer_energy / (total_energy + 1e-8))

        if (i + 1) % 10000 == 0:
            print(f"    Spatial frequency: {i+1}/{N}", flush=True)

    # 6. Color diversity: unique quantized color triplets in 8×8
    color_diversity = np.zeros(N, dtype=np.float32)
    for i in range(N):
        img_uint8 = images[i]  # (3, 32, 32) uint8
        # Downsample to 8×8 by block averaging
        img_ds = img_uint8.reshape(3, 8, 4, 8, 4).mean(axis=(2, 4)).astype(np.uint8)
        # Quantize to 4 bins per channel
        quantized = img_ds // 64  # 0-3 per channel
        # Flatten to triplets
        r_q = quantized[0].flatten()
        g_q = quantized[1].flatten()
        b_q = quantized[2].flatten()
        triplets = set(zip(r_q.tolist(), g_q.tolist(), b_q.tolist()))
        color_diversity[i] = float(len(triplets)) / 64.0

        if (i + 1) % 10000 == 0:
            print(f"    Color diversity: {i+1}/{N}", flush=True)

    properties = {
        'brightness': brightness,
        'saturation': saturation,
        'hue_conc': hue_conc,
        'edge_density': edge_density,
        'spatial_freq': spatial_freq,
        'color_diversity': color_diversity,
    }

    for name in PROPERTY_NAMES:
        v = properties[name]
        print(f"  {name:18s}: mean={v.mean():.3f}, std={v.std():.3f}, "
              f"range=[{v.min():.3f}, {v.max():.3f}]", flush=True)

    return properties


def compute_correlation_matrix(properties):
    """Compute and print 6×6 Pearson correlation matrix."""
    n = len(PROPERTY_NAMES)
    corr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            vi = properties[PROPERTY_NAMES[i]]
            vj = properties[PROPERTY_NAMES[j]]
            corr[i, j] = np.corrcoef(vi, vj)[0, 1]

    # Print
    header = "                " + "  ".join(f"{name[:8]:>8s}" for name in PROPERTY_NAMES)
    print(f"\n  Property correlation matrix:", flush=True)
    print(f"  {header}", flush=True)
    for i in range(n):
        row = f"  {PROPERTY_NAMES[i]:16s}"
        for j in range(n):
            row += f"  {corr[i, j]:>8.3f}"
        print(row, flush=True)

    # Flag high correlations
    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > 0.5:
                flagged.append((PROPERTY_NAMES[i], PROPERTY_NAMES[j],
                                float(corr[i, j])))
    if flagged:
        print(f"\n  WARNING: High correlations detected:", flush=True)
        for p1, p2, r in flagged:
            print(f"    {p1} <-> {p2}: r={r:.3f}", flush=True)

    return corr, flagged


def bin_properties(values, n_bins=N_BINS, label=""):
    """Bin continuous values into quintiles."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    bins = np.digitize(values, edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    if label:
        print(f"    {label}: {np.bincount(bins, minlength=n_bins).tolist()}", flush=True)
    return bins


def make_splits(n_total, train_frac=0.8, seed=42):
    """80/20 image split."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    n_train = int(n_total * train_frac)
    return indices[:n_train], indices[n_train:]


# ══════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════

class ImageSender(nn.Module):
    """Single-image sender: DINOv2 features → discrete message."""
    def __init__(self, input_dim=384, hidden_dim=128,
                 n_positions=6, vocab_size=5):
        super().__init__()
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_positions * vocab_size)

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        logits = self.head(h).view(-1, self.n_positions, self.vocab_size)

        if self.training:
            flat = logits.reshape(-1, self.vocab_size)
            tokens = F.gumbel_softmax(flat, tau=tau, hard=hard)
            tokens = tokens.reshape(-1, self.n_positions, self.vocab_size)
        else:
            idx = logits.argmax(dim=-1)
            tokens = F.one_hot(idx, self.vocab_size).float()

        message = tokens.reshape(-1, self.n_positions * self.vocab_size)
        return message, logits


class MultiPropertyReceiver(nn.Module):
    """Six-head receiver for property comparison."""
    def __init__(self, input_dim, hidden_dim=128, n_props=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_props)
        ])

    def forward(self, x):
        h = self.shared(x)
        return [head(h).squeeze(-1) for head in self.heads]


class OracleReceiver(nn.Module):
    """Direct MLP on raw features — no communication bottleneck."""
    def __init__(self, input_dim, n_props=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(n_props)
        ])

    def forward(self, x):
        h = self.net(x)
        return [head(h).squeeze(-1) for head in self.heads]


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def sample_pairs(scene_ids, batch_size, rng, all_bins=None,
                 min_diff=MIN_DIFFERING_PROPS):
    """Sample pairs where ≥min_diff properties differ in bin."""
    max_attempts = batch_size * 5
    idx_a_list = []
    idx_b_list = []

    for _ in range(max_attempts):
        if len(idx_a_list) >= batch_size:
            break
        ia = rng.choice(scene_ids)
        ib = rng.choice(scene_ids)
        if ia == ib:
            continue
        if all_bins is not None:
            n_diff = sum(1 for bins in all_bins if bins[ia] != bins[ib])
            if n_diff < min_diff:
                continue
        idx_a_list.append(ia)
        idx_b_list.append(ib)

    # Pad if needed (relax constraint)
    while len(idx_a_list) < batch_size:
        ia = rng.choice(scene_ids)
        ib = rng.choice(scene_ids)
        if ia != ib:
            idx_a_list.append(ia)
            idx_b_list.append(ib)

    return np.array(idx_a_list[:batch_size]), np.array(idx_b_list[:batch_size])


def _mutual_information(x, y):
    """Compute MI between discrete variables."""
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


def analyze_sender_mi_matrix(sender, features, all_bins, device,
                             n_positions, n_subsample=10000):
    """Compute full n_positions × 6 MI matrix for one sender."""
    sender.eval()
    all_tokens = []

    # Subsample for speed
    if len(features) > n_subsample:
        rng_sub = np.random.RandomState(42)
        sub_idx = rng_sub.choice(len(features), size=n_subsample, replace=False)
        sub_features = features[sub_idx]
        sub_bins = [b[sub_idx] for b in all_bins]
    else:
        sub_features = features
        sub_bins = all_bins

    with torch.no_grad():
        for i in range(0, len(sub_features), BATCH_SIZE):
            batch = sub_features[i:i + BATCH_SIZE].to(device)
            _, logits = sender(batch)
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)  # (n_sub, n_positions)

    # MI matrix: (n_positions, 6)
    mi_matrix = np.zeros((n_positions, N_PROPERTIES), dtype=np.float32)
    for p in range(n_positions):
        pos_tokens = all_tokens[:, p]
        for prop_idx in range(N_PROPERTIES):
            mi_matrix[p, prop_idx] = _mutual_information(
                pos_tokens, sub_bins[prop_idx])

    # Per-position specialization
    pos_spec = []
    for p in range(n_positions):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if len(sorted_mi) >= 2 and (sorted_mi[0] + sorted_mi[1]) > 1e-10:
            spec = float(abs(sorted_mi[0] - sorted_mi[1]) /
                         (sorted_mi[0] + sorted_mi[1]))
        else:
            spec = 0.0
        pos_spec.append(spec)

    # Total MI per property (sum across positions)
    total_mi_per_prop = mi_matrix.sum(axis=0).tolist()

    return {
        'mi_matrix': mi_matrix.tolist(),
        'pos_spec': pos_spec,
        'mean_pos_spec': float(np.mean(pos_spec)),
        'total_mi_per_prop': total_mi_per_prop,
    }


def evaluate_with_receiver(senders, receiver, features, all_bins_dev,
                           scene_ids, device, all_bins_np=None,
                           n_rounds=30):
    """Evaluate 6-property comparison accuracy."""
    rng = np.random.RandomState(999)

    for s in senders:
        s.eval()
    receiver.eval()

    features_on_dev = (features if features.device.type == device.type
                       else features.to(device))

    # Per-property counters
    correct = [0] * N_PROPERTIES
    total = [0] * N_PROPERTIES
    all_correct = 0
    all_total = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng, all_bins_np)

        with torch.no_grad():
            feat_a = features_on_dev[ia]
            feat_b = features_on_dev[ib]
            msg_a0, _ = senders[0](feat_a)
            msg_b1, _ = senders[1](feat_b)
            msg_b0, _ = senders[0](feat_b)
            msg_a1, _ = senders[1](feat_a)
            combined = torch.cat([msg_a0, msg_b1, msg_b0, msg_a1], dim=-1)
            preds = receiver(combined)  # list of 6 tensors

        all_ok = torch.ones(bs, dtype=torch.bool, device=device)
        any_valid = torch.zeros(bs, dtype=torch.bool, device=device)

        for prop_idx in range(N_PROPERTIES):
            bins_dev = all_bins_dev[prop_idx]
            label = (bins_dev[ia] > bins_dev[ib])
            valid = (bins_dev[ia] != bins_dev[ib])

            if valid.sum() > 0:
                pred_correct = (preds[prop_idx] > 0)[valid] == label[valid]
                correct[prop_idx] += pred_correct.sum().item()
                total[prop_idx] += valid.sum().item()
                # For all-correct: mark wrong where valid and incorrect
                all_ok[valid] &= ((preds[prop_idx] > 0)[valid] == label[valid])
                any_valid |= valid

        # all_correct only counts samples where at least one prop is valid
        if any_valid.sum() > 0:
            all_correct += all_ok[any_valid].sum().item()
            all_total += any_valid.sum().item()

    per_prop = {PROPERTY_NAMES[i]: correct[i] / max(total[i], 1)
                for i in range(N_PROPERTIES)}
    mean_acc = float(np.mean(list(per_prop.values())))
    all_acc = all_correct / max(all_total, 1)

    return {
        'per_prop': per_prop,
        'mean_acc': mean_acc,
        'all_correct': all_acc,
    }


def evaluate_population(senders, receivers, features, all_bins_dev,
                        scene_ids, device, all_bins_np=None, n_rounds=30):
    """Pick best receiver from population, then evaluate."""
    best_all = -1
    best_r = None
    for r in receivers:
        acc = evaluate_with_receiver(
            senders, r, features, all_bins_dev, scene_ids, device,
            all_bins_np, n_rounds=10)
        if acc['all_correct'] > best_all:
            best_all = acc['all_correct']
            best_r = r
    final = evaluate_with_receiver(
        senders, best_r, features, all_bins_dev, scene_ids, device,
        all_bins_np, n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_seed(seed, features, all_bins_np, all_bins_dev,
               train_ids, holdout_ids, device,
               n_positions, vocab_size):
    """Train 2-agent communication for one seed (6 properties)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    msg_dim = n_positions * vocab_size

    senders = [ImageSender(DINO_DIM, HIDDEN_DIM, n_positions, vocab_size).to(device)
               for _ in range(2)]

    recv_input_dim = 4 * msg_dim  # 2 agents × 2 images
    receivers = [MultiPropertyReceiver(recv_input_dim, HIDDEN_DIM, N_PROPERTIES).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_params = []
    for s in senders:
        sender_params.extend(list(s.parameters()))
    sender_opt = torch.optim.Adam(sender_params, lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    features_dev = features.to(device)
    max_entropy = math.log(vocab_size)

    best_holdout_all = 0.0
    best_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # Population IL: reset receivers
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = MultiPropertyReceiver(
                    recv_input_dim, HIDDEN_DIM, N_PROPERTIES).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        for s in senders:
            s.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(N_BATCHES_PER_EPOCH):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng, all_bins_np)

            feat_a = features_dev[ia]
            feat_b = features_dev[ib]

            msg_a0, lg_a0 = senders[0](feat_a, tau, hard)
            msg_b1, lg_b1 = senders[1](feat_b, tau, hard)
            msg_b0, lg_b0 = senders[0](feat_b, tau, hard)
            msg_a1, lg_a1 = senders[1](feat_a, tau, hard)

            combined = torch.cat([msg_a0, msg_b1, msg_b0, msg_a1], dim=-1)
            all_logits = [lg_a0, lg_b1, lg_b0, lg_a1]

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                preds = r(combined)
                r_loss = torch.tensor(0.0, device=device)
                for prop_idx in range(N_PROPERTIES):
                    bins_dev = all_bins_dev[prop_idx]
                    label = (bins_dev[ia] > bins_dev[ib]).float()
                    r_loss = r_loss + LOSS_WEIGHTS[prop_idx] * \
                        F.binary_cross_entropy_with_logits(
                            preds[prop_idx], label)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization
            for lg in all_logits:
                for p in range(n_positions):
                    pos_logits = lg[:, p, :]
                    log_probs = F.log_softmax(pos_logits, dim=-1)
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
                continue

            sender_opt.zero_grad()
            for opt in receiver_opts:
                opt.zero_grad()
            loss.backward()

            has_nan_grad = False
            all_params = list(sender_params)
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for param in all_params:
                if param.grad is not None and (torch.isnan(param.grad).any() or
                                               torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender_params, 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0 and DEVICE.type == 'mps':
            torch.mps.empty_cache()

        # Periodic evaluation
        if (epoch + 1) % 20 == 0:
            train_result, _ = evaluate_population(
                senders, receivers, features, all_bins_dev,
                train_ids, device, all_bins_np, n_rounds=10)
            holdout_result, _ = evaluate_population(
                senders, receivers, features, all_bins_dev,
                holdout_ids, device, all_bins_np, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: "
                  f"mean={train_result['mean_acc']:.1%}  "
                  f"all6={train_result['all_correct']:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['all_correct'] > best_holdout_all:
                best_holdout_all = holdout_result['all_correct']
                best_states = {
                    'senders': [
                        {k: v.cpu().clone() for k, v in s.state_dict().items()}
                        for s in senders
                    ],
                    'receivers': [
                        {k: v.cpu().clone() for k, v in r.state_dict().items()}
                        for r in receivers
                    ],
                }

    # Restore best
    if best_states is not None:
        for i, s in enumerate(senders):
            s.load_state_dict(best_states['senders'][i])
            s.to(device)
        for i, r in enumerate(receivers):
            r.load_state_dict(best_states['receivers'][i])
            r.to(device)

    # Final evaluation
    train_final, _ = evaluate_population(
        senders, receivers, features, all_bins_dev,
        train_ids, device, all_bins_np, n_rounds=30)
    holdout_final, _ = evaluate_population(
        senders, receivers, features, all_bins_dev,
        holdout_ids, device, all_bins_np, n_rounds=30)

    # MI analysis per sender
    msg_analysis = {}
    for si, sender in enumerate(senders):
        msg_analysis[f'sender_{si}'] = analyze_sender_mi_matrix(
            sender, features, all_bins_np, device, n_positions)

    return {
        'train': train_final,
        'holdout': holdout_final,
        'nan_count': nan_count,
        'msg_analysis': msg_analysis,
    }


# ══════════════════════════════════════════════════════════════════
# Oracle training
# ══════════════════════════════════════════════════════════════════

def train_oracle(features, all_bins_np, all_bins_dev,
                 train_ids, holdout_ids, device, seed):
    """Train oracle baseline on raw DINOv2 features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = DINO_DIM * 2
    model = OracleReceiver(input_dim=input_dim, n_props=N_PROPERTIES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    rng = np.random.RandomState(seed)
    features_dev = features.to(device)

    best_all = 0.0
    best_state = None

    for epoch in range(COMM_EPOCHS):
        model.train()
        for _ in range(N_BATCHES_PER_EPOCH):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng, all_bins_np)

            feat_a = features_dev[ia]
            feat_b = features_dev[ib]
            x = torch.cat([feat_a, feat_b], dim=-1)

            preds = model(x)
            loss = torch.tensor(0.0, device=device)
            for prop_idx in range(N_PROPERTIES):
                bins_dev = all_bins_dev[prop_idx]
                label = (bins_dev[ia] > bins_dev[ib]).float()
                loss = loss + LOSS_WEIGHTS[prop_idx] * \
                    F.binary_cross_entropy_with_logits(
                        preds[prop_idx], label)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 40 == 0:
            model.eval()
            eval_rng = np.random.RandomState(999)
            # Quick holdout eval
            prop_correct = [0] * N_PROPERTIES
            prop_total = [0] * N_PROPERTIES
            ac = at = 0

            for _ in range(20):
                bs = min(BATCH_SIZE, len(holdout_ids))
                ia, ib = sample_pairs(holdout_ids, bs, eval_rng, all_bins_np)
                with torch.no_grad():
                    feat_a = features_dev[ia]
                    feat_b = features_dev[ib]
                    x = torch.cat([feat_a, feat_b], dim=-1)
                    preds = model(x)

                all_ok = torch.ones(bs, dtype=torch.bool, device=device)
                for prop_idx in range(N_PROPERTIES):
                    bins_dev = all_bins_dev[prop_idx]
                    label = (bins_dev[ia] > bins_dev[ib])
                    valid = (bins_dev[ia] != bins_dev[ib])
                    if valid.sum() > 0:
                        pc = ((preds[prop_idx] > 0)[valid] == label[valid])
                        prop_correct[prop_idx] += pc.sum().item()
                        prop_total[prop_idx] += valid.sum().item()
                        all_ok[valid] &= pc
                ac += all_ok.sum().item()
                at += bs

            all_acc = ac / max(at, 1)
            if all_acc > best_all:
                best_all = all_acc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final eval on both splits
    results = {}
    for split_name, split_ids in [('train', train_ids), ('holdout', holdout_ids)]:
        model.eval()
        eval_rng = np.random.RandomState(999)
        prop_correct = [0] * N_PROPERTIES
        prop_total = [0] * N_PROPERTIES
        ac = at = 0

        for _ in range(30):
            bs = min(BATCH_SIZE, len(split_ids))
            ia, ib = sample_pairs(split_ids, bs, eval_rng, all_bins_np)
            with torch.no_grad():
                feat_a = features_dev[ia]
                feat_b = features_dev[ib]
                x = torch.cat([feat_a, feat_b], dim=-1)
                preds = model(x)

            all_ok = torch.ones(bs, dtype=torch.bool, device=device)
            for prop_idx in range(N_PROPERTIES):
                bins_dev = all_bins_dev[prop_idx]
                label = (bins_dev[ia] > bins_dev[ib])
                valid = (bins_dev[ia] != bins_dev[ib])
                if valid.sum() > 0:
                    pc = ((preds[prop_idx] > 0)[valid] == label[valid])
                    prop_correct[prop_idx] += pc.sum().item()
                    prop_total[prop_idx] += valid.sum().item()
                    all_ok[valid] &= pc
            ac += all_ok.sum().item()
            at += bs

        per_prop = {PROPERTY_NAMES[i]: prop_correct[i] / max(prop_total[i], 1)
                    for i in range(N_PROPERTIES)}
        results[split_name] = {
            'per_prop': per_prop,
            'mean_acc': float(np.mean(list(per_prop.values()))),
            'all_correct': ac / max(at, 1),
        }

    return results


# ══════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════

def _aggregate_results(seed_results, has_mi=True):
    """Aggregate per-seed results into summary."""
    summary = {}

    for split in ['train', 'holdout']:
        all_corr = [r[split]['all_correct'] for r in seed_results]
        mean_accs = [r[split]['mean_acc'] for r in seed_results]
        summary[f'{split}_all_correct_mean'] = float(np.mean(all_corr))
        summary[f'{split}_all_correct_std'] = float(np.std(all_corr))
        summary[f'{split}_mean_acc_mean'] = float(np.mean(mean_accs))
        summary[f'{split}_mean_acc_std'] = float(np.std(mean_accs))

        # Per-property averages
        for pname in PROPERTY_NAMES:
            vals = [r[split]['per_prop'][pname] for r in seed_results]
            summary[f'{split}_{pname}_mean'] = float(np.mean(vals))
            summary[f'{split}_{pname}_std'] = float(np.std(vals))

    if has_mi:
        sender_specs = {}
        for si in range(2):
            key = f'sender_{si}'
            mean_specs = [r['msg_analysis'][key]['mean_pos_spec']
                          for r in seed_results]
            # Average MI matrix across seeds
            mi_matrices = [np.array(r['msg_analysis'][key]['mi_matrix'])
                           for r in seed_results]
            avg_mi = np.mean(mi_matrices, axis=0).tolist()
            # Average total MI per property
            total_mi = [np.array(r['msg_analysis'][key]['total_mi_per_prop'])
                        for r in seed_results]
            avg_total_mi = np.mean(total_mi, axis=0).tolist()

            sender_specs[key] = {
                'mean_pos_spec': float(np.mean(mean_specs)),
                'mean_pos_spec_std': float(np.std(mean_specs)),
                'avg_mi_matrix': avg_mi,
                'avg_total_mi_per_prop': avg_total_mi,
            }
        summary['sender_specs'] = sender_specs

    summary['seeds'] = seed_results
    return summary


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 68b: Inverse Loss Weighting — Balanced Specialization",
          flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}", flush=True)
    print(f"  Properties: {PROPERTY_NAMES}", flush=True)
    print(f"  Loss weights (inverse oracle acc, normalized):", flush=True)
    for i, pname in enumerate(PROPERTY_NAMES):
        print(f"    {pname:18s}: {LOSS_WEIGHTS[i]:.3f}", flush=True)
    print(flush=True)

    t_global = time.time()

    # ── Stage 0: Load data ──
    print("[Stage 0] Loading data...", flush=True)
    images = load_cifar100_images()
    features = load_dino_features()

    # Extract all 6 properties
    print("\n  Extracting 6 visual properties...", flush=True)
    properties = extract_all_properties(images)

    # Correlation matrix
    corr_matrix, flagged_pairs = compute_correlation_matrix(properties)

    # Bin each property
    print("\n  Binning into quintiles...", flush=True)
    all_bins_np = []
    for pname in PROPERTY_NAMES:
        bins = bin_properties(properties[pname], label=pname)
        all_bins_np.append(bins)

    # Image split
    train_ids, holdout_ids = make_splits(len(images))
    print(f"\n  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    del images  # free memory

    # Move bins to device
    all_bins_dev = [torch.tensor(b, dtype=torch.float32).to(DEVICE)
                    for b in all_bins_np]

    all_results = {}
    total_t0 = time.time()
    total_seeds_all = N_SEEDS * len(CONDITIONS)
    seeds_done = 0

    # ── Trained conditions ──
    trained_conditions = ['comp_6pos_bal', 'holistic_bal']

    for ci, cname in enumerate(trained_conditions):
        cfg = CONDITION_CONFIGS[cname]
        print(f"\n{'='*60}", flush=True)
        print(f"[{ci+1}/{len(CONDITIONS)}] Condition: {cname.upper()} "
              f"({cfg['n_positions']} pos × vocab {cfg['vocab_size']})", flush=True)
        print(f"{'='*60}", flush=True)

        cond_seeds = []
        for si in range(N_SEEDS):
            seed = si * 100 + 42
            elapsed_total = time.time() - total_t0
            if seeds_done > 0:
                eta_total = elapsed_total / seeds_done * (total_seeds_all - seeds_done)
                eta_str = f"  total ETA {eta_total/60:.0f}min"
            else:
                eta_str = ""
            print(f"    [seed {si+1}/{N_SEEDS}, seed={seed}]{eta_str}", flush=True)

            result = train_seed(seed, features, all_bins_np, all_bins_dev,
                                train_ids, holdout_ids, DEVICE,
                                cfg['n_positions'], cfg['vocab_size'])
            seeds_done += 1

            h = result['holdout']
            ms = result['msg_analysis']
            print(f"      → holdout all6={h['all_correct']:.1%}  "
                  f"mean={h['mean_acc']:.1%}  "
                  f"spec0={ms['sender_0']['mean_pos_spec']:.3f}  "
                  f"spec1={ms['sender_1']['mean_pos_spec']:.3f}",
                  flush=True)
            cond_seeds.append(result)

        cond_summary = _aggregate_results(cond_seeds)
        all_results[cname] = cond_summary
        print(f"\n  {cname.upper()}: holdout all6="
              f"{cond_summary['holdout_all_correct_mean']:.1%} ± "
              f"{cond_summary['holdout_all_correct_std']:.1%}  "
              f"mean={cond_summary['holdout_mean_acc_mean']:.1%}",
              flush=True)

    # ── Oracle ──
    ci = len(trained_conditions)
    print(f"\n{'='*60}", flush=True)
    print(f"[{ci+1}/{len(CONDITIONS)}] Condition: ORACLE_BAL (raw features, weighted)", flush=True)
    print(f"{'='*60}", flush=True)

    oracle_seeds = []
    for si in range(N_SEEDS):
        seed = si * 100 + 42
        elapsed_total = time.time() - total_t0
        eta_total = elapsed_total / seeds_done * (total_seeds_all - seeds_done)
        print(f"    [seed {si+1}/{N_SEEDS}, seed={seed}]  "
              f"total ETA {eta_total/60:.0f}min", flush=True)

        result = train_oracle(features, all_bins_np, all_bins_dev,
                              train_ids, holdout_ids, DEVICE, seed)
        seeds_done += 1

        h = result['holdout']
        print(f"      → holdout all6={h['all_correct']:.1%}  "
              f"mean={h['mean_acc']:.1%}", flush=True)
        oracle_seeds.append(result)

    oracle_summary = _aggregate_results(oracle_seeds, has_mi=False)
    all_results['oracle_bal'] = oracle_summary
    print(f"\n  ORACLE_BAL: holdout all6="
          f"{oracle_summary['holdout_all_correct_mean']:.1%} ± "
          f"{oracle_summary['holdout_all_correct_std']:.1%}  "
          f"mean={oracle_summary['holdout_mean_acc_mean']:.1%}",
          flush=True)

    # ── Summary ──
    elapsed = time.time() - t_global
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    # Condition comparison
    print(f"\n{'Condition':<12s} {'Train All6':>10s} {'Hold All6':>10s} "
          f"{'Train Mean':>10s} {'Hold Mean':>10s}", flush=True)
    print("-" * 55, flush=True)
    for cname in CONDITIONS:
        a = all_results[cname]
        print(f"{cname:<12s} "
              f"{a['train_all_correct_mean']:>9.1%}  "
              f"{a['holdout_all_correct_mean']:>9.1%}  "
              f"{a['train_mean_acc_mean']:>9.1%}  "
              f"{a['holdout_mean_acc_mean']:>9.1%}", flush=True)

    # Per-property accuracy (holdout)
    print(f"\nPer-property holdout accuracy:", flush=True)
    header = f"{'Condition':<12s}"
    for pname in PROPERTY_NAMES:
        header += f"  {pname[:8]:>8s}"
    print(header, flush=True)
    print("-" * (12 + 10 * N_PROPERTIES), flush=True)
    for cname in CONDITIONS:
        a = all_results[cname]
        row = f"{cname:<12s}"
        for pname in PROPERTY_NAMES:
            row += f"  {a[f'holdout_{pname}_mean']:>7.1%} "
        print(row, flush=True)

    # Property difficulty (from oracle)
    orc = all_results['oracle_bal']
    diff_list = [(pname, orc[f'holdout_{pname}_mean'])
                 for pname in PROPERTY_NAMES]
    diff_list.sort(key=lambda x: x[1], reverse=True)
    print(f"\nProperty difficulty ranking (oracle holdout):", flush=True)
    for pname, acc in diff_list:
        print(f"  {pname:18s}: {acc:.1%}", flush=True)

    # MI specialization (COMP_6POS_BAL)
    if 'comp_6pos_bal' in all_results and 'sender_specs' in all_results['comp_6pos_bal']:
        print(f"\nCOMP_6POS_BAL MI Matrix (avg across seeds, sender_0):", flush=True)
        sp = all_results['comp_6pos_bal']['sender_specs']['sender_0']
        mi = np.array(sp['avg_mi_matrix'])
        header = f"{'':>8s}"
        for pname in PROPERTY_NAMES:
            header += f"  {pname[:8]:>8s}"
        print(f"  {header}", flush=True)
        for p in range(mi.shape[0]):
            row = f"  {'pos_'+str(p):>8s}"
            for prop_idx in range(N_PROPERTIES):
                row += f"  {mi[p, prop_idx]:>8.3f}"
            print(row, flush=True)

        # Total MI per property
        print(f"\n  Total MI per property (sender_0):", flush=True)
        total_mi = sp['avg_total_mi_per_prop']
        for i, pname in enumerate(PROPERTY_NAMES):
            print(f"    {pname:18s}: {total_mi[i]:.3f}", flush=True)

    # Bandwidth allocation: correlate MI with oracle accuracy
    if 'comp_6pos_bal' in all_results and 'sender_specs' in all_results['comp_6pos_bal']:
        sp0 = all_results['comp_6pos_bal']['sender_specs']['sender_0']
        sp1 = all_results['comp_6pos_bal']['sender_specs']['sender_1']
        mi0 = np.array(sp0['avg_total_mi_per_prop'])
        mi1 = np.array(sp1['avg_total_mi_per_prop'])
        total_mi_combined = mi0 + mi1
        oracle_accs = np.array([orc[f'holdout_{pname}_mean']
                                for pname in PROPERTY_NAMES])
        if np.std(total_mi_combined) > 1e-8 and np.std(oracle_accs) > 1e-8:
            bw_corr = np.corrcoef(total_mi_combined, oracle_accs)[0, 1]
            print(f"\n  Bandwidth allocation correlation "
                  f"(MI vs oracle acc): r={bw_corr:.3f}", flush=True)

    # Specialization comparison
    print(f"\nSpecialization (mean position spec):", flush=True)
    for cname in ['comp_6pos_bal', 'holistic_bal']:
        if cname in all_results and 'sender_specs' in all_results[cname]:
            sp = all_results[cname]['sender_specs']
            s0 = sp['sender_0']['mean_pos_spec']
            s1 = sp['sender_1']['mean_pos_spec']
            print(f"  {cname:<12s}: sender_0={s0:.3f}  sender_1={s1:.3f}",
                  flush=True)

    # Key comparison with Phase 68 (unweighted)
    comp_bal = all_results['comp_6pos_bal']['holdout_all_correct_mean']
    hol_bal = all_results['holistic_bal']['holdout_all_correct_mean']
    orc_bal = all_results['oracle_bal']['holdout_all_correct_mean']
    print(f"\nKey comparison (68b balanced vs 68 unweighted):", flush=True)
    print(f"  {'Condition':<22s} {'All-6':>8s} {'Mean':>8s} {'Spec':>8s} {'MI diag?':>10s}", flush=True)
    print(f"  {'-'*58}", flush=True)
    print(f"  {'68  COMP_6POS':<22s} {'40.5%':>8s} {'82.0%':>8s} {'0.20':>8s} {'No':>10s}", flush=True)
    comp_spec = all_results['comp_6pos_bal']['sender_specs']['sender_0']['mean_pos_spec'] \
        if 'sender_specs' in all_results['comp_6pos_bal'] else 0
    print(f"  {'68b COMP_6POS_BAL':<22s} {comp_bal:>7.1%}  "
          f"{all_results['comp_6pos_bal']['holdout_mean_acc_mean']:>7.1%}  "
          f"{comp_spec:>7.2f}   {'?':>8s}", flush=True)
    print(f"  {'68  HOLISTIC':<22s} {'35.5%':>8s} {'79.8%':>8s} {'0.14':>8s} {'—':>10s}", flush=True)
    print(f"  {'68b HOLISTIC_BAL':<22s} {hol_bal:>7.1%}  "
          f"{all_results['holistic_bal']['holdout_mean_acc_mean']:>7.1%}  "
          f"{'—':>8s}   {'—':>8s}", flush=True)
    print(f"  {'68  ORACLE':<22s} {'39.6%':>8s} {'81.9%':>8s} {'—':>8s} {'—':>10s}", flush=True)
    print(f"  {'68b ORACLE_BAL':<22s} {orc_bal:>7.1%}  "
          f"{all_results['oracle_bal']['holdout_mean_acc_mean']:>7.1%}  "
          f"{'—':>8s}   {'—':>8s}", flush=True)

    print(f"\n  Comp advantage (bal):  {comp_bal - hol_bal:+.1%}", flush=True)
    print(f"  Comp advantage (68):   +5.1%", flush=True)

    # Edge density improvement?
    ed_bal = all_results['comp_6pos_bal'].get('holdout_edge_density_mean', 0)
    print(f"\n  Edge density (comp_6pos):     68={71.1:.1f}%  68b={ed_bal:.1%}", flush=True)

    if flagged_pairs:
        print(f"\n  Correlated property pairs (|r|>0.5):", flush=True)
        for p1, p2, r in flagged_pairs:
            print(f"    {p1} <-> {p2}: r={r:.3f}", flush=True)

    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)

    # ── Save ──
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'properties': {
            'correlation_matrix': corr_matrix.tolist(),
            'flagged_pairs': flagged_pairs,
            'names': PROPERTY_NAMES,
        },
        'conditions': {},
    }
    for cname in CONDITIONS:
        cond = all_results[cname]
        output['conditions'][cname] = {
            k: v for k, v in cond.items() if k != 'seeds'
        }
        output['conditions'][cname]['seeds'] = cond['seeds']

    out_path = RESULTS_DIR / "phase68b_balanced.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()

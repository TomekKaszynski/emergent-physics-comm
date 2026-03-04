"""
Phase 67: Continuous Visual Attributes — Property Comparison on Natural Images
===============================================================================
Bridges physics results (Phases 54-65) to vision. Phase 66 showed categorical
tasks incentivize memorization. Phase 67 uses CONTINUOUS visual properties
(brightness, saturation) on CIFAR-100 images.

Communication pressure forces compositional encoding because the same class
(e.g. "cat") has different brightness/saturation across images — agents MUST
encode continuous values, not class identity.

Dataset: CIFAR-100 (60K images)
  - DINOv2 ViT-S/14 features (384-dim per image, cached from Phase 66)
  - Properties extracted from raw pixels:
    - Brightness: mean grayscale (0.299R + 0.587G + 0.114B)
    - Saturation: mean HSV saturation ((max-min)/max per pixel)
  - Each binned into 5 quintiles

Task: Property comparison
  - 2 agents, each sees one image's DINOv2 features
  - Receiver gets both messages → predicts which image is brighter / more saturated
  - Two BCE heads (same as Phase 62/64/65)

Four conditions × 15 seeds:
  (a) COMP_2POS: 2 positions × vocab 5
  (b) COMP_4POS: 4 positions × vocab 5
  (c) HOLISTIC:  1 position  × vocab 25
  (d) ORACLE:    raw features, no communication bottleneck

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase67_visual_attributes.py
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

COMM_EPOCHS = 200
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 2.0
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

N_RECEIVERS = 3
RECEIVER_RESET_INTERVAL = 30
N_BATCHES_PER_EPOCH = 32  # 32 × 64 = 2048 episodes/epoch

N_SEEDS = 15
SEEDS = list(range(N_SEEDS))

CONDITIONS = ['comp_2pos', 'comp_4pos', 'holistic', 'oracle']

CONDITION_CONFIGS = {
    'comp_2pos': {'n_positions': 2, 'vocab_size': 5},
    'comp_4pos': {'n_positions': 4, 'vocab_size': 5},
    'holistic':  {'n_positions': 1, 'vocab_size': 25},
}


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


def extract_properties(images):
    """Extract brightness and saturation from raw CIFAR-100 images.

    Args:
        images: (N, 3, 32, 32) uint8

    Returns:
        brightness: (N,) float32 — mean grayscale value
        saturation: (N,) float32 — mean HSV saturation
    """
    # Convert to float [0, 1]
    imgs = images.astype(np.float32) / 255.0  # (N, 3, 32, 32)

    # Brightness: grayscale = 0.299*R + 0.587*G + 0.114*B, mean over pixels
    gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]  # (N, 32, 32)
    brightness = gray.mean(axis=(1, 2))  # (N,)

    # Saturation: S = (max_rgb - min_rgb) / (max_rgb + 1e-8), mean over pixels
    max_rgb = imgs.max(axis=1)  # (N, 32, 32)
    min_rgb = imgs.min(axis=1)  # (N, 32, 32)
    sat_per_pixel = (max_rgb - min_rgb) / (max_rgb + 1e-8)  # (N, 32, 32)
    saturation = sat_per_pixel.mean(axis=(1, 2))  # (N,)

    print(f"  Brightness: mean={brightness.mean():.3f}, std={brightness.std():.3f}, "
          f"range=[{brightness.min():.3f}, {brightness.max():.3f}]", flush=True)
    print(f"  Saturation: mean={saturation.mean():.3f}, std={saturation.std():.3f}, "
          f"range=[{saturation.min():.3f}, {saturation.max():.3f}]", flush=True)

    return brightness, saturation


def bin_properties(values, n_bins=N_BINS):
    """Bin continuous values into quintiles."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    bins = np.digitize(values, edges[1:-1])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    print(f"  Bin distribution: {np.bincount(bins, minlength=n_bins).tolist()}", flush=True)
    return bins


def make_splits(n_total, train_frac=0.8, seed=42):
    """80/20 image split (not class-based)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    n_train = int(n_total * train_frac)
    train_ids = indices[:n_train]
    test_ids = indices[n_train:]
    return train_ids, test_ids


# ══════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════

class ImageSender(nn.Module):
    """Single-image sender: DINOv2 features → discrete message via Gumbel-Softmax."""
    def __init__(self, input_dim=384, hidden_dim=128,
                 n_positions=2, vocab_size=5):
        super().__init__()
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_positions * vocab_size)

    def forward(self, x, tau=1.0, hard=True):
        """
        x: (B, 384) DINOv2 features
        Returns: message (B, n_pos*vocab), logits (B, n_pos, vocab)
        """
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


class PropertyReceiver(nn.Module):
    """Two-head receiver for property comparison (from Phase 65)."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head_brightness = nn.Linear(hidden_dim // 2, 1)
        self.head_saturation = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        return (self.head_brightness(h).squeeze(-1),
                self.head_saturation(h).squeeze(-1))


class OracleReceiver(nn.Module):
    """Direct MLP on raw features — no communication bottleneck."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head_brightness = nn.Linear(64, 1)
        self.head_saturation = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return (self.head_brightness(h).squeeze(-1),
                self.head_saturation(h).squeeze(-1))


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def sample_pairs(scene_ids, batch_size, rng):
    """Sample pairs of different images."""
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


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


def analyze_sender_messages(sender, features, bright_bins, sat_bins,
                            device, n_positions, n_subsample=10000):
    """Compute MI of each message position with each property."""
    sender.eval()
    all_tokens = []

    # Subsample for speed
    if len(features) > n_subsample:
        rng_sub = np.random.RandomState(42)
        sub_idx = rng_sub.choice(len(features), size=n_subsample, replace=False)
        sub_features = features[sub_idx]
        sub_bright = bright_bins[sub_idx]
        sub_sat = sat_bins[sub_idx]
    else:
        sub_features = features
        sub_bright = bright_bins
        sub_sat = sat_bins

    with torch.no_grad():
        for i in range(0, len(sub_features), BATCH_SIZE):
            batch = sub_features[i:i + BATCH_SIZE].to(device)
            _, logits = sender(batch)
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)
    bright_bins = sub_bright
    sat_bins = sub_sat

    total_mi_bright = 0.0
    total_mi_sat = 0.0

    for p in range(n_positions):
        pos_tokens = all_tokens[:, p]
        total_mi_bright += _mutual_information(pos_tokens, bright_bins)
        total_mi_sat += _mutual_information(pos_tokens, sat_bins)

    denom = total_mi_bright + total_mi_sat
    spec_ratio = float(abs(total_mi_bright - total_mi_sat) / denom) if denom > 1e-10 else 0.0

    return {
        'total_mi_brightness': float(total_mi_bright),
        'total_mi_saturation': float(total_mi_sat),
        'spec_ratio': spec_ratio,
    }


def evaluate_with_receiver(senders, receiver, features,
                           bright_bins, sat_bins, scene_ids,
                           device, n_rounds=30):
    """Evaluate property comparison accuracy."""
    rng = np.random.RandomState(999)
    bright_dev = torch.tensor(bright_bins, dtype=torch.float32).to(device)
    sat_dev = torch.tensor(sat_bins, dtype=torch.float32).to(device)

    for s in senders:
        s.eval()
    receiver.eval()

    ca = cb = c_both = 0
    ta = tb = t_both = 0

    features_on_dev = features if features.device.type == device.type else features.to(device)

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        with torch.no_grad():
            feat_a = features_on_dev[ia]  # (B, 384)
            feat_b = features_on_dev[ib]  # (B, 384)
            msg_a1, _ = senders[0](feat_a)
            msg_a2, _ = senders[1](feat_b)
            msg_b1, _ = senders[0](feat_b)
            msg_b2, _ = senders[1](feat_a)
            # Agent 0 sees image A, Agent 1 sees image B
            # Then Agent 0 sees image B, Agent 1 sees image A
            # Combined: [msg_agent0(A), msg_agent1(B), msg_agent0(B), msg_agent1(A)]
            combined = torch.cat([msg_a1, msg_a2, msg_b1, msg_b2], dim=-1)
            pred_bright, pred_sat = receiver(combined)

        label_bright = (bright_dev[ia] > bright_dev[ib])
        label_sat = (sat_dev[ia] > sat_dev[ib])
        valid_bright = (bright_dev[ia] != bright_dev[ib])
        valid_sat = (sat_dev[ia] != sat_dev[ib])
        valid_both = valid_bright & valid_sat

        if valid_bright.sum() > 0:
            ca += ((pred_bright > 0)[valid_bright] == label_bright[valid_bright]).sum().item()
            ta += valid_bright.sum().item()
        if valid_sat.sum() > 0:
            cb += ((pred_sat > 0)[valid_sat] == label_sat[valid_sat]).sum().item()
            tb += valid_sat.sum().item()
        if valid_both.sum() > 0:
            both_ok = ((pred_bright > 0)[valid_both] == label_bright[valid_both]) & \
                      ((pred_sat > 0)[valid_both] == label_sat[valid_both])
            c_both += both_ok.sum().item()
            t_both += valid_both.sum().item()

    return {
        'brightness_acc': ca / max(ta, 1),
        'saturation_acc': cb / max(tb, 1),
        'both_acc': c_both / max(t_both, 1),
    }


def evaluate_population(senders, receivers, features,
                        bright_bins, sat_bins, scene_ids,
                        device, n_rounds=30):
    """Pick best receiver from population, then evaluate."""
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate_with_receiver(
            senders, r, features, bright_bins, sat_bins,
            scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate_with_receiver(
        senders, best_r, features, bright_bins, sat_bins,
        scene_ids, device, n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_seed(seed, features, bright_bins, sat_bins,
               train_ids, holdout_ids, device,
               n_positions, vocab_size):
    """Train 2-agent communication for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    msg_dim = n_positions * vocab_size

    # 2 senders (one per image)
    senders = [ImageSender(DINO_DIM, HIDDEN_DIM, n_positions, vocab_size).to(device)
               for _ in range(2)]

    # Receiver input: 2 agents × 2 images × msg_dim = 4 * msg_dim
    recv_input_dim = 4 * msg_dim
    receivers = [PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_params = []
    for s in senders:
        sender_params.extend(list(s.parameters()))
    sender_opt = torch.optim.Adam(sender_params, lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    bright_dev = torch.tensor(bright_bins, dtype=torch.float32).to(device)
    sat_dev = torch.tensor(sat_bins, dtype=torch.float32).to(device)
    features_dev = features.to(device)
    max_entropy = math.log(vocab_size)

    best_holdout_both = 0.0
    best_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # Population IL: reset receivers
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = PropertyReceiver(recv_input_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        for s in senders:
            s.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(N_BATCHES_PER_EPOCH):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_bright = (bright_dev[ia] > bright_dev[ib]).float()
            label_sat = (sat_dev[ia] > sat_dev[ib]).float()

            feat_a = features_dev[ia]
            feat_b = features_dev[ib]

            # Agent 0 sees image A, Agent 1 sees image B
            msg_a0, lg_a0 = senders[0](feat_a, tau, hard)
            msg_b1, lg_b1 = senders[1](feat_b, tau, hard)
            # Agent 0 sees image B, Agent 1 sees image A
            msg_b0, lg_b0 = senders[0](feat_b, tau, hard)
            msg_a1, lg_a1 = senders[1](feat_a, tau, hard)

            combined = torch.cat([msg_a0, msg_b1, msg_b0, msg_a1], dim=-1)
            all_logits = [lg_a0, lg_b1, lg_b0, lg_a1]

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_bright, pred_sat = r(combined)
                r_loss = F.binary_cross_entropy_with_logits(pred_bright, label_bright) + \
                         F.binary_cross_entropy_with_logits(pred_sat, label_sat)
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

            # NaN gradient check
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
                senders, receivers, features, bright_bins, sat_bins,
                train_ids, device, n_rounds=10)
            holdout_result, _ = evaluate_population(
                senders, receivers, features, bright_bins, sat_bins,
                holdout_ids, device, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: "
                  f"bright={train_result['brightness_acc']:.1%}  "
                  f"sat={train_result['saturation_acc']:.1%}  "
                  f"both={train_result['both_acc']:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
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

    # Final evaluation on both splits
    train_final, _ = evaluate_population(
        senders, receivers, features, bright_bins, sat_bins,
        train_ids, device, n_rounds=30)
    holdout_final, _ = evaluate_population(
        senders, receivers, features, bright_bins, sat_bins,
        holdout_ids, device, n_rounds=30)

    # MI analysis per sender
    msg_analysis = {}
    for si, sender in enumerate(senders):
        msg_analysis[f'sender_{si}'] = analyze_sender_messages(
            sender, features, bright_bins, sat_bins, device, n_positions)

    return {
        'train_brightness': train_final['brightness_acc'],
        'train_saturation': train_final['saturation_acc'],
        'train_both': train_final['both_acc'],
        'holdout_brightness': holdout_final['brightness_acc'],
        'holdout_saturation': holdout_final['saturation_acc'],
        'holdout_both': holdout_final['both_acc'],
        'nan_count': nan_count,
        'msg_analysis': msg_analysis,
    }


# ══════════════════════════════════════════════════════════════════
# Oracle training
# ══════════════════════════════════════════════════════════════════

def train_oracle(features, bright_bins, sat_bins,
                 train_ids, holdout_ids, device, seed):
    """Train oracle baseline on raw DINOv2 features (no communication)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Input: concatenate both images' DINOv2 features
    input_dim = DINO_DIM * 2  # 384 * 2 = 768
    model = OracleReceiver(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    rng = np.random.RandomState(seed)
    bright_dev = torch.tensor(bright_bins, dtype=torch.float32).to(device)
    sat_dev = torch.tensor(sat_bins, dtype=torch.float32).to(device)
    features_dev = features.to(device)

    best_both = 0.0
    best_state = None

    for epoch in range(COMM_EPOCHS):
        model.train()
        for _ in range(N_BATCHES_PER_EPOCH):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_bright = (bright_dev[ia] > bright_dev[ib]).float()
            label_sat = (sat_dev[ia] > sat_dev[ib]).float()

            feat_a = features_dev[ia]
            feat_b = features_dev[ib]
            x = torch.cat([feat_a, feat_b], dim=-1)

            pred_bright, pred_sat = model(x)
            loss = F.binary_cross_entropy_with_logits(pred_bright, label_bright) + \
                   F.binary_cross_entropy_with_logits(pred_sat, label_sat)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 40 == 0:
            model.eval()
            eval_rng = np.random.RandomState(999)
            ca = cb = c_both = 0
            ta_cnt = tb_cnt = t_both_cnt = 0

            for _ in range(30):
                bs = min(BATCH_SIZE, len(holdout_ids))
                ia, ib = sample_pairs(holdout_ids, bs, eval_rng)
                with torch.no_grad():
                    feat_a = features_dev[ia]
                    feat_b = features_dev[ib]
                    x = torch.cat([feat_a, feat_b], dim=-1)
                    pred_bright, pred_sat = model(x)

                lb = (bright_dev[ia] > bright_dev[ib])
                ls = (sat_dev[ia] > sat_dev[ib])
                vb = (bright_dev[ia] != bright_dev[ib])
                vs = (sat_dev[ia] != sat_dev[ib])
                v_both = vb & vs

                if vb.sum() > 0:
                    ca += ((pred_bright > 0)[vb] == lb[vb]).sum().item()
                    ta_cnt += vb.sum().item()
                if vs.sum() > 0:
                    cb += ((pred_sat > 0)[vs] == ls[vs]).sum().item()
                    tb_cnt += vs.sum().item()
                if v_both.sum() > 0:
                    both_ok = ((pred_bright > 0)[v_both] == lb[v_both]) & \
                              ((pred_sat > 0)[v_both] == ls[v_both])
                    c_both += both_ok.sum().item()
                    t_both_cnt += v_both.sum().item()

            both_acc = c_both / max(t_both_cnt, 1)
            if both_acc > best_both:
                best_both = both_acc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # Restore best and final eval
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    results = {}
    for split_name, split_ids in [('train', train_ids), ('holdout', holdout_ids)]:
        model.eval()
        eval_rng = np.random.RandomState(999)
        ca = cb = c_both = 0
        ta_cnt = tb_cnt = t_both_cnt = 0

        for _ in range(30):
            bs = min(BATCH_SIZE, len(split_ids))
            ia, ib = sample_pairs(split_ids, bs, eval_rng)
            with torch.no_grad():
                feat_a = features_dev[ia]
                feat_b = features_dev[ib]
                x = torch.cat([feat_a, feat_b], dim=-1)
                pred_bright, pred_sat = model(x)

            lb = (bright_dev[ia] > bright_dev[ib])
            ls = (sat_dev[ia] > sat_dev[ib])
            vb = (bright_dev[ia] != bright_dev[ib])
            vs = (sat_dev[ia] != sat_dev[ib])
            v_both = vb & vs

            if vb.sum() > 0:
                ca += ((pred_bright > 0)[vb] == lb[vb]).sum().item()
                ta_cnt += vb.sum().item()
            if vs.sum() > 0:
                cb += ((pred_sat > 0)[vs] == ls[vs]).sum().item()
                tb_cnt += vs.sum().item()
            if v_both.sum() > 0:
                both_ok = ((pred_bright > 0)[v_both] == lb[v_both]) & \
                          ((pred_sat > 0)[v_both] == ls[v_both])
                c_both += both_ok.sum().item()
                t_both_cnt += v_both.sum().item()

        results[f'{split_name}_brightness'] = ca / max(ta_cnt, 1)
        results[f'{split_name}_saturation'] = cb / max(tb_cnt, 1)
        results[f'{split_name}_both'] = c_both / max(t_both_cnt, 1)

    return results


# ══════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════

def _aggregate_results(seed_results, has_mi=True):
    """Aggregate per-seed results into summary."""
    keys = ['train_brightness', 'train_saturation', 'train_both',
            'holdout_brightness', 'holdout_saturation', 'holdout_both']
    summary = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))

    if has_mi:
        sender_specs = {}
        for si in range(2):
            key = f'sender_{si}'
            mi_brights = [r['msg_analysis'][key]['total_mi_brightness']
                          for r in seed_results]
            mi_sats = [r['msg_analysis'][key]['total_mi_saturation']
                       for r in seed_results]
            specs = [r['msg_analysis'][key]['spec_ratio']
                     for r in seed_results]
            sender_specs[key] = {
                'mi_brightness_mean': float(np.mean(mi_brights)),
                'mi_saturation_mean': float(np.mean(mi_sats)),
                'spec_ratio_mean': float(np.mean(specs)),
                'spec_ratio_std': float(np.std(specs)),
            }
        summary['sender_specs'] = sender_specs

    summary['seeds'] = seed_results
    return summary


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 67: Continuous Visual Attributes — Property Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}", flush=True)
    print(flush=True)

    t_global = time.time()

    # ── Stage 0: Load data ──
    print("[Stage 0] Loading data...", flush=True)
    images = load_cifar100_images()
    features = load_dino_features()

    # Extract properties
    print("\n  Extracting visual properties...", flush=True)
    brightness, saturation = extract_properties(images)

    print("\n  Binning into quintiles...", flush=True)
    print("  Brightness bins:", flush=True)
    bright_bins = bin_properties(brightness)
    print("  Saturation bins:", flush=True)
    sat_bins = bin_properties(saturation)

    # Image split
    train_ids, holdout_ids = make_splits(len(images))
    print(f"\n  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Verify property distributions in splits
    for name, ids in [('Train', train_ids), ('Holdout', holdout_ids)]:
        b_dist = np.bincount(bright_bins[ids], minlength=N_BINS)
        s_dist = np.bincount(sat_bins[ids], minlength=N_BINS)
        print(f"  {name}: bright_bins={b_dist.tolist()}, sat_bins={s_dist.tolist()}", flush=True)

    del images  # free memory

    all_results = {}
    total_t0 = time.time()

    # Count total seeds for ETA
    total_seeds_all = N_SEEDS * len(CONDITIONS)
    seeds_done = 0

    # ── Condition 1: COMP_2POS ────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[1/{len(CONDITIONS)}] Condition: COMP_2POS (2 pos × vocab 5)", flush=True)
    print(f"{'='*60}", flush=True)

    cfg = CONDITION_CONFIGS['comp_2pos']
    comp2_seeds = []
    for si in range(N_SEEDS):
        seed = si * 100 + 42
        elapsed_total = time.time() - total_t0
        if seeds_done > 0:
            eta_total = elapsed_total / seeds_done * (total_seeds_all - seeds_done)
            eta_str = f"  total ETA {eta_total/60:.0f}min"
        else:
            eta_str = ""
        print(f"    [seed {si+1}/{N_SEEDS}, seed={seed}]{eta_str}", flush=True)

        result = train_seed(seed, features, bright_bins, sat_bins,
                            train_ids, holdout_ids, DEVICE,
                            cfg['n_positions'], cfg['vocab_size'])
        seeds_done += 1

        print(f"      → train both={result['train_both']:.1%}  "
              f"holdout both={result['holdout_both']:.1%}  "
              f"spec0={result['msg_analysis']['sender_0']['spec_ratio']:.3f}  "
              f"spec1={result['msg_analysis']['sender_1']['spec_ratio']:.3f}",
              flush=True)
        comp2_seeds.append(result)

    comp2_summary = _aggregate_results(comp2_seeds)
    all_results['comp_2pos'] = comp2_summary
    print(f"\n  COMP_2POS: holdout both={comp2_summary['holdout_both_mean']:.1%} ± "
          f"{comp2_summary['holdout_both_std']:.1%}", flush=True)

    # ── Condition 2: COMP_4POS ────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[2/{len(CONDITIONS)}] Condition: COMP_4POS (4 pos × vocab 5)", flush=True)
    print(f"{'='*60}", flush=True)

    cfg = CONDITION_CONFIGS['comp_4pos']
    comp4_seeds = []
    for si in range(N_SEEDS):
        seed = si * 100 + 42
        elapsed_total = time.time() - total_t0
        eta_total = elapsed_total / seeds_done * (total_seeds_all - seeds_done)
        print(f"    [seed {si+1}/{N_SEEDS}, seed={seed}]  "
              f"total ETA {eta_total/60:.0f}min", flush=True)

        result = train_seed(seed, features, bright_bins, sat_bins,
                            train_ids, holdout_ids, DEVICE,
                            cfg['n_positions'], cfg['vocab_size'])
        seeds_done += 1

        print(f"      → train both={result['train_both']:.1%}  "
              f"holdout both={result['holdout_both']:.1%}  "
              f"spec0={result['msg_analysis']['sender_0']['spec_ratio']:.3f}  "
              f"spec1={result['msg_analysis']['sender_1']['spec_ratio']:.3f}",
              flush=True)
        comp4_seeds.append(result)

    comp4_summary = _aggregate_results(comp4_seeds)
    all_results['comp_4pos'] = comp4_summary
    print(f"\n  COMP_4POS: holdout both={comp4_summary['holdout_both_mean']:.1%} ± "
          f"{comp4_summary['holdout_both_std']:.1%}", flush=True)

    # ── Condition 3: HOLISTIC ─────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[3/{len(CONDITIONS)}] Condition: HOLISTIC (1 pos × vocab 25)", flush=True)
    print(f"{'='*60}", flush=True)

    cfg = CONDITION_CONFIGS['holistic']
    hol_seeds = []
    for si in range(N_SEEDS):
        seed = si * 100 + 42
        elapsed_total = time.time() - total_t0
        eta_total = elapsed_total / seeds_done * (total_seeds_all - seeds_done)
        print(f"    [seed {si+1}/{N_SEEDS}, seed={seed}]  "
              f"total ETA {eta_total/60:.0f}min", flush=True)

        result = train_seed(seed, features, bright_bins, sat_bins,
                            train_ids, holdout_ids, DEVICE,
                            cfg['n_positions'], cfg['vocab_size'])
        seeds_done += 1

        print(f"      → train both={result['train_both']:.1%}  "
              f"holdout both={result['holdout_both']:.1%}  "
              f"spec0={result['msg_analysis']['sender_0']['spec_ratio']:.3f}  "
              f"spec1={result['msg_analysis']['sender_1']['spec_ratio']:.3f}",
              flush=True)
        hol_seeds.append(result)

    hol_summary = _aggregate_results(hol_seeds)
    all_results['holistic'] = hol_summary
    print(f"\n  HOLISTIC: holdout both={hol_summary['holdout_both_mean']:.1%} ± "
          f"{hol_summary['holdout_both_std']:.1%}", flush=True)

    # ── Condition 4: ORACLE ───────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[4/{len(CONDITIONS)}] Condition: ORACLE (raw features)", flush=True)
    print(f"{'='*60}", flush=True)

    oracle_seeds = []
    for si in range(N_SEEDS):
        seed = si * 100 + 42
        elapsed_total = time.time() - total_t0
        eta_total = elapsed_total / seeds_done * (total_seeds_all - seeds_done)
        print(f"    [seed {si+1}/{N_SEEDS}, seed={seed}]  "
              f"total ETA {eta_total/60:.0f}min", flush=True)

        result = train_oracle(features, bright_bins, sat_bins,
                              train_ids, holdout_ids, DEVICE, seed)
        seeds_done += 1

        print(f"      → train both={result['train_both']:.1%}  "
              f"holdout both={result['holdout_both']:.1%}", flush=True)
        oracle_seeds.append(result)

    oracle_summary = _aggregate_results(oracle_seeds, has_mi=False)
    all_results['oracle'] = oracle_summary
    print(f"\n  ORACLE: holdout both={oracle_summary['holdout_both_mean']:.1%} ± "
          f"{oracle_summary['holdout_both_std']:.1%}", flush=True)

    # ── Summary ──
    elapsed = time.time() - t_global
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Condition':<15s} {'Train Bright':>12s} {'Train Sat':>10s} "
          f"{'Train Both':>11s} {'Hold Bright':>12s} {'Hold Sat':>9s} "
          f"{'Hold Both':>10s}", flush=True)
    print("-" * 80, flush=True)

    for cname in CONDITIONS:
        a = all_results[cname]
        print(f"{cname:<15s} "
              f"{a['train_brightness_mean']:>11.1%}  "
              f"{a['train_saturation_mean']:>9.1%}  "
              f"{a['train_both_mean']:>10.1%}  "
              f"{a['holdout_brightness_mean']:>11.1%}  "
              f"{a['holdout_saturation_mean']:>8.1%}  "
              f"{a['holdout_both_mean']:>9.1%}", flush=True)

    # Specialization analysis
    print(f"\nSpecialization (MI-based):", flush=True)
    print(f"{'Condition':<15s} {'Sender':>8s} {'MI(bright)':>10s} "
          f"{'MI(sat)':>8s} {'Spec':>6s}", flush=True)
    print("-" * 52, flush=True)

    for cname in ['comp_2pos', 'comp_4pos', 'holistic']:
        a = all_results[cname]
        if 'sender_specs' not in a:
            continue
        for si in range(2):
            key = f'sender_{si}'
            sp = a['sender_specs'][key]
            print(f"{cname:<15s} {key:>8s} "
                  f"{sp['mi_brightness_mean']:>9.3f}  "
                  f"{sp['mi_saturation_mean']:>7.3f}  "
                  f"{sp['spec_ratio_mean']:>5.3f}", flush=True)

    # Cross-domain comparison
    print(f"\nCross-domain comparison (holdout both-correct):", flush=True)
    print(f"  Phase 62 (physics/ramp):     ~92%", flush=True)
    print(f"  Phase 64 (abstract scenes):  ~87%", flush=True)
    print(f"  Phase 65 (temporal physics):  ~88%", flush=True)
    c2 = all_results['comp_2pos']['holdout_both_mean']
    c4 = all_results['comp_4pos']['holdout_both_mean']
    print(f"  Phase 67 COMP_2POS:          {c2:.1%}", flush=True)
    print(f"  Phase 67 COMP_4POS:          {c4:.1%}", flush=True)

    # Key comparisons
    comp_best = max(c2, c4)
    hol_both = all_results['holistic']['holdout_both_mean']
    orc_both = all_results['oracle']['holdout_both_mean']
    print(f"\nKey comparisons:", flush=True)
    print(f"  Best compositional: {comp_best:.1%}", flush=True)
    print(f"  Holistic:           {hol_both:.1%}", flush=True)
    print(f"  Oracle (ceiling):   {orc_both:.1%}", flush=True)
    print(f"  Comp advantage:     {comp_best - hol_both:+.1%}", flush=True)
    print(f"  Ceiling captured:   {comp_best / max(orc_both, 0.01):.1%}", flush=True)

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

    # Remove seed details from top-level for cleaner JSON
    output = {}
    for cname in CONDITIONS:
        cond = all_results[cname]
        output[cname] = {k: v for k, v in cond.items() if k != 'seeds'}
        output[cname]['seeds'] = cond['seeds']

    out_path = RESULTS_DIR / "phase67_visual_attributes.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()

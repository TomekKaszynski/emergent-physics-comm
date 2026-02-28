"""
Phase 56: Cross-Physics Transfer (Ramp → Flat Drop)
=====================================================
Tests whether compositional protocols learned on ramp scenes transfer to
a novel physics environment (flat drop).

Three conditions × 20 seeds:
  (a) TRANSFER: Frozen ramp sender + new receiver trained on flat drop
  (b) NATIVE:   Fresh sender + receiver trained from scratch on flat drop
  (c) RANDOM:   Frozen random sender + new receiver (floor baseline)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase56_transfer.py
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
import os
import cv2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

# Ramp sender training (same as Phase 54f)
ORACLE_EPOCHS = 100
ORACLE_LR = 1e-3
RAMP_COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

# Transfer receiver training
TRANSFER_EPOCHS = 200
TRANSFER_LR = 3e-3

SEEDS = list(range(20))


# ══════════════════════════════════════════════════════════════════
# Architecture (same as Phase 54f)
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
# Data helpers
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
# Feature extraction for flat drop
# ══════════════════════════════════════════════════════════════════

def extract_flat_drop_features():
    """Extract DINOv2 features from flat drop dataset."""
    cache_path = RESULTS_DIR / "phase56_flat_dino_features.pt"
    if cache_path.exists():
        print("  Loading cached flat drop features...", flush=True)
        data = torch.load(cache_path, weights_only=False)
        return data['features'], data['e_bins'], data['f_bins']

    print("  Extracting DINOv2 features from flat drop dataset...", flush=True)
    dataset_dir = Path("kubric/output/flat_drop_dataset")
    index_path = dataset_dir / "index.json"

    with open(index_path) as f:
        metadata = json.load(f)

    n_scenes = len(metadata)
    n_frames = 8
    frame_indices = np.linspace(0, 35, n_frames).astype(int)  # 36 frames

    print(f"  {n_scenes} scenes, {n_frames} frames each", flush=True)

    # Load DINOv2
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino = dino.to(DEVICE).eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    e_bins = []
    f_bins = []

    for si, meta in enumerate(metadata):
        scene_dir = dataset_dir / f"scene_{meta['scene_id']:04d}"
        frames = []
        for fi in frame_indices:
            img_path = scene_dir / f"rgba_{fi:05d}.png"
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: missing {img_path}", flush=True)
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(transform(img))

        batch = torch.stack(frames).to(DEVICE)
        with torch.no_grad():
            feat = dino(batch)  # (n_frames, 384)
        all_features.append(feat.cpu())
        e_bins.append(meta['elasticity_bin'])
        f_bins.append(meta['friction_bin'])

        if (si + 1) % 50 == 0:
            print(f"    [{si+1}/{n_scenes}]", flush=True)
            torch.mps.empty_cache()

    features = torch.stack(all_features)  # (300, 8, 384)
    e_bins = np.array(e_bins)
    f_bins = np.array(f_bins)

    torch.save({'features': features, 'e_bins': e_bins, 'f_bins': f_bins},
               cache_path)
    print(f"  Saved to {cache_path}: {features.shape}", flush=True)

    del dino
    torch.mps.empty_cache()
    return features, e_bins, f_bins


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

    return {
        'e': correct_e / max(total_e, 1),
        'f': correct_f / max(total_f, 1),
        'both': correct_both / max(total_both, 1),
    }


def evaluate_population(sender, receivers, data_t, e_bins, f_bins,
                        scene_ids, device, n_rounds=30):
    """Evaluate with best receiver (highest both accuracy)."""
    best_both = 0
    best_r = None
    for r in receivers:
        acc = evaluate_accuracy(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if acc['both'] > best_both:
            best_both = acc['both']
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

    # Entropy
    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(VOCAB_SIZE)))

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
# Stage 1: Train ramp senders and save checkpoints
# ══════════════════════════════════════════════════════════════════

def train_ramp_sender(seed, ramp_data, ramp_e_bins, ramp_f_bins,
                      train_ids, holdout_ids, device):
    """Train a compositional sender on ramp data (Phase 54f style). Returns sender."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_entropy = math.log(VOCAB_SIZE)
    msg_dim = VOCAB_SIZE * N_HEADS
    n_batches = max(1, len(train_ids) // BATCH_SIZE)
    e_dev = torch.tensor(ramp_e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(ramp_f_bins, dtype=torch.float32).to(device)

    # Oracle
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
    oracle_opt = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = ramp_data[ia].to(device), ramp_data[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            oracle_opt.zero_grad()
            loss.backward()
            oracle_opt.step()
        if epoch % 20 == 0:
            torch.mps.empty_cache()

    oracle_enc_state = oracle.enc_a.state_dict()

    # Sender with oracle encoder init
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)

    # Population of receivers
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    rng = np.random.RandomState(seed + 1000)

    best_both_acc = 0.0
    best_sender_state = None
    nan_count = 0

    for epoch in range(RAMP_COMM_EPOCHS):
        # Simultaneous IL
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, RAMP_COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = ramp_data[ia].to(device), ramp_data[ib].to(device)
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

            # Entropy regularization
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
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if (epoch + 1) % 50 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                acc, best_r = evaluate_population(
                    sender, receivers, ramp_data, ramp_e_bins, ramp_f_bins,
                    train_ids, device, n_rounds=20)
            if acc['both'] > best_both_acc:
                best_both_acc = acc['both']
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)

    del oracle, receivers, receiver_opts
    torch.mps.empty_cache()
    return sender


# ══════════════════════════════════════════════════════════════════
# Stage 2: Train receiver only (frozen sender) on target domain
# ══════════════════════════════════════════════════════════════════

def train_receiver_frozen_sender(sender, data_t, e_bins, f_bins,
                                 train_ids, device, seed, epochs=TRANSFER_EPOCHS):
    """Train a new receiver against a frozen sender on target data."""
    msg_dim = VOCAB_SIZE * N_HEADS
    receiver = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=TRANSFER_LR)

    rng = np.random.RandomState(seed + 2000)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    sender.eval()
    for p in sender.parameters():
        p.requires_grad_(False)

    for epoch in range(epochs):
        receiver.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            with torch.no_grad():
                msg_a, _ = sender(da)
                msg_b, _ = sender(db)

            pred_e, pred_f = receiver(msg_a, msg_b)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(receiver.parameters(), 1.0)
            optimizer.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

    return receiver


# ══════════════════════════════════════════════════════════════════
# Stage 3: Train sender+receiver from scratch (native baseline)
# ══════════════════════════════════════════════════════════════════

def train_native(data_t, e_bins, f_bins, train_ids, holdout_ids, device, seed):
    """Full native training on target domain (same architecture, no transfer)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_entropy = math.log(VOCAB_SIZE)
    msg_dim = VOCAB_SIZE * N_HEADS
    n_batches = max(1, len(train_ids) // BATCH_SIZE)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    # Oracle for encoder init
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
    oracle_opt = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)

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
            oracle_opt.zero_grad()
            loss.backward()
            oracle_opt.step()
        if epoch % 20 == 0:
            torch.mps.empty_cache()

    oracle_enc_state = oracle.enc_a.state_dict()

    # Sender
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)

    # Population IL training (same as ramp)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    rng = np.random.RandomState(seed + 1000)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_states = None

    for epoch in range(RAMP_COMM_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, RAMP_COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

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
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if (epoch + 1) % 50 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (acc), best_r = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
            tb = acc['both']
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

    return sender, receivers


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 56: Cross-Physics Transfer (Ramp → Flat Drop)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Conditions: TRANSFER, NATIVE, RANDOM", flush=True)

    t_total = time.time()

    # ── Load ramp features ──────────────────────────────────────
    print("\n  Loading ramp features...", flush=True)
    ramp_cache = str(RESULTS_DIR / "phase54b_dino_features.pt")
    ramp_data, ramp_e_bins, ramp_f_bins = load_cached_features(ramp_cache)
    ramp_train_ids, ramp_holdout_ids = create_splits(
        ramp_e_bins, ramp_f_bins, HOLDOUT_CELLS)
    print(f"  Ramp: {ramp_data.shape}, {len(ramp_train_ids)} train, "
          f"{len(ramp_holdout_ids)} holdout", flush=True)

    # ── Extract/load flat drop features ─────────────────────────
    print("\n  Loading flat drop features...", flush=True)
    flat_data, flat_e_bins, flat_f_bins = extract_flat_drop_features()
    flat_train_ids, flat_holdout_ids = create_splits(
        flat_e_bins, flat_f_bins, HOLDOUT_CELLS)
    print(f"  Flat: {flat_data.shape}, {len(flat_train_ids)} train, "
          f"{len(flat_holdout_ids)} holdout", flush=True)

    # ── Flat drop oracle (shared across seeds) ──────────────────
    print("\n  Training flat drop oracle...", flush=True)
    torch.manual_seed(42)
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    flat_oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(DEVICE)
    flat_oracle_opt = torch.optim.Adam(flat_oracle.parameters(), lr=ORACLE_LR)
    rng_oracle = np.random.RandomState(42)

    n_batches = max(1, len(flat_train_ids) // BATCH_SIZE)
    e_dev = torch.tensor(flat_e_bins, dtype=torch.float32).to(DEVICE)
    f_dev = torch.tensor(flat_f_bins, dtype=torch.float32).to(DEVICE)

    for epoch in range(ORACLE_EPOCHS):
        flat_oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(flat_train_ids, BATCH_SIZE, rng_oracle)
            da, db = flat_data[ia].to(DEVICE), flat_data[ib].to(DEVICE)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = flat_oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            flat_oracle_opt.zero_grad()
            loss.backward()
            flat_oracle_opt.step()
        if epoch % 20 == 0:
            torch.mps.empty_cache()

    flat_oracle.eval()
    with torch.no_grad():
        oracle_train = evaluate_accuracy(
            None, None, flat_data, flat_e_bins, flat_f_bins,
            flat_train_ids, DEVICE, oracle_model=flat_oracle, n_rounds=50)
        oracle_holdout = evaluate_accuracy(
            None, None, flat_data, flat_e_bins, flat_f_bins,
            flat_holdout_ids, DEVICE, oracle_model=flat_oracle, n_rounds=50)
    print(f"  Flat oracle: train={oracle_train['both']:.1%}, "
          f"holdout={oracle_holdout['both']:.1%}", flush=True)

    # ── Per-seed experiments ────────────────────────────────────
    all_results = []
    msg_dim = VOCAB_SIZE * N_HEADS

    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n  --- Seed {seed} ---", flush=True)
        result = {'seed': seed}

        # (a) TRANSFER: Train ramp sender, freeze, train new receiver on flat
        print(f"    [TRANSFER] Training ramp sender...", flush=True)
        t0 = time.time()
        ramp_sender = train_ramp_sender(
            seed, ramp_data, ramp_e_bins, ramp_f_bins,
            ramp_train_ids, ramp_holdout_ids, DEVICE)

        # Evaluate ramp sender on ramp first
        ramp_sender.eval()
        with torch.no_grad():
            comp_ramp = compute_compositionality(
                ramp_sender, ramp_data, ramp_e_bins, ramp_f_bins, DEVICE)
        result['ramp_pos_dis'] = comp_ramp['pos_dis']
        result['ramp_topsim'] = comp_ramp['topsim']
        print(f"    Ramp sender: PosDis={comp_ramp['pos_dis']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        # Freeze ramp sender, train new receiver on flat drop
        print(f"    [TRANSFER] Training receiver on flat drop...", flush=True)
        t0 = time.time()
        transfer_receiver = train_receiver_frozen_sender(
            ramp_sender, flat_data, flat_e_bins, flat_f_bins,
            flat_train_ids, DEVICE, seed)

        # Evaluate transfer
        ramp_sender.eval()
        transfer_receiver.eval()
        with torch.no_grad():
            transfer_train = evaluate_accuracy(
                ramp_sender, transfer_receiver, flat_data, flat_e_bins, flat_f_bins,
                flat_train_ids, DEVICE, n_rounds=50)
            transfer_holdout = evaluate_accuracy(
                ramp_sender, transfer_receiver, flat_data, flat_e_bins, flat_f_bins,
                flat_holdout_ids, DEVICE, n_rounds=50)
        result['transfer_train'] = transfer_train
        result['transfer_holdout'] = transfer_holdout
        print(f"    Transfer: train={transfer_train['both']:.1%}, "
              f"holdout={transfer_holdout['both']:.1%} ({time.time()-t0:.0f}s)", flush=True)

        # Compositionality of ramp sender messages on flat drop data
        with torch.no_grad():
            comp_flat = compute_compositionality(
                ramp_sender, flat_data, flat_e_bins, flat_f_bins, DEVICE)
        result['transfer_pos_dis'] = comp_flat['pos_dis']
        result['transfer_topsim'] = comp_flat['topsim']

        del ramp_sender, transfer_receiver
        torch.mps.empty_cache()

        # (b) NATIVE: Full training on flat drop from scratch
        print(f"    [NATIVE] Full training on flat drop...", flush=True)
        t0 = time.time()
        native_sender, native_receivers = train_native(
            flat_data, flat_e_bins, flat_f_bins,
            flat_train_ids, flat_holdout_ids, DEVICE, seed)

        native_sender.eval()
        for r in native_receivers:
            r.eval()
        with torch.no_grad():
            (native_train_acc), native_best_r = evaluate_population(
                native_sender, native_receivers, flat_data, flat_e_bins, flat_f_bins,
                flat_train_ids, DEVICE, n_rounds=50)
            native_holdout_acc = evaluate_accuracy(
                native_sender, native_best_r, flat_data, flat_e_bins, flat_f_bins,
                flat_holdout_ids, DEVICE, n_rounds=50)
            comp_native = compute_compositionality(
                native_sender, flat_data, flat_e_bins, flat_f_bins, DEVICE)

        result['native_train'] = native_train_acc
        result['native_holdout'] = native_holdout_acc
        result['native_pos_dis'] = comp_native['pos_dis']
        result['native_topsim'] = comp_native['topsim']
        print(f"    Native: train={native_train_acc['both']:.1%}, "
              f"holdout={native_holdout_acc['both']:.1%}, "
              f"PosDis={comp_native['pos_dis']:.3f} ({time.time()-t0:.0f}s)", flush=True)

        del native_sender, native_receivers
        torch.mps.empty_cache()

        # (c) RANDOM: Frozen random sender + new receiver (floor baseline)
        print(f"    [RANDOM] Random sender baseline...", flush=True)
        t0 = time.time()
        torch.manual_seed(seed + 5000)
        random_encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        random_sender = CompositionalSender(
            random_encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(DEVICE)
        random_sender.eval()

        random_receiver = train_receiver_frozen_sender(
            random_sender, flat_data, flat_e_bins, flat_f_bins,
            flat_train_ids, DEVICE, seed)

        random_sender.eval()
        random_receiver.eval()
        with torch.no_grad():
            random_train = evaluate_accuracy(
                random_sender, random_receiver, flat_data, flat_e_bins, flat_f_bins,
                flat_train_ids, DEVICE, n_rounds=50)
            random_holdout = evaluate_accuracy(
                random_sender, random_receiver, flat_data, flat_e_bins, flat_f_bins,
                flat_holdout_ids, DEVICE, n_rounds=50)
        result['random_train'] = random_train
        result['random_holdout'] = random_holdout
        print(f"    Random: train={random_train['both']:.1%}, "
              f"holdout={random_holdout['both']:.1%} ({time.time()-t0:.0f}s)", flush=True)

        del random_sender, random_receiver
        torch.mps.empty_cache()

        dt_seed = time.time() - t_seed
        print(f"    Seed {seed} total: {dt_seed:.0f}s", flush=True)
        all_results.append(result)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"RESULTS SUMMARY: Phase 56 Cross-Physics Transfer", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n  Flat drop oracle: train={oracle_train['both']:.1%}, "
          f"holdout={oracle_holdout['both']:.1%}", flush=True)

    header = (f"  {'Seed':>4} | {'Transfer':>8} | {'Native':>8} | "
              f"{'Random':>8} | {'Ramp PD':>7} | {'Xfer PD':>7}")
    print(f"\n{header}", flush=True)
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'--------':>8}-+-"
          f"{'--------':>8}-+-{'-------':>7}-+-{'-------':>7}", flush=True)

    t_hb, n_hb, r_hb = [], [], []
    t_pd, n_pd, r_pd = [], [], []
    for r in all_results:
        t_h = r['transfer_holdout']['both']
        n_h = r['native_holdout']['both']
        r_h = r['random_holdout']['both']
        tag = " *" if r['ramp_pos_dis'] > 0.4 else ""
        print(f"  {r['seed']:>4} | {t_h:>7.1%} | {n_h:>7.1%} | "
              f"{r_h:>7.1%} | {r['ramp_pos_dis']:>7.3f} | "
              f"{r.get('transfer_pos_dis', 0):>7.3f}{tag}", flush=True)
        t_hb.append(t_h)
        n_hb.append(n_h)
        r_hb.append(r_h)
        t_pd.append(r.get('transfer_pos_dis', 0))
        n_pd.append(r.get('native_pos_dis', 0))

    print(f"  {'----':>4}-+-{'--------':>8}-+-{'--------':>8}-+-"
          f"{'--------':>8}-+-{'-------':>7}-+-{'-------':>7}", flush=True)
    print(f"  {'Mean':>4} | {np.mean(t_hb):>7.1%} | {np.mean(n_hb):>7.1%} | "
          f"{np.mean(r_hb):>7.1%} |         |", flush=True)
    print(f"  {'Std':>4} | {np.std(t_hb):>7.1%} | {np.std(n_hb):>7.1%} | "
          f"{np.std(r_hb):>7.1%} |         |", flush=True)

    # Compositional vs holistic transfer
    comp_seeds = [r for r in all_results if r['ramp_pos_dis'] > 0.4]
    hol_seeds = [r for r in all_results if r['ramp_pos_dis'] <= 0.4]

    if comp_seeds:
        ct = [r['transfer_holdout']['both'] for r in comp_seeds]
        print(f"\n  Compositional ramp senders ({len(comp_seeds)} seeds):", flush=True)
        print(f"    Transfer holdout: {np.mean(ct):.1%} +/- {np.std(ct):.1%}", flush=True)
    if hol_seeds:
        ht = [r['transfer_holdout']['both'] for r in hol_seeds]
        print(f"  Holistic ramp senders ({len(hol_seeds)} seeds):", flush=True)
        print(f"    Transfer holdout: {np.mean(ht):.1%} +/- {np.std(ht):.1%}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'ramp_comm_epochs': RAMP_COMM_EPOCHS,
            'transfer_epochs': TRANSFER_EPOCHS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'vocab_size': VOCAB_SIZE,
            'n_heads': N_HEADS,
        },
        'flat_oracle': {
            'train': oracle_train,
            'holdout': oracle_holdout,
        },
        'per_seed': all_results,
        'summary': {
            'transfer_holdout_mean': float(np.mean(t_hb)),
            'transfer_holdout_std': float(np.std(t_hb)),
            'native_holdout_mean': float(np.mean(n_hb)),
            'native_holdout_std': float(np.std(n_hb)),
            'random_holdout_mean': float(np.mean(r_hb)),
            'random_holdout_std': float(np.std(r_hb)),
            'compositional_seeds': len(comp_seeds),
            'holistic_seeds': len(hol_seeds),
        },
    }

    save_path = RESULTS_DIR / "phase56_results.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/len(SEEDS):.0f}s per seed)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

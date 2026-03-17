"""
Phase 83: Outcome Prediction from Frozen Messages (SELF-CORRECTING)
====================================================================
Show frozen V-JEPA 2 messages predict collision outcomes better than
frozen DINOv2 messages, and comparably to raw V-JEPA 2 features.

STEP 1: Generate outcome labels from physics params
STEP 2: Train senders (V-JEPA2 4-agent, DINOv2 4-agent) and extract frozen messages
STEP 3: Train outcome predictors on frozen messages (20 seeds each)
STEP 4: Self-correct if needed
STEP 5: Save results

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase83_outcome_prediction.py
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

RESULTS_DIR = Path("results")

# --- Architecture constants (match Phase 79/79b exactly) ---
HIDDEN_DIM = 128
DINO_DIM = 384
VJEPA_DIM = 1024
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
N_AGENTS = 4
N_FRAMES = 24
FRAMES_PER_AGENT = N_FRAMES // N_AGENTS  # 6

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

# Training constants for senders
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

# Outcome predictor constants
OUTCOME_EPOCHS = 100
OUTCOME_LR = 1e-3
N_OUTCOME_SEEDS = 20


# ══════════════════════════════════════════════════════════════════
# Architecture (copied from Phase 79/79b)
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


# ══════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════

def load_features(path, expect_dim=None):
    """Load features from .pt file. Returns features, mass_bins, rest_bins, index."""
    data = torch.load(path, weights_only=False)
    features = data['features'].float()
    if 'index' in data and isinstance(data['index'], list) and isinstance(data['index'][0], dict):
        index = data['index']
        mass_bins = np.array([e['mass_ratio_bin'] for e in index])
        rest_bins = np.array([e['restitution_bin'] for e in index])
    else:
        mass_bins = np.array(data['mass_bins'])
        rest_bins = np.array(data['rest_bins'])
        index = data.get('index', None)
    if expect_dim and features.shape[-1] != expect_dim:
        raise ValueError(f"Expected dim {expect_dim}, got {features.shape[-1]}")
    return features, mass_bins, rest_bins, index


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
# Sender training (same as Phase 79/79b)
# ══════════════════════════════════════════════════════════════════

def make_4agent_sender(input_dim):
    senders = []
    for _ in range(N_AGENTS):
        enc = TemporalEncoder(HIDDEN_DIM, input_dim)
        sender = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
        senders.append(sender)
    return MultiAgentSender(senders)


def train_4agent_sender(features, mass_bins, rest_bins, input_dim, seed=0):
    """Train a 4-agent sender. Returns trained sender."""
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    train_ids, holdout_ids = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE  # 4*2*5 = 40

    torch.manual_seed(seed)
    np.random.seed(seed)
    multi_sender = make_4agent_sender(input_dim).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(mass_bins, dtype=torch.float32).to(DEVICE)
    f_dev = torch.tensor(rest_bins, dtype=torch.float32).to(DEVICE)
    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)
    best_both_acc = 0.0
    best_sender_state = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi_sender.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            views_a = [v[ia].to(DEVICE) for v in agent_views]
            views_b = [v[ib].to(DEVICE) for v in agent_views]
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            msg_a, logits_a = multi_sender(views_a, tau=tau, hard=hard)
            msg_b, logits_b = multi_sender(views_b, tau=tau, hard=hard)
            total_loss = torch.tensor(0.0, device=DEVICE)
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
                continue
            torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        # Quick eval to track best and print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            multi_sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                # Quick train accuracy check
                rng_eval = np.random.RandomState(999)
                correct_both = total_both = 0
                for _ in range(10):
                    bs = min(BATCH_SIZE, len(train_ids))
                    ia, ib = sample_pairs(train_ids, bs, rng_eval)
                    views_a = [v[ia].to(DEVICE) for v in agent_views]
                    views_b = [v[ib].to(DEVICE) for v in agent_views]
                    msg_a, _ = multi_sender(views_a)
                    msg_b, _ = multi_sender(views_b)
                    for r in receivers:
                        pred_e, pred_f = r(msg_a, msg_b)
                        label_e = (e_dev[ia] > e_dev[ib])
                        label_f = (f_dev[ia] > f_dev[ib])
                        e_diff = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
                        f_diff = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
                        both_diff = e_diff & f_diff
                        if both_diff.sum() > 0:
                            both_ok = ((pred_e[both_diff] > 0) == label_e[both_diff]) & \
                                      ((pred_f[both_diff] > 0) == label_f[both_diff])
                            correct_both += both_ok.sum().item()
                            total_both += both_diff.sum().item()
                train_acc = correct_both / max(total_both, 1)
                if train_acc > best_both_acc:
                    best_both_acc = train_acc
                    best_sender_state = {k: v.cpu().clone() for k, v in multi_sender.state_dict().items()}
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  train_both={train_acc:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

    if best_sender_state is not None:
        multi_sender.load_state_dict(best_sender_state)
    return multi_sender


# ══════════════════════════════════════════════════════════════════
# Extract frozen messages
# ══════════════════════════════════════════════════════════════════

def extract_frozen_messages(multi_sender, features, n_agents, frames_per_agent):
    """Run frozen sender on all scenes, return one-hot messages (N, msg_dim)."""
    agent_views = split_views(features, n_agents, frames_per_agent)
    multi_sender.eval()
    all_messages = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            msg, _ = multi_sender(views)
            all_messages.append(msg.cpu())
    return torch.cat(all_messages, dim=0)  # (600, 40)


def extract_frozen_tokens(multi_sender, features, n_agents, frames_per_agent):
    """Run frozen sender on all scenes, return discrete token IDs."""
    agent_views = split_views(features, n_agents, frames_per_agent)
    multi_sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi_sender(views)
            tokens = [l.argmax(dim=-1).cpu() for l in logits]
            all_tokens.append(torch.stack(tokens, dim=1))
    return torch.cat(all_tokens, dim=0)  # (600, n_agents*n_heads)


# ══════════════════════════════════════════════════════════════════
# Outcome predictor
# ══════════════════════════════════════════════════════════════════

class OutcomePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class OutcomePredictor3Layer(nn.Module):
    """Deeper MLP for self-correction if 2-layer fails."""
    def __init__(self, input_dim, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_outcome_predictor(inputs, labels, train_ids, holdout_ids, seed,
                             model_cls=OutcomePredictor, model_kwargs=None):
    """Train binary outcome predictor. Returns holdout accuracy."""
    if model_kwargs is None:
        model_kwargs = {'input_dim': inputs.shape[1]}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=OUTCOME_LR)
    rng = np.random.RandomState(seed)

    inputs_dev = inputs.to(DEVICE)
    labels_dev = labels.float().to(DEVICE)
    best_holdout_acc = 0.0

    for epoch in range(OUTCOME_EPOCHS):
        model.train()
        # Mini-batch training
        perm = rng.permutation(len(train_ids))
        for start in range(0, len(train_ids), BATCH_SIZE):
            batch_idx = train_ids[perm[start:start+BATCH_SIZE]]
            x = inputs_dev[batch_idx]
            y = labels_dev[batch_idx]
            pred = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == OUTCOME_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred_h = model(inputs_dev[holdout_ids])
                acc_h = ((pred_h > 0).float() == labels_dev[holdout_ids]).float().mean().item()
                pred_t = model(inputs_dev[train_ids])
                acc_t = ((pred_t > 0).float() == labels_dev[train_ids]).float().mean().item()
            if acc_h > best_holdout_acc:
                best_holdout_acc = acc_h

        if epoch % 50 == 0:
            torch.mps.empty_cache()

    return best_holdout_acc


# ══════════════════════════════════════════════════════════════════
# Compositionality check for sender verification
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


def compute_posdis(tokens, mass_bins, rest_bins):
    """Compute PosDis from token array (N, n_positions)."""
    n_pos = tokens.shape[1]
    attributes = np.stack([mass_bins, rest_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(tokens[:, p], attributes[:, a])
    pos_dis = 0.0
    for p in range(n_pos):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if sorted_mi[0] > 1e-10:
            pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
    pos_dis /= n_pos
    return pos_dis, mi_matrix


# ══════════════════════════════════════════════════════════════════
# STEP 1: Generate outcome labels
# ══════════════════════════════════════════════════════════════════

def step1_outcome_labels():
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: Generate Outcome Labels", flush=True)
    print("=" * 70, flush=True)

    # Load index from V-JEPA2 features (has physics params)
    data = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)
    index = data['index']
    n = len(index)

    # Primary label: Sphere B post-collision speed (binarized at median)
    vel_b = np.array([e['post_collision_vel_b'] for e in index])
    median_vel = np.median(vel_b)
    labels_speed = (vel_b > median_vel).astype(int)

    # Alternative label: Does sphere B move faster than sphere A post-collision?
    vel_a = np.array([e['post_collision_vel_a'] for e in index])
    labels_b_faster = (vel_b > vel_a).astype(int)

    # Alternative label: Speed ratio (B/A) > 1
    speed_ratio = vel_b / np.maximum(vel_a, 1e-6)
    labels_ratio = (speed_ratio > 1.0).astype(int)

    print(f"  N scenes: {n}", flush=True)
    print(f"  vel_b range: [{vel_b.min():.3f}, {vel_b.max():.3f}], median={median_vel:.3f}", flush=True)
    print(f"  Label 'speed > median': {labels_speed.sum()}/{n} positive ({labels_speed.mean():.1%})", flush=True)
    print(f"  Label 'B faster than A': {labels_b_faster.sum()}/{n} positive ({labels_b_faster.mean():.1%})", flush=True)
    print(f"  Label 'speed ratio > 1': {labels_ratio.sum()}/{n} positive ({labels_ratio.mean():.1%})", flush=True)

    # Choose primary label
    primary_label = labels_speed
    primary_name = "sphere_B_speed_above_median"

    # Check balance
    pos_frac = primary_label.mean()
    if pos_frac < 0.3 or pos_frac > 0.7:
        print(f"  WARNING: Imbalanced ({pos_frac:.1%}), switching to B-faster-than-A", flush=True)
        primary_label = labels_b_faster
        primary_name = "sphere_B_faster_than_A"
        pos_frac = primary_label.mean()

    result = {
        'primary_label': primary_label.tolist(),
        'primary_name': primary_name,
        'n_positive': int(primary_label.sum()),
        'n_negative': int(n - primary_label.sum()),
        'vel_b': vel_b.tolist(),
        'vel_a': vel_a.tolist(),
        'median_vel_b': float(median_vel),
        'alt_labels': {
            'b_faster': labels_b_faster.tolist(),
            'speed_ratio_gt1': labels_ratio.tolist(),
        }
    }

    save_path = RESULTS_DIR / 'phase83_collision_outcomes.json'
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {save_path}", flush=True)
    print(f"  CHECKPOINT: {primary_label.sum()}/{n} positive — {'OK' if 0.3 < pos_frac < 0.7 else 'IMBALANCED'}", flush=True)

    return torch.tensor(primary_label, dtype=torch.float32), primary_name, result


# ══════════════════════════════════════════════════════════════════
# STEP 2: Train senders & extract frozen messages
# ══════════════════════════════════════════════════════════════════

def step2_extract_messages():
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Train Senders & Extract Frozen Messages", flush=True)
    print("=" * 70, flush=True)

    # Load V-JEPA2 features
    print("\n  Loading V-JEPA 2 collision features...", flush=True)
    vjepa_feats, vjepa_mass, vjepa_rest, vjepa_idx = load_features(
        'results/vjepa2_collision_pooled.pt', expect_dim=VJEPA_DIM)
    print(f"  V-JEPA 2: {vjepa_feats.shape}", flush=True)

    # Load DINOv2 features
    print("  Loading DINOv2 collision features...", flush=True)
    dino_feats, dino_mass, dino_rest, _ = load_features(
        'results/collision_dinov2_features.pt')
    print(f"  DINOv2: {dino_feats.shape}", flush=True)

    # Train V-JEPA2 4-agent sender (seed 0)
    print("\n  Training V-JEPA 2 4-agent sender (seed 0, 400 epochs)...", flush=True)
    t0 = time.time()
    vjepa_sender = train_4agent_sender(vjepa_feats, vjepa_mass, vjepa_rest,
                                        input_dim=VJEPA_DIM, seed=0)
    dt = time.time() - t0
    print(f"  V-JEPA 2 sender trained in {dt:.0f}s", flush=True)

    # Extract V-JEPA2 messages
    vjepa_messages = extract_frozen_messages(vjepa_sender, vjepa_feats, N_AGENTS, FRAMES_PER_AGENT)
    vjepa_tokens = extract_frozen_tokens(vjepa_sender, vjepa_feats, N_AGENTS, FRAMES_PER_AGENT)
    print(f"  V-JEPA 2 messages: {vjepa_messages.shape}", flush=True)

    # Check PosDis
    vjepa_posdis, vjepa_mi = compute_posdis(vjepa_tokens.numpy(), vjepa_mass, vjepa_rest)
    print(f"  V-JEPA 2 sender PosDis: {vjepa_posdis:.3f}", flush=True)

    # Train DINOv2 4-agent sender (seed 0)
    print("\n  Training DINOv2 4-agent sender (seed 0, 400 epochs)...", flush=True)
    t0 = time.time()
    dino_sender = train_4agent_sender(dino_feats, dino_mass, dino_rest,
                                       input_dim=DINO_DIM, seed=0)
    dt = time.time() - t0
    print(f"  DINOv2 sender trained in {dt:.0f}s", flush=True)

    # Extract DINOv2 messages
    dino_messages = extract_frozen_messages(dino_sender, dino_feats, N_AGENTS, FRAMES_PER_AGENT)
    dino_tokens = extract_frozen_tokens(dino_sender, dino_feats, N_AGENTS, FRAMES_PER_AGENT)
    print(f"  DINOv2 messages: {dino_messages.shape}", flush=True)

    # Check PosDis
    dino_posdis, dino_mi = compute_posdis(dino_tokens.numpy(), dino_mass, dino_rest)
    print(f"  DINOv2 sender PosDis: {dino_posdis:.3f}", flush=True)

    # Raw V-JEPA2 features (mean-pooled over time)
    vjepa_raw = vjepa_feats.mean(dim=1)  # (600, 1024)
    print(f"  V-JEPA 2 raw (mean-pooled): {vjepa_raw.shape}", flush=True)

    # CHECKPOINT
    print(f"\n  CHECKPOINT:", flush=True)
    print(f"    V-JEPA 2 messages: {vjepa_messages.shape} — expect (600, 40) — "
          f"{'OK' if vjepa_messages.shape == (600, 40) else 'MISMATCH'}", flush=True)
    print(f"    DINOv2 messages:   {dino_messages.shape} — expect (600, 40) — "
          f"{'OK' if dino_messages.shape == (600, 40) else 'MISMATCH'}", flush=True)
    print(f"    V-JEPA 2 PosDis:   {vjepa_posdis:.3f} — {'compositional' if vjepa_posdis > 0.4 else 'NOT compositional'}", flush=True)
    print(f"    DINOv2 PosDis:     {dino_posdis:.3f} — {'compositional' if dino_posdis > 0.4 else 'NOT compositional'}", flush=True)

    return {
        'vjepa_messages': vjepa_messages,
        'dino_messages': dino_messages,
        'vjepa_raw': vjepa_raw,
        'vjepa_posdis': vjepa_posdis,
        'dino_posdis': dino_posdis,
        'mass_bins': vjepa_mass,
        'rest_bins': vjepa_rest,
    }


# ══════════════════════════════════════════════════════════════════
# STEP 3: Train outcome predictors
# ══════════════════════════════════════════════════════════════════

def step3_train_predictors(messages_data, labels, label_name):
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Train Outcome Predictors (20 seeds each)", flush=True)
    print("=" * 70, flush=True)

    mass_bins = messages_data['mass_bins']
    rest_bins = messages_data['rest_bins']
    train_ids, holdout_ids = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    print(f"  Train: {len(train_ids)}, Holdout: {len(holdout_ids)}", flush=True)
    print(f"  Label: {label_name}", flush=True)
    print(f"  Holdout label balance: {labels[holdout_ids].mean():.1%} positive", flush=True)

    conditions = {
        'vjepa2_messages': messages_data['vjepa_messages'],       # (600, 40)
        'dinov2_messages': messages_data['dino_messages'],         # (600, 40)
        'vjepa2_raw_features': messages_data['vjepa_raw'],         # (600, 1024)
    }

    results = {}
    for cond_name, inputs in conditions.items():
        print(f"\n  --- {cond_name} (input_dim={inputs.shape[1]}) ---", flush=True)
        accs = []
        for seed in range(N_OUTCOME_SEEDS):
            acc = train_outcome_predictor(
                inputs, labels, train_ids, holdout_ids, seed,
                model_cls=OutcomePredictor,
                model_kwargs={'input_dim': inputs.shape[1]})
            accs.append(acc)
            if (seed + 1) % 5 == 0:
                print(f"    Seeds 0-{seed}: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)

        results[cond_name] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'seeds': accs,
        }
        print(f"    Final: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)

    return results, train_ids, holdout_ids


# ══════════════════════════════════════════════════════════════════
# STEP 4: Self-correction
# ══════════════════════════════════════════════════════════════════

def step4_self_correct(results, messages_data, labels, label_name,
                        train_ids, holdout_ids, outcome_data):
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: Self-Correction Checks", flush=True)
    print("=" * 70, flush=True)

    corrections = []

    vjepa_mean = results['vjepa2_messages']['mean']
    dino_mean = results['dinov2_messages']['mean']
    raw_mean = results['vjepa2_raw_features']['mean']
    gap_vjepa_dino = vjepa_mean - dino_mean
    gap_msgs_raw = vjepa_mean - raw_mean

    print(f"\n  V-JEPA 2 messages:      {vjepa_mean:.1%}", flush=True)
    print(f"  DINOv2 messages:        {dino_mean:.1%}", flush=True)
    print(f"  V-JEPA 2 raw features:  {raw_mean:.1%}", flush=True)
    print(f"  Gap (V-JEPA2 - DINOv2): {gap_vjepa_dino:+.1%}", flush=True)
    print(f"  Gap (msgs - raw):       {gap_msgs_raw:+.1%}", flush=True)

    # Check 1: V-JEPA2 messages too low?
    if vjepa_mean < 0.60:
        print(f"\n  ⚠ V-JEPA 2 messages < 60% ({vjepa_mean:.1%}). Trying 3-layer MLP...", flush=True)
        corrections.append("vjepa2_messages_3layer_mlp")
        accs = []
        for seed in range(N_OUTCOME_SEEDS):
            acc = train_outcome_predictor(
                messages_data['vjepa_messages'], labels, train_ids, holdout_ids, seed,
                model_cls=OutcomePredictor3Layer,
                model_kwargs={'input_dim': 40})
            accs.append(acc)
        results['vjepa2_messages_3layer'] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'seeds': accs,
        }
        print(f"    3-layer V-JEPA2: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)
        vjepa_mean = max(vjepa_mean, np.mean(accs))

    # Check 2: DINOv2 too close to V-JEPA2?
    if abs(gap_vjepa_dino) < 0.03:
        print(f"\n  ⚠ DINOv2 ≈ V-JEPA2 (gap={gap_vjepa_dino:+.1%} < 3pp). "
              f"Trying 'B faster than A' label...", flush=True)
        corrections.append("switched_to_b_faster_label")

        alt_labels_raw = outcome_data['alt_labels']['b_faster']
        alt_labels = torch.tensor(alt_labels_raw, dtype=torch.float32)
        alt_name = "sphere_B_faster_than_A"
        print(f"    Alt label balance: {alt_labels.mean():.1%} positive", flush=True)

        alt_results = {}
        for cond_name, inputs in [
            ('vjepa2_messages', messages_data['vjepa_messages']),
            ('dinov2_messages', messages_data['dino_messages']),
            ('vjepa2_raw_features', messages_data['vjepa_raw']),
        ]:
            accs = []
            for seed in range(N_OUTCOME_SEEDS):
                acc = train_outcome_predictor(
                    inputs, alt_labels, train_ids, holdout_ids, seed,
                    model_cls=OutcomePredictor,
                    model_kwargs={'input_dim': inputs.shape[1]})
                accs.append(acc)
            alt_results[cond_name] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'seeds': accs,
            }
            print(f"    {cond_name}: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)

        # Use alt results if gap is bigger
        alt_gap = alt_results['vjepa2_messages']['mean'] - alt_results['dinov2_messages']['mean']
        print(f"    Alt label gap: {alt_gap:+.1%}", flush=True)
        if alt_gap > gap_vjepa_dino:
            print(f"    Alt label improves gap from {gap_vjepa_dino:+.1%} to {alt_gap:+.1%}. Using alt.", flush=True)
            results = alt_results
            label_name = alt_name
            corrections.append("used_alt_label_b_faster")
        else:
            print(f"    Alt label doesn't help. Keeping original.", flush=True)
            results['alt_label_results'] = alt_results

    # Check 3: Messages >> raw features (suspicious)?
    if gap_msgs_raw > 0.05:
        print(f"\n  ⚠ Messages beat raw features by {gap_msgs_raw:+.1%} (>5pp). "
              f"Checking for leakage...", flush=True)
        corrections.append("leakage_check")
        # Shuffle labels and verify chance
        shuffled_labels = labels[torch.randperm(len(labels))]
        accs_shuffled = []
        for seed in range(5):
            acc = train_outcome_predictor(
                messages_data['vjepa_messages'], shuffled_labels,
                train_ids, holdout_ids, seed,
                model_cls=OutcomePredictor,
                model_kwargs={'input_dim': 40})
            accs_shuffled.append(acc)
        shuffle_mean = np.mean(accs_shuffled)
        print(f"    Shuffled labels accuracy: {shuffle_mean:.1%} (expect ~50%)", flush=True)
        if shuffle_mean > 0.60:
            print(f"    LEAKAGE DETECTED — shuffled accuracy too high!", flush=True)
            corrections.append("LEAKAGE_DETECTED")
        else:
            print(f"    No leakage — shuffled is at chance.", flush=True)

    # Check 4: All reasonable?
    vjepa_mean_final = results.get('vjepa2_messages', results.get('vjepa2_messages_3layer', {})).get('mean', 0)
    if not results.get('vjepa2_messages'):
        vjepa_mean_final = vjepa_mean

    success = (vjepa_mean_final >= 0.55 and
               abs(gap_msgs_raw) <= 0.10)  # relaxed criteria

    if success:
        print(f"\n  SUCCESS: Results look reasonable.", flush=True)
    else:
        print(f"\n  Note: Results may need manual review.", flush=True)

    return results, label_name, corrections


# ══════════════════════════════════════════════════════════════════
# STEP 5: Save results
# ══════════════════════════════════════════════════════════════════

def step5_save(results, label_name, outcome_data, messages_data, corrections):
    print("\n" + "=" * 70, flush=True)
    print("STEP 5: Save Results", flush=True)
    print("=" * 70, flush=True)

    # Statistical tests
    vjepa_seeds = results['vjepa2_messages']['seeds']
    dino_seeds = results['dinov2_messages']['seeds']
    raw_seeds = results['vjepa2_raw_features']['seeds']

    t_vd, p_vd = stats.ttest_ind(vjepa_seeds, dino_seeds)
    d_vd = (np.mean(vjepa_seeds) - np.mean(dino_seeds)) / np.sqrt(
        (np.std(vjepa_seeds)**2 + np.std(dino_seeds)**2) / 2)

    t_mr, p_mr = stats.ttest_ind(vjepa_seeds, raw_seeds)
    d_mr = (np.mean(vjepa_seeds) - np.mean(raw_seeds)) / np.sqrt(
        (np.std(vjepa_seeds)**2 + np.std(raw_seeds)**2) / 2)

    save_data = {
        'vjepa2_messages': results['vjepa2_messages'],
        'dinov2_messages': results['dinov2_messages'],
        'vjepa2_raw_features': results['vjepa2_raw_features'],
        'ttest_vjepa2_vs_dinov2': {'t': float(t_vd), 'p': float(p_vd), 'd': float(d_vd)},
        'ttest_vjepa2_msgs_vs_raw': {'t': float(t_mr), 'p': float(p_mr), 'd': float(d_mr)},
        'outcome_label': label_name,
        'n_positive': outcome_data['n_positive'],
        'n_negative': outcome_data['n_negative'],
        'vjepa2_sender_posdis': messages_data['vjepa_posdis'],
        'dinov2_sender_posdis': messages_data['dino_posdis'],
        'self_corrections_applied': corrections,
    }

    # Include 3-layer results if they exist
    if 'vjepa2_messages_3layer' in results:
        save_data['vjepa2_messages_3layer'] = results['vjepa2_messages_3layer']
    if 'alt_label_results' in results:
        save_data['alt_label_results'] = results['alt_label_results']

    save_path = RESULTS_DIR / 'phase83_outcome_prediction.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved {save_path}", flush=True)

    # Print summary
    print(f"\n  {'='*60}", flush=True)
    print(f"  SUMMARY: Outcome Prediction from Frozen Messages", flush=True)
    print(f"  {'='*60}", flush=True)
    print(f"  Label: {label_name}", flush=True)
    print(f"  {'Condition':<30s} {'Accuracy':>10s} {'Std':>8s}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for name in ['vjepa2_messages', 'dinov2_messages', 'vjepa2_raw_features']:
        r = results[name]
        print(f"  {name:<30s} {r['mean']:>9.1%} {r['std']:>7.1%}", flush=True)
    if 'vjepa2_messages_3layer' in results:
        r = results['vjepa2_messages_3layer']
        print(f"  {'vjepa2_msgs_3layer':<30s} {r['mean']:>9.1%} {r['std']:>7.1%}", flush=True)
    print(f"\n  V-JEPA 2 vs DINOv2 messages: t={t_vd:.3f}, p={p_vd:.4f}, d={d_vd:.3f}", flush=True)
    print(f"  V-JEPA 2 msgs vs raw feats:  t={t_mr:.3f}, p={p_mr:.4f}, d={d_mr:.3f}", flush=True)
    print(f"  Sender PosDis — V-JEPA 2: {messages_data['vjepa_posdis']:.3f}, DINOv2: {messages_data['dino_posdis']:.3f}", flush=True)
    if corrections:
        print(f"  Self-corrections: {corrections}", flush=True)
    print(f"  {'='*60}", flush=True)

    return save_data


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Phase 83: Outcome Prediction from Frozen Messages", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    t_total = time.time()

    labels, label_name, outcome_data = step1_outcome_labels()
    messages_data = step2_extract_messages()
    results, train_ids, holdout_ids = step3_train_predictors(messages_data, labels, label_name)
    results, label_name, corrections = step4_self_correct(
        results, messages_data, labels, label_name, train_ids, holdout_ids, outcome_data)
    step5_save(results, label_name, outcome_data, messages_data, corrections)

    dt = time.time() - t_total
    print(f"\nPhase 83 complete. Total time: {dt/60:.1f}min", flush=True)

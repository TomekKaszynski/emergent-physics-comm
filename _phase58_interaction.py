"""
Phase 58: Interaction-Dependent Communication Task
===================================================
Test whether compositional property encoding enables reasoning about
property INTERACTIONS, not just individual properties.

Task: "which ball has longer trajectory?" — requires knowing both
elasticity AND friction (e-only: 70%, f-only: 78%, both: 100%).

Three conditions × 20 seeds:
(a) FACTORED 2×5: fresh sender+receiver on interaction task, population IL
(b) HOLISTIC 1×25: same but holistic control
(c) TRANSFER: frozen Phase 54f-style sender, new receiver only

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase58_interaction.py
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

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
CONTROL_VOCAB = 25
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

ORACLE_EPOCHS = 100
ORACLE_LR = 1e-3
COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TRANSFER_RECEIVER_LR = 3e-3
TRANSFER_EPOCHS = 200
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

SEEDS = list(range(20))


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


class InteractionReceiver(nn.Module):
    """Single-head receiver for binary interaction task."""
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.net(torch.cat([msg_a, msg_b], dim=-1))
        return self.head(h).squeeze(-1)


class PropertyReceiver(nn.Module):
    """Two-head receiver for individual property prediction (Phase 54f style)."""
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


class InteractionOracle(nn.Module):
    def __init__(self, encoder_cls, encoder_kwargs, hidden_dim):
        super().__init__()
        self.enc_a = encoder_cls(**encoder_kwargs)
        self.enc_b = encoder_cls(**encoder_kwargs)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_a, x_b):
        ha = self.enc_a(x_a)
        hb = self.enc_b(x_b)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return self.head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def load_cached_features(cache_path):
    data = torch.load(cache_path, weights_only=False)
    return data['features'], data['e_bins'], data['f_bins']


def compute_trajectory_lengths():
    """Compute total trajectory length for each of 300 ramp scenes."""
    dataset_dir = Path("kubric/output/ramp_dataset")
    traj_lens = []
    for si in range(300):
        pos = np.load(dataset_dir / f"scene_{si:04d}" / "positions.npy")
        diffs = np.diff(pos, axis=0)
        traj_len = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        traj_lens.append(traj_len)
    return np.array(traj_lens, dtype=np.float32)


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

def evaluate_interaction(sender, receiver, data_t, traj_lens_t,
                         scene_ids, device, n_rounds=30,
                         oracle_model=None):
    """Evaluate accuracy on interaction task (which travels further?)."""
    rng = np.random.RandomState(999)
    correct = 0
    total = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)

        label = (traj_lens_t[ia] > traj_lens_t[ib]).float().to(device)
        # Skip tied pairs
        diff_mask = traj_lens_t[ia] != traj_lens_t[ib]
        if diff_mask.sum() == 0:
            continue

        if oracle_model is not None:
            pred = oracle_model(da, db)
        else:
            msg_a, _ = sender(da)
            msg_b, _ = sender(db)
            pred = receiver(msg_a, msg_b)

        pred_bin = (pred > 0).float()
        correct += (pred_bin[diff_mask] == label[diff_mask]).sum().item()
        total += diff_mask.sum().item()

    return correct / max(total, 1)


def evaluate_interaction_population(sender, receivers, data_t, traj_lens_t,
                                    scene_ids, device, n_rounds=30):
    """Pick best receiver from population, return accuracy."""
    best_acc = 0
    best_r = None
    for r in receivers:
        acc = evaluate_interaction(
            sender, r, data_t, traj_lens_t, scene_ids, device, n_rounds=10)
        if acc > best_acc:
            best_acc = acc
            best_r = r
    final_acc = evaluate_interaction(
        sender, best_r, data_t, traj_lens_t, scene_ids, device, n_rounds=n_rounds)
    return final_acc, best_r


def evaluate_property_accuracy(sender, receiver, data_t, e_bins, f_bins,
                               scene_ids, device, n_rounds=30):
    """Evaluate e/f/both accuracy (Phase 54f style)."""
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


def compute_compositionality(sender, data_t, e_bins, f_bins, device,
                             vocab_size=None):
    """Compute PosDis, TopSim, MI matrix, entropies."""
    if vocab_size is None:
        vocab_size = sender.vocab_size
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
        v = vocab_size if n_pos > 1 else CONTROL_VOCAB
        counts = np.bincount(all_tokens[:, p], minlength=v)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(v)))

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
# Training: Oracle
# ══════════════════════════════════════════════════════════════════

def train_oracle(data_t, traj_lens_t, train_ids, device, seed):
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = InteractionOracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)

    best_acc = 0.0
    best_state = None
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label = (traj_lens_t[ia] > traj_lens_t[ib]).float().to(device)
            pred = oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                acc = evaluate_interaction(
                    None, None, data_t, traj_lens_t, train_ids, device,
                    oracle_model=oracle, n_rounds=20)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone()
                              for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


# ══════════════════════════════════════════════════════════════════
# Training: Property sender (Phase 54f style, for transfer condition)
# ══════════════════════════════════════════════════════════════════

def train_property_sender(data_t, e_bins, f_bins, train_ids, holdout_ids,
                          device, seed):
    """Train a Phase 54f-style sender on individual property prediction.
    Returns frozen sender state dict and PosDis."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Oracle for encoder init
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    prop_oracle = nn.Module()
    prop_oracle.enc_a = TemporalEncoder(**enc_kwargs)
    prop_oracle.enc_b = TemporalEncoder(**enc_kwargs)
    prop_oracle.shared = nn.Sequential(
        nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU())
    prop_oracle.elast_head = nn.Linear(HIDDEN_DIM, 1)
    prop_oracle.friction_head = nn.Linear(HIDDEN_DIM, 1)
    prop_oracle = prop_oracle.to(device)

    prop_oracle_opt = torch.optim.Adam(prop_oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        prop_oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            ha = prop_oracle.enc_a(da)
            hb = prop_oracle.enc_b(db)
            h = prop_oracle.shared(torch.cat([ha, hb], dim=-1))
            pred_e = prop_oracle.elast_head(h).squeeze(-1)
            pred_f = prop_oracle.friction_head(h).squeeze(-1)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            prop_oracle_opt.zero_grad()
            loss.backward()
            prop_oracle_opt.step()
        if epoch % 20 == 0:
            torch.mps.empty_cache()

    oracle_enc_state = prop_oracle.enc_a.state_dict()

    # Build sender with oracle encoder
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Population of property receivers
    receivers = [PropertyReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    max_entropy = math.log(VOCAB_SIZE)
    best_both_acc = 0.0
    best_sender_state = None
    nan_count = 0

    for epoch in range(COMM_EPOCHS):
        # Simultaneous IL
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = PropertyReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
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
                if p.grad is not None and (torch.isnan(p.grad).any() or
                                           torch.isinf(p.grad).any()):
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

        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                best_both = 0
                for r in receivers:
                    _, _, both = evaluate_property_accuracy(
                        sender, r, data_t, e_bins, f_bins,
                        train_ids, device, n_rounds=10)
                    if both > best_both:
                        best_both = both

            if best_both > best_both_acc:
                best_both_acc = best_both
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)

    # Compute compositionality
    sender.eval()
    with torch.no_grad():
        comp = compute_compositionality(sender, data_t, e_bins, f_bins, device)

    return sender, comp['pos_dis']


# ══════════════════════════════════════════════════════════════════
# Training: Interaction task (conditions a and b)
# ══════════════════════════════════════════════════════════════════

def train_interaction_population(sender, receivers, data_t, traj_lens_t,
                                 e_bins, f_bins, train_ids, holdout_ids,
                                 device, msg_dim, seed, vocab_size,
                                 condition_name=""):
    """Train sender+receivers on interaction task with population IL."""
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    traj_dev = traj_lens_t.to(device)
    max_entropy = math.log(vocab_size)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_acc = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # Simultaneous IL
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = InteractionReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label = (traj_dev[ia] > traj_dev[ib]).float()

            msg_a, logits_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b = sender(db, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred, label)
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
                if p.grad is not None and (torch.isnan(p.grad).any() or
                                           torch.isinf(p.grad).any()):
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

        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                train_acc, _ = evaluate_interaction_population(
                    sender, receivers, data_t, traj_lens_t,
                    train_ids, device, n_rounds=15)
                holdout_acc, _ = evaluate_interaction_population(
                    sender, receivers, data_t, traj_lens_t,
                    holdout_ids, device, n_rounds=15)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"      Ep {epoch+1:3d}: train={train_acc:.1%}  "
                  f"holdout={holdout_acc:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_acc > best_holdout_acc:
                best_holdout_acc = holdout_acc
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
# Training: Transfer receiver (condition c)
# ══════════════════════════════════════════════════════════════════

def train_transfer_receiver(frozen_sender, data_t, traj_lens_t,
                            train_ids, holdout_ids, device, msg_dim, seed):
    """Train receiver only against frozen sender on interaction task."""
    torch.manual_seed(seed + 1000)
    receiver = InteractionReceiver(msg_dim, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=TRANSFER_RECEIVER_LR)
    rng = np.random.RandomState(seed + 1000)
    traj_dev = traj_lens_t.to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_acc = 0.0
    best_state = None

    frozen_sender.eval()

    for epoch in range(TRANSFER_EPOCHS):
        receiver.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label = (traj_dev[ia] > traj_dev[ib]).float()

            with torch.no_grad():
                msg_a, _ = frozen_sender(da)
                msg_b, _ = frozen_sender(db)

            pred = receiver(msg_a, msg_b)
            loss = F.binary_cross_entropy_with_logits(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(receiver.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            receiver.eval()
            with torch.no_grad():
                holdout_acc = evaluate_interaction(
                    frozen_sender, receiver, data_t, traj_lens_t,
                    holdout_ids, device, n_rounds=20)
            if holdout_acc > best_holdout_acc:
                best_holdout_acc = holdout_acc
                best_state = {k: v.cpu().clone()
                              for k, v in receiver.state_dict().items()}

        if epoch % 50 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        receiver.load_state_dict(best_state)
    return receiver


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    print("=" * 70, flush=True)
    print("Phase 58: Interaction-Dependent Communication Task", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Task: which ball has longer trajectory?", flush=True)
    print(f"  Conditions: FACTORED 2×{VOCAB_SIZE}, HOLISTIC 1×{CONTROL_VOCAB}, "
          f"TRANSFER", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Comm epochs: {COMM_EPOCHS} (a,b), {TRANSFER_EPOCHS} (c)", flush=True)
    print(f"  Population IL: {N_RECEIVERS} receivers, "
          f"reset every {RECEIVER_RESET_INTERVAL}", flush=True)

    # Load features
    print("\n  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins = load_cached_features(
        RESULTS_DIR / "phase54b_dino_features.pt")
    data_t = features.clone()
    print(f"  Features: {data_t.shape}", flush=True)

    # Compute trajectory lengths
    print("  Computing trajectory lengths...", flush=True)
    traj_lens = compute_trajectory_lengths()
    traj_lens_t = torch.tensor(traj_lens, dtype=torch.float32)
    print(f"  Trajectory lengths: min={traj_lens.min():.2f}, "
          f"max={traj_lens.max():.2f}, mean={traj_lens.mean():.2f}", flush=True)

    # Variance decomposition
    ss_total = np.sum((traj_lens - traj_lens.mean())**2)
    e_means = [traj_lens[e_bins == i].mean() for i in range(5)]
    f_means = [traj_lens[f_bins == i].mean() for i in range(5)]
    pred_e = np.array([e_means[i] for i in e_bins])
    pred_f = np.array([f_means[i] for i in f_bins])
    r2_e = np.sum((pred_e - traj_lens.mean())**2) / ss_total
    r2_f = np.sum((pred_f - traj_lens.mean())**2) / ss_total
    print(f"  Variance: R²(e)={r2_e:.3f}, R²(f)={r2_f:.3f}, "
          f"sum={r2_e+r2_f:.3f}", flush=True)

    # Splits
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Oracle
    print("\n  Training interaction oracle...", flush=True)
    oracle, oracle_train_acc = train_oracle(
        data_t, traj_lens_t, train_ids, DEVICE, seed=42)
    oracle.eval()
    with torch.no_grad():
        oracle_holdout_acc = evaluate_interaction(
            None, None, data_t, traj_lens_t, holdout_ids, DEVICE,
            oracle_model=oracle, n_rounds=50)
    print(f"  Oracle: train={oracle_train_acc:.1%}, "
          f"holdout={oracle_holdout_acc:.1%}", flush=True)

    # Train transfer sender (Phase 54f style, one seed)
    print("\n  Training property sender for transfer (seed 0)...", flush=True)
    t_xfer = time.time()
    transfer_sender, transfer_posdis = train_property_sender(
        data_t, e_bins, f_bins, train_ids, holdout_ids, DEVICE, seed=0)
    transfer_sender.eval()
    for p in transfer_sender.parameters():
        p.requires_grad_(False)
    print(f"  Transfer sender: PosDis={transfer_posdis:.3f} "
          f"({time.time()-t_xfer:.0f}s)", flush=True)

    # Run all seeds
    msg_dim_factored = VOCAB_SIZE * N_HEADS   # 10
    msg_dim_holistic = CONTROL_VOCAB * 1       # 25

    all_results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        t_seed = time.time()

        print(f"\n  --- Seed {seed} ---", flush=True)

        # (a) FACTORED 2×5
        print(f"    [FACTORED 2×{VOCAB_SIZE}] Training on interaction task...",
              flush=True)
        encoder_a = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        sender_a = CompositionalSender(
            encoder_a, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(DEVICE)
        receivers_a = [InteractionReceiver(msg_dim_factored, HIDDEN_DIM).to(DEVICE)
                       for _ in range(N_RECEIVERS)]
        receivers_a, nan_a = train_interaction_population(
            sender_a, receivers_a, data_t, traj_lens_t, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim_factored, seed,
            VOCAB_SIZE, "FACTORED")

        sender_a.eval()
        for r in receivers_a:
            r.eval()
        with torch.no_grad():
            train_acc_a, _ = evaluate_interaction_population(
                sender_a, receivers_a, data_t, traj_lens_t,
                train_ids, DEVICE, n_rounds=50)
            holdout_acc_a, _ = evaluate_interaction_population(
                sender_a, receivers_a, data_t, traj_lens_t,
                holdout_ids, DEVICE, n_rounds=50)
            comp_a = compute_compositionality(
                sender_a, data_t, e_bins, f_bins, DEVICE)

        print(f"    Factored: train={train_acc_a:.1%}, holdout={holdout_acc_a:.1%}, "
              f"PosDis={comp_a['pos_dis']:.3f}", flush=True)

        # (b) HOLISTIC 1×25
        print(f"    [HOLISTIC 1×{CONTROL_VOCAB}] Training on interaction task...",
              flush=True)
        encoder_b = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        sender_b = CompositionalSender(
            encoder_b, HIDDEN_DIM, CONTROL_VOCAB, 1).to(DEVICE)
        receivers_b = [InteractionReceiver(msg_dim_holistic, HIDDEN_DIM).to(DEVICE)
                       for _ in range(N_RECEIVERS)]
        receivers_b, nan_b = train_interaction_population(
            sender_b, receivers_b, data_t, traj_lens_t, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim_holistic, seed + 100,
            CONTROL_VOCAB, "HOLISTIC")

        sender_b.eval()
        for r in receivers_b:
            r.eval()
        with torch.no_grad():
            train_acc_b, _ = evaluate_interaction_population(
                sender_b, receivers_b, data_t, traj_lens_t,
                train_ids, DEVICE, n_rounds=50)
            holdout_acc_b, _ = evaluate_interaction_population(
                sender_b, receivers_b, data_t, traj_lens_t,
                holdout_ids, DEVICE, n_rounds=50)
            comp_b = compute_compositionality(
                sender_b, data_t, e_bins, f_bins, DEVICE,
                vocab_size=CONTROL_VOCAB)

        print(f"    Holistic: train={train_acc_b:.1%}, holdout={holdout_acc_b:.1%}, "
              f"PosDis={comp_b['pos_dis']:.3f}", flush=True)

        # (c) TRANSFER
        print(f"    [TRANSFER] Training receiver on interaction task...", flush=True)
        t_c = time.time()
        receiver_c = train_transfer_receiver(
            transfer_sender, data_t, traj_lens_t,
            train_ids, holdout_ids, DEVICE, msg_dim_factored, seed)

        receiver_c.eval()
        with torch.no_grad():
            train_acc_c = evaluate_interaction(
                transfer_sender, receiver_c, data_t, traj_lens_t,
                train_ids, DEVICE, n_rounds=50)
            holdout_acc_c = evaluate_interaction(
                transfer_sender, receiver_c, data_t, traj_lens_t,
                holdout_ids, DEVICE, n_rounds=50)

        print(f"    Transfer: train={train_acc_c:.1%}, holdout={holdout_acc_c:.1%} "
              f"({time.time()-t_c:.0f}s)", flush=True)

        dt = time.time() - t_seed
        print(f"    Seed {seed} total: {dt:.0f}s", flush=True)

        all_results.append({
            'seed': seed,
            'factored': {
                'train': float(train_acc_a),
                'holdout': float(holdout_acc_a),
                'pos_dis': comp_a['pos_dis'],
                'topsim': comp_a['topsim'],
                'entropies': comp_a['entropies'],
                'mi_matrix': comp_a['mi_matrix'].tolist(),
                'nan_count': nan_a,
            },
            'holistic': {
                'train': float(train_acc_b),
                'holdout': float(holdout_acc_b),
                'pos_dis': comp_b['pos_dis'],
                'topsim': comp_b['topsim'],
                'nan_count': nan_b,
            },
            'transfer': {
                'train': float(train_acc_c),
                'holdout': float(holdout_acc_c),
            },
            'time_sec': dt,
        })

        torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY: Phase 58 Interaction Task", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  Oracle: train={oracle_train_acc:.1%}, "
          f"holdout={oracle_holdout_acc:.1%}", flush=True)
    print(f"  Transfer sender PosDis: {transfer_posdis:.3f}", flush=True)

    print(f"\n  Seed | Factored | Holistic | Transfer | Fact PD | Hol PD",
          flush=True)
    print(f"  -----+----------+----------+----------+---------+--------",
          flush=True)

    fact_holdouts = []
    hol_holdouts = []
    xfer_holdouts = []
    fact_posdis = []

    for r in all_results:
        s = r['seed']
        fh = r['factored']['holdout']
        hh = r['holistic']['holdout']
        xh = r['transfer']['holdout']
        fpd = r['factored']['pos_dis']
        hpd = r['holistic']['pos_dis']

        fact_holdouts.append(fh)
        hol_holdouts.append(hh)
        xfer_holdouts.append(xh)
        fact_posdis.append(fpd)

        print(f"  {s:4d} | {fh:6.1%}  | {hh:6.1%}  | {xh:6.1%}  | "
              f"{fpd:6.3f}  | {hpd:6.3f}", flush=True)

    print(f"  -----+----------+----------+----------+---------+--------",
          flush=True)

    fact_holdouts = np.array(fact_holdouts)
    hol_holdouts = np.array(hol_holdouts)
    xfer_holdouts = np.array(xfer_holdouts)
    fact_posdis = np.array(fact_posdis)

    print(f"  Mean | {fact_holdouts.mean():6.1%}  | {hol_holdouts.mean():6.1%}  | "
          f"{xfer_holdouts.mean():6.1%}  |", flush=True)
    print(f"   Std | {fact_holdouts.std():6.1%}  | {hol_holdouts.std():6.1%}  | "
          f"{xfer_holdouts.std():6.1%}  |", flush=True)

    # Key comparison: transfer vs factored ceiling
    gap = xfer_holdouts.mean() - hol_holdouts.mean()
    ceiling = fact_holdouts.mean() - hol_holdouts.mean()
    frac = gap / ceiling if abs(ceiling) > 0.001 else float('nan')
    print(f"\n  Transfer vs holistic gap: {gap:+.1%}", flush=True)
    print(f"  Factored vs holistic gap: {ceiling:+.1%}", flush=True)
    print(f"  Transfer captures {frac:.0%} of factored advantage", flush=True)

    # Compositionality on interaction task
    comp_seeds = sum(1 for pd in fact_posdis if pd > 0.4)
    print(f"\n  Factored senders compositional (PosDis>0.4): "
          f"{comp_seeds}/20", flush=True)
    print(f"  Factored mean PosDis: {fact_posdis.mean():.3f} ± "
          f"{fact_posdis.std():.3f}", flush=True)

    # Save
    output = {
        'config': {
            'task': 'trajectory_length_comparison',
            'outcome_variable': 'trajectory_length',
            'r2_elasticity': float(r2_e),
            'r2_friction': float(r2_f),
            'oracle_train': float(oracle_train_acc),
            'oracle_holdout': float(oracle_holdout_acc),
            'transfer_sender_posdis': float(transfer_posdis),
            'transfer_sender_seed': 0,
            'comm_epochs': COMM_EPOCHS,
            'transfer_epochs': TRANSFER_EPOCHS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'vocab_size': VOCAB_SIZE,
            'n_heads': N_HEADS,
            'control_vocab': CONTROL_VOCAB,
        },
        'per_seed': all_results,
        'summary': {
            'factored_holdout_mean': float(fact_holdouts.mean()),
            'factored_holdout_std': float(fact_holdouts.std()),
            'holistic_holdout_mean': float(hol_holdouts.mean()),
            'holistic_holdout_std': float(hol_holdouts.std()),
            'transfer_holdout_mean': float(xfer_holdouts.mean()),
            'transfer_holdout_std': float(xfer_holdouts.std()),
            'factored_posdis_mean': float(fact_posdis.mean()),
            'factored_posdis_std': float(fact_posdis.std()),
            'factored_compositional_count': int(comp_seeds),
        },
    }

    out_path = RESULTS_DIR / "phase58_interaction.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

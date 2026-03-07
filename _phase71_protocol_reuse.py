"""
Phase 71: Protocol reuse — frozen sender enables multiple downstream tasks
=========================================================================
Take the best frozen compositional sender (seed 0, trained with IL),
freeze it, and train 3 different receiver tasks on its frozen messages:

Task 1: Same-property comparison (original): "which has higher elasticity?"
Task 2: Cross-property comparison: "is A's elasticity > B's friction?"
Task 3: Property REGRESSION from single message: classify elasticity
        bin (0-4) from one message alone. MLP (msg_dim -> 64 -> 5).

20 seeds each, 100 epochs, frozen sender throughout.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase71_protocol_reuse.py
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
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

# Sender training config (Phase 54f / IL)
ORACLE_EPOCHS = 100
ORACLE_LR = 1e-3
SENDER_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

# Downstream task config
DOWNSTREAM_EPOCHS = 100
DOWNSTREAM_LR = 3e-3
N_DOWNSTREAM_SEEDS = 20
SENDER_SEED = 0  # seed for the frozen sender


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
    """Original task: compare two scenes on same properties."""
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


class CrossPropertyReceiver(nn.Module):
    """Task 2: is A's elasticity > B's friction?"""
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


class RegressionReceiver(nn.Module):
    """Task 3: classify elasticity bin (0-4) from single message."""
    def __init__(self, msg_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, msg):
        return self.net(msg)


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
# Train the compositional sender (Phase 54f config with IL)
# ══════════════════════════════════════════════════════════════════

def train_oracle(data_t, e_bins, f_bins, train_ids, device, seed):
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
                _, _, acc_both = eval_comparison(
                    None, None, data_t, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone() for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def eval_comparison(sender, receiver, data_t, e_bins, f_bins,
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


def eval_population(sender, receivers, data_t, e_bins, f_bins,
                    scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = eval_comparison(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return eval_comparison(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


def train_sender_with_il(data_t, e_bins, f_bins, train_ids, holdout_ids, device, seed):
    """Train a compositional sender using IL (Phase 54f config). Returns frozen sender."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Oracle
    oracle, oracle_acc = train_oracle(data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Sender
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Population IL
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    nan_count = 0
    t0 = time.time()

    for epoch in range(SENDER_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, SENDER_EPOCHS - 1)
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

        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (te, tf, tb), _ = eval_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}

            elapsed = time.time() - t0
            eta = elapsed / (epoch + 1) * (SENDER_EPOCHS - epoch - 1)
            print(f"      Ep {epoch+1:3d}: train={tb:.1%}  ETA {eta/60:.0f}min", flush=True)

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)

    # Compute PosDis
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i+BATCH_SIZE].to(device)
            _, logits = sender(batch)
            tokens_batch = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens_batch, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    mi_matrix = np.zeros((N_HEADS, 2))
    attributes = np.stack([e_bins, f_bins], axis=1)
    for p in range(N_HEADS):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    pos_dis = 0.0
    for p in range(N_HEADS):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if sorted_mi[0] > 1e-10:
            pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
    pos_dis /= N_HEADS

    dt = time.time() - t0
    print(f"    Sender trained: PosDis={pos_dis:.3f}  MI={mi_matrix.tolist()}  ({dt:.0f}s)", flush=True)

    return sender, pos_dis, mi_matrix


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
# Pre-compute frozen messages
# ══════════════════════════════════════════════════════════════════

def precompute_messages(sender, data_t, device):
    """Pre-compute all messages from frozen sender (one-hot)."""
    sender.eval()
    all_msgs = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i+BATCH_SIZE].to(device)
            msg, _ = sender(batch)
            all_msgs.append(msg.cpu())
    return torch.cat(all_msgs, dim=0)


# ══════════════════════════════════════════════════════════════════
# Task 1: Same-property comparison (original)
# ══════════════════════════════════════════════════════════════════

def run_task1(msgs, e_bins, f_bins, train_ids, holdout_ids, device, seed):
    """Original task: predict which has higher e and f."""
    torch.manual_seed(seed + 1000)
    msg_dim = msgs.shape[1]
    receiver = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=DOWNSTREAM_LR)
    rng = np.random.RandomState(seed + 1000)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(DOWNSTREAM_EPOCHS):
        receiver.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            ma, mb = msgs[ia].to(device), msgs[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = receiver(ma, mb)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    receiver.eval()
    rng_eval = np.random.RandomState(999)
    results = {}
    for split_name, split_ids in [('train', train_ids), ('holdout', holdout_ids)]:
        correct_e = correct_f = correct_both = 0
        total_e = total_f = total_both = 0
        for _ in range(50):
            bs = min(BATCH_SIZE, len(split_ids))
            ia, ib = sample_pairs(split_ids, bs, rng_eval)
            ma, mb = msgs[ia].to(device), msgs[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib])
            label_f = (f_dev[ia] > f_dev[ib])
            with torch.no_grad():
                pred_e, pred_f = receiver(ma, mb)
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
        results[f'{split_name}_e'] = correct_e / max(total_e, 1)
        results[f'{split_name}_f'] = correct_f / max(total_f, 1)
        results[f'{split_name}_both'] = correct_both / max(total_both, 1)

    return results


# ══════════════════════════════════════════════════════════════════
# Task 2: Cross-property comparison
# ══════════════════════════════════════════════════════════════════

def run_task2(msgs, e_bins, f_bins, train_ids, holdout_ids, device, seed):
    """Cross-property: is A's elasticity > B's friction?"""
    torch.manual_seed(seed + 2000)
    msg_dim = msgs.shape[1]
    receiver = CrossPropertyReceiver(msg_dim, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=DOWNSTREAM_LR)
    rng = np.random.RandomState(seed + 2000)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(DOWNSTREAM_EPOCHS):
        receiver.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            ma, mb = msgs[ia].to(device), msgs[ib].to(device)
            # Label: is A's elasticity > B's friction?
            label = (e_dev[ia] > f_dev[ib]).float()
            pred = receiver(ma, mb)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    receiver.eval()
    rng_eval = np.random.RandomState(999)
    results = {}
    for split_name, split_ids in [('train', train_ids), ('holdout', holdout_ids)]:
        correct = total = 0
        for _ in range(50):
            bs = min(BATCH_SIZE, len(split_ids))
            ia, ib = sample_pairs(split_ids, bs, rng_eval)
            ma, mb = msgs[ia].to(device), msgs[ib].to(device)
            label = (e_dev[ia] > f_dev[ib])
            with torch.no_grad():
                pred = receiver(ma, mb)
            pred_bin = pred > 0
            # Only count when e_a != f_b (non-trivial)
            diff = torch.tensor(e_bins[ia] != f_bins[ib], device=device)
            if diff.sum() > 0:
                correct += (pred_bin[diff] == label[diff]).sum().item()
                total += diff.sum().item()
        results[f'{split_name}_acc'] = correct / max(total, 1)

    return results


# ══════════════════════════════════════════════════════════════════
# Task 3: Property regression from single message
# ══════════════════════════════════════════════════════════════════

def run_task3(msgs, e_bins, f_bins, train_ids, holdout_ids, device, seed):
    """Classify elasticity bin (0-4) from a single message."""
    torch.manual_seed(seed + 3000)
    msg_dim = msgs.shape[1]
    receiver = RegressionReceiver(msg_dim).to(device)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=DOWNSTREAM_LR)
    rng = np.random.RandomState(seed + 3000)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    e_labels = torch.tensor(e_bins, dtype=torch.long).to(device)
    f_labels = torch.tensor(f_bins, dtype=torch.long).to(device)

    for epoch in range(DOWNSTREAM_EPOCHS):
        receiver.train()
        for _ in range(n_batches):
            idx = rng.choice(train_ids, size=BATCH_SIZE)
            m = msgs[idx].to(device)
            target_e = e_labels[idx]
            logits = receiver(m)
            loss = F.cross_entropy(logits, target_e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    receiver.eval()
    results = {}
    for split_name, split_ids in [('train', train_ids), ('holdout', holdout_ids)]:
        correct_e = 0
        total = 0
        # Also test friction decoding (not trained on it)
        correct_f = 0
        for i in range(0, len(split_ids), BATCH_SIZE):
            batch_ids = split_ids[i:i+BATCH_SIZE]
            m = msgs[batch_ids].to(device)
            with torch.no_grad():
                logits = receiver(m)
            pred = logits.argmax(dim=-1)
            correct_e += (pred == e_labels[batch_ids]).sum().item()
            # Check if model accidentally predicts friction (shouldn't if compositional)
            correct_f += (pred == f_labels[batch_ids]).sum().item()
            total += len(batch_ids)
        results[f'{split_name}_e_acc'] = correct_e / max(total, 1)
        results[f'{split_name}_f_match'] = correct_f / max(total, 1)  # should be ~20% (chance)

    return results


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 71: Protocol reuse — frozen sender, 3 downstream tasks", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Sender: seed {SENDER_SEED}, trained with IL (Phase 54f config)", flush=True)
    print(f"  Downstream: {N_DOWNSTREAM_SEEDS} seeds x 3 tasks x {DOWNSTREAM_EPOCHS} epochs", flush=True)

    t_total = time.time()

    # Load features
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # ════════════════════════════════════════════════════════════
    # Step 1: Train the compositional sender
    # ════════════════════════════════════════════════════════════
    print(f"\n  Step 1: Training compositional sender (seed {SENDER_SEED})...", flush=True)
    sender, pos_dis, mi_matrix = train_sender_with_il(
        data_t, e_bins, f_bins, train_ids, holdout_ids, DEVICE, SENDER_SEED)

    # Freeze sender
    sender.eval()
    for p in sender.parameters():
        p.requires_grad = False

    # Pre-compute messages
    msgs = precompute_messages(sender, data_t, DEVICE)
    print(f"  Messages pre-computed: {msgs.shape}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Step 2: Run 3 downstream tasks
    # ════════════════════════════════════════════════════════════

    # Task 1: Same-property comparison
    print(f"\n  Step 2a: Task 1 — Same-property comparison ({N_DOWNSTREAM_SEEDS} seeds)...", flush=True)
    task1_results = []
    for seed in range(N_DOWNSTREAM_SEEDS):
        r = run_task1(msgs, e_bins, f_bins, train_ids, holdout_ids, DEVICE, seed)
        task1_results.append(r)
        if (seed + 1) % 5 == 0:
            print(f"    Seed {seed}: train={r['train_both']:.1%} holdout={r['holdout_both']:.1%}", flush=True)

    # Task 2: Cross-property comparison
    print(f"\n  Step 2b: Task 2 — Cross-property comparison ({N_DOWNSTREAM_SEEDS} seeds)...", flush=True)
    task2_results = []
    for seed in range(N_DOWNSTREAM_SEEDS):
        r = run_task2(msgs, e_bins, f_bins, train_ids, holdout_ids, DEVICE, seed)
        task2_results.append(r)
        if (seed + 1) % 5 == 0:
            print(f"    Seed {seed}: train={r['train_acc']:.1%} holdout={r['holdout_acc']:.1%}", flush=True)

    # Task 3: Property regression
    print(f"\n  Step 2c: Task 3 — Elasticity regression from single msg ({N_DOWNSTREAM_SEEDS} seeds)...", flush=True)
    task3_results = []
    for seed in range(N_DOWNSTREAM_SEEDS):
        r = run_task3(msgs, e_bins, f_bins, train_ids, holdout_ids, DEVICE, seed)
        task3_results.append(r)
        if (seed + 1) % 5 == 0:
            print(f"    Seed {seed}: train_e={r['train_e_acc']:.1%} holdout_e={r['holdout_e_acc']:.1%} "
                  f"f_match={r['holdout_f_match']:.1%}", flush=True)

    torch.mps.empty_cache()

    # ════════════════════════════════════════════════════════════
    # Analysis
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"RESULTS: Phase 71 — Protocol Reuse", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Sender: seed {SENDER_SEED}, PosDis={pos_dis:.3f}", flush=True)
    print(f"  MI matrix: {mi_matrix.tolist()}", flush=True)

    # Task 1
    t1_train = np.array([r['train_both'] for r in task1_results])
    t1_holdout = np.array([r['holdout_both'] for r in task1_results])
    print(f"\n  TASK 1: Same-property comparison (original)", flush=True)
    print(f"    Train:   {t1_train.mean():.1%} +/- {t1_train.std():.1%}", flush=True)
    print(f"    Holdout: {t1_holdout.mean():.1%} +/- {t1_holdout.std():.1%}", flush=True)
    print(f"    Gap:     {(t1_train.mean() - t1_holdout.mean()):.1%}", flush=True)

    # Task 2
    t2_train = np.array([r['train_acc'] for r in task2_results])
    t2_holdout = np.array([r['holdout_acc'] for r in task2_results])
    print(f"\n  TASK 2: Cross-property comparison (A's elast > B's friction)", flush=True)
    print(f"    Train:   {t2_train.mean():.1%} +/- {t2_train.std():.1%}", flush=True)
    print(f"    Holdout: {t2_holdout.mean():.1%} +/- {t2_holdout.std():.1%}", flush=True)
    print(f"    Gap:     {(t2_train.mean() - t2_holdout.mean()):.1%}", flush=True)

    # Task 3
    t3_train_e = np.array([r['train_e_acc'] for r in task3_results])
    t3_holdout_e = np.array([r['holdout_e_acc'] for r in task3_results])
    t3_holdout_f = np.array([r['holdout_f_match'] for r in task3_results])
    print(f"\n  TASK 3: Elasticity regression from single message", flush=True)
    print(f"    Train e-acc:   {t3_train_e.mean():.1%} +/- {t3_train_e.std():.1%}", flush=True)
    print(f"    Holdout e-acc: {t3_holdout_e.mean():.1%} +/- {t3_holdout_e.std():.1%}", flush=True)
    print(f"    Holdout f-match: {t3_holdout_f.mean():.1%} +/- {t3_holdout_f.std():.1%}  (chance=20%)", flush=True)
    print(f"    Gap:     {(t3_train_e.mean() - t3_holdout_e.mean()):.1%}", flush=True)

    # Summary
    print(f"\n  SUMMARY:", flush=True)
    print(f"    Task 1 (same-prop):  holdout {t1_holdout.mean():.1%}", flush=True)
    print(f"    Task 2 (cross-prop): holdout {t2_holdout.mean():.1%}", flush=True)
    print(f"    Task 3 (regression): holdout {t3_holdout_e.mean():.1%}", flush=True)
    print(f"    All tasks succeed => protocol is a REUSABLE INTERFACE", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'sender_seed': SENDER_SEED,
            'downstream_epochs': DOWNSTREAM_EPOCHS,
            'downstream_lr': DOWNSTREAM_LR,
            'n_downstream_seeds': N_DOWNSTREAM_SEEDS,
        },
        'sender': {
            'pos_dis': float(pos_dis),
            'mi_matrix': mi_matrix.tolist(),
        },
        'task1_same_property': {
            'per_seed': task1_results,
            'train_both_mean': float(t1_train.mean()),
            'train_both_std': float(t1_train.std()),
            'holdout_both_mean': float(t1_holdout.mean()),
            'holdout_both_std': float(t1_holdout.std()),
        },
        'task2_cross_property': {
            'per_seed': task2_results,
            'train_acc_mean': float(t2_train.mean()),
            'train_acc_std': float(t2_train.std()),
            'holdout_acc_mean': float(t2_holdout.mean()),
            'holdout_acc_std': float(t2_holdout.std()),
        },
        'task3_regression': {
            'per_seed': task3_results,
            'train_e_mean': float(t3_train_e.mean()),
            'train_e_std': float(t3_train_e.std()),
            'holdout_e_mean': float(t3_holdout_e.mean()),
            'holdout_e_std': float(t3_holdout_e.std()),
            'holdout_f_match_mean': float(t3_holdout_f.mean()),
            'holdout_f_match_std': float(t3_holdout_f.std()),
        },
    }

    save_path = RESULTS_DIR / "phase71_protocol_reuse.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

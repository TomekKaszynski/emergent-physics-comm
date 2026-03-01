"""
Phase 58b: Cross-Property Interaction Task
==========================================
Task: "is ball A's elasticity > ball B's friction?"

This CROSSES property dimensions — the receiver needs elasticity from
message A and friction from message B. Impossible to solve additively.

Single-property baselines: e-only 80%, f-only 80%, both 100%.

Three conditions × 20 seeds:
(a) FACTORED 2×5: fresh sender+receiver on cross-property task
(b) HOLISTIC 1×25: control
(c) TRANSFER: frozen compositional property sender, new receiver only

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase58b_cross_property.py
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
TRANSFER_CANDIDATE_SEEDS = [0, 1, 2, 3, 4]  # Try 5 seeds, pick best PosDis


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 54f / 58)
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


class CrossPropertyReceiver(nn.Module):
    """Single-head receiver for cross-property binary task."""
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


class CrossPropertyOracle(nn.Module):
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

def evaluate_cross_property(sender, receiver, data_t, e_bins, f_bins,
                            scene_ids, device, n_rounds=30,
                            oracle_model=None):
    """Evaluate: is e(A) > f(B)?  Skip tied pairs."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    correct = 0
    total = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)

        label = (e_dev[ia] > f_dev[ib]).float()
        valid = (e_dev[ia] != f_dev[ib])
        if valid.sum() == 0:
            continue

        if oracle_model is not None:
            pred = oracle_model(da, db)
        else:
            msg_a, _ = sender(da)
            msg_b, _ = sender(db)
            pred = receiver(msg_a, msg_b)

        pred_bin = (pred > 0).float()
        correct += (pred_bin[valid] == label[valid]).sum().item()
        total += valid.sum().item()

    return correct / max(total, 1)


def evaluate_cross_property_population(sender, receivers, data_t, e_bins, f_bins,
                                       scene_ids, device, n_rounds=30):
    """Pick best receiver from population."""
    best_acc = 0
    best_r = None
    for r in receivers:
        acc = evaluate_cross_property(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if acc > best_acc:
            best_acc = acc
            best_r = r
    final_acc = evaluate_cross_property(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds)
    return final_acc, best_r


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
# Ablation: which message positions does the receiver use?
# ══════════════════════════════════════════════════════════════════

def ablation_analysis(sender, receiver, data_t, e_bins, f_bins,
                      scene_ids, device, vocab_size, n_heads,
                      n_rounds=50):
    """Zero out individual positions and measure accuracy drop.

    For factored 2×5: msg = [pos0(5), pos1(5)] = 10 dims.
    Ablate pos0 of msg_A, pos1 of msg_A, pos0 of msg_B, pos1 of msg_B.

    Returns dict with ablation results.
    """
    if n_heads == 1:
        return {}  # Can't ablate holistic

    sender.eval()
    receiver.eval()
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    # Full accuracy (baseline)
    full_acc = evaluate_cross_property(
        sender, receiver, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds)

    # Ablation for each position × each message
    results = {'full': full_acc}
    for msg_label, msg_idx in [('A', 0), ('B', 1)]:
        for pos in range(n_heads):
            correct = 0
            total = 0
            for _ in range(n_rounds):
                bs = min(BATCH_SIZE, len(scene_ids))
                ia, ib = sample_pairs(scene_ids, bs, rng)
                da, db = data_t[ia].to(device), data_t[ib].to(device)

                label = (e_dev[ia] > f_dev[ib]).float()
                valid = (e_dev[ia] != f_dev[ib])
                if valid.sum() == 0:
                    continue

                with torch.no_grad():
                    msg_a, _ = sender(da)
                    msg_b, _ = sender(db)

                    # Zero out position `pos` of message `msg_idx`
                    if msg_idx == 0:
                        msg_a_abl = msg_a.clone()
                        msg_a_abl[:, pos*vocab_size:(pos+1)*vocab_size] = 0
                        pred = receiver(msg_a_abl, msg_b)
                    else:
                        msg_b_abl = msg_b.clone()
                        msg_b_abl[:, pos*vocab_size:(pos+1)*vocab_size] = 0
                        pred = receiver(msg_a, msg_b_abl)

                pred_bin = (pred > 0).float()
                correct += (pred_bin[valid] == label[valid]).sum().item()
                total += valid.sum().item()

            abl_acc = correct / max(total, 1)
            drop = full_acc - abl_acc
            results[f'ablate_{msg_label}_pos{pos}'] = abl_acc
            results[f'drop_{msg_label}_pos{pos}'] = drop

    return results


# ══════════════════════════════════════════════════════════════════
# Training: Oracle
# ══════════════════════════════════════════════════════════════════

def train_oracle(data_t, e_bins, f_bins, train_ids, device, seed):
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = CrossPropertyOracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
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
            label = (e_dev[ia] > f_dev[ib]).float()
            pred = oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                acc = evaluate_cross_property(
                    None, None, data_t, e_bins, f_bins, train_ids, device,
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
# Training: Property sender (Phase 54f style, for transfer)
# ══════════════════════════════════════════════════════════════════

def train_property_sender(data_t, e_bins, f_bins, train_ids, holdout_ids,
                          device, seed):
    """Train Phase 54f-style sender. Returns sender + PosDis."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Quick oracle for encoder init
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = CrossPropertyOracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
    oracle_opt = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    # Train oracle on e/f separately (not cross-property)
    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            ha = oracle.enc_a(da)
            hb = oracle.enc_b(db)
            h = oracle.shared(torch.cat([ha, hb], dim=-1))
            pred = oracle.head(h).squeeze(-1)
            # Train on combined e+f signal
            loss = F.binary_cross_entropy_with_logits(pred, label_e) + \
                   F.binary_cross_entropy_with_logits(pred, label_f)
            oracle_opt.zero_grad()
            loss.backward()
            oracle_opt.step()
        if epoch % 20 == 0:
            torch.mps.empty_cache()

    oracle_enc_state = oracle.enc_a.state_dict()

    # Build sender
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Population of PROPERTY receivers (e/f task, not cross-property)
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

        # Track best by property accuracy
        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                best_both = 0
                for r in receivers:
                    e_dev_np = e_bins
                    f_dev_np = f_bins
                    # Quick property eval
                    eval_rng = np.random.RandomState(888)
                    ce = cf = cb = 0
                    te = tf = tb = 0
                    for _ in range(10):
                        tia, tib = sample_pairs(train_ids, BATCH_SIZE, eval_rng)
                        tda = data_t[tia].to(device)
                        tdb = data_t[tib].to(device)
                        tmsg_a, _ = sender(tda)
                        tmsg_b, _ = sender(tdb)
                        tpe, tpf = r(tmsg_a, tmsg_b)
                        tle = (torch.tensor(e_bins[tia]).to(device) >
                               torch.tensor(e_bins[tib]).to(device))
                        tlf = (torch.tensor(f_bins[tia]).to(device) >
                               torch.tensor(f_bins[tib]).to(device))
                        ed = torch.tensor(e_bins[tia] != e_bins[tib], device=device)
                        fd = torch.tensor(f_bins[tia] != f_bins[tib], device=device)
                        bd = ed & fd
                        if ed.sum() > 0:
                            ce += ((tpe > 0)[ed] == tle[ed]).sum().item()
                            te += ed.sum().item()
                        if fd.sum() > 0:
                            cf += ((tpf > 0)[fd] == tlf[fd]).sum().item()
                            tf += fd.sum().item()
                        if bd.sum() > 0:
                            bok = ((tpe > 0)[bd] == tle[bd]) & \
                                  ((tpf > 0)[bd] == tlf[bd])
                            cb += bok.sum().item()
                            tb += bd.sum().item()
                    both = cb / max(tb, 1)
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
# Training: Cross-property task (conditions a and b)
# ══════════════════════════════════════════════════════════════════

def train_cross_property_population(sender, receivers, data_t, e_bins, f_bins,
                                    train_ids, holdout_ids, device,
                                    msg_dim, seed, vocab_size,
                                    condition_name=""):
    """Train sender+receivers on cross-property task with population IL."""
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
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
                receivers[i] = CrossPropertyReceiver(msg_dim, HIDDEN_DIM).to(device)
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
            label = (e_dev[ia] > f_dev[ib]).float()

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
                train_acc, _ = evaluate_cross_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=15)
                holdout_acc, _ = evaluate_cross_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
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

def train_transfer_receiver(frozen_sender, data_t, e_bins, f_bins,
                            train_ids, holdout_ids, device, msg_dim, seed):
    """Train receiver only against frozen sender on cross-property task."""
    torch.manual_seed(seed + 1000)
    receiver = CrossPropertyReceiver(msg_dim, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(receiver.parameters(), lr=TRANSFER_RECEIVER_LR)
    rng = np.random.RandomState(seed + 1000)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_acc = 0.0
    best_state = None

    frozen_sender.eval()

    for epoch in range(TRANSFER_EPOCHS):
        receiver.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label = (e_dev[ia] > f_dev[ib]).float()

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
                holdout_acc = evaluate_cross_property(
                    frozen_sender, receiver, data_t, e_bins, f_bins,
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
    print("Phase 58b: Cross-Property Interaction Task", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Task: is e(A) > f(B)?", flush=True)
    print(f"  Conditions: FACTORED 2×{VOCAB_SIZE}, HOLISTIC 1×{CONTROL_VOCAB}, "
          f"TRANSFER", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Comm epochs: {COMM_EPOCHS} (a,b), {TRANSFER_EPOCHS} (c)", flush=True)

    # Load features
    print("\n  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins = load_cached_features(
        RESULTS_DIR / "phase54b_dino_features.pt")
    data_t = features.clone()
    print(f"  Features: {data_t.shape}", flush=True)

    # Splits
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Oracle
    print("\n  Training cross-property oracle...", flush=True)
    oracle, oracle_train_acc = train_oracle(
        data_t, e_bins, f_bins, train_ids, DEVICE, seed=42)
    oracle.eval()
    with torch.no_grad():
        oracle_holdout_acc = evaluate_cross_property(
            None, None, data_t, e_bins, f_bins, holdout_ids, DEVICE,
            oracle_model=oracle, n_rounds=50)
    print(f"  Oracle: train={oracle_train_acc:.1%}, "
          f"holdout={oracle_holdout_acc:.1%}", flush=True)

    # Train transfer senders — try multiple seeds, pick best PosDis
    print(f"\n  Training property senders for transfer "
          f"(seeds {TRANSFER_CANDIDATE_SEEDS})...", flush=True)
    best_transfer_sender = None
    best_transfer_posdis = -1
    best_transfer_seed = -1

    for cs in TRANSFER_CANDIDATE_SEEDS:
        t_cs = time.time()
        sender_cs, posdis_cs = train_property_sender(
            data_t, e_bins, f_bins, train_ids, holdout_ids, DEVICE, seed=cs)
        print(f"    Seed {cs}: PosDis={posdis_cs:.3f} ({time.time()-t_cs:.0f}s)",
              flush=True)
        if posdis_cs > best_transfer_posdis:
            best_transfer_posdis = posdis_cs
            best_transfer_sender = sender_cs
            best_transfer_seed = cs

    transfer_sender = best_transfer_sender
    transfer_sender.eval()
    for p in transfer_sender.parameters():
        p.requires_grad_(False)
    print(f"  Selected transfer sender: seed {best_transfer_seed}, "
          f"PosDis={best_transfer_posdis:.3f}", flush=True)

    # Get MI matrix for transfer sender
    with torch.no_grad():
        xfer_comp = compute_compositionality(
            transfer_sender, data_t, e_bins, f_bins, DEVICE)
    mi = xfer_comp['mi_matrix']
    print(f"  Transfer MI matrix:", flush=True)
    print(f"    pos0: MI(e)={mi[0,0]:.3f}  MI(f)={mi[0,1]:.3f}", flush=True)
    print(f"    pos1: MI(e)={mi[1,0]:.3f}  MI(f)={mi[1,1]:.3f}", flush=True)

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
        print(f"    [FACTORED 2×{VOCAB_SIZE}] Training on cross-property task...",
              flush=True)
        encoder_a = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        sender_a = CompositionalSender(
            encoder_a, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(DEVICE)
        receivers_a = [CrossPropertyReceiver(msg_dim_factored, HIDDEN_DIM).to(DEVICE)
                       for _ in range(N_RECEIVERS)]
        receivers_a, nan_a = train_cross_property_population(
            sender_a, receivers_a, data_t, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim_factored, seed,
            VOCAB_SIZE, "FACTORED")

        sender_a.eval()
        for r in receivers_a:
            r.eval()
        with torch.no_grad():
            train_acc_a, best_r_a = evaluate_cross_property_population(
                sender_a, receivers_a, data_t, e_bins, f_bins,
                train_ids, DEVICE, n_rounds=50)
            holdout_acc_a, _ = evaluate_cross_property_population(
                sender_a, receivers_a, data_t, e_bins, f_bins,
                holdout_ids, DEVICE, n_rounds=50)
            comp_a = compute_compositionality(
                sender_a, data_t, e_bins, f_bins, DEVICE)

        # Ablation analysis for factored
        abl_a = ablation_analysis(
            sender_a, best_r_a, data_t, e_bins, f_bins,
            holdout_ids, DEVICE, VOCAB_SIZE, N_HEADS, n_rounds=50)

        abl_str = ""
        if abl_a:
            abl_str = (f"  abl: A0={abl_a.get('drop_A_pos0',0):+.1%} "
                       f"A1={abl_a.get('drop_A_pos1',0):+.1%} "
                       f"B0={abl_a.get('drop_B_pos0',0):+.1%} "
                       f"B1={abl_a.get('drop_B_pos1',0):+.1%}")

        print(f"    Factored: train={train_acc_a:.1%}, holdout={holdout_acc_a:.1%}, "
              f"PosDis={comp_a['pos_dis']:.3f}{abl_str}", flush=True)

        # (b) HOLISTIC 1×25
        print(f"    [HOLISTIC 1×{CONTROL_VOCAB}] Training on cross-property task...",
              flush=True)
        encoder_b = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        sender_b = CompositionalSender(
            encoder_b, HIDDEN_DIM, CONTROL_VOCAB, 1).to(DEVICE)
        receivers_b = [CrossPropertyReceiver(msg_dim_holistic, HIDDEN_DIM).to(DEVICE)
                       for _ in range(N_RECEIVERS)]
        receivers_b, nan_b = train_cross_property_population(
            sender_b, receivers_b, data_t, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim_holistic, seed + 100,
            CONTROL_VOCAB, "HOLISTIC")

        sender_b.eval()
        for r in receivers_b:
            r.eval()
        with torch.no_grad():
            train_acc_b, _ = evaluate_cross_property_population(
                sender_b, receivers_b, data_t, e_bins, f_bins,
                train_ids, DEVICE, n_rounds=50)
            holdout_acc_b, _ = evaluate_cross_property_population(
                sender_b, receivers_b, data_t, e_bins, f_bins,
                holdout_ids, DEVICE, n_rounds=50)
            comp_b = compute_compositionality(
                sender_b, data_t, e_bins, f_bins, DEVICE,
                vocab_size=CONTROL_VOCAB)

        print(f"    Holistic: train={train_acc_b:.1%}, holdout={holdout_acc_b:.1%}, "
              f"PosDis={comp_b['pos_dis']:.3f}", flush=True)

        # (c) TRANSFER
        print(f"    [TRANSFER] Training receiver on cross-property task...",
              flush=True)
        t_c = time.time()
        receiver_c = train_transfer_receiver(
            transfer_sender, data_t, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim_factored, seed)

        receiver_c.eval()
        with torch.no_grad():
            train_acc_c = evaluate_cross_property(
                transfer_sender, receiver_c, data_t, e_bins, f_bins,
                train_ids, DEVICE, n_rounds=50)
            holdout_acc_c = evaluate_cross_property(
                transfer_sender, receiver_c, data_t, e_bins, f_bins,
                holdout_ids, DEVICE, n_rounds=50)

        # Ablation for transfer
        abl_c = ablation_analysis(
            transfer_sender, receiver_c, data_t, e_bins, f_bins,
            holdout_ids, DEVICE, VOCAB_SIZE, N_HEADS, n_rounds=50)

        xfer_abl_str = ""
        if abl_c:
            xfer_abl_str = (f"  abl: A0={abl_c.get('drop_A_pos0',0):+.1%} "
                            f"A1={abl_c.get('drop_A_pos1',0):+.1%} "
                            f"B0={abl_c.get('drop_B_pos0',0):+.1%} "
                            f"B1={abl_c.get('drop_B_pos1',0):+.1%}")

        print(f"    Transfer: train={train_acc_c:.1%}, holdout={holdout_acc_c:.1%} "
              f"({time.time()-t_c:.0f}s){xfer_abl_str}", flush=True)

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
                'ablation': abl_a,
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
                'ablation': abl_c,
            },
            'time_sec': dt,
        })

        torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY: Phase 58b Cross-Property Task", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  Oracle: train={oracle_train_acc:.1%}, "
          f"holdout={oracle_holdout_acc:.1%}", flush=True)
    print(f"  Transfer sender: seed {best_transfer_seed}, "
          f"PosDis={best_transfer_posdis:.3f}", flush=True)

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

    # Key comparisons
    gap_fh = fact_holdouts.mean() - hol_holdouts.mean()
    gap_xh = xfer_holdouts.mean() - hol_holdouts.mean()
    print(f"\n  Factored vs holistic: {gap_fh:+.1%}", flush=True)
    print(f"  Transfer vs holistic: {gap_xh:+.1%}", flush=True)

    # Ablation summary for factored
    print(f"\n  === Ablation Analysis (factored, holdout) ===", flush=True)
    print(f"  Zeroing a position: accuracy drop (negative = position needed)",
          flush=True)
    abl_drops = {'A_pos0': [], 'A_pos1': [], 'B_pos0': [], 'B_pos1': []}
    for r in all_results:
        abl = r['factored']['ablation']
        if abl:
            for key in abl_drops:
                abl_drops[key].append(abl.get(f'drop_{key}', 0))
    for key in ['A_pos0', 'A_pos1', 'B_pos0', 'B_pos1']:
        vals = abl_drops[key]
        if vals:
            print(f"    {key}: mean drop = {np.mean(vals):+.1%} "
                  f"± {np.std(vals):.1%}", flush=True)

    # Ablation summary for transfer
    print(f"\n  === Ablation Analysis (transfer, holdout) ===", flush=True)
    xfer_drops = {'A_pos0': [], 'A_pos1': [], 'B_pos0': [], 'B_pos1': []}
    for r in all_results:
        abl = r['transfer']['ablation']
        if abl:
            for key in xfer_drops:
                xfer_drops[key].append(abl.get(f'drop_{key}', 0))
    for key in ['A_pos0', 'A_pos1', 'B_pos0', 'B_pos1']:
        vals = xfer_drops[key]
        if vals:
            print(f"    {key}: mean drop = {np.mean(vals):+.1%} "
                  f"± {np.std(vals):.1%}", flush=True)

    # Compositionality
    comp_seeds = sum(1 for pd in fact_posdis if pd > 0.4)
    print(f"\n  Factored compositional (PosDis>0.4): {comp_seeds}/20", flush=True)
    print(f"  Factored mean PosDis: {fact_posdis.mean():.3f} ± "
          f"{fact_posdis.std():.3f}", flush=True)

    # Save
    output = {
        'config': {
            'task': 'cross_property_comparison',
            'description': 'is e(A) > f(B)?',
            'oracle_train': float(oracle_train_acc),
            'oracle_holdout': float(oracle_holdout_acc),
            'transfer_sender_seed': best_transfer_seed,
            'transfer_sender_posdis': float(best_transfer_posdis),
            'transfer_sender_mi_matrix': xfer_comp['mi_matrix'].tolist(),
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
            'factored_vs_holistic_gap': float(gap_fh),
            'transfer_vs_holistic_gap': float(gap_xh),
        },
        'ablation_summary': {
            'factored': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                         for k, v in abl_drops.items() if v},
            'transfer': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                         for k, v in xfer_drops.items() if v},
        },
    }

    out_path = RESULTS_DIR / "phase58b_cross_property.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

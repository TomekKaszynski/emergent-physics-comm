"""
Phase 59: Emergent Message Structure
=====================================
Do agents discover the right vocabulary factorization when given overcomplete capacity?

Four conditions × 20 seeds, all on the 2-property ramp task (e + f comparison):
(a) 2×5 baseline — reproduction of Phase 54f (5²=25 messages for 25 combos)
(b) 4×5 overcomplete positions — (5⁴=625 messages for 25 combos)
(c) 2×10 overcomplete vocab — (10²=100 messages for 25 combos)
(d) 6×3 minimal symbols — (3⁶=729 messages for 25 combos)

Key questions:
1. In 4×5: do agents collapse to ~2 active positions?
2. In 2×10: do agents cluster symbols into ~5 effective categories?
3. In 6×3: do agents pair positions to compensate for small vocab?
4. Does overcomplete capacity help or hurt compositionality?

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase59_emergent_structure.py
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
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

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

CONDITIONS = [
    ('2x5',  2,  5),   # baseline
    ('4x5',  4,  5),   # overcomplete positions
    ('2x10', 2, 10),   # overcomplete vocab
    ('6x3',  6,  3),   # minimal symbols
]


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 54f / 58b)
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


class PropertyReceiver(nn.Module):
    """Two-head receiver for property comparison (e and f)."""
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

def evaluate_property(sender, receiver, data_t, e_bins, f_bins,
                      scene_ids, device, n_rounds=30):
    """Evaluate property comparison: e accuracy, f accuracy, both accuracy."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    ce = cf = cb = 0
    te = tf = tb = 0

    sender.eval()
    receiver.eval()

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)

        with torch.no_grad():
            msg_a, _ = sender(da)
            msg_b, _ = sender(db)
            pred_e, pred_f = receiver(msg_a, msg_b)

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])
        valid_e = (e_dev[ia] != e_dev[ib])
        valid_f = (f_dev[ia] != f_dev[ib])
        valid_both = valid_e & valid_f

        if valid_e.sum() > 0:
            ce += ((pred_e > 0)[valid_e] == label_e[valid_e]).sum().item()
            te += valid_e.sum().item()
        if valid_f.sum() > 0:
            cf += ((pred_f > 0)[valid_f] == label_f[valid_f]).sum().item()
            tf += valid_f.sum().item()
        if valid_both.sum() > 0:
            both_ok = ((pred_e > 0)[valid_both] == label_e[valid_both]) & \
                      ((pred_f > 0)[valid_both] == label_f[valid_both])
            cb += both_ok.sum().item()
            tb += valid_both.sum().item()

    return {
        'e_acc': ce / max(te, 1),
        'f_acc': cf / max(tf, 1),
        'both_acc': cb / max(tb, 1),
    }


def evaluate_property_population(sender, receivers, data_t, e_bins, f_bins,
                                 scene_ids, device, n_rounds=30):
    """Pick best receiver from population, return accuracies + best receiver."""
    best_both = -1
    best_r = None
    for r in receivers:
        acc = evaluate_property(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if acc['both_acc'] > best_both:
            best_both = acc['both_acc']
            best_r = r
    final = evaluate_property(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Message structure analysis
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


def analyze_messages(sender, data_t, e_bins, f_bins, device):
    """Comprehensive analysis of message structure.

    Returns per-position: entropy, effective vocab, MI with e/f, symbol usage.
    Also: PosDis, active PosDis, TopSim, unique message count.
    """
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i + BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, K)
    n_pos = all_tokens.shape[1]
    vocab_size = sender.vocab_size

    # Per-position analysis
    per_position = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=vocab_size)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(vocab_size) if vocab_size > 1 else 0.0

        eff_vocab = int(np.sum(probs > 0.05))
        mi_e = _mutual_information(all_tokens[:, p], e_bins)
        mi_f = _mutual_information(all_tokens[:, p], f_bins)

        per_position.append({
            'entropy': float(norm_ent),
            'eff_vocab': eff_vocab,
            'mi_e': float(mi_e),
            'mi_f': float(mi_f),
            'usage': counts.tolist(),
        })

    # MI matrix
    mi_matrix = np.array([[pp['mi_e'], pp['mi_f']] for pp in per_position])

    # PosDis (standard: averaged over ALL positions)
    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    # Active positions: MI > 0.1 for at least one property
    active_mask = mi_matrix.max(axis=1) > 0.1
    active_positions = int(active_mask.sum())

    # Active PosDis (only over positions with MI > 0.1)
    if active_positions >= 2:
        active_pos_dis = 0.0
        for p in range(n_pos):
            if active_mask[p]:
                sorted_mi = np.sort(mi_matrix[p])[::-1]
                if sorted_mi[0] > 1e-10:
                    active_pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        active_pos_dis /= active_positions
    elif active_positions == 1:
        # Single active position can't be compositional
        active_pos_dis = 0.0
    else:
        active_pos_dis = 0.0

    # Unique messages
    n_unique = len(set(map(tuple, all_tokens)))

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
        'active_pos_dis': float(active_pos_dis),
        'active_positions': active_positions,
        'n_unique_messages': n_unique,
        'topsim': float(topsim),
        'per_position': per_position,
        'mi_matrix': mi_matrix.tolist(),
    }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_property_population(n_heads, vocab_size, data_t, e_bins, f_bins,
                              train_ids, holdout_ids, device, seed):
    """Train sender+receivers on property comparison task with population IL.

    Returns (sender, receivers, nan_count).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    msg_dim = n_heads * vocab_size
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    sender = CompositionalSender(encoder, HIDDEN_DIM, vocab_size, n_heads).to(device)
    receivers = [PropertyReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    max_entropy = math.log(vocab_size)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_both = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        # Simultaneous IL: reset all receivers together
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

            # Entropy regularization (per head)
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

        # Evaluate every 40 epochs
        if (epoch + 1) % 40 == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                train_result, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=10)
                holdout_result, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: train={train_result['both_acc']:.1%}  "
                  f"holdout={holdout_result['both_acc']:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if holdout_result['both_acc'] > best_holdout_both:
                best_holdout_both = holdout_result['both_acc']
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}
                best_receiver_states = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]

    # Restore best
    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return sender, receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    print("=" * 70, flush=True)
    print("Phase 59: Emergent Message Structure", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Task: 2-property comparison (e + f)", flush=True)
    cond_str = ", ".join(f"{name}" for name, _, _ in CONDITIONS)
    print(f"  Conditions: {cond_str}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Comm epochs: {COMM_EPOCHS}", flush=True)

    # Load features
    print("\n  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins = load_cached_features(
        RESULTS_DIR / "phase54b_dino_features.pt")
    data_t = features.clone()
    print(f"  Features: {data_t.shape}", flush=True)

    # Splits
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # Capacity analysis
    print("\n  Capacity analysis:", flush=True)
    for name, nh, vs in CONDITIONS:
        capacity = vs ** nh
        print(f"    {name}: {nh} heads × {vs} symbols = {capacity} messages "
              f"(need 25)", flush=True)

    # Run all seeds × conditions
    all_results = []

    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n  {'='*60}", flush=True)
        print(f"  Seed {seed}", flush=True)
        print(f"  {'='*60}", flush=True)

        seed_result = {'seed': seed, 'conditions': {}}

        for cond_name, n_heads, vocab_size in CONDITIONS:
            t_cond = time.time()
            print(f"    [{cond_name}] {n_heads} heads × {vocab_size} symbols...",
                  flush=True)

            sender, receivers, nan_count = train_property_population(
                n_heads, vocab_size, data_t, e_bins, f_bins,
                train_ids, holdout_ids, DEVICE, seed)

            sender.eval()
            for r in receivers:
                r.eval()

            with torch.no_grad():
                train_eval, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, DEVICE, n_rounds=50)
                holdout_eval, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, DEVICE, n_rounds=50)

                analysis = analyze_messages(
                    sender, data_t, e_bins, f_bins, DEVICE)

            dt_cond = time.time() - t_cond
            eff_str = str([pp['eff_vocab'] for pp in analysis['per_position']])
            print(f"    {cond_name}: holdout e={holdout_eval['e_acc']:.1%} "
                  f"f={holdout_eval['f_acc']:.1%} "
                  f"both={holdout_eval['both_acc']:.1%}  "
                  f"PD={analysis['pos_dis']:.3f}  "
                  f"act={analysis['active_positions']}/{n_heads}  "
                  f"uniq={analysis['n_unique_messages']}  "
                  f"effV={eff_str}  ({dt_cond:.0f}s)", flush=True)

            seed_result['conditions'][cond_name] = {
                'n_heads': n_heads,
                'vocab_size': vocab_size,
                'train': train_eval,
                'holdout': holdout_eval,
                'analysis': analysis,
                'nan_count': nan_count,
                'time_sec': dt_cond,
            }

            torch.mps.empty_cache()

        dt_seed = time.time() - t_seed
        print(f"    Seed {seed} total: {dt_seed:.0f}s", flush=True)
        all_results.append(seed_result)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY: Phase 59 Emergent Message Structure", flush=True)
    print("=" * 70, flush=True)

    # Per-condition detail
    for cond_name, n_heads, vocab_size in CONDITIONS:
        e_accs = [r['conditions'][cond_name]['holdout']['e_acc']
                  for r in all_results]
        f_accs = [r['conditions'][cond_name]['holdout']['f_acc']
                  for r in all_results]
        both_accs = [r['conditions'][cond_name]['holdout']['both_acc']
                     for r in all_results]
        pd_vals = [r['conditions'][cond_name]['analysis']['pos_dis']
                   for r in all_results]
        apd_vals = [r['conditions'][cond_name]['analysis']['active_pos_dis']
                    for r in all_results]
        act_vals = [r['conditions'][cond_name]['analysis']['active_positions']
                    for r in all_results]
        uniq_vals = [r['conditions'][cond_name]['analysis']['n_unique_messages']
                     for r in all_results]
        ts_vals = [r['conditions'][cond_name]['analysis']['topsim']
                   for r in all_results]

        cap = vocab_size ** n_heads
        print(f"\n  --- {cond_name} ({n_heads}h × {vocab_size}v = {cap} capacity) ---",
              flush=True)
        print(f"    Holdout: e={np.mean(e_accs):.1%}±{np.std(e_accs):.1%}  "
              f"f={np.mean(f_accs):.1%}±{np.std(f_accs):.1%}  "
              f"both={np.mean(both_accs):.1%}±{np.std(both_accs):.1%}", flush=True)
        print(f"    PosDis={np.mean(pd_vals):.3f}±{np.std(pd_vals):.3f}  "
              f"ActivePD={np.mean(apd_vals):.3f}±{np.std(apd_vals):.3f}  "
              f"TopSim={np.mean(ts_vals):.3f}±{np.std(ts_vals):.3f}", flush=True)
        print(f"    Active: {np.mean(act_vals):.1f}±{np.std(act_vals):.1f} "
              f"of {n_heads}  "
              f"Unique msgs: {np.mean(uniq_vals):.1f}±{np.std(uniq_vals):.1f} "
              f"(need 25)", flush=True)

        # Per-position averages
        print(f"    Per-position (averaged over {len(SEEDS)} seeds):", flush=True)
        print(f"      {'pos':>4} | {'H(norm)':>8} | {'effV':>5} | "
              f"{'MI(e)':>7} | {'MI(f)':>7} | {'specializes':>12}", flush=True)
        print(f"      {'-'*4}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}",
              flush=True)
        for p in range(n_heads):
            ents = [r['conditions'][cond_name]['analysis']['per_position'][p]['entropy']
                    for r in all_results]
            evs = [r['conditions'][cond_name]['analysis']['per_position'][p]['eff_vocab']
                   for r in all_results]
            mis_e = [r['conditions'][cond_name]['analysis']['per_position'][p]['mi_e']
                     for r in all_results]
            mis_f = [r['conditions'][cond_name]['analysis']['per_position'][p]['mi_f']
                     for r in all_results]
            # Determine specialization
            me, mf = np.mean(mis_e), np.mean(mis_f)
            if me > 0.1 and mf > 0.1:
                if me > mf * 1.5:
                    spec = "e"
                elif mf > me * 1.5:
                    spec = "f"
                else:
                    spec = "both"
            elif me > 0.1:
                spec = "e"
            elif mf > 0.1:
                spec = "f"
            else:
                spec = "inactive"
            print(f"      pos{p} |   {np.mean(ents):.3f}  |  {np.mean(evs):.1f}  | "
                  f" {np.mean(mis_e):.3f}  |  {np.mean(mis_f):.3f}  | "
                  f"{spec:>12}", flush=True)

    # Comparison table
    print(f"\n  === Comparison Table ===", flush=True)
    print(f"  {'Cond':>5} | {'Both':>12} | {'PosDis':>12} | "
          f"{'ActPD':>12} | {'Active':>8} | {'Unique':>7} | {'TopSim':>8}",
          flush=True)
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-"
          f"{'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}", flush=True)

    for cond_name, n_heads, vocab_size in CONDITIONS:
        both = [r['conditions'][cond_name]['holdout']['both_acc']
                for r in all_results]
        pd = [r['conditions'][cond_name]['analysis']['pos_dis']
              for r in all_results]
        apd = [r['conditions'][cond_name]['analysis']['active_pos_dis']
               for r in all_results]
        act = [r['conditions'][cond_name]['analysis']['active_positions']
               for r in all_results]
        uniq = [r['conditions'][cond_name]['analysis']['n_unique_messages']
                for r in all_results]
        ts = [r['conditions'][cond_name]['analysis']['topsim']
              for r in all_results]
        print(f"  {cond_name:>5} | "
              f"{np.mean(both):.1%}±{np.std(both):.1%} | "
              f"{np.mean(pd):.3f}±{np.std(pd):.3f} | "
              f"{np.mean(apd):.3f}±{np.std(apd):.3f} | "
              f"{np.mean(act):4.1f}/{n_heads}   | "
              f"{np.mean(uniq):5.1f} | "
              f"{np.mean(ts):.3f}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════

    output = {
        'config': {
            'task': 'property_comparison',
            'conditions': [{'name': n, 'n_heads': nh, 'vocab_size': vs,
                           'capacity': vs ** nh}
                          for n, nh, vs in CONDITIONS],
            'comm_epochs': COMM_EPOCHS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'entropy_threshold': ENTROPY_THRESHOLD,
            'entropy_coef': ENTROPY_COEF,
            'n_seeds': len(SEEDS),
        },
        'per_seed': all_results,
        'summary': {},
    }

    for cond_name, n_heads, vocab_size in CONDITIONS:
        e = [r['conditions'][cond_name]['holdout']['e_acc'] for r in all_results]
        f = [r['conditions'][cond_name]['holdout']['f_acc'] for r in all_results]
        both = [r['conditions'][cond_name]['holdout']['both_acc']
                for r in all_results]
        pd = [r['conditions'][cond_name]['analysis']['pos_dis']
              for r in all_results]
        apd = [r['conditions'][cond_name]['analysis']['active_pos_dis']
               for r in all_results]
        act = [r['conditions'][cond_name]['analysis']['active_positions']
               for r in all_results]
        uniq = [r['conditions'][cond_name]['analysis']['n_unique_messages']
                for r in all_results]
        ts = [r['conditions'][cond_name]['analysis']['topsim']
              for r in all_results]

        output['summary'][cond_name] = {
            'e_holdout_mean': float(np.mean(e)),
            'e_holdout_std': float(np.std(e)),
            'f_holdout_mean': float(np.mean(f)),
            'f_holdout_std': float(np.std(f)),
            'both_holdout_mean': float(np.mean(both)),
            'both_holdout_std': float(np.std(both)),
            'pos_dis_mean': float(np.mean(pd)),
            'pos_dis_std': float(np.std(pd)),
            'active_pos_dis_mean': float(np.mean(apd)),
            'active_pos_dis_std': float(np.std(apd)),
            'active_positions_mean': float(np.mean(act)),
            'active_positions_std': float(np.std(act)),
            'n_unique_mean': float(np.mean(uniq)),
            'n_unique_std': float(np.std(uniq)),
            'topsim_mean': float(np.mean(ts)),
            'topsim_std': float(np.std(ts)),
        }

    out_path = RESULTS_DIR / "phase59_emergent_structure.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

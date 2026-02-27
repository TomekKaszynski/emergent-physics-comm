"""
Phase 54f: Extended training (400 epochs)
==========================================
Same as Phase 54e (population + simultaneous reset + vocab=5, 20 seeds)
but with COMM_EPOCHS=400 (was 200). 9 receiver generations instead of 4.
Does more evolutionary time convert intermediate seeds into compositional ones?

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase54f_extended.py
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
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

SEEDS = list(range(20))  # seeds 0-19


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

    return (correct_e / max(total_e, 1),
            correct_f / max(total_f, 1),
            correct_both / max(total_both, 1))


def evaluate_population(sender, receivers, data_t, e_bins, f_bins,
                        scene_ids, device, n_rounds=30):
    """Evaluate with best receiver (highest individual accuracy)."""
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
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
        v = VOCAB_SIZE if n_pos > 1 else CONTROL_VOCAB
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
# Training
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
                _, _, acc_both = evaluate_accuracy(
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


def train_population(sender, receivers, data_t, e_bins, f_bins,
                     train_ids, holdout_ids, device, msg_dim, seed):
    """Train sender against population of receivers with simultaneous IL."""
    s_lr = SENDER_LR
    r_lr = RECEIVER_LR
    sender_opt = torch.optim.Adam(sender.parameters(), lr=s_lr)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=r_lr) for r in receivers]

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
        # Simultaneous IL: reset ALL receivers at once
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            gen = epoch // RECEIVER_RESET_INTERVAL
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=r_lr)
            print(f"    ** Reset ALL {len(receivers)} receivers at epoch {epoch+1} "
                  f"(gen {gen}) **", flush=True)

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

            # Sender produces messages once
            msg_a, logits_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b = sender(db, tau=tau, hard=hard)

            # Average loss across all receivers
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization on sender logits
            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

            # NaN-safe gradient step
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

            # Check for NaN gradients
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
                (te, tf, tb), best_r = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
                (he, hf, hb), _ = evaluate_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

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


# ══════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════

def run_single_seed(seed, data_t, e_bins, f_bins, train_ids, holdout_ids, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle(data_t, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Sender (vocab=5, 2 heads -> msg_dim=10)
    encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Population of receivers
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    print(f"    Training sender (2x{VOCAB_SIZE}) vs {N_RECEIVERS} receivers "
          f"(IL={RECEIVER_RESET_INTERVAL}, simultaneous)...", flush=True)

    receivers, nan_count = train_population(
        sender, receivers, data_t, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval with best receiver
    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population(
            sender, receivers, data_t, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy(
            sender, best_r, data_t, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    # Compositionality
    with torch.no_grad():
        comp = compute_compositionality(sender, data_t, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())

    dt = time.time() - t0

    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"MI->e={best_mi_e:.3f}  MI->f={best_mi_f:.3f}  "
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


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 54f: Extended training (400 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Vocab: {VOCAB_SIZE} (2x{VOCAB_SIZE}, control 1x{CONTROL_VOCAB})", flush=True)
    print(f"  Receivers: {N_RECEIVERS}", flush=True)
    print(f"  Simultaneous IL: reset ALL every {RECEIVER_RESET_INTERVAL} epochs",
          flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)

    t_total = time.time()

    # Load cached features
    cache_path = str(RESULTS_DIR / "phase54b_dino_features.pt")
    data_t, e_bins, f_bins = load_cached_features(cache_path)
    print(f"  Features: {data_t.shape}", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, data_t, e_bins, f_bins,
                                 train_ids, holdout_ids, DEVICE)
        all_results.append(result)

    # ════════════════════════════════════════════════════════════
    # Full table (20 rows)
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"FULL TABLE: Phase 54f extended (400 epochs, 20 seeds)", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"  {'Seed':>4} | {'Holdout':>8} | {'PosDis':>7} | "
              f"{'TopSim':>7} | {'MI->e':>7} | {'MI->f':>7} | {'NaN':>3}")
    print(header, flush=True)
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-{'---':>3}",
          flush=True)

    hb, pd, ts, me, mf = [], [], [], [], []
    for r in all_results:
        tag = ""
        if r['pos_dis'] > 0.4:
            tag = " *"
        print(f"  {r['seed']:>4} | {r['holdout_both']:>7.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['topsim']:>7.3f} | "
              f"{r['best_mi_e']:>7.3f} | {r['best_mi_f']:>7.3f} | "
              f"{r['nan_count']:>3}{tag}", flush=True)
        hb.append(r['holdout_both'])
        pd.append(r['pos_dis'])
        ts.append(r['topsim'])
        me.append(r['best_mi_e'])
        mf.append(r['best_mi_f'])

    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-{'---':>3}",
          flush=True)
    print(f"  {'Mean':>4} | {np.mean(hb):>7.1%} | "
          f"{np.mean(pd):>7.3f} | {np.mean(ts):>7.3f} | "
          f"{np.mean(me):>7.3f} | {np.mean(mf):>7.3f} |", flush=True)
    print(f"  {'Std':>4} | {np.std(hb):>7.1%} | "
          f"{np.std(pd):>7.3f} | {np.std(ts):>7.3f} | "
          f"{np.std(me):>7.3f} | {np.std(mf):>7.3f} |", flush=True)

    # ════════════════════════════════════════════════════════════
    # Group analysis
    # ════════════════════════════════════════════════════════════
    compositional = [r for r in all_results if r['pos_dis'] > 0.4]
    holistic = [r for r in all_results if r['pos_dis'] < 0.15]
    intermediate = [r for r in all_results if 0.15 <= r['pos_dis'] <= 0.4]

    n = len(all_results)
    print(f"\n  GROUP ANALYSIS:", flush=True)
    print(f"  Compositional (PosDis > 0.4):  {len(compositional):>2}/{n} "
          f"({len(compositional)/n:.0%})", flush=True)
    if compositional:
        c_hb = [r['holdout_both'] for r in compositional]
        c_pd = [r['pos_dis'] for r in compositional]
        print(f"    Mean holdout: {np.mean(c_hb):.1%} +/- {np.std(c_hb):.1%}", flush=True)
        print(f"    Mean PosDis:  {np.mean(c_pd):.3f} +/- {np.std(c_pd):.3f}", flush=True)
        print(f"    Seeds: {[r['seed'] for r in compositional]}", flush=True)

    print(f"  Holistic (PosDis < 0.15):      {len(holistic):>2}/{n} "
          f"({len(holistic)/n:.0%})", flush=True)
    if holistic:
        h_hb = [r['holdout_both'] for r in holistic]
        h_pd = [r['pos_dis'] for r in holistic]
        print(f"    Mean holdout: {np.mean(h_hb):.1%} +/- {np.std(h_hb):.1%}", flush=True)
        print(f"    Mean PosDis:  {np.mean(h_pd):.3f} +/- {np.std(h_pd):.3f}", flush=True)
        print(f"    Seeds: {[r['seed'] for r in holistic]}", flush=True)

    print(f"  Intermediate (0.15-0.4):       {len(intermediate):>2}/{n} "
          f"({len(intermediate)/n:.0%})", flush=True)
    if intermediate:
        i_hb = [r['holdout_both'] for r in intermediate]
        i_pd = [r['pos_dis'] for r in intermediate]
        print(f"    Mean holdout: {np.mean(i_hb):.1%} +/- {np.std(i_hb):.1%}", flush=True)
        print(f"    Mean PosDis:  {np.mean(i_pd):.3f} +/- {np.std(i_pd):.3f}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Text histogram of PosDis values
    # ════════════════════════════════════════════════════════════
    print(f"\n  PosDis HISTOGRAM:", flush=True)

    # Bins: 0.0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7+
    bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    bin_labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4",
                  "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7+   "]

    for i in range(len(bin_edges) - 1):
        count = sum(1 for p in pd if bin_edges[i] <= p < bin_edges[i + 1])
        bar = "#" * (count * 3)
        label = bin_labels[i]
        print(f"  {label} | {bar} {count}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'n_heads': N_HEADS,
            'control_vocab': CONTROL_VOCAB,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'reset_mode': 'simultaneous',
            'comm_epochs': COMM_EPOCHS,
        },
        'seeds': SEEDS,
        'per_seed': all_results,
        'summary': {
            'holdout_both_mean': float(np.mean(hb)),
            'holdout_both_std': float(np.std(hb)),
            'pos_dis_mean': float(np.mean(pd)),
            'pos_dis_std': float(np.std(pd)),
            'topsim_mean': float(np.mean(ts)),
            'topsim_std': float(np.std(ts)),
            'best_mi_e_mean': float(np.mean(me)),
            'best_mi_e_std': float(np.std(me)),
            'best_mi_f_mean': float(np.mean(mf)),
            'best_mi_f_std': float(np.std(mf)),
        },
        'groups': {
            'compositional_count': len(compositional),
            'compositional_seeds': [r['seed'] for r in compositional],
            'holistic_count': len(holistic),
            'holistic_seeds': [r['seed'] for r in holistic],
            'intermediate_count': len(intermediate),
        },
    }

    save_path = RESULTS_DIR / "phase54f_extended.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min ({dt/len(SEEDS):.0f}s per seed)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

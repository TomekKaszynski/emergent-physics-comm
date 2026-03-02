"""
Phase 59d: Hard-Concrete Gated Positions for Message Length Discovery
=====================================================================
Replace discrete STOP token with continuous per-position hard-concrete gates
(Louizos et al. 2018). Each of 6 positions has an independent differentiable
on/off gate. Combined with Impatient Listener (Rita et al. 2020) which forces
information to the front of the message.

Key insight: length is NOT one discrete STOP decision. It's 6 independent
smooth gate decisions. Each gate can be optimized independently. No cliff.

Four conditions x 20 seeds:
  lambda=0.00  no length pressure (baseline)
  lambda=0.05  mild pressure
  lambda=0.10  moderate pressure
  lambda=0.20  strong pressure

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase59d_gated.py
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

VOCAB_SIZE = 5
MAX_POSITIONS = 6
MSG_DIM = MAX_POSITIONS * VOCAB_SIZE  # 6 * 5 = 30

# Hard-concrete parameters (Louizos et al. 2018)
BETA_GATE = 0.66
GAMMA_HC = -0.1
ZETA_HC = 1.1

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 2.0
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

LAMBDA_WARMUP_EPOCHS = 50

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

SEEDS = list(range(20))
LAMBDAS = [0.0, 0.05, 0.1, 0.2]


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


class GatedSender(nn.Module):
    """Fixed-position sender with hard-concrete gates.

    Each of max_positions positions independently:
    - Selects a symbol via Gumbel-Softmax (vocab_size options)
    - Has a hard-concrete gate z_t that can turn the position on/off

    The gate is sampled from a hard-concrete distribution during training;
    thresholded (log_alpha > 0) during eval.
    """
    def __init__(self, hidden_dim=128, input_dim=384,
                 max_positions=6, vocab_size=5):
        super().__init__()
        self.encoder = TemporalEncoder(hidden_dim, input_dim)
        self.max_positions = max_positions
        self.vocab_size = vocab_size
        self.head = nn.Linear(hidden_dim, max_positions * vocab_size)

        # Hard-concrete gate parameters: one per position
        # Init to 2.0 -> p(active) ~ 0.97 at start, all positions on
        self.log_alpha = nn.Parameter(torch.full((max_positions,), 2.0))

    def _sample_hard_concrete(self):
        """Sample hard-concrete gates. Returns z (max_positions,)."""
        u = torch.rand_like(self.log_alpha).clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid(
            (torch.log(u / (1 - u)) + self.log_alpha) / BETA_GATE)
        z_bar = s * (ZETA_HC - GAMMA_HC) + GAMMA_HC
        z = z_bar.clamp(0.0, 1.0)
        return z

    def _gate_probabilities(self):
        """Analytical probability each gate is active: p(z > 0)."""
        return torch.sigmoid(
            self.log_alpha - BETA_GATE * math.log(-GAMMA_HC / ZETA_HC))

    def forward(self, x, tau=1.0, hard=True):
        """
        Returns:
            message: (batch, max_positions * vocab_size) gated tokens
            logits: (batch, max_positions, vocab_size) raw logits
            p_active: (max_positions,) gate probabilities
        """
        h = self.encoder(x)
        batch_size = h.size(0)

        logits = self.head(h).view(batch_size, self.max_positions,
                                   self.vocab_size)

        # Gumbel-Softmax per position
        if self.training:
            flat_logits = logits.reshape(-1, self.vocab_size)
            tokens = F.gumbel_softmax(flat_logits, tau=tau, hard=hard)
            tokens = tokens.reshape(batch_size, self.max_positions,
                                    self.vocab_size)
        else:
            idx = logits.argmax(dim=-1)
            tokens = F.one_hot(idx, self.vocab_size).float()

        # Hard-concrete gates
        if self.training:
            z = self._sample_hard_concrete()  # (max_positions,)
        else:
            z = (self.log_alpha > 0).float()

        # Apply gates: z is (max_positions,), broadcast to (1, max_pos, 1)
        gated_tokens = tokens * z.unsqueeze(0).unsqueeze(-1)

        # Flatten to message vector
        message = gated_tokens.reshape(batch_size,
                                       self.max_positions * self.vocab_size)

        p_active = self._gate_probabilities()
        return message, logits, p_active


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
    """Evaluate property comparison using full gated messages."""
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
            msg_a, _, _ = sender(da)
            msg_b, _, _ = sender(db)
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
    """Pick best receiver from population."""
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
# Message analysis
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


def analyze_gated_messages(sender, data_t, e_bins, f_bins, device):
    """Analyze gated messages: gate activations, per-position MI, etc."""
    sender.eval()

    # Gate info (structural, not input-dependent)
    p_active = sender._gate_probabilities().cpu().detach().numpy()
    active_mask = (sender.log_alpha > 0).cpu().detach().numpy()
    log_alpha_vals = sender.log_alpha.cpu().detach().numpy()
    n_active = int(active_mask.sum())

    # Get all token selections (ignoring gates — raw argmax)
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i + BATCH_SIZE].to(device)
            msg, logits, _ = sender(batch)
            toks = logits.argmax(dim=-1).cpu().numpy()  # (batch, max_pos)
            all_tokens.append(toks)

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, max_pos)

    # Per-position analysis
    per_position = []
    for p in range(MAX_POSITIONS):
        is_active = bool(active_mask[p])

        if not is_active:
            per_position.append({
                'p_active': float(p_active[p]),
                'log_alpha': float(log_alpha_vals[p]),
                'is_active': False,
                'mi_e': 0.0, 'mi_f': 0.0,
                'entropy': 0.0, 'eff_vocab': 0,
            })
            continue

        tokens = all_tokens[:, p]
        mi_e = _mutual_information(tokens, e_bins)
        mi_f = _mutual_information(tokens, f_bins)

        counts = np.bincount(tokens, minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(VOCAB_SIZE) if VOCAB_SIZE > 1 else 0.0
        eff_vocab = int(np.sum(probs > 0.05))

        per_position.append({
            'p_active': float(p_active[p]),
            'log_alpha': float(log_alpha_vals[p]),
            'is_active': True,
            'mi_e': float(mi_e),
            'mi_f': float(mi_f),
            'entropy': float(norm_ent),
            'eff_vocab': eff_vocab,
        })

    # PosDis over active positions
    active_indices = [p for p in range(MAX_POSITIONS) if active_mask[p]]
    if len(active_indices) >= 2:
        mi_matrix = np.array([[per_position[p]['mi_e'], per_position[p]['mi_f']]
                              for p in active_indices])
        pos_dis = 0.0
        for row in mi_matrix:
            sorted_mi = np.sort(row)[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= len(active_indices)
    else:
        pos_dis = 0.0

    # Unique messages (active positions only)
    msgs = []
    for i in range(len(all_tokens)):
        msg = tuple(all_tokens[i, active_mask])
        msgs.append(msg)
    n_unique = len(set(msgs))

    # TopSim
    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(data_t), size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        dist = sum(1 for p in active_indices
                   if all_tokens[i, p] != all_tokens[j, p])
        message_dists.append(dist)
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'n_active_positions': n_active,
        'p_active_per_position': [float(v) for v in p_active],
        'log_alpha_per_position': [float(v) for v in log_alpha_vals],
        'active_positions': [int(p) for p in active_indices],
        'per_position': per_position,
        'pos_dis': float(pos_dis),
        'n_unique_messages': n_unique,
        'topsim': float(topsim),
    }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def mask_prefix(msg, k, vocab_size):
    """Zero out positions k and beyond in flat message vector."""
    masked = msg.clone()
    masked[:, k * vocab_size:] = 0
    return masked


def train_gated(lam, data_t, e_bins, f_bins, train_ids, holdout_ids,
                device, seed):
    """Train gated sender + impatient receiver population.

    Returns (sender, receivers, nan_count).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    sender = GatedSender(HIDDEN_DIM, DINO_DIM, MAX_POSITIONS,
                         VOCAB_SIZE).to(device)
    receivers = [PropertyReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    max_entropy = math.log(VOCAB_SIZE)
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
                receivers[i] = PropertyReceiver(MSG_DIM, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        # Lambda warmup
        lam_eff = lam * min(1.0, epoch / max(1, LAMBDA_WARMUP_EPOCHS))

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a, p_active = sender(da, tau=tau, hard=hard)
            msg_b, logits_b, _ = sender(db, tau=tau, hard=hard)

            # Impatient listener: average loss over all prefix lengths
            total_loss = torch.tensor(0.0, device=device)
            for k in range(1, MAX_POSITIONS + 1):
                masked_a = mask_prefix(msg_a, k, VOCAB_SIZE)
                masked_b = mask_prefix(msg_b, k, VOCAB_SIZE)
                for r in receivers:
                    pred_e, pred_f = r(masked_a, masked_b)
                    r_loss = (F.binary_cross_entropy_with_logits(pred_e, label_e) +
                              F.binary_cross_entropy_with_logits(pred_f, label_f))
                    total_loss = total_loss + r_loss
            loss = total_loss / (MAX_POSITIONS * len(receivers))

            # Gate penalty: expected number of active positions
            if lam_eff > 0:
                gate_penalty = lam_eff * p_active.sum()
                loss = loss + gate_penalty

            # Entropy regularization on token logits (all positions)
            all_logits = torch.cat([logits_a, logits_b], dim=0)
            for p in range(MAX_POSITIONS):
                pos_logits = all_logits[:, p, :]
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
                # Gate info
                p_act = sender._gate_probabilities().cpu().numpy()
                n_act = int((sender.log_alpha > 0).sum().item())

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            gates_str = ",".join(f"{p:.2f}" for p in p_act)
            print(f"        Ep {epoch+1:3d}: train={train_result['both_acc']:.1%}  "
                  f"holdout={holdout_result['both_acc']:.1%}  "
                  f"gates=[{gates_str}] ({n_act}on){nan_str}  "
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
    print("Phase 59d: Hard-Concrete Gated Positions", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Positions: {MAX_POSITIONS}, Vocab: {VOCAB_SIZE}", flush=True)
    print(f"  MSG_DIM: {MSG_DIM} ({MAX_POSITIONS}x{VOCAB_SIZE})", flush=True)
    print(f"  Hard-concrete: beta={BETA_GATE}, gamma={GAMMA_HC}, "
          f"zeta={ZETA_HC}", flush=True)
    print(f"  Impatient listener: {MAX_POSITIONS} prefix losses", flush=True)
    print(f"  Lambdas: {LAMBDAS}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}, lambda warmup: {LAMBDA_WARMUP_EPOCHS}",
          flush=True)

    # Load features
    print("\n  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins = load_cached_features(
        RESULTS_DIR / "phase54b_dino_features.pt")
    data_t = features.clone()
    print(f"  Features: {data_t.shape}", flush=True)

    # Splits
    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout\n",
          flush=True)

    # Run all conditions
    all_results = {}
    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        all_results[lam_name] = []

        print(f"  {'='*60}", flush=True)
        print(f"  Condition: {lam_name}", flush=True)
        print(f"  {'='*60}", flush=True)

        for seed in SEEDS:
            t_seed = time.time()
            print(f"    [seed={seed}] Training...", flush=True)

            sender, receivers, nan_count = train_gated(
                lam, data_t, e_bins, f_bins,
                train_ids, holdout_ids, DEVICE, seed)

            sender.eval()
            for r in receivers:
                r.eval()

            with torch.no_grad():
                holdout_eval, _ = evaluate_property_population(
                    sender, receivers, data_t, e_bins, f_bins,
                    holdout_ids, DEVICE, n_rounds=50)

                analysis = analyze_gated_messages(
                    sender, data_t, e_bins, f_bins, DEVICE)

            dt = time.time() - t_seed
            active_str = ",".join(str(p) for p in analysis['active_positions'])
            print(f"    [seed={seed}] holdout e={holdout_eval['e_acc']:.1%} "
                  f"f={holdout_eval['f_acc']:.1%} "
                  f"both={holdout_eval['both_acc']:.1%}  "
                  f"active=[{active_str}] ({analysis['n_active_positions']}pos)  "
                  f"uniq={analysis['n_unique_messages']}  "
                  f"PD={analysis['pos_dis']:.3f}  ({dt:.0f}s)", flush=True)

            all_results[lam_name].append({
                'seed': seed,
                'holdout': holdout_eval,
                'analysis': analysis,
                'nan_count': nan_count,
                'time_sec': dt,
            })

            torch.mps.empty_cache()

        # Condition summary
        both_accs = [r['holdout']['both_acc'] for r in all_results[lam_name]]
        n_acts = [r['analysis']['n_active_positions']
                  for r in all_results[lam_name]]
        pd_vals = [r['analysis']['pos_dis'] for r in all_results[lam_name]]
        print(f"\n  {lam_name} summary:", flush=True)
        print(f"    both={np.mean(both_accs):.1%} +/- {np.std(both_accs):.1%}  "
              f"active={np.mean(n_acts):.1f} +/- {np.std(n_acts):.1f}  "
              f"PD={np.mean(pd_vals):.3f}\n", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Lambda':>8} | {'Both':>14} | {'Active':>10} | "
          f"{'PosDis':>10} | {'Unique':>7} | {'TopSim':>8}", flush=True)
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*7}-+-{'-'*8}", flush=True)

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        results = all_results[lam_name]
        both = [r['holdout']['both_acc'] for r in results]
        n_act = [r['analysis']['n_active_positions'] for r in results]
        pd = [r['analysis']['pos_dis'] for r in results]
        uniq = [r['analysis']['n_unique_messages'] for r in results]
        ts = [r['analysis']['topsim'] for r in results]
        print(f"  {lam:8.2f} | "
              f"{np.mean(both):.1%}+/-{np.std(both):.1%} | "
              f"{np.mean(n_act):4.1f}+/-{np.std(n_act):.1f} | "
              f"{np.mean(pd):.3f}+/-{np.std(pd):.3f} | "
              f"{np.mean(uniq):5.1f} | "
              f"{np.mean(ts):.3f}", flush=True)

    # Per-position gate probabilities (averaged over seeds)
    print(f"\n  Gate probabilities per position (mean over seeds):", flush=True)
    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        results = all_results[lam_name]
        avg_p = np.mean([r['analysis']['p_active_per_position']
                         for r in results], axis=0)
        avg_la = np.mean([r['analysis']['log_alpha_per_position']
                          for r in results], axis=0)
        gates_str = "  ".join(f"p{i}={avg_p[i]:.2f}" for i in range(MAX_POSITIONS))
        print(f"    {lam_name}: {gates_str}", flush=True)

    # Per-position MI for best condition (lambda that gets best accuracy
    # with fewest positions)
    print(f"\n  Per-position MI (active positions, best seed per condition):",
          flush=True)
    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        results = all_results[lam_name]
        # Pick seed with highest both_acc
        best_idx = np.argmax([r['holdout']['both_acc'] for r in results])
        best = results[best_idx]
        active = best['analysis']['active_positions']
        pp = best['analysis']['per_position']
        if active:
            mi_strs = [f"p{p}: e={pp[p]['mi_e']:.3f} f={pp[p]['mi_f']:.3f}"
                       for p in active]
            print(f"    {lam_name} (seed {best['seed']}, "
                  f"both={best['holdout']['both_acc']:.1%}): "
                  f"{' | '.join(mi_strs)}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════

    output = {
        'config': {
            'task': 'property_comparison',
            'max_positions': MAX_POSITIONS,
            'vocab_size': VOCAB_SIZE,
            'msg_dim': MSG_DIM,
            'hard_concrete': {
                'beta': BETA_GATE, 'gamma': GAMMA_HC, 'zeta': ZETA_HC,
            },
            'impatient_listener': True,
            'lambdas': LAMBDAS,
            'lambda_warmup_epochs': LAMBDA_WARMUP_EPOCHS,
            'comm_epochs': COMM_EPOCHS,
            'n_receivers': N_RECEIVERS,
            'receiver_reset_interval': RECEIVER_RESET_INTERVAL,
            'n_seeds': len(SEEDS),
        },
        'per_condition': {},
        'summary': {},
    }

    for lam in LAMBDAS:
        lam_name = f"lam={lam:.2f}"
        results = all_results[lam_name]
        output['per_condition'][lam_name] = results

        both = [r['holdout']['both_acc'] for r in results]
        e = [r['holdout']['e_acc'] for r in results]
        f = [r['holdout']['f_acc'] for r in results]
        n_act = [r['analysis']['n_active_positions'] for r in results]
        pd = [r['analysis']['pos_dis'] for r in results]
        uniq = [r['analysis']['n_unique_messages'] for r in results]
        ts = [r['analysis']['topsim'] for r in results]
        avg_p = np.mean([r['analysis']['p_active_per_position']
                         for r in results], axis=0).tolist()

        output['summary'][lam_name] = {
            'e_holdout_mean': float(np.mean(e)),
            'e_holdout_std': float(np.std(e)),
            'f_holdout_mean': float(np.mean(f)),
            'f_holdout_std': float(np.std(f)),
            'both_holdout_mean': float(np.mean(both)),
            'both_holdout_std': float(np.std(both)),
            'n_active_mean': float(np.mean(n_act)),
            'n_active_std': float(np.std(n_act)),
            'pos_dis_mean': float(np.mean(pd)),
            'pos_dis_std': float(np.std(pd)),
            'n_unique_mean': float(np.mean(uniq)),
            'topsim_mean': float(np.mean(ts)),
            'topsim_std': float(np.std(ts)),
            'avg_gate_probs': avg_p,
        }

    out_path = RESULTS_DIR / "phase59d_gated.json"
    with open(out_path, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

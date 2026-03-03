"""
Phase 63: Novel Property Introduction — Protocol Adaptation Mid-Training
=========================================================================
Train 4 agents (1 frame each, 2 positions, vocab=5) on 2 properties for
200 epochs, then introduce a THIRD property and continue for 200 more.

Third property = "interaction": compare balls by e_bin + f_bin (sum).
This requires information about BOTH elasticity AND friction, forcing
agents to encode a conjunction they previously encoded separately.

Four conditions × 15 seeds:
  (a) CURRICULUM: 2 props for 200ep, then add 3rd for 200ep
  (b) JOINT: all 3 props from start for 400ep
  (c) TWO_ONLY: only 2 props for 400ep (baseline)
  (d) INTERACTION_ONLY: only interaction for 400ep

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase63_adaptation.py
"""

import time
import json
import math
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

HIDDEN_DIM = 128
DINO_DIM = 384
BATCH_SIZE = 64

N_AGENTS = 4
AGENT_FRAMES = [[0], [1], [2], [3]]
N_POSITIONS = 2
VOCAB_SIZE = 5
MSG_DIM = N_POSITIONS * VOCAB_SIZE  # 10 per sender

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

COMM_EPOCHS = 400
SWITCH_EPOCH = 200  # when curriculum adds the 3rd property
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 2.0
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

N_SEEDS = 15
SEEDS = list(range(N_SEEDS))

CONDITIONS = ['curriculum', 'joint', 'two_only', 'interaction_only']

# Checkpoints for curriculum tracking
CURRICULUM_CHECKPOINTS = [199, 210, 220, 250, 300, 350, 399]


# ══════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Encodes 1+ frames into a hidden vector."""
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


class FixedSender(nn.Module):
    """Fixed-length Gumbel-Softmax sender."""
    def __init__(self, hidden_dim=128, input_dim=384,
                 n_positions=2, vocab_size=5):
        super().__init__()
        self.encoder = TemporalEncoder(hidden_dim, input_dim)
        self.n_positions = n_positions
        self.vocab_size = vocab_size
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


class ThreeHeadReceiver(nn.Module):
    """Receiver with 2 or 3 output heads for property comparison.

    Starts with e and f heads. The interaction head can be added later
    via add_interaction_head() without resetting shared layers.
    """
    def __init__(self, input_dim, hidden_dim=128, include_interaction=False):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim // 2, 1)
        self.friction_head = nn.Linear(hidden_dim // 2, 1)
        self.interaction_head = None
        if include_interaction:
            self.interaction_head = nn.Linear(hidden_dim // 2, 1)

    def add_interaction_head(self):
        """Add interaction head without touching existing weights."""
        if self.interaction_head is None:
            self.interaction_head = nn.Linear(
                self.shared[-2].out_features, 1)
            # Initialize on same device as existing params
            device = self.elast_head.weight.device
            self.interaction_head = self.interaction_head.to(device)

    def forward(self, x):
        h = self.shared(x)
        pred_e = self.elast_head(h).squeeze(-1)
        pred_f = self.friction_head(h).squeeze(-1)
        pred_i = None
        if self.interaction_head is not None:
            pred_i = self.interaction_head(h).squeeze(-1)
        return pred_e, pred_f, pred_i


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def load_data():
    """Load cached DINOv2 features."""
    cache_path = RESULTS_DIR / "phase54b_dino_features.pt"
    data = torch.load(cache_path, weights_only=False)
    features = data['features'][:, :4, :]  # (300, 4, 384)
    e_bins = data['e_bins']  # (300,) int64, values 0-4
    f_bins = data['f_bins']  # (300,) int64, values 0-4

    # Interaction score = e_bin + f_bin (range 0-8)
    interaction_scores = e_bins + f_bins  # (300,) int64

    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)

    return (features, e_bins, f_bins, interaction_scores,
            np.array(train_ids), np.array(holdout_ids))


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

def evaluate_with_receiver(senders, receiver, features, e_bins, f_bins,
                           interaction_scores, scene_ids, device,
                           include_interaction=False, n_rounds=30):
    """Evaluate property comparison accuracy."""
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    i_dev = torch.tensor(interaction_scores, dtype=torch.float32).to(device)

    for s in senders:
        s.eval()
    receiver.eval()

    ce = cf = ci = cb2 = cb3 = 0
    te = tf = ti = tb2 = tb3 = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        with torch.no_grad():
            parts = []
            for si, sender in enumerate(senders):
                frames = AGENT_FRAMES[si]
                feat_a = features[ia][:, frames, :].to(device)
                feat_b = features[ib][:, frames, :].to(device)
                msg_a, _ = sender(feat_a)
                msg_b, _ = sender(feat_b)
                parts.extend([msg_a, msg_b])

            combined = torch.cat(parts, dim=-1)
            pred_e, pred_f, pred_i = receiver(combined)

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])
        label_i = (i_dev[ia] > i_dev[ib])
        valid_e = (e_dev[ia] != e_dev[ib])
        valid_f = (f_dev[ia] != f_dev[ib])
        valid_i = (i_dev[ia] != i_dev[ib])
        valid_2 = valid_e & valid_f
        valid_3 = valid_e & valid_f & valid_i

        if valid_e.sum() > 0:
            ce += ((pred_e > 0)[valid_e] == label_e[valid_e]).sum().item()
            te += valid_e.sum().item()
        if valid_f.sum() > 0:
            cf += ((pred_f > 0)[valid_f] == label_f[valid_f]).sum().item()
            tf += valid_f.sum().item()
        if include_interaction and pred_i is not None and valid_i.sum() > 0:
            ci += ((pred_i > 0)[valid_i] == label_i[valid_i]).sum().item()
            ti += valid_i.sum().item()
        if valid_2.sum() > 0:
            both2_ok = ((pred_e > 0)[valid_2] == label_e[valid_2]) & \
                       ((pred_f > 0)[valid_2] == label_f[valid_2])
            cb2 += both2_ok.sum().item()
            tb2 += valid_2.sum().item()
        if include_interaction and pred_i is not None and valid_3.sum() > 0:
            all3_ok = ((pred_e > 0)[valid_3] == label_e[valid_3]) & \
                      ((pred_f > 0)[valid_3] == label_f[valid_3]) & \
                      ((pred_i > 0)[valid_3] == label_i[valid_3])
            cb3 += all3_ok.sum().item()
            tb3 += valid_3.sum().item()

    result = {
        'e_acc': ce / max(te, 1),
        'f_acc': cf / max(tf, 1),
        'both2_acc': cb2 / max(tb2, 1),
    }
    if include_interaction:
        result['interaction_acc'] = ci / max(ti, 1)
        result['all3_acc'] = cb3 / max(tb3, 1)
    return result


def evaluate_population(senders, receivers, features, e_bins, f_bins,
                        interaction_scores, scene_ids, device,
                        include_interaction=False, n_rounds=30):
    """Pick best receiver from population, then evaluate fully."""
    best_score = -1
    best_r = None
    for r in receivers:
        acc = evaluate_with_receiver(
            senders, r, features, e_bins, f_bins, interaction_scores,
            scene_ids, device, include_interaction=include_interaction,
            n_rounds=10)
        score = acc.get('all3_acc', acc['both2_acc'])
        if score > best_score:
            best_score = score
            best_r = r
    final = evaluate_with_receiver(
        senders, best_r, features, e_bins, f_bins, interaction_scores,
        scene_ids, device, include_interaction=include_interaction,
        n_rounds=n_rounds)
    return final, best_r


# ══════════════════════════════════════════════════════════════════
# Message Analysis
# ══════════════════════════════════════════════════════════════════

def _mutual_information(x, y):
    """Compute MI between discrete arrays x and y."""
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


def analyze_sender_messages(sender, features, e_bins, f_bins,
                            interaction_scores, device, label=""):
    """Compute MI of each message position with each property."""
    sender.eval()
    all_tokens = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch = features[i:i + BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)

    result = {'label': label, 'per_position': []}
    total_mi_e = 0.0
    total_mi_f = 0.0
    total_mi_i = 0.0

    for p in range(N_POSITIONS):
        pos_tokens = all_tokens[:, p]
        mi_e = _mutual_information(pos_tokens, e_bins)
        mi_f = _mutual_information(pos_tokens, f_bins)
        mi_i = _mutual_information(pos_tokens, interaction_scores)
        total_mi_e += mi_e
        total_mi_f += mi_f
        total_mi_i += mi_i

        counts = np.bincount(pos_tokens, minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        raw_ent = -np.sum(probs_nz * np.log(probs_nz))
        norm_ent = raw_ent / np.log(VOCAB_SIZE) if VOCAB_SIZE > 1 else 0.0

        result['per_position'].append({
            'mi_e': float(mi_e), 'mi_f': float(mi_f),
            'mi_i': float(mi_i), 'entropy': float(norm_ent),
        })

    result['total_mi_e'] = float(total_mi_e)
    result['total_mi_f'] = float(total_mi_f)
    result['total_mi_i'] = float(total_mi_i)

    # Specialization ratios
    denom_ef = total_mi_e + total_mi_f
    if denom_ef > 1e-10:
        result['spec_ratio_ef'] = float(abs(total_mi_e - total_mi_f) / denom_ef)
    else:
        result['spec_ratio_ef'] = 0.0

    return result


def analyze_all_senders(senders, features, e_bins, f_bins,
                        interaction_scores, device):
    """Analyze all senders."""
    results = {}
    for si, sender in enumerate(senders):
        frames = AGENT_FRAMES[si]
        sender_features = features[:, frames, :]
        label = f"agent_{si} (frames {frames})"
        results[f'agent_{si}'] = analyze_sender_messages(
            sender, sender_features, e_bins, f_bins,
            interaction_scores, device, label=label)
    return results


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_condition(condition, features, e_bins, f_bins, interaction_scores,
                    train_ids, holdout_ids, device, seed):
    """Train one condition for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine if interaction head is included from the start
    include_interaction_from_start = condition in ('joint', 'interaction_only')

    # Create senders
    senders = []
    for i in range(N_AGENTS):
        senders.append(
            FixedSender(HIDDEN_DIM, DINO_DIM, N_POSITIONS, VOCAB_SIZE).to(device))

    # Receiver input: 4 agents × 2 balls × msg_dim
    recv_input_dim = N_AGENTS * 2 * MSG_DIM  # 80
    receivers = [ThreeHeadReceiver(
        recv_input_dim, HIDDEN_DIM,
        include_interaction=include_interaction_from_start
    ).to(device) for _ in range(N_RECEIVERS)]

    # Optimizers
    sender_params = []
    for s in senders:
        sender_params.extend(list(s.parameters()))
    sender_opt = torch.optim.Adam(sender_params, lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    i_dev = torch.tensor(interaction_scores, dtype=torch.float32).to(device)
    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_holdout_score = 0.0
    best_states = None
    nan_count = 0
    t_start = time.time()

    # Track whether interaction is currently active
    interaction_active = include_interaction_from_start

    # Curriculum tracking
    curriculum_history = []  # list of (epoch, metrics_dict)
    mi_at_switch = None  # MI analysis just before switch

    for epoch in range(COMM_EPOCHS):
        # CURRICULUM: add interaction head at SWITCH_EPOCH
        if condition == 'curriculum' and epoch == SWITCH_EPOCH:
            # Record MI just before switch
            mi_at_switch = analyze_all_senders(
                senders, features, e_bins, f_bins,
                interaction_scores, device)

            # Add interaction head to all receivers
            for r in receivers:
                r.add_interaction_head()
            # Rebuild receiver optimizers to include new head params
            receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                             for r in receivers]
            interaction_active = True
            print(f"        >>> Epoch {epoch}: Added interaction head", flush=True)

        # IL reset
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = ThreeHeadReceiver(
                    recv_input_dim, HIDDEN_DIM,
                    include_interaction=interaction_active
                ).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        for s in senders:
            s.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            label_i = (i_dev[ia] > i_dev[ib]).float()

            # Forward through all senders
            parts = []
            all_logits = []
            for si, sender in enumerate(senders):
                frames = AGENT_FRAMES[si]
                feat_a = features[ia][:, frames, :].to(device)
                feat_b = features[ib][:, frames, :].to(device)
                msg_a, lg_a = sender(feat_a, tau, hard)
                msg_b, lg_b = sender(feat_b, tau, hard)
                parts.extend([msg_a, msg_b])
                all_logits.extend([lg_a, lg_b])

            combined = torch.cat(parts, dim=-1)

            # Task loss
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f, pred_i = r(combined)

                r_loss = torch.tensor(0.0, device=device)
                if condition != 'interaction_only':
                    r_loss = r_loss + F.binary_cross_entropy_with_logits(pred_e, label_e)
                    r_loss = r_loss + F.binary_cross_entropy_with_logits(pred_f, label_f)
                if interaction_active and pred_i is not None:
                    r_loss = r_loss + F.binary_cross_entropy_with_logits(pred_i, label_i)

                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization
            for lg in all_logits:
                for p in range(N_POSITIONS):
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

            # NaN grad check
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

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        # Evaluate at checkpoints
        is_regular = (epoch + 1) % 40 == 0
        is_curriculum_ckpt = (condition == 'curriculum' and
                              epoch in CURRICULUM_CHECKPOINTS)
        if is_regular or is_curriculum_ckpt:
            holdout_result, _ = evaluate_population(
                senders, receivers, features, e_bins, f_bins,
                interaction_scores, holdout_ids, device,
                include_interaction=interaction_active, n_rounds=10)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)

            # Build log string
            log_parts = [f"Ep {epoch+1:3d}: e={holdout_result['e_acc']:.1%}",
                         f"f={holdout_result['f_acc']:.1%}"]
            if interaction_active:
                log_parts.append(f"i={holdout_result.get('interaction_acc', 0):.1%}")
            log_parts.append(f"b2={holdout_result['both2_acc']:.1%}")
            if interaction_active:
                log_parts.append(f"a3={holdout_result.get('all3_acc', 0):.1%}")
            if nan_count > 0:
                log_parts.append(f"NaN={nan_count}")
            log_parts.append(f"ETA {eta/60:.0f}min")

            if is_regular:
                print(f"        {'  '.join(log_parts)}", flush=True)

            # Track for curriculum
            if is_curriculum_ckpt:
                curriculum_history.append({
                    'epoch': epoch,
                    **holdout_result,
                })

            # Best model tracking
            score = holdout_result.get('all3_acc', holdout_result['both2_acc'])
            if score > best_holdout_score:
                best_holdout_score = score
                states = {
                    'senders': [
                        {k: v.cpu().clone() for k, v in s.state_dict().items()}
                        for s in senders
                    ],
                    'receivers': [
                        {k: v.cpu().clone() for k, v in r.state_dict().items()}
                        for r in receivers
                    ],
                }
                best_states = states

    # Restore best
    if best_states is not None:
        for i, s in enumerate(senders):
            s.load_state_dict(best_states['senders'][i])
            s.to(device)
        for i, r in enumerate(receivers):
            # Best state may be from before interaction head was added
            # (curriculum condition). Use strict=False to handle this.
            missing, unexpected = r.load_state_dict(
                best_states['receivers'][i], strict=False)
            r.to(device)

    # Final evaluation
    final_result, best_r = evaluate_population(
        senders, receivers, features, e_bins, f_bins,
        interaction_scores, holdout_ids, device,
        include_interaction=interaction_active, n_rounds=30)

    # Message analysis
    msg_analysis = analyze_all_senders(
        senders, features, e_bins, f_bins,
        interaction_scores, device)

    result = {
        'e_acc': final_result['e_acc'],
        'f_acc': final_result['f_acc'],
        'both2_acc': final_result['both2_acc'],
        'nan_count': nan_count,
        'msg_analysis': msg_analysis,
    }
    if interaction_active:
        result['interaction_acc'] = final_result.get('interaction_acc', 0)
        result['all3_acc'] = final_result.get('all3_acc', 0)

    # Curriculum-specific data
    if condition == 'curriculum':
        result['curriculum_history'] = curriculum_history
        result['mi_at_switch'] = mi_at_switch

    return result


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 63: Novel Property Introduction", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Agents: {N_AGENTS}, frames={AGENT_FRAMES}, "
          f"{N_POSITIONS} pos, vocab={VOCAB_SIZE}", flush=True)
    print(f"  Conditions: {CONDITIONS}", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Epochs: {COMM_EPOCHS}, switch at {SWITCH_EPOCH}", flush=True)
    print(f"  Interaction = e_bin + f_bin comparison", flush=True)
    print(flush=True)

    # Load data
    print("  Loading cached DINOv2 features...", flush=True)
    features, e_bins, f_bins, interaction_scores, train_ids, holdout_ids = load_data()
    print(f"  Features: {features.shape}", flush=True)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)
    print(f"  Interaction scores: {np.unique(interaction_scores)} "
          f"(e_bin + f_bin, range 0-8)", flush=True)
    print(flush=True)

    all_results = {}
    total_t0 = time.time()

    for ci, condition in enumerate(CONDITIONS):
        print(f"  {'='*60}", flush=True)
        print(f"  [{ci+1}/{len(CONDITIONS)}] Condition: {condition.upper()}", flush=True)
        print(f"  {'='*60}", flush=True)

        condition_seeds = []
        for seed in SEEDS:
            t0 = time.time()
            total_elapsed = time.time() - total_t0
            done_seeds = ci * N_SEEDS + seed
            total_seeds = len(CONDITIONS) * N_SEEDS
            if done_seeds > 0:
                per_seed = total_elapsed / done_seeds
                remaining = (total_seeds - done_seeds) * per_seed
                eta_str = f"  total ETA {remaining/60:.0f}min"
            else:
                eta_str = ""

            print(f"    [seed={seed}] Training...{eta_str}", flush=True)

            result = train_condition(
                condition, features, e_bins, f_bins, interaction_scores,
                train_ids, holdout_ids, DEVICE, seed)

            elapsed = time.time() - t0
            parts = [f"e={result['e_acc']:.1%}",
                     f"f={result['f_acc']:.1%}"]
            if 'interaction_acc' in result:
                parts.append(f"i={result['interaction_acc']:.1%}")
            parts.append(f"b2={result['both2_acc']:.1%}")
            if 'all3_acc' in result:
                parts.append(f"a3={result['all3_acc']:.1%}")
            print(f"    [seed={seed}] holdout {'  '.join(parts)}  "
                  f"({elapsed:.0f}s)", flush=True)
            condition_seeds.append(result)

        # Aggregate
        e_accs = [r['e_acc'] for r in condition_seeds]
        f_accs = [r['f_acc'] for r in condition_seeds]
        b2_accs = [r['both2_acc'] for r in condition_seeds]

        summary = {
            'e_mean': float(np.mean(e_accs)),
            'e_std': float(np.std(e_accs)),
            'f_mean': float(np.mean(f_accs)),
            'f_std': float(np.std(f_accs)),
            'both2_mean': float(np.mean(b2_accs)),
            'both2_std': float(np.std(b2_accs)),
        }

        if 'interaction_acc' in condition_seeds[0]:
            i_accs = [r['interaction_acc'] for r in condition_seeds]
            a3_accs = [r['all3_acc'] for r in condition_seeds]
            summary['interaction_mean'] = float(np.mean(i_accs))
            summary['interaction_std'] = float(np.std(i_accs))
            summary['all3_mean'] = float(np.mean(a3_accs))
            summary['all3_std'] = float(np.std(a3_accs))

        # Per-agent specialization
        agent_specs = {}
        for si in range(N_AGENTS):
            key = f'agent_{si}'
            mi_es = [r['msg_analysis'][key]['total_mi_e']
                     for r in condition_seeds]
            mi_fs = [r['msg_analysis'][key]['total_mi_f']
                     for r in condition_seeds]
            mi_is = [r['msg_analysis'][key]['total_mi_i']
                     for r in condition_seeds]
            specs = [r['msg_analysis'][key]['spec_ratio_ef']
                     for r in condition_seeds]
            agent_specs[key] = {
                'mi_e_mean': float(np.mean(mi_es)),
                'mi_f_mean': float(np.mean(mi_fs)),
                'mi_i_mean': float(np.mean(mi_is)),
                'spec_ratio_ef_mean': float(np.mean(specs)),
                'frames': AGENT_FRAMES[si],
            }
        summary['agent_specs'] = agent_specs

        # Curriculum-specific: aggregate history
        if condition == 'curriculum':
            # Aggregate curriculum history across seeds
            agg_history = {}
            for r in condition_seeds:
                for ckpt in r.get('curriculum_history', []):
                    ep = ckpt['epoch']
                    if ep not in agg_history:
                        agg_history[ep] = {'e': [], 'f': [], 'b2': [],
                                           'i': [], 'a3': []}
                    agg_history[ep]['e'].append(ckpt.get('e_acc', 0))
                    agg_history[ep]['f'].append(ckpt.get('f_acc', 0))
                    agg_history[ep]['b2'].append(ckpt.get('both2_acc', 0))
                    agg_history[ep]['i'].append(ckpt.get('interaction_acc', 0))
                    agg_history[ep]['a3'].append(ckpt.get('all3_acc', 0))

            summary['curriculum_history'] = {
                str(ep): {
                    'e_mean': float(np.mean(v['e'])),
                    'f_mean': float(np.mean(v['f'])),
                    'b2_mean': float(np.mean(v['b2'])),
                    'i_mean': float(np.mean(v['i'])),
                    'a3_mean': float(np.mean(v['a3'])),
                } for ep, v in sorted(agg_history.items())
            }

            # MI shift: before vs after switch
            mi_before = {}
            mi_after = {}
            for si in range(N_AGENTS):
                key = f'agent_{si}'
                mi_before_e = [r['mi_at_switch'][key]['total_mi_e']
                               for r in condition_seeds if r.get('mi_at_switch')]
                mi_before_f = [r['mi_at_switch'][key]['total_mi_f']
                               for r in condition_seeds if r.get('mi_at_switch')]
                mi_before_i = [r['mi_at_switch'][key]['total_mi_i']
                               for r in condition_seeds if r.get('mi_at_switch')]
                mi_after_e = [r['msg_analysis'][key]['total_mi_e']
                              for r in condition_seeds]
                mi_after_f = [r['msg_analysis'][key]['total_mi_f']
                              for r in condition_seeds]
                mi_after_i = [r['msg_analysis'][key]['total_mi_i']
                              for r in condition_seeds]
                mi_before[key] = {
                    'mi_e': float(np.mean(mi_before_e)) if mi_before_e else 0,
                    'mi_f': float(np.mean(mi_before_f)) if mi_before_f else 0,
                    'mi_i': float(np.mean(mi_before_i)) if mi_before_i else 0,
                }
                mi_after[key] = {
                    'mi_e': float(np.mean(mi_after_e)),
                    'mi_f': float(np.mean(mi_after_f)),
                    'mi_i': float(np.mean(mi_after_i)),
                }
            summary['mi_before_switch'] = mi_before
            summary['mi_after_switch'] = mi_after

        summary['seeds'] = condition_seeds
        all_results[condition] = summary

        # Print summary
        print(f"\n  {condition.upper()} summary:", flush=True)
        parts = [f"e={summary['e_mean']:.1%} ± {summary['e_std']:.1%}",
                 f"f={summary['f_mean']:.1%} ± {summary['f_std']:.1%}",
                 f"b2={summary['both2_mean']:.1%} ± {summary['both2_std']:.1%}"]
        if 'interaction_mean' in summary:
            parts.append(f"i={summary['interaction_mean']:.1%} ± {summary['interaction_std']:.1%}")
            parts.append(f"a3={summary['all3_mean']:.1%} ± {summary['all3_std']:.1%}")
        print(f"    {'  '.join(parts)}", flush=True)

        for si in range(N_AGENTS):
            key = f'agent_{si}'
            asp = agent_specs[key]
            print(f"    {key} (frames {asp['frames']}): "
                  f"MI(e)={asp['mi_e_mean']:.3f}  MI(f)={asp['mi_f_mean']:.3f}  "
                  f"MI(i)={asp['mi_i_mean']:.3f}", flush=True)

        if condition == 'curriculum':
            print(f"    Curriculum adaptation curve:", flush=True)
            for ep_str, v in sorted(summary.get('curriculum_history', {}).items(),
                                     key=lambda x: int(x[0])):
                ep = int(ep_str)
                marker = " <<<" if ep == 199 else ""
                parts = [f"e={v['e_mean']:.1%}", f"f={v['f_mean']:.1%}",
                         f"b2={v['b2_mean']:.1%}"]
                if ep >= SWITCH_EPOCH:
                    parts.extend([f"i={v['i_mean']:.1%}",
                                  f"a3={v['a3_mean']:.1%}"])
                print(f"      ep {ep+1:3d}: {'  '.join(parts)}{marker}", flush=True)

            print(f"    MI shift (before → after switch):", flush=True)
            for si in range(N_AGENTS):
                key = f'agent_{si}'
                b = summary['mi_before_switch'][key]
                a = summary['mi_after_switch'][key]
                print(f"      {key}: MI(e) {b['mi_e']:.3f}→{a['mi_e']:.3f}  "
                      f"MI(f) {b['mi_f']:.3f}→{a['mi_f']:.3f}  "
                      f"MI(i) {b['mi_i']:.3f}→{a['mi_i']:.3f}", flush=True)

        print(flush=True)

    total_elapsed = time.time() - total_t0

    # Save results
    output_path = RESULTS_DIR / "phase63_adaptation.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_str = json.dumps(all_results, indent=2, default=convert)
    with open(output_path, 'w') as f:
        f.write(json_str)
    print(f"  Saved results to {output_path}", flush=True)

    # Final summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  Accuracy comparison:", flush=True)
    header = f"  {'Condition':<20s} {'E':>8s} {'F':>8s} {'Both2':>8s} {'Inter':>8s} {'All3':>8s}"
    print(header, flush=True)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    for cond in CONDITIONS:
        s = all_results[cond]
        e_str = f"{s['e_mean']:.1%}"
        f_str = f"{s['f_mean']:.1%}"
        b2_str = f"{s['both2_mean']:.1%}"
        i_str = f"{s.get('interaction_mean', 0):.1%}" if 'interaction_mean' in s else "—"
        a3_str = f"{s.get('all3_mean', 0):.1%}" if 'all3_mean' in s else "—"
        print(f"  {cond:<20s} {e_str:>8s} {f_str:>8s} {b2_str:>8s} "
              f"{i_str:>8s} {a3_str:>8s}", flush=True)

    # Key questions
    print(f"\n  Key questions:", flush=True)

    if 'curriculum' in all_results and 'two_only' in all_results:
        c = all_results['curriculum']
        t = all_results['two_only']
        diff_e = c['e_mean'] - t['e_mean']
        diff_f = c['f_mean'] - t['f_mean']
        print(f"    1. Catastrophic forgetting? curriculum vs two_only: "
              f"e {diff_e:+.1%}, f {diff_f:+.1%}", flush=True)

    if 'curriculum' in all_results and 'joint' in all_results:
        c = all_results['curriculum']
        j = all_results['joint']
        if 'interaction_mean' in c and 'interaction_mean' in j:
            diff_i = c['interaction_mean'] - j['interaction_mean']
            print(f"    2. Adaptation quality? curriculum vs joint interaction: "
                  f"{diff_i:+.1%}", flush=True)

    if 'interaction_only' in all_results:
        io = all_results['interaction_only']
        if 'interaction_mean' in io:
            print(f"    4. Conjunction complexity: interaction_only = "
                  f"{io['interaction_mean']:.1%}", flush=True)

    print(f"\n  Total runtime: {total_elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""
Phase 84c: Discrete vs Continuous Communication
=================================================
Replace Gumbel-Softmax (one-hot in R^5) with tanh projection to R^5.
Same total dims (40), same encoder, same receiver, same IL schedule.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase84c_discrete_vs_continuous.py
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

HIDDEN_DIM = 128
VJEPA_DIM = 1024
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
N_AGENTS = 4
N_FRAMES = 24
FRAMES_PER_AGENT = 6

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
N_SEEDS = 20

OUTCOME_EPOCHS = 100
OUTCOME_LR = 1e-3
N_OUTCOME_SEEDS = 20


# ══════════════════════════════════════════════════════════════════
# Architecture
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


class DiscreteHead(nn.Module):
    """Standard Gumbel-Softmax head."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, h, tau=1.0, hard=True):
        logits = self.fc(h)
        if self.training:
            msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            idx = logits.argmax(dim=-1)
            msg = F.one_hot(idx, self.vocab_size).float()
        return msg, logits


class ContinuousHead(nn.Module):
    """Continuous tanh head — same output dim as discrete."""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, tau=1.0, hard=True):
        # tau and hard ignored for continuous
        msg = torch.tanh(self.fc(h))
        return msg, None  # No logits for continuous


class AgentSender(nn.Module):
    """Single agent sender with either discrete or continuous heads."""
    def __init__(self, encoder, hidden_dim, n_heads, vocab_size, continuous=False):
        super().__init__()
        self.encoder = encoder
        self.continuous = continuous
        if continuous:
            self.heads = nn.ModuleList([
                ContinuousHead(hidden_dim, vocab_size) for _ in range(n_heads)
            ])
        else:
            self.heads = nn.ModuleList([
                DiscreteHead(hidden_dim, vocab_size) for _ in range(n_heads)
            ])

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        messages = []
        all_logits = []
        for head in self.heads:
            msg, logits = head(h, tau=tau, hard=hard)
            messages.append(msg)
            if logits is not None:
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


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def load_data():
    data = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)
    features = data['features'].float()
    index = data['index']
    mass_bins = np.array([e['mass_ratio_bin'] for e in index])
    rest_bins = np.array([e['restitution_bin'] for e in index])
    vel_b = np.array([e['post_collision_vel_b'] for e in index])
    outcome_labels = torch.tensor((vel_b > np.median(vel_b)).astype(np.float32))
    return features, mass_bins, rest_bins, outcome_labels


def create_splits(e_bins, f_bins):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
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
# Training
# ══════════════════════════════════════════════════════════════════

def train_sender(features, mass_bins, rest_bins, seed, continuous=False):
    """Train a 4-agent sender (discrete or continuous)."""
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    train_ids, holdout_ids = create_splits(mass_bins, rest_bins)
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE  # 40

    torch.manual_seed(seed)
    np.random.seed(seed)

    senders = []
    for _ in range(N_AGENTS):
        enc = TemporalEncoder(HIDDEN_DIM, VJEPA_DIM)
        s = AgentSender(enc, HIDDEN_DIM, N_HEADS, VOCAB_SIZE, continuous=continuous)
        senders.append(s)
    multi_sender = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(mass_bins, dtype=torch.float32).to(DEVICE)
    f_dev = torch.tensor(rest_bins, dtype=torch.float32).to(DEVICE)
    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)
    best_both = 0.0
    best_sender_state = None
    best_receiver_states = None
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

            # Entropy regularization (only for discrete)
            if not continuous:
                for logits in logits_a + logits_b:
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

            # NaN grad check
            has_nan = False
            all_params = list(multi_sender.parameters())
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan = True
                    break
            if has_nan:
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

        # Evaluate
        if (epoch + 1) % 50 == 0 or epoch == 0:
            multi_sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                correct_both = total_both = 0
                eval_rng = np.random.RandomState(999)
                for r in receivers:
                    for _ in range(10):
                        bs = min(BATCH_SIZE, len(holdout_ids))
                        ia, ib = sample_pairs(holdout_ids, bs, eval_rng)
                        views_a = [v[ia].to(DEVICE) for v in agent_views]
                        views_b = [v[ib].to(DEVICE) for v in agent_views]
                        msg_a, _ = multi_sender(views_a)
                        msg_b, _ = multi_sender(views_b)
                        pred_e, pred_f = r(msg_a, msg_b)
                        label_e = e_dev[ia] > e_dev[ib]
                        label_f = f_dev[ia] > f_dev[ib]
                        e_diff = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
                        f_diff = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
                        both_diff = e_diff & f_diff
                        if both_diff.sum() > 0:
                            both_ok = ((pred_e[both_diff] > 0) == label_e[both_diff]) & \
                                      ((pred_f[both_diff] > 0) == label_f[both_diff])
                            correct_both += both_ok.sum().item()
                            total_both += both_diff.sum().item()
                acc = correct_both / max(total_both, 1)
                if acc > best_both:
                    best_both = acc
                    best_sender_state = {k: v.cpu().clone() for k, v in multi_sender.state_dict().items()}
                    best_receiver_states = [{k: v.cpu().clone() for k, v in r.state_dict().items()} for r in receivers]

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            mode = "CONT" if continuous else "DISC"
            print(f"    [{mode}] Ep {epoch+1:3d}: tau={tau:.2f}  holdout_both={acc:.1%}{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

    if best_sender_state is not None:
        multi_sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return multi_sender, receivers, best_both, nan_count


def extract_messages(multi_sender, features):
    """Extract frozen messages for all scenes."""
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    multi_sender.eval()
    all_msgs = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            msg, _ = multi_sender(views)
            all_msgs.append(msg.cpu())
    return torch.cat(all_msgs, dim=0)


def block_zero_ablation(multi_sender, receivers, features, mass_bins, rest_bins, holdout_ids):
    """Zero contiguous 10-dim blocks and measure selective disruption."""
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    multi_sender.eval()
    e_dev = torch.tensor(mass_bins, dtype=torch.float32).to(DEVICE)
    f_dev = torch.tensor(rest_bins, dtype=torch.float32).to(DEVICE)

    # Find best receiver
    best_r = receivers[0]

    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE  # 40

    results = {'baseline': {}, 'zeroed': []}

    with torch.no_grad():
        # Baseline (no zeroing)
        eval_rng = np.random.RandomState(999)
        correct_e = correct_f = total_e = total_f = 0
        for _ in range(30):
            bs = min(BATCH_SIZE, len(holdout_ids))
            ia, ib = sample_pairs(holdout_ids, bs, eval_rng)
            views_a = [v[ia].to(DEVICE) for v in agent_views]
            views_b = [v[ib].to(DEVICE) for v in agent_views]
            msg_a, _ = multi_sender(views_a)
            msg_b, _ = multi_sender(views_b)
            pred_e, pred_f = best_r(msg_a, msg_b)
            label_e = e_dev[ia] > e_dev[ib]
            label_f = f_dev[ia] > f_dev[ib]
            e_diff = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
            f_diff = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
            if e_diff.sum() > 0:
                correct_e += ((pred_e[e_diff] > 0) == label_e[e_diff]).sum().item()
                total_e += e_diff.sum().item()
            if f_diff.sum() > 0:
                correct_f += ((pred_f[f_diff] > 0) == label_f[f_diff]).sum().item()
                total_f += f_diff.sum().item()
        results['baseline'] = {
            'mass_acc': correct_e / max(total_e, 1),
            'rest_acc': correct_f / max(total_f, 1),
        }

        # Zero each 10-dim block (4 agents × 2 heads × 5 vocab = 4 blocks of 10)
        for block_idx in range(4):
            start = block_idx * 10
            end = start + 10
            eval_rng = np.random.RandomState(999)
            correct_e = correct_f = total_e = total_f = 0
            for _ in range(30):
                bs = min(BATCH_SIZE, len(holdout_ids))
                ia, ib = sample_pairs(holdout_ids, bs, eval_rng)
                views_a = [v[ia].to(DEVICE) for v in agent_views]
                views_b = [v[ib].to(DEVICE) for v in agent_views]
                msg_a, _ = multi_sender(views_a)
                msg_b, _ = multi_sender(views_b)
                # Zero the block
                msg_a_z = msg_a.clone()
                msg_b_z = msg_b.clone()
                msg_a_z[:, start:end] = 0
                msg_b_z[:, start:end] = 0
                pred_e, pred_f = best_r(msg_a_z, msg_b_z)
                label_e = e_dev[ia] > e_dev[ib]
                label_f = f_dev[ia] > f_dev[ib]
                e_diff = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
                f_diff = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
                if e_diff.sum() > 0:
                    correct_e += ((pred_e[e_diff] > 0) == label_e[e_diff]).sum().item()
                    total_e += e_diff.sum().item()
                if f_diff.sum() > 0:
                    correct_f += ((pred_f[f_diff] > 0) == label_f[f_diff]).sum().item()
                    total_f += f_diff.sum().item()
            results['zeroed'].append({
                'block': block_idx,
                'dims': f"{start}-{end}",
                'mass_acc': correct_e / max(total_e, 1),
                'rest_acc': correct_f / max(total_f, 1),
                'mass_drop': results['baseline']['mass_acc'] - correct_e / max(total_e, 1),
                'rest_drop': results['baseline']['rest_acc'] - correct_f / max(total_f, 1),
            })

    return results


def train_outcome_predictor(inputs, labels, train_ids, holdout_ids, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OutcomePredictor(inputs.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=OUTCOME_LR)
    rng = np.random.RandomState(seed)
    inputs_dev = inputs.to(DEVICE)
    labels_dev = labels.float().to(DEVICE)
    best_acc = 0.0
    for epoch in range(OUTCOME_EPOCHS):
        model.train()
        perm = rng.permutation(len(train_ids))
        for start in range(0, len(train_ids), BATCH_SIZE):
            idx = train_ids[perm[start:start+BATCH_SIZE]]
            pred = model(inputs_dev[idx])
            loss = F.binary_cross_entropy_with_logits(pred, labels_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_h = model(inputs_dev[holdout_ids])
                acc = ((pred_h > 0).float() == labels_dev[holdout_ids]).float().mean().item()
            if acc > best_acc:
                best_acc = acc
        if epoch % 50 == 0:
            torch.mps.empty_cache()
    return best_acc


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Phase 84c: Discrete vs Continuous Communication", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 70, flush=True)
    t_total = time.time()

    features, mass_bins, rest_bins, outcome_labels = load_data()
    train_ids, holdout_ids = create_splits(mass_bins, rest_bins)
    print(f"Features: {features.shape}", flush=True)
    print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}", flush=True)

    all_results = {
        'discrete': {'holdout': [], 'outcome': [], 'ablation': []},
        'continuous': {'holdout': [], 'outcome': [], 'ablation': []},
    }

    for mode in ['discrete', 'continuous']:
        is_cont = (mode == 'continuous')
        print(f"\n{'='*70}", flush=True)
        print(f"{'CONTINUOUS' if is_cont else 'DISCRETE'} Communication ({N_SEEDS} seeds)", flush=True)
        print(f"{'='*70}", flush=True)

        for seed in range(N_SEEDS):
            t0 = time.time()
            print(f"\n  --- Seed {seed} ---", flush=True)

            # Train sender
            sender, receivers, holdout_acc, nans = train_sender(
                features, mass_bins, rest_bins, seed, continuous=is_cont)
            all_results[mode]['holdout'].append(holdout_acc)

            # Extract messages
            messages = extract_messages(sender, features)

            # Block-zeroing ablation (for seeds 0-4 only to save time)
            if seed < 5:
                ablation = block_zero_ablation(sender, receivers, features,
                                               mass_bins, rest_bins, holdout_ids)
                all_results[mode]['ablation'].append(ablation)
                print(f"    Ablation baseline: mass={ablation['baseline']['mass_acc']:.1%} "
                      f"rest={ablation['baseline']['rest_acc']:.1%}", flush=True)
                for z in ablation['zeroed']:
                    print(f"    Zero block {z['block']}: mass_drop={z['mass_drop']:+.1%} "
                          f"rest_drop={z['rest_drop']:+.1%}", flush=True)

            # Outcome prediction from frozen messages
            outcome_acc = train_outcome_predictor(
                messages, outcome_labels, train_ids, holdout_ids, seed)
            all_results[mode]['outcome'].append(outcome_acc)

            dt = time.time() - t0
            print(f"    holdout={holdout_acc:.1%}  outcome={outcome_acc:.1%}  "
                  f"NaN={nans}  ({dt:.0f}s)", flush=True)

            if (seed + 1) % 5 == 0:
                h = all_results[mode]['holdout']
                o = all_results[mode]['outcome']
                print(f"    Running avg: holdout={np.mean(h):.1%}±{np.std(h):.1%}  "
                      f"outcome={np.mean(o):.1%}±{np.std(o):.1%}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Discrete vs Continuous", flush=True)
    print("=" * 70, flush=True)

    for mode in ['discrete', 'continuous']:
        h = all_results[mode]['holdout']
        o = all_results[mode]['outcome']
        print(f"\n  {mode.upper()}:", flush=True)
        print(f"    Comparison holdout: {np.mean(h):.1%} ± {np.std(h):.1%}", flush=True)
        print(f"    Outcome prediction: {np.mean(o):.1%} ± {np.std(o):.1%}", flush=True)

    # Statistical tests
    dh = all_results['discrete']['holdout']
    ch = all_results['continuous']['holdout']
    do = all_results['discrete']['outcome']
    co = all_results['continuous']['outcome']

    t_h, p_h = stats.ttest_ind(dh, ch)
    d_h = (np.mean(dh) - np.mean(ch)) / np.sqrt((np.std(dh)**2 + np.std(ch)**2) / 2)
    t_o, p_o = stats.ttest_ind(do, co)
    d_o = (np.mean(do) - np.mean(co)) / np.sqrt((np.std(do)**2 + np.std(co)**2) / 2)

    print(f"\n  Comparison holdout: t={t_h:.3f}, p={p_h:.4f}, d={d_h:.3f}", flush=True)
    print(f"  Outcome prediction: t={t_o:.3f}, p={p_o:.4f}, d={d_o:.3f}", flush=True)

    # Ablation analysis
    print(f"\n  ABLATION (block-zeroing, 5 seeds):", flush=True)
    for mode in ['discrete', 'continuous']:
        ablations = all_results[mode]['ablation']
        if not ablations:
            continue
        print(f"  {mode.upper()}:", flush=True)
        for block_idx in range(4):
            mass_drops = [a['zeroed'][block_idx]['mass_drop'] for a in ablations]
            rest_drops = [a['zeroed'][block_idx]['rest_drop'] for a in ablations]
            print(f"    Block {block_idx}: mass_drop={np.mean(mass_drops):+.1%}±{np.std(mass_drops):.1%}  "
                  f"rest_drop={np.mean(rest_drops):+.1%}±{np.std(rest_drops):.1%}", flush=True)

        # Selectivity: how asymmetric are the drops?
        all_mass_drops = []
        all_rest_drops = []
        for a in ablations:
            for z in a['zeroed']:
                all_mass_drops.append(abs(z['mass_drop']))
                all_rest_drops.append(abs(z['rest_drop']))
        # Compute selectivity: max(mass_drop, rest_drop) / (mass_drop + rest_drop)
        selectivities = []
        for a in ablations:
            for z in a['zeroed']:
                total = abs(z['mass_drop']) + abs(z['rest_drop'])
                if total > 0.01:
                    selectivities.append(max(abs(z['mass_drop']), abs(z['rest_drop'])) / total)
        if selectivities:
            print(f"    Selectivity: {np.mean(selectivities):.3f} ± {np.std(selectivities):.3f} "
                  f"(1.0 = perfectly selective, 0.5 = uniform)", flush=True)

    # Save
    save_data = {
        'discrete': {
            'holdout_mean': float(np.mean(dh)),
            'holdout_std': float(np.std(dh)),
            'holdout_seeds': dh,
            'outcome_mean': float(np.mean(do)),
            'outcome_std': float(np.std(do)),
            'outcome_seeds': do,
        },
        'continuous': {
            'holdout_mean': float(np.mean(ch)),
            'holdout_std': float(np.std(ch)),
            'holdout_seeds': ch,
            'outcome_mean': float(np.mean(co)),
            'outcome_std': float(np.std(co)),
            'outcome_seeds': co,
        },
        'tests': {
            'holdout': {'t': float(t_h), 'p': float(p_h), 'd': float(d_h)},
            'outcome': {'t': float(t_o), 'p': float(p_o), 'd': float(d_o)},
        },
    }

    # Add ablation data
    for mode in ['discrete', 'continuous']:
        ablations = all_results[mode]['ablation']
        if ablations:
            abl_summary = []
            for block_idx in range(4):
                mass_drops = [a['zeroed'][block_idx]['mass_drop'] for a in ablations]
                rest_drops = [a['zeroed'][block_idx]['rest_drop'] for a in ablations]
                abl_summary.append({
                    'block': block_idx,
                    'mass_drop_mean': float(np.mean(mass_drops)),
                    'mass_drop_std': float(np.std(mass_drops)),
                    'rest_drop_mean': float(np.mean(rest_drops)),
                    'rest_drop_std': float(np.std(rest_drops)),
                })
            save_data[mode]['ablation'] = abl_summary

    save_path = RESULTS_DIR / 'phase84c_discrete_vs_continuous.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\nPhase 84c complete. Total time: {dt/60:.1f}min", flush=True)

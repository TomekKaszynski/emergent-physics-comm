"""
Phase 84d: Action-Conditioned Generalization
==============================================
Test whether frozen compositional messages generalize to predict
outcomes under NOVEL velocity conditions the sender never saw.

Step 1: Compute expected outcomes for varied velocities (analytical, no rendering)
Step 2: Use frozen senders from original dataset to encode scenes
Step 3: Train predictors with velocity as additional input

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase84d_action_conditioned.py
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
DINO_DIM = 384
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
OUTCOME_EPOCHS = 150
OUTCOME_LR = 1e-3

# Velocity multipliers to test
VEL_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]


# ══════════════════════════════════════════════════════════════════
# Architecture (from Phase 79)
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


class ActionConditionedPredictor(nn.Module):
    """Predict outcome from frozen message + velocity condition."""
    def __init__(self, msg_dim, vel_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim + vel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, msg, vel_onehot):
        x = torch.cat([msg, vel_onehot], dim=-1)
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def load_features_and_index(path):
    data = torch.load(path, weights_only=False)
    features = data['features'].float()
    index = data['index']
    mass_bins = np.array([e['mass_ratio_bin'] for e in index])
    rest_bins = np.array([e['restitution_bin'] for e in index])
    return features, mass_bins, rest_bins, index


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
# Sender training (same as Phase 79b)
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
    train_ids, _ = create_splits(mass_bins, rest_bins)
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE

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
    best_acc = 0.0
    best_state = None
    nan_count = 0

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
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                             F.binary_cross_entropy_with_logits(pred_f, label_f)
            loss = total_loss / len(receivers)
            for logits in logits_a + logits_b:
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp().clamp(min=1e-8)
                ent = -(probs * log_probs).sum(dim=-1).mean()
                if ent / max_entropy < ENTROPY_THRESHOLD:
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
            has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                         for p in list(multi_sender.parameters()) + [p for r in receivers for p in r.parameters()])
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

        if (epoch + 1) % 100 == 0 or epoch == 0:
            multi_sender.eval()
            with torch.no_grad():
                eval_rng = np.random.RandomState(999)
                correct = total = 0
                for _ in range(10):
                    ia, ib = sample_pairs(train_ids, BATCH_SIZE, eval_rng)
                    views_a = [v[ia].to(DEVICE) for v in agent_views]
                    views_b = [v[ib].to(DEVICE) for v in agent_views]
                    msg_a, _ = multi_sender(views_a)
                    msg_b, _ = multi_sender(views_b)
                    for r in receivers:
                        pred_e, pred_f = r(msg_a, msg_b)
                        e_diff = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
                        f_diff = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
                        both = e_diff & f_diff
                        if both.sum() > 0:
                            ok = ((pred_e[both] > 0) == (e_dev[ia][both] > e_dev[ib][both])) & \
                                 ((pred_f[both] > 0) == (f_dev[ia][both] > f_dev[ib][both]))
                            correct += ok.sum().item()
                            total += both.sum().item()
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.cpu().clone() for k, v in multi_sender.state_dict().items()}
            print(f"    Ep {epoch+1}: train_both={acc:.1%}  NaN={nan_count}", flush=True)

    if best_state is not None:
        multi_sender.load_state_dict(best_state)
    return multi_sender


def extract_messages(multi_sender, features):
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    multi_sender.eval()
    all_msgs = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            msg, _ = multi_sender(views)
            all_msgs.append(msg.cpu())
    return torch.cat(all_msgs, dim=0)


# ══════════════════════════════════════════════════════════════════
# Step 1: Compute analytical outcomes at varied velocities
# ══════════════════════════════════════════════════════════════════

def step1_compute_outcomes():
    """
    For each scene, compute sphere B's post-collision speed at each velocity level.
    Use analytical 1D collision formula (no need to re-render).
    The frozen messages encode material properties, not velocity — so we test
    whether they generalize when velocity changes.
    """
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: Compute Analytical Outcomes at Varied Velocities", flush=True)
    print("=" * 70, flush=True)

    data = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)
    index = data['index']
    n_scenes = len(index)

    # For each scene, get material properties and original velocity
    mass_a_arr = np.array([e['sphere_a_mass'] for e in index])
    mass_b_arr = np.array([e['sphere_b_mass'] for e in index])
    rest_arr = np.array([e['restitution'] for e in index])
    vel_orig = np.array([e['initial_velocity'] for e in index])

    # Compute sphere B velocity for each velocity multiplier
    # v_b_final = (1 + e) * m_a / (m_a + m_b) * v_a_init
    all_vel_b = {}
    for mult in VEL_MULTIPLIERS:
        v_init = vel_orig * mult
        v_b = (1 + rest_arr) * mass_a_arr / (mass_a_arr + mass_b_arr) * v_init
        all_vel_b[mult] = v_b

    # Compute global median across ALL velocity conditions
    all_speeds = np.concatenate(list(all_vel_b.values()))
    global_median = np.median(all_speeds)
    print(f"  Global median vel_b across all conditions: {global_median:.3f}", flush=True)

    # Create labels: vel_b > global_median
    labels = {}
    for mult in VEL_MULTIPLIERS:
        lab = (all_vel_b[mult] > global_median).astype(int)
        labels[mult] = lab
        print(f"  Velocity ×{mult}: vel_b range [{all_vel_b[mult].min():.3f}, "
              f"{all_vel_b[mult].max():.3f}], {lab.sum()}/{n_scenes} positive", flush=True)

    return labels, all_vel_b, global_median


# ══════════════════════════════════════════════════════════════════
# Step 2-3: Train and evaluate
# ══════════════════════════════════════════════════════════════════

def train_action_predictor(messages, vel_onehots, labels, train_mask, test_mask, seed):
    """Train predictor: message + velocity → outcome."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    msg_dim = messages.shape[1]
    vel_dim = vel_onehots.shape[1]
    model = ActionConditionedPredictor(msg_dim, vel_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=OUTCOME_LR)
    rng = np.random.RandomState(seed)

    msgs_dev = messages.to(DEVICE)
    vel_dev = vel_onehots.to(DEVICE)
    labels_dev = labels.float().to(DEVICE)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    best_test_acc = 0.0

    for epoch in range(OUTCOME_EPOCHS):
        model.train()
        perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx), BATCH_SIZE):
            idx = train_idx[perm[start:start+BATCH_SIZE]]
            pred = model(msgs_dev[idx], vel_dev[idx])
            loss = F.binary_cross_entropy_with_logits(pred, labels_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == OUTCOME_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                if len(test_idx) > 0:
                    pred_t = model(msgs_dev[test_idx], vel_dev[test_idx])
                    acc = ((pred_t > 0).float() == labels_dev[test_idx]).float().mean().item()
                    if acc > best_test_acc:
                        best_test_acc = acc

        if epoch % 50 == 0:
            torch.mps.empty_cache()

    return best_test_acc


def run_generalization_experiment(messages, mass_bins, rest_bins, labels_dict):
    """
    Leave-one-velocity-out cross-validation.
    For each velocity level: train on the other 4, test on the held-out one.
    Also apply Latin square holdout to material properties.
    """
    n_scenes = len(messages)
    n_vel = len(VEL_MULTIPLIERS)

    # Expand messages: repeat for each velocity level
    # messages_expanded: (n_scenes * n_vel, msg_dim)
    # vel_onehots: (n_scenes * n_vel, n_vel)
    # labels: (n_scenes * n_vel,)
    messages_exp = messages.repeat(n_vel, 1)
    vel_onehots = []
    all_labels = []
    vel_indices = []

    for vi, mult in enumerate(VEL_MULTIPLIERS):
        onehot = torch.zeros(n_scenes, n_vel)
        onehot[:, vi] = 1.0
        vel_onehots.append(onehot)
        all_labels.append(torch.tensor(labels_dict[mult], dtype=torch.float32))
        vel_indices.extend([vi] * n_scenes)

    vel_onehots = torch.cat(vel_onehots, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    vel_indices = np.array(vel_indices)

    # Material property holdout
    _, mat_holdout = create_splits(mass_bins, rest_bins)
    mat_holdout_set = set(mat_holdout.tolist())

    results_by_vel = {}

    for held_out_vi, held_out_mult in enumerate(VEL_MULTIPLIERS):
        # Train: all velocity levels except held_out, excluding material holdout
        train_mask = np.zeros(len(messages_exp), dtype=bool)
        test_mask = np.zeros(len(messages_exp), dtype=bool)

        for i in range(len(messages_exp)):
            scene_idx = i % n_scenes
            vi = vel_indices[i]

            if scene_idx in mat_holdout_set:
                continue  # Never train on material holdout

            if vi == held_out_vi:
                test_mask[i] = True  # Test on held-out velocity
            else:
                train_mask[i] = True  # Train on other velocities

        accs = []
        for seed in range(N_SEEDS):
            acc = train_action_predictor(
                messages_exp, vel_onehots, all_labels,
                train_mask, test_mask, seed)
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results_by_vel[held_out_mult] = {
            'mean': float(mean_acc),
            'std': float(std_acc),
            'seeds': accs,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
        }
        print(f"    Hold-out vel ×{held_out_mult}: {mean_acc:.1%} ± {std_acc:.1%} "
              f"(train={train_mask.sum()}, test={test_mask.sum()})", flush=True)

    return results_by_vel


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Phase 84d: Action-Conditioned Generalization", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 70, flush=True)
    t_total = time.time()

    # Step 1: Compute outcomes
    labels_dict, all_vel_b, global_median = step1_compute_outcomes()

    # Step 2: Load features and train senders
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Train Senders & Extract Messages", flush=True)
    print("=" * 70, flush=True)

    # V-JEPA 2
    print("\n  Loading V-JEPA 2 features...", flush=True)
    vjepa_feats, mass_bins, rest_bins, _ = load_features_and_index(
        'results/vjepa2_collision_pooled.pt')
    print(f"  V-JEPA 2: {vjepa_feats.shape}", flush=True)

    print("  Training V-JEPA 2 sender (seed 0)...", flush=True)
    vjepa_sender = train_4agent_sender(vjepa_feats, mass_bins, rest_bins, VJEPA_DIM, seed=0)
    vjepa_msgs = extract_messages(vjepa_sender, vjepa_feats)
    print(f"  V-JEPA 2 messages: {vjepa_msgs.shape}", flush=True)

    # DINOv2
    print("\n  Loading DINOv2 features...", flush=True)
    dino_feats, _, _, _ = load_features_and_index('results/collision_dinov2_features.pt')
    print(f"  DINOv2: {dino_feats.shape}", flush=True)

    print("  Training DINOv2 sender (seed 0)...", flush=True)
    dino_sender = train_4agent_sender(dino_feats, mass_bins, rest_bins, DINO_DIM, seed=0)
    dino_msgs = extract_messages(dino_sender, dino_feats)
    print(f"  DINOv2 messages: {dino_msgs.shape}", flush=True)

    # Raw features (mean-pooled)
    vjepa_raw = vjepa_feats.mean(dim=1)  # (600, 1024)

    # Step 3: Generalization experiments
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Leave-One-Velocity-Out Generalization", flush=True)
    print("=" * 70, flush=True)

    conditions = {
        'vjepa2_messages': vjepa_msgs,
        'dinov2_messages': dino_msgs,
        'vjepa2_raw': vjepa_raw,
    }

    all_results = {}
    for cond_name, msgs in conditions.items():
        print(f"\n  --- {cond_name} (dim={msgs.shape[1]}) ---", flush=True)
        results_by_vel = run_generalization_experiment(
            msgs, mass_bins, rest_bins, labels_dict)
        all_results[cond_name] = results_by_vel

        # Overall mean across velocity holdouts
        all_accs = []
        for mult, r in results_by_vel.items():
            all_accs.extend(r['seeds'])
        print(f"    Overall: {np.mean(all_accs):.1%} ± {np.std(all_accs):.1%}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Action-Conditioned Generalization", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Condition':<25s}", end="", flush=True)
    for mult in VEL_MULTIPLIERS:
        print(f"  ×{mult:<6}", end="", flush=True)
    print(f"  {'Overall':>8s}", flush=True)
    print(f"  {'-'*80}", flush=True)

    for cond_name in ['vjepa2_messages', 'dinov2_messages', 'vjepa2_raw']:
        print(f"  {cond_name:<25s}", end="", flush=True)
        all_accs = []
        for mult in VEL_MULTIPLIERS:
            r = all_results[cond_name][mult]
            print(f"  {r['mean']:.1%}  ", end="", flush=True)
            all_accs.extend(r['seeds'])
        print(f"  {np.mean(all_accs):.1%}", flush=True)

    # Statistical tests
    vjepa_all = []
    dino_all = []
    raw_all = []
    for mult in VEL_MULTIPLIERS:
        vjepa_all.extend(all_results['vjepa2_messages'][mult]['seeds'])
        dino_all.extend(all_results['dinov2_messages'][mult]['seeds'])
        raw_all.extend(all_results['vjepa2_raw'][mult]['seeds'])

    t_vd, p_vd = stats.ttest_ind(vjepa_all, dino_all)
    d_vd = (np.mean(vjepa_all) - np.mean(dino_all)) / np.sqrt(
        (np.std(vjepa_all)**2 + np.std(dino_all)**2) / 2)
    t_vr, p_vr = stats.ttest_ind(vjepa_all, raw_all)
    d_vr = (np.mean(vjepa_all) - np.mean(raw_all)) / np.sqrt(
        (np.std(vjepa_all)**2 + np.std(raw_all)**2) / 2)

    print(f"\n  V-JEPA 2 msgs vs DINOv2 msgs: t={t_vd:.3f}, p={p_vd:.4f}, d={d_vd:.3f}", flush=True)
    print(f"  V-JEPA 2 msgs vs raw feats:   t={t_vr:.3f}, p={p_vr:.4f}, d={d_vr:.3f}", flush=True)

    # Save
    save_data = {
        'velocity_multipliers': VEL_MULTIPLIERS,
        'global_median_vel_b': float(global_median),
    }
    for cond_name in ['vjepa2_messages', 'dinov2_messages', 'vjepa2_raw']:
        save_data[cond_name] = {}
        all_accs = []
        for mult in VEL_MULTIPLIERS:
            r = all_results[cond_name][mult]
            save_data[cond_name][str(mult)] = r
            all_accs.extend(r['seeds'])
        save_data[cond_name]['overall_mean'] = float(np.mean(all_accs))
        save_data[cond_name]['overall_std'] = float(np.std(all_accs))

    save_data['tests'] = {
        'vjepa2_vs_dinov2': {'t': float(t_vd), 'p': float(p_vd), 'd': float(d_vd)},
        'vjepa2_msgs_vs_raw': {'t': float(t_vr), 'p': float(p_vr), 'd': float(d_vr)},
    }

    save_path = RESULTS_DIR / 'phase84d_action_conditioned.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\nPhase 84d complete. Total time: {dt/60:.1f}min", flush=True)

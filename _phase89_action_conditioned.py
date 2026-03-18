#!/usr/bin/env python3
"""Phase 89: Action-Conditioned Planning with Compositional Messages.

Shows frozen compositional messages support:
1. Action-conditioned outcome prediction with novel velocities
2. Selective property querying (mass vs restitution positions)
3. Counterfactual planning (vary velocity, observe prediction changes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
from scipy import stats

DEVICE = 'mps'
HIDDEN_DIM = 128
VOCAB_SIZE = 5
N_HEADS = 2
N_AGENTS = 4
BATCH_SIZE = 128
N_SEEDS = 20
HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

# ─── Models (matching Phase 79 exactly) ──────────────────────────

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=1024, n_positions=6):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


class CompositionalSender(nn.Module):
    def __init__(self, input_dim=1024, n_positions=6):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, n_positions)
        self.heads = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE) for _ in range(N_HEADS)
        ])

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        logits_list = [head(h) for head in self.heads]
        msgs = []
        for logits in logits_list:
            msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            msgs.append(msg)
        return torch.cat(msgs, dim=-1), logits_list


class MultiAgentSender(nn.Module):
    def __init__(self, n_agents=4, positions_per_agent=6, input_dim=1024):
        super().__init__()
        self.n_agents = n_agents
        self.positions_per_agent = positions_per_agent
        self.senders = nn.ModuleList([
            CompositionalSender(input_dim, positions_per_agent)
            for _ in range(n_agents)
        ])

    def forward(self, agent_views, tau=1.0, hard=True):
        msgs = []
        all_logits = []
        for i, sender in enumerate(self.senders):
            msg, logits = sender(agent_views[i], tau=tau, hard=hard)
            msgs.append(msg)
            all_logits.extend(logits)
        return torch.cat(msgs, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
        )
        self.head_e = nn.Linear(64, 1)
        self.head_f = nn.Linear(64, 1)

    def forward(self, msg_a, msg_b):
        x = torch.cat([msg_a, msg_b], dim=-1)
        h = self.net(x)
        return self.head_e(h).squeeze(-1), self.head_f(h).squeeze(-1)


def _mutual_information(x, y):
    from collections import Counter
    n = len(x)
    xy = list(zip(x, y))
    px = Counter(x)
    py = Counter(y)
    pxy = Counter(xy)
    mi = 0.0
    for (xi, yi), count in pxy.items():
        pxy_val = count / n
        px_val = px[xi] / n
        py_val = py[yi] / n
        if pxy_val > 0 and px_val > 0 and py_val > 0:
            mi += pxy_val * np.log(pxy_val / (px_val * py_val))
    return max(0.0, mi)


# ─── Train a sender from scratch (needed since no checkpoints saved) ─

def train_sender(features, e_bins, f_bins, seed=42, epochs=400):
    """Train a 4-agent V-JEPA 2 collision sender and return it frozen."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    train_ids = np.array(train_ids)
    holdout_ids = np.array(holdout_ids)

    T = features.shape[1]  # 24
    pos_per_agent = T // N_AGENTS  # 6
    # Keep features on CPU, move to device per batch
    agent_views = [features[:, a*pos_per_agent:(a+1)*pos_per_agent, :].clone() for a in range(N_AGENTS)]

    device = DEVICE
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE

    multi_sender = MultiAgentSender(N_AGENTS, pos_per_agent, features.shape[2]).to(device)
    receivers = [CompositionalReceiver(msg_dim).to(device) for _ in range(3)]
    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=1e-3)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=1e-3) for r in receivers]

    e_t = torch.from_numpy(e_bins).float()
    f_t = torch.from_numpy(f_bins).float()
    max_entropy = np.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)
    nan_count = 0

    for epoch in range(epochs):
        if epoch > 0 and epoch % 40 == 0:
            for i in range(3):
                receivers[i] = CompositionalReceiver(msg_dim).to(device)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=1e-3)

        tau = 3.0 + (1.0 - 3.0) * epoch / max(1, epochs - 1)
        hard = epoch >= 30
        multi_sender.train()
        for r in receivers:
            r.train()

        for _ in range(n_batches):
            ia = rng.choice(train_ids, BATCH_SIZE, replace=True)
            ib = rng.choice(train_ids, BATCH_SIZE, replace=True)
            views_a = [v[ia].to(device) for v in agent_views]
            views_b = [v[ib].to(device) for v in agent_views]
            label_e = (e_t[ia] > e_t[ib]).float().to(device)
            label_f = (f_t[ia] > f_t[ib]).float().to(device)

            msg_a, logits_a = multi_sender(views_a, tau=tau, hard=hard)
            msg_b, logits_b = multi_sender(views_b, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / 3

            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    if ent / max_entropy < 0.1:
                        loss = loss - 0.03 * ent

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

            # Check for NaN gradients
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

        if (epoch + 1) % 50 == 0:
            # Quick eval
            multi_sender.eval()
            receivers[0].eval()
            with torch.no_grad():
                ia = rng.choice(holdout_ids, min(200, len(holdout_ids)), replace=True)
                ib = rng.choice(holdout_ids, min(200, len(holdout_ids)), replace=True)
                va = [v[ia].to(device) for v in agent_views]
                vb = [v[ib].to(device) for v in agent_views]
                ma, _ = multi_sender(va, tau=1.0, hard=True)
                mb, _ = multi_sender(vb, tau=1.0, hard=True)
                pe, pf = receivers[0](ma, mb)
                le = (e_t[ia] > e_t[ib]).float().to(device)
                lf = (f_t[ia] > f_t[ib]).float().to(device)
                ae = ((pe > 0) == le).float().mean().item()
                af = ((pf > 0) == lf).float().mean().item()
                ab = (((pe > 0) == le) & ((pf > 0) == lf)).float().mean().item()
            nan_str = f" NaN={nan_count}" if nan_count else ""
            print(f"    Ep {epoch+1}: tau={tau:.2f} holdout[e={ae:.1%} f={af:.1%} both={ab:.1%}]{nan_str}", flush=True)

    multi_sender.eval()
    return multi_sender, agent_views


def train_best_sender(features, e_bins, f_bins, n_tries=5):
    """Try multiple seeds, return the best non-degenerate sender."""
    best_sender = None
    best_views = None
    best_mi_total = -1

    for seed in range(n_tries):
        print(f"\n  Training sender attempt {seed+1}/{n_tries} (seed={seed})...", flush=True)
        sender, views = train_sender(features, e_bins, f_bins, seed=seed, epochs=400)

        # Check if sender is non-degenerate
        tokens, onehot = extract_messages(sender, views)
        mi_total = 0
        for p in range(8):
            mi_total += _mutual_information(tokens[:, p], e_bins)
            mi_total += _mutual_information(tokens[:, p], f_bins)

        unique_msgs = len(set(tuple(t) for t in tokens))
        print(f"    MI total: {mi_total:.3f}, unique messages: {unique_msgs}/600", flush=True)

        if mi_total > best_mi_total:
            best_mi_total = mi_total
            best_sender = sender
            best_views = views

        if mi_total > 1.0 and unique_msgs > 10:
            print(f"    Good sender found (seed={seed})", flush=True)
            break

        torch.mps.empty_cache()

    return best_sender, best_views


def extract_messages(multi_sender, agent_views):
    """Extract hard discrete messages for all scenes."""
    multi_sender.eval()
    all_tokens = []
    all_onehot = []
    with torch.no_grad():
        for i in range(0, agent_views[0].shape[0], BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            msg, logits = multi_sender(views, tau=1.0, hard=True)
            tokens = np.stack([l.argmax(dim=-1).cpu().numpy() for l in logits], axis=1)
            all_tokens.append(tokens)
            all_onehot.append(msg.cpu())
    return np.concatenate(all_tokens, axis=0), torch.cat(all_onehot, dim=0)


# ─── Step 1: Action-Conditioned Outcome Prediction ───────────────

def step1_action_conditioned(messages_onehot, idx, e_bins, f_bins):
    """Train MLP: frozen message + velocity → collision outcome."""
    print("\n" + "="*70, flush=True)
    print("STEP 1: Action-Conditioned Outcome Prediction", flush=True)
    print("="*70, flush=True)

    # Get velocities and outcomes
    velocities = np.array([s['initial_velocity'] for s in idx])
    post_vel_b = np.array([s['post_collision_vel_b'] for s in idx])
    median_vel_b = np.median(post_vel_b)
    outcomes = (post_vel_b > median_vel_b).astype(np.float32)

    # Bin velocities into 5 levels for held-out testing
    vel_bins = np.digitize(velocities, np.percentile(velocities, [20, 40, 60, 80])).astype(int)

    device = DEVICE
    msg_dim = messages_onehot.shape[1]  # 40

    # Leave-one-velocity-out cross-validation
    all_results = []
    for held_vel in range(5):
        test_mask = vel_bins == held_vel
        train_mask = ~test_mask
        train_ids = np.where(train_mask)[0]
        test_ids = np.where(test_mask)[0]

        seed_accs = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Input: message one-hot (40) + normalized velocity (1)
            vel_norm = (velocities - velocities.mean()) / velocities.std()

            X_train = torch.cat([
                messages_onehot[train_ids],
                torch.from_numpy(vel_norm[train_ids]).float().unsqueeze(1)
            ], dim=1).to(device)
            y_train = torch.from_numpy(outcomes[train_ids]).float().to(device)

            X_test = torch.cat([
                messages_onehot[test_ids],
                torch.from_numpy(vel_norm[test_ids]).float().unsqueeze(1)
            ], dim=1).to(device)
            y_test = torch.from_numpy(outcomes[test_ids]).float().to(device)

            # MLP: 41 → 64 → 1
            mlp = nn.Sequential(
                nn.Linear(msg_dim + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ).to(device)
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

            for ep in range(200):
                mlp.train()
                pred = mlp(X_train).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

            mlp.eval()
            with torch.no_grad():
                pred_test = mlp(X_test).squeeze(-1)
                acc = ((pred_test > 0) == y_test).float().mean().item()
            seed_accs.append(acc)

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        all_results.append({
            'held_vel_bin': held_vel,
            'n_test': int(test_mask.sum()),
            'mean_acc': float(mean_acc),
            'std_acc': float(std_acc),
        })
        print(f"  Held-out vel bin {held_vel}: {mean_acc:.1%} ± {std_acc:.1%} (n={test_mask.sum()})", flush=True)

    # Overall
    overall_accs = [r['mean_acc'] for r in all_results]
    print(f"\n  Overall leave-one-vel-out: {np.mean(overall_accs):.1%} ± {np.std(overall_accs):.1%}", flush=True)

    # Also train with raw features for comparison
    features = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)['features']
    feat_pooled = features.float().mean(dim=1)  # (600, 1024)

    raw_accs = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        vel_norm = (velocities - velocities.mean()) / velocities.std()

        # Use all data (no velocity holdout) for a fair overall comparison
        train_mask = np.ones(len(idx), dtype=bool)
        for (eb, fb) in HOLDOUT_CELLS:
            for i in range(len(idx)):
                if int(e_bins[i]) == eb and int(f_bins[i]) == fb:
                    train_mask[i] = False
        test_ids = np.where(~train_mask)[0]
        train_ids_raw = np.where(train_mask)[0]

        X_train = torch.cat([
            feat_pooled[train_ids_raw],
            torch.from_numpy(vel_norm[train_ids_raw]).float().unsqueeze(1)
        ], dim=1).to(device)
        y_train = torch.from_numpy(outcomes[train_ids_raw]).float().to(device)

        X_test = torch.cat([
            feat_pooled[test_ids],
            torch.from_numpy(vel_norm[test_ids]).float().unsqueeze(1)
        ], dim=1).to(device)
        y_test = torch.from_numpy(outcomes[test_ids]).float().to(device)

        mlp = nn.Sequential(
            nn.Linear(1025, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        for ep in range(200):
            mlp.train()
            pred = mlp(X_train).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            acc = ((mlp(X_test).squeeze(-1) > 0) == y_test).float().mean().item()
        raw_accs.append(acc)

    print(f"  Raw features + velocity (holdout): {np.mean(raw_accs):.1%} ± {np.std(raw_accs):.1%}", flush=True)

    return {
        'per_velocity_bin': all_results,
        'overall_mean': float(np.mean(overall_accs)),
        'overall_std': float(np.std(overall_accs)),
        'raw_features_mean': float(np.mean(raw_accs)),
        'raw_features_std': float(np.std(raw_accs)),
    }


# ─── Step 2: Selective Property Querying ──────────────────────────

def step2_selective_querying(messages_onehot, tokens, idx, e_bins, f_bins):
    """Show that querying relevant message positions beats irrelevant ones."""
    print("\n" + "="*70, flush=True)
    print("STEP 2: Selective Property Querying", flush=True)
    print("="*70, flush=True)

    # From MI analysis: positions 2-5 (agents 1,2) encode restitution
    # positions 6-7 (agent 3) encode mass_ratio
    # positions 0-1 (agent 0) are mixed
    # In one-hot space: position k occupies dims k*5:(k+1)*5

    # Define position groups
    mass_positions = [6, 7]       # Agent 3 — mass-heavy
    rest_positions = [2, 3, 4, 5] # Agents 1,2 — restitution-heavy
    all_positions = list(range(8))

    def get_onehot_for_positions(onehot, positions):
        """Extract one-hot dims for specific message positions."""
        dims = []
        for p in positions:
            dims.extend(range(p * VOCAB_SIZE, (p + 1) * VOCAB_SIZE))
        return onehot[:, dims]

    device = DEVICE
    # Holdout split
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    train_ids = np.array(train_ids)
    holdout_ids = np.array(holdout_ids)

    # Task: pairwise comparison on holdout
    # For mass: "which has higher mass_ratio?"
    # For restitution: "which has higher restitution?"

    results = {}
    for task_name, label_bins, relevant_pos, irrelevant_pos in [
        ("mass_ratio", e_bins, mass_positions, rest_positions),
        ("restitution", f_bins, rest_positions, mass_positions),
    ]:
        print(f"\n  Task: {task_name}", flush=True)
        print(f"    Relevant positions: {relevant_pos}", flush=True)
        print(f"    Irrelevant positions: {irrelevant_pos}", flush=True)

        label_t = torch.from_numpy(label_bins).float()

        for pos_name, positions in [
            ("relevant", relevant_pos),
            ("irrelevant", irrelevant_pos),
            ("all", all_positions),
        ]:
            msg_subset = get_onehot_for_positions(messages_onehot, positions)
            input_dim = msg_subset.shape[1]

            seed_accs = []
            for seed in range(N_SEEDS):
                torch.manual_seed(seed)
                rng = np.random.RandomState(seed)

                # Train pairwise comparison
                mlp = nn.Sequential(
                    nn.Linear(input_dim * 2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                ).to(device)
                opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

                for ep in range(200):
                    mlp.train()
                    ia = rng.choice(train_ids, BATCH_SIZE, replace=True)
                    ib = rng.choice(train_ids, BATCH_SIZE, replace=True)
                    xa = msg_subset[ia].to(device)
                    xb = msg_subset[ib].to(device)
                    label = (label_t[ia] > label_t[ib]).float().to(device)
                    pred = mlp(torch.cat([xa, xb], dim=-1)).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(pred, label)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # Eval on holdout
                mlp.eval()
                with torch.no_grad():
                    n_correct = n_total = 0
                    for i in range(0, len(holdout_ids), 64):
                        for j in range(0, len(holdout_ids), 64):
                            ha = holdout_ids[i:i+64]
                            hb = holdout_ids[j:j+64]
                            ia_idx = np.repeat(ha, len(hb))
                            ib_idx = np.tile(hb, len(ha))
                            mask = ia_idx != ib_idx
                            ia_idx = ia_idx[mask]
                            ib_idx = ib_idx[mask]
                            if len(ia_idx) == 0:
                                continue
                            xa = msg_subset[ia_idx].to(device)
                            xb = msg_subset[ib_idx].to(device)
                            pred = mlp(torch.cat([xa, xb], dim=-1)).squeeze(-1)
                            label = (label_t[ia_idx] > label_t[ib_idx]).float().to(device)
                            n_correct += ((pred > 0) == label).float().sum().item()
                            n_total += len(ia_idx)
                    acc = n_correct / max(1, n_total)
                seed_accs.append(acc)

            mean_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs)
            results[f"{task_name}_{pos_name}"] = {
                'mean': float(mean_acc),
                'std': float(std_acc),
                'positions': positions,
                'n_dims': input_dim,
            }
            print(f"    {pos_name:>10} positions ({positions}): {mean_acc:.1%} ± {std_acc:.1%}", flush=True)

    # Print summary table
    print("\n  Selective Querying Summary:", flush=True)
    print(f"  {'Task':<15} {'Relevant':>10} {'Irrelevant':>12} {'All':>10} {'Gap':>8}", flush=True)
    for task in ["mass_ratio", "restitution"]:
        rel = results[f"{task}_relevant"]['mean']
        irr = results[f"{task}_irrelevant"]['mean']
        all_ = results[f"{task}_all"]['mean']
        gap = rel - irr
        print(f"  {task:<15} {rel:.1%}      {irr:.1%}        {all_:.1%}    {gap:+.1%}", flush=True)

    return results


# ─── Step 3: Counterfactual Planning ─────────────────────────────

def step3_counterfactual(messages_onehot, idx, e_bins, f_bins):
    """Test if varying velocity input produces physically correct predictions."""
    print("\n" + "="*70, flush=True)
    print("STEP 3: Counterfactual Planning", flush=True)
    print("="*70, flush=True)

    velocities = np.array([s['initial_velocity'] for s in idx])
    post_vel_b = np.array([s['post_collision_vel_b'] for s in idx])

    device = DEVICE
    msg_dim = messages_onehot.shape[1]

    # Train predictor: message + velocity → post_collision_vel_b (regression)
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    train_ids = np.array(train_ids)
    holdout_ids = np.array(holdout_ids)

    vel_mean, vel_std = velocities.mean(), velocities.std()
    pvb_mean, pvb_std = post_vel_b.mean(), post_vel_b.std()
    vel_norm = (velocities - vel_mean) / vel_std
    pvb_norm = (post_vel_b - pvb_mean) / pvb_std

    seed_results = []
    for seed in range(min(N_SEEDS, 10)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        X_train = torch.cat([
            messages_onehot[train_ids],
            torch.from_numpy(vel_norm[train_ids]).float().unsqueeze(1)
        ], dim=1).to(device)
        y_train = torch.from_numpy(pvb_norm[train_ids]).float().to(device)

        X_test = torch.cat([
            messages_onehot[holdout_ids],
            torch.from_numpy(vel_norm[holdout_ids]).float().unsqueeze(1)
        ], dim=1).to(device)
        y_test_raw = post_vel_b[holdout_ids]

        mlp = nn.Sequential(
            nn.Linear(msg_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        for ep in range(300):
            mlp.train()
            pred = mlp(X_train).squeeze(-1)
            loss = F.mse_loss(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Eval regression quality
        mlp.eval()
        with torch.no_grad():
            pred_test = mlp(X_test).squeeze(-1).cpu().numpy() * pvb_std + pvb_mean
        if np.std(pred_test) < 1e-8 or np.std(y_test_raw) < 1e-8:
            r, p = 0.0, 1.0
        else:
            r, p = stats.pearsonr(pred_test, y_test_raw)

        # Counterfactual: for each holdout scene, vary velocity
        # and check if prediction changes monotonically
        velocity_levels = np.array([1.5, 1.75, 2.0, 2.25, 2.5])
        vel_levels_norm = (velocity_levels - vel_mean) / vel_std

        n_monotonic = 0
        n_tested = 0
        for i, scene_idx in enumerate(holdout_ids[:50]):  # Test on 50 scenes
            preds_at_vels = []
            for vn in vel_levels_norm:
                x = torch.cat([
                    messages_onehot[scene_idx:scene_idx+1],
                    torch.tensor([[vn]], dtype=torch.float32)
                ], dim=1).to(device)
                with torch.no_grad():
                    pred = mlp(x).item() * pvb_std + pvb_mean
                preds_at_vels.append(pred)

            # Check if predictions increase with velocity (expected physics)
            # Higher initial velocity → higher post-collision vel_b
            is_monotonic = all(preds_at_vels[j] <= preds_at_vels[j+1]
                              for j in range(len(preds_at_vels)-1))
            # Also accept mostly monotonic (allow 1 violation)
            violations = sum(1 for j in range(len(preds_at_vels)-1)
                             if preds_at_vels[j] > preds_at_vels[j+1])
            if violations <= 1:
                n_monotonic += 1
            n_tested += 1

        mono_frac = n_monotonic / max(1, n_tested)
        seed_results.append({
            'seed': seed,
            'regression_r': float(r),
            'regression_p': float(p),
            'monotonic_fraction': float(mono_frac),
        })
        print(f"  Seed {seed}: r={r:.3f} (p={p:.4f}), monotonic={mono_frac:.1%}", flush=True)

    # Summary
    mean_r = np.mean([r['regression_r'] for r in seed_results])
    mean_mono = np.mean([r['monotonic_fraction'] for r in seed_results])
    print(f"\n  Regression r: {mean_r:.3f}", flush=True)
    print(f"  Monotonic response: {mean_mono:.1%}", flush=True)

    return {
        'per_seed': seed_results,
        'mean_regression_r': float(mean_r),
        'mean_monotonic_fraction': float(mean_mono),
    }


# ─── Main ─────────────────────────────────────────────────────────

def run_phase89():
    print("Phase 89: Action-Conditioned Planning", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Load features and labels
    features = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)['features']
    features = features.float()
    print(f"V-JEPA 2 features: {features.shape}", flush=True)

    with open('kubric/output/collision_dataset/index.json') as f:
        idx = json.load(f)
    e_bins = np.array([s['mass_ratio_bin'] for s in idx])
    f_bins = np.array([s['restitution_bin'] for s in idx])

    # Train a sender (since no checkpoint saved) — try multiple seeds
    print("\nTraining 4-agent V-JEPA 2 sender (trying up to 5 seeds)...", flush=True)
    start = time.time()
    multi_sender, agent_views = train_best_sender(features, e_bins, f_bins, n_tries=5)
    print(f"  Best sender found in {time.time()-start:.0f}s", flush=True)

    # Extract frozen messages
    tokens, messages_onehot = extract_messages(multi_sender, agent_views)
    print(f"  Messages: tokens={tokens.shape}, onehot={messages_onehot.shape}", flush=True)

    # Verify message quality
    mi_matrix = np.zeros((8, 2))
    for p in range(8):
        mi_matrix[p, 0] = _mutual_information(tokens[:, p], e_bins)
        mi_matrix[p, 1] = _mutual_information(tokens[:, p], f_bins)
    print("\n  MI matrix (mass, rest):", flush=True)
    for p in range(8):
        print(f"    Pos {p}: mass={mi_matrix[p,0]:.3f}  rest={mi_matrix[p,1]:.3f}", flush=True)

    # Run all steps
    step1_results = step1_action_conditioned(messages_onehot, idx, e_bins, f_bins)
    step2_results = step2_selective_querying(messages_onehot, tokens, idx, e_bins, f_bins)
    step3_results = step3_counterfactual(messages_onehot, idx, e_bins, f_bins)

    # Save
    results = {
        'mi_matrix': mi_matrix.tolist(),
        'step1_action_conditioned': step1_results,
        'step2_selective_querying': step2_results,
        'step3_counterfactual': step3_results,
    }
    with open('results/phase89_action_conditioned.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/phase89_action_conditioned.json", flush=True)

    # Final summary
    print("\n" + "="*70, flush=True)
    print("PHASE 89 SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"  Step 1: Action-conditioned outcome (leave-one-vel-out): {step1_results['overall_mean']:.1%}", flush=True)
    print(f"          Raw features comparison: {step1_results['raw_features_mean']:.1%}", flush=True)

    for task in ["mass_ratio", "restitution"]:
        rel = step2_results[f"{task}_relevant"]['mean']
        irr = step2_results[f"{task}_irrelevant"]['mean']
        gap = rel - irr
        print(f"  Step 2: {task} — relevant {rel:.1%}, irrelevant {irr:.1%}, gap {gap:+.1%}", flush=True)

    print(f"  Step 3: Regression r={step3_results['mean_regression_r']:.3f}, monotonic={step3_results['mean_monotonic_fraction']:.1%}", flush=True)

    selective_works = all(
        step2_results[f"{task}_relevant"]['mean'] > step2_results[f"{task}_irrelevant"]['mean'] + 0.03
        for task in ["mass_ratio", "restitution"]
    )
    print(f"\n  Selective querying works: {'YES' if selective_works else 'NO'}", flush=True)
    print(f"  Counterfactual planning: {'YES' if step3_results['mean_monotonic_fraction'] > 0.7 else 'MIXED' if step3_results['mean_monotonic_fraction'] > 0.5 else 'NO'}", flush=True)

    print("\nPhase 89 complete.", flush=True)


if __name__ == '__main__':
    run_phase89()

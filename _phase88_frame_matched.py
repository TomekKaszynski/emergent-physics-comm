#!/usr/bin/env python3
"""Phase 88: Frame-matched DINOv2 backbone control.

Extract DINOv2 ViT-S features from ALL 48 collision frames,
then train 4-agent communication to compare with V-JEPA 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
from PIL import Image
from torchvision import transforms
from scipy import stats

# ─── Config ───────────────────────────────────────────────────────
DEVICE = 'mps'
N_SCENES = 600
N_FRAMES_48 = 48
N_FRAMES_24 = 24
FEAT_DIM = 384  # DINOv2 ViT-S
HIDDEN_DIM = 128
VOCAB_SIZE = 5
N_HEADS = 2
N_AGENTS = 4
BATCH_SIZE = 128
COMM_EPOCHS = 400
N_SEEDS = 20
N_RECEIVERS = 3
RECEIVER_RESET = 40
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESH = 0.1
ENTROPY_COEF = 0.03
SENDER_LR = 1e-3
RECEIVER_LR = 1e-3
GRAD_CLIP = 1.0

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}
COLLISION_DIR = 'kubric/output/collision_dataset'

# ─── Step 1: Feature Extraction ──────────────────────────────────

def extract_dinov2_48frame():
    """Extract DINOv2 ViT-S CLS tokens from all 48 frames per scene."""
    print("STEP 1: Extracting DINOv2 ViT-S features (48 frames per scene)", flush=True)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()

    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_features = []
    start = time.time()

    for scene_id in range(N_SCENES):
        scene_dir = os.path.join(COLLISION_DIR, f'scene_{scene_id:04d}')
        frames = []
        for fr in range(N_FRAMES_48):
            img = Image.open(os.path.join(scene_dir, f'rgba_{fr:05d}.png')).convert('RGB')
            frames.append(t(img))

        batch = torch.stack(frames)
        with torch.no_grad():
            feats = model(batch)  # (48, 384)
        all_features.append(feats.cpu())

        if (scene_id + 1) % 100 == 0:
            elapsed = time.time() - start
            eta = elapsed / (scene_id + 1) * (N_SCENES - scene_id - 1)
            print(f"  Scene {scene_id+1}/{N_SCENES} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    features_48 = torch.stack(all_features)  # (600, 48, 384)

    # Also create 24-frame temporally aggregated version
    # Average adjacent pairs: (f0+f1)/2, (f2+f3)/2, ...
    features_agg = (features_48[:, 0::2, :] + features_48[:, 1::2, :]) / 2  # (600, 24, 384)

    # Load labels from index
    with open(os.path.join(COLLISION_DIR, 'index.json')) as f:
        idx = json.load(f)
    mass_bins = np.array([s['mass_ratio_bin'] for s in idx])
    rest_bins = np.array([s['restitution_bin'] for s in idx])

    torch.save({
        'features_48': features_48,
        'features_agg24': features_agg,
        'mass_bins': torch.from_numpy(mass_bins),
        'rest_bins': torch.from_numpy(rest_bins),
        'model': 'dinov2_vits14',
    }, 'results/phase88_dinov2_48frame_features.pt')

    elapsed = time.time() - start
    print(f"  Saved features_48: {features_48.shape}, features_agg24: {features_agg.shape} ({elapsed:.0f}s)", flush=True)
    return features_48, features_agg, mass_bins, rest_bins


# ─── Models ───────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=384, n_positions=12):
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
        # x: (B, T, 384)
        x = x.permute(0, 2, 1)  # (B, 384, T)
        x = self.temporal(x).squeeze(-1)  # (B, 128)
        return self.fc(x)


class CompositionalSender(nn.Module):
    def __init__(self, input_dim=384, n_positions=12):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, n_positions)
        self.heads = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE) for _ in range(N_HEADS)
        ])

    def forward(self, x, tau=1.0, hard=False):
        h = self.encoder(x)
        logits_list = [head(h) for head in self.heads]
        msgs = []
        for logits in logits_list:
            if hard:
                msg = F.gumbel_softmax(logits, tau=tau, hard=True)
            else:
                msg = F.gumbel_softmax(logits, tau=tau, hard=False)
            msgs.append(msg)
        return torch.cat(msgs, dim=-1), logits_list


class MultiAgentSender(nn.Module):
    def __init__(self, n_agents, positions_per_agent, input_dim=384):
        super().__init__()
        self.n_agents = n_agents
        self.positions_per_agent = positions_per_agent
        self.senders = nn.ModuleList([
            CompositionalSender(input_dim, positions_per_agent)
            for _ in range(n_agents)
        ])

    def forward(self, agent_views, tau=1.0, hard=False):
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


# ─── Training ─────────────────────────────────────────────────────

def create_splits(e_bins, f_bins):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


def _mutual_information(x, y):
    """Compute MI between discrete x and discrete y."""
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


def compute_metrics(multi_sender, agent_views, e_bins, f_bins, device):
    """Compute PosDis, TopSim, MI matrix."""
    multi_sender.eval()
    n_scenes = len(e_bins)
    all_tokens = []

    with torch.no_grad():
        for i in range(0, n_scenes, BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(device) for v in agent_views]
            _, logits = multi_sender(views, tau=1.0, hard=True)
            tokens = np.stack([l.argmax(dim=-1).cpu().numpy() for l in logits], axis=1)
            all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, 8)
    n_pos = all_tokens.shape[1]
    max_ent = np.log(VOCAB_SIZE)

    # Entropies
    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / max_ent))

    # MI matrix
    attributes = np.stack([e_bins, f_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    # PosDis (per-agent best)
    per_agent_posdis = []
    for agent_idx in range(N_AGENTS):
        start_p = agent_idx * N_HEADS
        agent_mi = mi_matrix[start_p:start_p + N_HEADS]
        agent_pd = 0.0
        for p in range(N_HEADS):
            sorted_mi = np.sort(agent_mi[p])[::-1]
            if sorted_mi[0] > 1e-10:
                agent_pd += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        agent_pd /= N_HEADS
        per_agent_posdis.append(float(agent_pd))
    pos_dis = max(per_agent_posdis)

    # Global PosDis
    global_pd = 0.0
    for p in range(n_pos):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if sorted_mi[0] > 1e-10:
            global_pd += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
    global_pd /= n_pos

    # TopSim
    rng = np.random.RandomState(42)
    n_pairs = min(5000, n_scenes * (n_scenes - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(n_scenes, size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'pos_dis': float(pos_dis),
        'pos_dis_global': float(global_pd),
        'topsim': float(topsim),
        'entropies': entropies,
        'mi_matrix': mi_matrix.tolist(),
    }


def train_seed(features, e_bins, f_bins, seed, n_positions_per_agent, label=""):
    """Train one seed of 4-agent communication."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    train_ids, holdout_ids = create_splits(e_bins, f_bins)
    n_total = len(features)
    T = features.shape[1]  # 48 or 24

    # Normalize features to prevent NaN (zero mean, unit variance per feature dim)
    feat_mean = features.mean(dim=(0, 1), keepdim=True)
    feat_std = features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    features_norm = (features - feat_mean) / feat_std

    # Split temporal positions across agents
    agent_views = []
    for a in range(N_AGENTS):
        start_t = a * n_positions_per_agent
        end_t = start_t + n_positions_per_agent
        agent_views.append(features_norm[:, start_t:end_t, :])

    device = DEVICE
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE

    multi_sender = MultiAgentSender(N_AGENTS, n_positions_per_agent, FEAT_DIM).to(device)
    receivers = [CompositionalReceiver(msg_dim).to(device) for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

    e_t = torch.from_numpy(e_bins).float()
    f_t = torch.from_numpy(f_bins).float()
    max_entropy = np.log(VOCAB_SIZE)

    n_batches = max(1, len(train_ids) // BATCH_SIZE)
    best_holdout = 0.0
    nan_count = 0
    start_time = time.time()

    for epoch in range(COMM_EPOCHS):
        # Receiver reset
        if epoch > 0 and epoch % RECEIVER_RESET == 0:
            for i in range(N_RECEIVERS):
                receivers[i] = CompositionalReceiver(msg_dim).to(device)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        multi_sender.train()
        for r in receivers:
            r.train()

        epoch_nan = 0
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
            loss = total_loss / N_RECEIVERS

            # Entropy reg
            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESH:
                        loss = loss - ENTROPY_COEF * ent

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

            # Check for NaN gradients (matching Phase 79)
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
                epoch_nan += 1
                continue

            torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), GRAD_CLIP)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), GRAD_CLIP)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        # Eval every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == COMM_EPOCHS - 1:
            multi_sender.eval()
            receivers[0].eval()

            with torch.no_grad():
                # Holdout
                n_correct_e = n_correct_f = n_correct_b = n_total_h = 0
                for i in range(0, len(holdout_ids), BATCH_SIZE):
                    for j in range(0, len(holdout_ids), BATCH_SIZE):
                        ha = holdout_ids[i:i+BATCH_SIZE]
                        hb = holdout_ids[j:j+BATCH_SIZE]
                        if len(ha) == 0 or len(hb) == 0:
                            continue
                        # Use all pairs
                        ia_idx = np.repeat(ha, len(hb))
                        ib_idx = np.tile(hb, len(ha))
                        # Filter same
                        mask = ia_idx != ib_idx
                        ia_idx = ia_idx[mask]
                        ib_idx = ib_idx[mask]
                        if len(ia_idx) == 0:
                            continue

                        views_a = [v[ia_idx].to(device) for v in agent_views]
                        views_b = [v[ib_idx].to(device) for v in agent_views]
                        msg_a, _ = multi_sender(views_a, tau=1.0, hard=True)
                        msg_b, _ = multi_sender(views_b, tau=1.0, hard=True)
                        pred_e, pred_f = receivers[0](msg_a, msg_b)

                        le = (e_t[ia_idx] > e_t[ib_idx]).float().to(device)
                        lf = (f_t[ia_idx] > f_t[ib_idx]).float().to(device)

                        ce = ((pred_e > 0) == le).float().sum().item()
                        cf = ((pred_f > 0) == lf).float().sum().item()
                        cb = (((pred_e > 0) == le) & ((pred_f > 0) == lf)).float().sum().item()
                        n_correct_e += ce
                        n_correct_f += cf
                        n_correct_b += cb
                        n_total_h += len(ia_idx)

                h_e = n_correct_e / max(1, n_total_h)
                h_f = n_correct_f / max(1, n_total_h)
                h_b = n_correct_b / max(1, n_total_h)
                best_holdout = max(best_holdout, h_b)

            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count else ""
            print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  holdout[e={h_e:.1%} f={h_f:.1%} both={h_b:.1%}]{nan_str}  ETA {eta/60:.0f}min", flush=True)

    # Compute metrics
    metrics = compute_metrics(multi_sender, agent_views, e_bins, f_bins, device)

    result = {
        'seed': seed,
        'holdout_both': float(best_holdout),
        'holdout_e': float(h_e),
        'holdout_f': float(h_f),
        'pos_dis': metrics['pos_dis'],
        'pos_dis_global': metrics['pos_dis_global'],
        'topsim': metrics['topsim'],
        'entropies': metrics['entropies'],
        'mi_matrix': metrics['mi_matrix'],
        'nan_count': nan_count,
        'time_sec': time.time() - start_time,
    }

    elapsed = time.time() - start_time
    print(f"    -> holdout={best_holdout:.1%}  PosDis={metrics['pos_dis']:.3f}  TopSim={metrics['topsim']:.3f}  NaN={nan_count}  ({elapsed:.0f}s)", flush=True)

    if (seed + 1) % 5 == 0:
        torch.mps.empty_cache()

    return result


def run_condition(features, e_bins, f_bins, n_positions_per_agent, label, save_path):
    """Run 20 seeds for one condition."""
    print(f"\n{'='*70}", flush=True)
    print(f"  {label} (20 seeds, {COMM_EPOCHS} epochs)", flush=True)
    print(f"  Features: {features.shape}, {n_positions_per_agent} positions/agent", flush=True)
    print(f"{'='*70}", flush=True)

    results = []
    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed} ---", flush=True)
        r = train_seed(features, e_bins, f_bins, seed, n_positions_per_agent, label)
        results.append(r)

    holdouts = [r['holdout_both'] for r in results]
    posdis = [r['pos_dis'] for r in results]
    topsim = [r['topsim'] for r in results]

    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis)),
        'pos_dis_std': float(np.std(posdis)),
        'topsim_mean': float(np.mean(topsim)),
        'topsim_std': float(np.std(topsim)),
        'compositional_count': sum(1 for p in posdis if p > 0.4),
        'n_seeds': N_SEEDS,
    }

    output = {'config': label, 'per_seed': results, 'summary': summary}
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  {label} summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']*100:.1f}% ± {summary['holdout_both_std']*100:.1f}%", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    TopSim:  {summary['topsim_mean']:.3f} ± {summary['topsim_std']:.3f}", flush=True)
    print(f"    Comp:    {summary['compositional_count']}/{N_SEEDS}", flush=True)
    print(f"  Saved {save_path}", flush=True)

    return summary


# ─── Main ─────────────────────────────────────────────────────────

def run_phase88():
    print("Phase 88: Frame-Matched DINOv2 Backbone Control", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Step 1: Extract features
    feat_path = 'results/phase88_dinov2_48frame_features.pt'
    if os.path.exists(feat_path):
        print("Loading cached 48-frame features...", flush=True)
        d = torch.load(feat_path, weights_only=True)
        features_48 = d['features_48'].float()
        features_agg = d['features_agg24'].float()
        e_bins = d['mass_bins'].numpy()
        f_bins = d['rest_bins'].numpy()
    else:
        features_48, features_agg, e_bins, f_bins = extract_dinov2_48frame()
        features_48 = features_48.float()
        features_agg = features_agg.float()

    print(f"Features 48-frame: {features_48.shape}", flush=True)
    print(f"Features agg-24: {features_agg.shape}", flush=True)

    # Step 2a: 48-frame condition (12 positions per agent)
    summary_48 = run_condition(
        features_48, e_bins, f_bins,
        n_positions_per_agent=12,
        label="DINOv2 ViT-S 48-frame",
        save_path='results/phase88_dinov2_48frame_4agent.json'
    )

    # Step 2b: Temporally aggregated 24-frame condition (6 positions per agent)
    summary_agg = run_condition(
        features_agg, e_bins, f_bins,
        n_positions_per_agent=6,
        label="DINOv2 ViT-S agg-24",
        save_path='results/phase88_dinov2_agg24_4agent.json'
    )

    # Step 3: Comparison table
    print("\n" + "="*70, flush=True)
    print("STEP 3: FRAME-MATCHED COMPARISON", flush=True)
    print("="*70, flush=True)

    # Load existing baselines
    with open('results/phase79_dinov2_4agent_collision.json') as f:
        r79 = json.load(f)
    with open('results/phase79b_vjepa2_4agent_collision.json') as f:
        r79b = json.load(f)
    with open('results/phase82_dinov2_vitl_collision_4agent.json') as f:
        r82 = json.load(f)

    vjepa2_holdouts = [s['holdout_both'] for s in r79b['per_seed']]
    dinov2_24_holdouts = [s['holdout_both'] for s in r79['per_seed']]
    vitl_holdouts = [s['holdout_both'] for s in r82['per_seed']]

    # Load our new results
    with open('results/phase88_dinov2_48frame_4agent.json') as f:
        r48 = json.load(f)
    with open('results/phase88_dinov2_agg24_4agent.json') as f:
        ragg = json.load(f)

    dinov2_48_holdouts = [s['holdout_both'] for s in r48['per_seed']]
    dinov2_agg_holdouts = [s['holdout_both'] for s in ragg['per_seed']]

    conditions = [
        ("DINOv2 ViT-S (24fr)", 24, "22M", dinov2_24_holdouts, r79['summary']),
        ("DINOv2 ViT-S (48fr)", 48, "22M", dinov2_48_holdouts, r48['summary']),
        ("DINOv2 ViT-S (agg24)", "24*", "22M", dinov2_agg_holdouts, ragg['summary']),
        ("DINOv2 ViT-L (24fr)", 24, "304M", vitl_holdouts, r82['summary']),
        ("V-JEPA 2 ViT-L (48fr)", 48, "304M", vjepa2_holdouts, r79b['summary']),
    ]

    print(f"\n  {'Backbone':<24} {'Frames':>6} {'Params':>7} {'Holdout':>12} {'PosDis':>8} {'d vs VJEPA2':>12}", flush=True)
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*12}", flush=True)

    for name, frames, params, holdouts, summary in conditions:
        mean_h = np.mean(holdouts)
        std_h = np.std(holdouts)
        pd = summary.get('pos_dis_mean', summary.get('posdis_mean', 0))

        # Cohen's d vs V-JEPA2
        if name != "V-JEPA 2 ViT-L (48fr)":
            pooled_std = np.sqrt((np.std(holdouts)**2 + np.std(vjepa2_holdouts)**2) / 2)
            d_val = (np.mean(holdouts) - np.mean(vjepa2_holdouts)) / max(pooled_std, 1e-10)
            t_stat, p_val = stats.ttest_ind(holdouts, vjepa2_holdouts)
            d_str = f"{d_val:+.2f} (p={p_val:.4f})"
        else:
            d_str = "---"

        print(f"  {name:<24} {str(frames):>6} {params:>7} {mean_h*100:.1f}% ± {std_h*100:.1f}% {pd:.3f}    {d_str}", flush=True)

    # Statistical tests
    print("\n  --- Statistical Tests ---", flush=True)

    t1, p1 = stats.ttest_ind(dinov2_48_holdouts, vjepa2_holdouts)
    d1 = (np.mean(dinov2_48_holdouts) - np.mean(vjepa2_holdouts)) / np.sqrt((np.std(dinov2_48_holdouts)**2 + np.std(vjepa2_holdouts)**2) / 2)
    print(f"  DINOv2 48fr vs V-JEPA2: t={t1:.3f}, p={p1:.6f}, d={d1:.3f}", flush=True)

    t2, p2 = stats.ttest_ind(dinov2_48_holdouts, dinov2_24_holdouts)
    d2 = (np.mean(dinov2_48_holdouts) - np.mean(dinov2_24_holdouts)) / np.sqrt((np.std(dinov2_48_holdouts)**2 + np.std(dinov2_24_holdouts)**2) / 2)
    print(f"  DINOv2 48fr vs 24fr: t={t2:.3f}, p={p2:.6f}, d={d2:.3f}", flush=True)

    t3, p3 = stats.ttest_ind(dinov2_agg_holdouts, dinov2_24_holdouts)
    d3 = (np.mean(dinov2_agg_holdouts) - np.mean(dinov2_24_holdouts)) / np.sqrt((np.std(dinov2_agg_holdouts)**2 + np.std(dinov2_24_holdouts)**2) / 2)
    print(f"  DINOv2 agg24 vs 24fr: t={t3:.3f}, p={p3:.6f}, d={d3:.3f}", flush=True)

    # Interpretation
    gap_48 = np.mean(vjepa2_holdouts) - np.mean(dinov2_48_holdouts)
    print(f"\n  V-JEPA2 advantage over DINOv2-48fr: {gap_48*100:.1f}pp", flush=True)
    if np.mean(dinov2_48_holdouts) < 0.82:
        print("  → Frame gap survives: video-native pretraining drives the advantage.", flush=True)
    else:
        print("  → Frame equalization narrows the gap. Report honestly.", flush=True)

    # Save combined results
    combined = {
        'dinov2_48frame': r48['summary'],
        'dinov2_agg24': ragg['summary'],
        'dinov2_24frame': {'holdout_both_mean': np.mean(dinov2_24_holdouts), 'holdout_both_std': np.std(dinov2_24_holdouts)},
        'dinov2_vitl_24frame': {'holdout_both_mean': np.mean(vitl_holdouts), 'holdout_both_std': np.std(vitl_holdouts)},
        'vjepa2_48frame': {'holdout_both_mean': np.mean(vjepa2_holdouts), 'holdout_both_std': np.std(vjepa2_holdouts)},
        'tests': {
            '48fr_vs_vjepa2': {'t': float(t1), 'p': float(p1), 'd': float(d1)},
            '48fr_vs_24fr': {'t': float(t2), 'p': float(p2), 'd': float(d2)},
            'agg24_vs_24fr': {'t': float(t3), 'p': float(p3), 'd': float(d3)},
        }
    }
    with open('results/phase88_frame_matched.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print("\n  Saved results/phase88_frame_matched.json", flush=True)

    print("\nPhase 88 complete.", flush=True)


if __name__ == '__main__':
    run_phase88()

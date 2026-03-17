"""
Phase 85: Real-Video Transfer — CoPhy CollisionCF
===================================================
Zero-shot transfer: frozen V-JEPA 2 sender (trained on synthetic Kubric)
applied to real CoPhy collision videos.

STEP 1: Load CoPhy CollisionCF videos and confounders
STEP 2: Extract V-JEPA 2 features
STEP 3: Run frozen sender (zero-shot)
STEP 4: Analyze symbol-property correlations
STEP 5: Save results

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase85_cophy_transfer.py
"""

import time
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from PIL import Image
import imageio

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

# Sender architecture constants (must match Phase 79b exactly)
HIDDEN_DIM = 128
VJEPA_DIM = 1024
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
N_AGENTS = 4
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
HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

# CoPhy paths
COPHY_DIR = Path("cophy/CoPhy_224/collisionCF")
COPHY_SPLITS_DIR = Path("/tmp/cophy_repo/dataloaders/splits")


# ══════════════════════════════════════════════════════════════════
# Sender Architecture (exact copy from Phase 79b)
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


# ══════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════

def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


def create_splits(e_bins, f_bins, holdout_cells):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in holdout_cells:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


# ══════════════════════════════════════════════════════════════════
# STEP 1: Load CoPhy data
# ══════════════════════════════════════════════════════════════════

def step1_load_cophy():
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: Load CoPhy CollisionCF Data", flush=True)
    print("=" * 70, flush=True)

    # Find all CollisionCF trials
    cophy_root = COPHY_DIR
    if not cophy_root.exists():
        # Try alternate locations
        for alt in [Path("cophy/collisionCF"), Path("cophy/CoPhy_224/collisionCF")]:
            if alt.exists():
                cophy_root = alt
                break
        else:
            raise FileNotFoundError(f"CoPhy not found at {COPHY_DIR}")

    print(f"  CoPhy root: {cophy_root}", flush=True)

    # List all trial directories
    trial_dirs = sorted([d for d in cophy_root.iterdir()
                         if d.is_dir() and not d.name.startswith('.')])
    print(f"  Found {len(trial_dirs)} trials", flush=True)

    # Load confounders and identify valid trials
    valid_trials = []
    all_confounders = []
    all_ids = []

    for trial_dir in trial_dirs:
        conf_path = trial_dir / 'confounders.npy'
        cd_rgb = trial_dir / 'cd' / 'rgb.mp4'
        ab_rgb = trial_dir / 'ab' / 'rgb.mp4'

        if not conf_path.exists():
            continue

        # Use the 'cd' sequence (counterfactual) as it has a proper collision
        # Or 'ab' — either works for feature extraction
        rgb_path = cd_rgb if cd_rgb.exists() else ab_rgb
        if not rgb_path.exists():
            continue

        confounders = np.load(conf_path)  # (K, 3) = (num_objects, mass/friction/restitution)
        valid_trials.append({
            'dir': trial_dir,
            'rgb_path': str(rgb_path),
            'confounders': confounders,
            'id': trial_dir.name,
        })
        all_confounders.append(confounders)
        all_ids.append(trial_dir.name)

    n_trials = len(valid_trials)
    print(f"  Valid trials with confounders + video: {n_trials}", flush=True)

    if n_trials == 0:
        raise RuntimeError("No valid CoPhy trials found!")

    # Limit for computational feasibility
    MAX_TRIALS = 500
    if n_trials > MAX_TRIALS:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_trials, MAX_TRIALS, replace=False)
        valid_trials = [valid_trials[i] for i in sorted(indices)]
        all_confounders = [all_confounders[i] for i in sorted(indices)]
        n_trials = MAX_TRIALS
        print(f"  Subsampled to {n_trials} trials", flush=True)

    # Stack confounders
    confounders = np.stack(all_confounders)  # (N, K, 3)
    print(f"  Confounders shape: {confounders.shape}", flush=True)
    print(f"  Properties per object: mass, friction, restitution", flush=True)

    # Print distributions
    n_objects = confounders.shape[1]
    for obj_idx in range(min(n_objects, 4)):
        for prop_idx, prop_name in enumerate(['mass', 'friction', 'restitution']):
            vals = confounders[:, obj_idx, prop_idx]
            unique = np.unique(vals)
            print(f"    Object {obj_idx} {prop_name}: unique={unique}, "
                  f"range=[{vals.min():.2f}, {vals.max():.2f}]", flush=True)

    # Load a sample video to check frame count
    sample_video = valid_trials[0]['rgb_path']
    reader = imageio.get_reader(sample_video)
    n_frames_sample = reader.count_frames()
    reader.close()
    print(f"  Sample video frames: {n_frames_sample}", flush=True)

    return valid_trials, confounders


# ══════════════════════════════════════════════════════════════════
# STEP 2: Extract V-JEPA 2 features
# ══════════════════════════════════════════════════════════════════

def step2_extract_features(valid_trials):
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Extract V-JEPA 2 Features from CoPhy Videos", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModel
    from torchvision import transforms

    # Check for cached features
    cache_path = RESULTS_DIR / "phase85_cophy_vjepa2_features.pt"
    if cache_path.exists():
        print(f"  Loading cached features from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=True)
        features = data['features'].float()
        print(f"  Loaded: {features.shape}", flush=True)
        return features

    print("  Loading V-JEPA 2 model...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained('facebook/vjepa2-vitl-fpc64-256')
    model = model.to(DEVICE)
    model.eval()
    dt = time.time() - t0
    print(f"  Model loaded in {dt:.0f}s", flush=True)

    # V-JEPA 2 preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    n_trials = len(valid_trials)
    # V-JEPA 2 fpc64-256: CoPhy has 15 frames. Using 16 frames → 8 temporal
    # tokens, then interpolate to 24 to match synthetic sender expectations.
    # 48 frames OOMs on MPS, so use 16 and adapt.
    N_FRAMES_INPUT = 16   # Feasible on MPS
    N_TEMPORAL_RAW = 8    # 16 / tubelet_size 2 = 8
    N_SPATIAL = 256       # 256x256 / 16 patch = 16x16 = 256

    # Extract features
    all_features = []
    t_start = time.time()

    for si, trial in enumerate(valid_trials):
        try:
            reader = imageio.get_reader(trial['rgb_path'])
            n_total = reader.count_frames()

            # Read all available frames
            raw_frames = []
            for fi in range(n_total):
                frame = reader.get_data(fi)
                img = Image.fromarray(frame)
                raw_frames.append(transform(img))
            reader.close()

            # Sample to reach N_FRAMES_INPUT (16)
            # 15 frames → sample 16 (one frame repeated)
            indices = np.linspace(0, len(raw_frames) - 1, N_FRAMES_INPUT).astype(int)
            frames = [raw_frames[i] for i in indices]

            # Stack: (T, C, H, W) → (1, T, C, H, W)
            video = torch.stack(frames, dim=0).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

            with torch.no_grad():
                output = model(pixel_values_videos=video)
                hidden = output.last_hidden_state  # (1, T*spatial, D)

                n_tokens = hidden.shape[1]
                feat_dim = hidden.shape[2]
                n_temporal = n_tokens // N_SPATIAL  # Should be N_TEMPORAL_RAW
                if n_temporal < 1:
                    n_temporal = 1
                n_spatial = n_tokens // n_temporal
                hidden = hidden.view(1, n_temporal, n_spatial, feat_dim)
                pooled = hidden.mean(dim=2)  # (1, T_temporal, D)

                all_features.append(pooled.squeeze(0).cpu().half())

        except Exception as e:
            print(f"  ERROR on trial {si}: {e}", flush=True)
            # Fill with zeros
            all_features.append(torch.zeros(8, VJEPA_DIM).half())

        if (si + 1) % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (si + 1) * (n_trials - si - 1)
            print(f"  [{si+1}/{n_trials}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
            torch.mps.empty_cache()

    features = torch.stack(all_features)  # (N, T_temporal, 1024)
    print(f"  Features shape: {features.shape} {features.dtype}", flush=True)

    # Verify features are non-degenerate
    feat_var = features.float().var(dim=0).mean().item()
    print(f"  Feature variance (mean): {feat_var:.4f}", flush=True)
    if feat_var < 1e-6:
        print("  WARNING: Features have near-zero variance — extraction may have failed!", flush=True)

    # Compare to synthetic features
    synth = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)
    synth_feats = synth['features'].float()
    synth_var = synth_feats.var(dim=0).mean().item()
    print(f"  Synthetic feature variance: {synth_var:.4f}", flush=True)
    print(f"  Variance ratio (CoPhy/Synthetic): {feat_var/max(synth_var,1e-8):.2f}", flush=True)

    # Save cache
    torch.save({
        'features': features,
        'n_trials': n_trials,
        'n_temporal': features.shape[1],
    }, cache_path)
    print(f"  Saved {cache_path}", flush=True)

    del model
    torch.mps.empty_cache()

    return features.float()


# ══════════════════════════════════════════════════════════════════
# STEP 3: Train sender on synthetic data, then run on CoPhy
# ══════════════════════════════════════════════════════════════════

def make_4agent_sender(input_dim, n_agents, frames_per_agent):
    """Create a multi-agent sender."""
    senders = []
    for _ in range(n_agents):
        enc = TemporalEncoder(HIDDEN_DIM, input_dim)
        sender = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
        senders.append(sender)
    return MultiAgentSender(senders)


def train_sender_on_synthetic(seed=0):
    """Train a 4-agent sender on the synthetic collision dataset."""
    print("\n  Training sender on synthetic collision data (seed 0)...", flush=True)

    data = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)
    features = data['features'].float()  # (600, 24, 1024)
    index = data['index']
    mass_bins = np.array([e['mass_ratio_bin'] for e in index])
    rest_bins = np.array([e['restitution_bin'] for e in index])

    n_frames = features.shape[1]
    frames_per_agent = n_frames // N_AGENTS

    agent_views = [features[:, i*frames_per_agent:(i+1)*frames_per_agent, :]
                   for i in range(N_AGENTS)]
    train_ids, _ = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE

    torch.manual_seed(seed)
    np.random.seed(seed)
    multi_sender = make_4agent_sender(VJEPA_DIM, N_AGENTS, frames_per_agent).to(DEVICE)
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

    for epoch in range(COMM_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi_sender.train()
        for r in receivers: r.train()
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
                for opt in receiver_opts: opt.zero_grad()
                continue
            sender_opt.zero_grad()
            for opt in receiver_opts: opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), 1.0)
            for r in receivers: torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts: opt.step()

        if epoch % 50 == 0: torch.mps.empty_cache()

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
            print(f"    Ep {epoch+1}: train_both={acc:.1%}", flush=True)

    if best_state is not None:
        multi_sender.load_state_dict(best_state)
    print(f"  Sender trained: best accuracy = {best_acc:.1%}", flush=True)

    return multi_sender, features.shape[1]  # return n_frames for the synthetic data


def step3_zero_shot_transfer(sender, cophy_features, n_synth_frames):
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Zero-Shot Transfer — Run Frozen Sender on CoPhy", flush=True)
    print("=" * 70, flush=True)

    n_cophy_temporal = cophy_features.shape[1]
    n_synth_temporal = n_synth_frames
    synth_frames_per_agent = n_synth_temporal // N_AGENTS

    print(f"  CoPhy temporal positions: {n_cophy_temporal}", flush=True)
    print(f"  Synthetic temporal positions: {n_synth_temporal}", flush=True)
    print(f"  Synthetic frames per agent: {synth_frames_per_agent}", flush=True)

    # Adapt CoPhy features to match synthetic frame count
    # CoPhy has 8 temporal positions, synthetic has 24
    # Interpolate/repeat to match
    if n_cophy_temporal != n_synth_temporal:
        print(f"  Adapting {n_cophy_temporal} → {n_synth_temporal} temporal positions...", flush=True)
        # Use linear interpolation along temporal dimension
        cophy_adapted = F.interpolate(
            cophy_features.permute(0, 2, 1),  # (N, D, T)
            size=n_synth_temporal,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)  # (N, T_new, D)
        print(f"  Adapted features: {cophy_adapted.shape}", flush=True)
    else:
        cophy_adapted = cophy_features

    # Split into agent views
    frames_per_agent = n_synth_temporal // N_AGENTS
    agent_views = [cophy_adapted[:, i*frames_per_agent:(i+1)*frames_per_agent, :]
                   for i in range(N_AGENTS)]

    # Extract messages
    sender.eval()
    all_tokens = []
    all_messages = []
    n_scenes = len(cophy_features)

    with torch.no_grad():
        for i in range(0, n_scenes, BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            msg, logits = sender(views)
            tokens = [l.argmax(dim=-1).cpu() for l in logits]
            all_tokens.append(torch.stack(tokens, dim=1))
            all_messages.append(msg.cpu())

    tokens = torch.cat(all_tokens, dim=0)  # (N, 8) = 4 agents × 2 heads
    messages = torch.cat(all_messages, dim=0)  # (N, 40) one-hot

    print(f"  Tokens shape: {tokens.shape}", flush=True)
    print(f"  Messages shape: {messages.shape}", flush=True)

    # Check for degeneracy
    print(f"\n  Symbol frequency distribution:", flush=True)
    for pos in range(tokens.shape[1]):
        counts = torch.bincount(tokens[:, pos], minlength=VOCAB_SIZE)
        freqs = counts.float() / counts.sum()
        max_freq = freqs.max().item()
        print(f"    Position {pos}: {counts.tolist()} (max_freq={max_freq:.1%})", flush=True)

    # Check if collapsed
    max_freqs = []
    for pos in range(tokens.shape[1]):
        counts = torch.bincount(tokens[:, pos], minlength=VOCAB_SIZE)
        max_freqs.append(counts.float().max().item() / counts.sum().item())
    avg_max_freq = np.mean(max_freqs)
    collapsed = avg_max_freq > 0.80

    if collapsed:
        print(f"\n  WARNING: Messages appear collapsed (avg max freq = {avg_max_freq:.1%})", flush=True)
        print(f"  This is a NEGATIVE transfer result.", flush=True)
    else:
        print(f"\n  Messages are non-degenerate (avg max freq = {avg_max_freq:.1%})", flush=True)

    return tokens.numpy(), messages, collapsed


# ══════════════════════════════════════════════════════════════════
# STEP 4: Analysis
# ══════════════════════════════════════════════════════════════════

def step4_analysis(tokens, confounders, cophy_features):
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: Analysis", flush=True)
    print("=" * 70, flush=True)

    n_scenes = len(tokens)
    n_positions = tokens.shape[1]
    n_objects = confounders.shape[1]
    prop_names = ['mass', 'friction', 'restitution']

    results = {}

    # ─── Analysis 1: Symbol-Property Correlation ───
    print("\n  --- ANALYSIS 1: Symbol-Property Correlations ---", flush=True)

    correlations = {}
    n_significant = 0

    for pos in range(n_positions):
        for obj_idx in range(min(n_objects, 4)):
            for prop_idx, prop_name in enumerate(prop_names):
                prop_vals = confounders[:, obj_idx, prop_idx]

                # Skip if property is constant
                if np.std(prop_vals) < 1e-6:
                    continue

                # Skip constant token positions
                if np.std(tokens[:, pos]) < 1e-6:
                    continue

                # Spearman correlation
                rho, p_value = stats.spearmanr(tokens[:, pos], prop_vals)

                # Permutation test (1000 shuffles)
                perm_rhos = []
                rng = np.random.RandomState(42)
                for _ in range(1000):
                    shuffled = rng.permutation(prop_vals)
                    r, _ = stats.spearmanr(tokens[:, pos], shuffled)
                    perm_rhos.append(abs(r))
                p_perm = np.mean(np.array(perm_rhos) >= abs(rho))

                key = f"pos{pos}_obj{obj_idx}_{prop_name}"
                correlations[key] = {
                    'rho': float(rho),
                    'p_spearman': float(p_value),
                    'p_permutation': float(p_perm),
                }

                if p_perm < 0.05 and abs(rho) > 0.05:
                    n_significant += 1
                    print(f"    * pos={pos} obj={obj_idx} {prop_name}: "
                          f"rho={rho:+.3f} p_perm={p_perm:.4f}", flush=True)

    print(f"\n  Significant correlations (p_perm<0.05): {n_significant}", flush=True)
    results['correlations'] = correlations
    results['n_significant'] = n_significant

    # ─── Analysis 2: Clustering ───
    print("\n  --- ANALYSIS 2: Message Clustering ---", flush=True)

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score

    # Cluster messages
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tokens)

    # Compute AMI with property bins
    ami_scores = {}
    for obj_idx in range(min(n_objects, 4)):
        for prop_idx, prop_name in enumerate(prop_names):
            prop_vals = confounders[:, obj_idx, prop_idx]
            if np.std(prop_vals) < 1e-6:
                continue
            # Bin into 5 levels
            prop_bins = np.digitize(prop_vals, np.quantile(prop_vals, [0.2, 0.4, 0.6, 0.8]))
            ami = adjusted_mutual_info_score(clusters, prop_bins)
            key = f"obj{obj_idx}_{prop_name}"
            ami_scores[key] = float(ami)
            print(f"    AMI(clusters, obj{obj_idx}_{prop_name}): {ami:.3f}", flush=True)

    results['clustering_ami'] = ami_scores

    # ─── Analysis 3: TopSim ───
    print("\n  --- ANALYSIS 3: TopSim ---", flush=True)

    # Compute meaning distances using confounders
    # Use object 0 mass and friction as primary properties
    rng = np.random.RandomState(42)
    n_pairs = min(5000, n_scenes * (n_scenes - 1) // 2)
    meaning_dists = []
    message_dists = []

    for _ in range(n_pairs):
        i, j = rng.choice(n_scenes, size=2, replace=False)
        # L1 distance on all confounders (flattened)
        md = np.sum(np.abs(confounders[i].flatten() - confounders[j].flatten()))
        meaning_dists.append(md)
        # Hamming distance on tokens
        hd = int(np.sum(tokens[i] != tokens[j]))
        message_dists.append(hd)

    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0
    print(f"  TopSim: {topsim:.3f} (synthetic V-JEPA 2: 0.738)", flush=True)
    results['topsim'] = float(topsim)

    # ─── Analysis 4: Outcome Prediction ───
    print("\n  --- ANALYSIS 4: Outcome Prediction from Messages ---", flush=True)

    # Use mass of object 0 as binary target (above/below median)
    mass_obj0 = confounders[:, 0, 0]  # First object's mass
    if np.std(mass_obj0) > 1e-6:
        median_mass = np.median(mass_obj0)
        labels = torch.tensor((mass_obj0 > median_mass).astype(np.float32))

        # Simple 80/20 split
        n_train = int(0.8 * n_scenes)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_scenes)

        msg_tensor = torch.tensor(tokens, dtype=torch.float32)

        accs = []
        for seed in range(20):
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(n_positions, 64), nn.ReLU(), nn.Linear(64, 1))
            model = model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            msg_dev = msg_tensor.to(DEVICE)
            lab_dev = labels.to(DEVICE)

            for epoch in range(100):
                model.train()
                pred = model(msg_dev[train_idx]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, lab_dev[train_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_test = model(msg_dev[test_idx]).squeeze(-1)
                acc = ((pred_test > 0).float() == lab_dev[test_idx]).float().mean().item()
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"  Outcome prediction (mass > median): {mean_acc:.1%} ± {std_acc:.1%} (chance=50%)", flush=True)
        results['outcome_prediction'] = {'mean': float(mean_acc), 'std': float(std_acc)}
    else:
        print(f"  Cannot test outcome prediction — mass has no variance", flush=True)
        results['outcome_prediction'] = None

    return results


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Phase 85: Real-Video Transfer — CoPhy CollisionCF", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    t_total = time.time()

    # Step 1: Load CoPhy
    valid_trials, confounders = step1_load_cophy()

    # Step 2: Extract features
    cophy_features = step2_extract_features(valid_trials)

    # Step 3: Train sender + zero-shot transfer
    sender, n_synth_frames = train_sender_on_synthetic(seed=0)
    tokens, messages, collapsed = step3_zero_shot_transfer(sender, cophy_features, n_synth_frames)

    # Step 4: Analysis
    analysis = step4_analysis(tokens, confounders, cophy_features)

    # Determine result
    if collapsed:
        transfer_result = "negative"
        summary = "Frozen sender produces degenerate messages on CoPhy — distribution-specific mapping"
    elif analysis['n_significant'] >= 3:
        transfer_result = "positive"
        summary = f"Frozen sender messages correlate with CoPhy properties ({analysis['n_significant']} significant), TopSim={analysis['topsim']:.3f}"
    elif analysis['n_significant'] >= 1:
        transfer_result = "mixed"
        summary = f"Partial transfer: {analysis['n_significant']} significant correlations, TopSim={analysis['topsim']:.3f}"
    else:
        transfer_result = "negative"
        summary = "Non-degenerate messages but no significant property alignment — distribution-specific"

    # Save
    save_data = {
        'n_scenes': len(tokens),
        'feature_shape': list(cophy_features.shape),
        'symbol_frequencies': {},
        'spearman_correlations': analysis['correlations'],
        'n_significant_correlations': analysis['n_significant'],
        'clustering_ami': analysis['clustering_ami'],
        'outcome_prediction': analysis.get('outcome_prediction'),
        'topsim': analysis['topsim'],
        'transfer_result': transfer_result,
        'summary': summary,
        'collapsed': bool(collapsed),
    }

    # Add symbol frequencies
    for pos in range(tokens.shape[1]):
        counts = np.bincount(tokens[:, pos], minlength=VOCAB_SIZE)
        save_data['symbol_frequencies'][f'position_{pos}'] = counts.tolist()

    save_path = RESULTS_DIR / 'phase85_cophy_transfer.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 85 RESULT: {transfer_result.upper()}", flush=True)
    print(f"Key finding: {summary}", flush=True)
    print(f"Include in paper: {'YES' if transfer_result != 'negative' else 'AS BOUNDARY CONDITION'}", flush=True)
    if transfer_result == 'positive':
        print(f"Suggested placement: Section 5 (Discussion) as zero-shot transfer evidence", flush=True)
    elif transfer_result == 'mixed':
        print(f"Suggested placement: Section 5 (Discussion) with caveats", flush=True)
    else:
        print(f"Suggested placement: Future work paragraph", flush=True)
    print(f"Total time: {dt/60:.1f}min", flush=True)
    print(f"{'='*70}", flush=True)

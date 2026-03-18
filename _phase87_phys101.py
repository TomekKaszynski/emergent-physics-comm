"""
Phase 87: Physics 101 — Real-Video Emergent Communication
==========================================================
Multi-agent communication about physical properties (mass, density)
trained ENTIRELY on real camera footage from MIT's Physics 101 dataset.

Steps:
  1. Extract V-JEPA 2 + DINOv2 features from spring/fall/ramp videos
  1.5. Probe gate: dynamics vs appearance
  2. Spring mass communication (multi-agent)
  3. Fall restitution extraction (bounce tracking)
  4. Two-property compositionality (mass + restitution on fall)
  5. Sim-to-real transfer (frozen Kubric sender on Physics 101)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase87_phys101.py
"""

import time, json, math, os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from collections import defaultdict
import cv2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
PHYS101_DIR = Path("phys101")

# ═══ Architecture constants (match Phase 79b/86b) ═══
HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
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

# ═══════════════════════════════════════════════════════════════
# Architecture (identical to Phase 79b/86b)
# ═══════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, n_frames)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)]
        )

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        messages, all_logits = [], []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                msg = F.one_hot(logits.argmax(dim=-1), self.vocab_size).float()
            messages.append(msg)
            all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits


class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)

    def forward(self, views, tau=1.0, hard=True):
        messages, all_logits = [], []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    """Receiver for single-property comparison task."""
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.head(h).squeeze(-1)


class TwoPropertyReceiver(nn.Module):
    """Receiver for two-property comparison task."""
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.head_a = nn.Linear(hidden_dim // 2, 1)
        self.head_b = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════

def parse_phys101_labels():
    """Parse mass and volume labels."""
    mass, vol = {}, {}
    with open(PHYS101_DIR / "objects" / "mass") as f:
        for line in f:
            name, val = line.strip().split()
            mass[name] = float(val)
    with open(PHYS101_DIR / "objects" / "vol") as f:
        for line in f:
            name, val = line.strip().split()
            vol[name] = float(val)
    density = {k: mass[k] / vol[k] for k in mass if k in vol and vol[k] > 0}
    return mass, vol, density


def inventory_scenario(scenario_name):
    """List all trials for a scenario with object IDs."""
    base = PHYS101_DIR / "scenarios" / scenario_name
    trials = []
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj_name = obj_dir.name
        # Walk to find Camera_1.mp4 files
        for cam_file in sorted(obj_dir.rglob("Camera_1.mp4")):
            trial_dir = cam_file.parent
            # Parse sub-conditions from path
            rel = trial_dir.relative_to(obj_dir)
            trials.append({
                "obj": obj_name,
                "path": str(cam_file),
                "trial_dir": str(trial_dir),
                "sub_condition": str(rel),
            })
    return trials


def read_video_frames(video_path, n_frames=16, target_size=256):
    """Read video and sample n_frames uniformly. Returns (n_frames, 3, H, W) tensor."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Sample frame indices uniformly
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = np.arange(total)
        # Pad by repeating last frame
        indices = np.concatenate([indices, np.full(n_frames - total, total - 1)])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, (target_size, target_size))
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    # Stack to (n_frames, H, W, 3) then (n_frames, 3, H, W)
    frames = np.stack(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return torch.from_numpy(frames)


# ═══════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════

def extract_vjepa2_features(trials, scenario_name, batch_size=4):
    """Extract V-JEPA 2 features for all trials using HF AutoModel."""
    save_path = RESULTS_DIR / f"phase87_phys101_{scenario_name}_features.pt"
    if save_path.exists():
        print(f"  Loading cached {save_path}", flush=True)
        data = torch.load(save_path, weights_only=False)
        return data["features"], data["obj_names"], data["mass_values"]

    print(f"  Extracting V-JEPA 2 features for {scenario_name} ({len(trials)} trials)...", flush=True)

    from transformers import AutoModel
    from torchvision import transforms
    from PIL import Image

    model = AutoModel.from_pretrained('facebook/vjepa2-vitl-fpc64-256')
    model = model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    N_FRAMES_INPUT = 16
    N_SPATIAL = 256  # 256/16 = 16 patches per dim -> 16*16 = 256

    all_features = []
    obj_names = []
    mass_dict, vol_dict, density_dict = parse_phys101_labels()
    mass_values = []
    failed = 0
    t_start = time.time()

    for i, trial in enumerate(trials):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(i, 1) * (len(trials) - i)
            print(f"    [{i+1}/{len(trials)}] {trial['obj']}  ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)", flush=True)

        try:
            cap = cv2.VideoCapture(trial["path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                cap.release()
                failed += 1
                continue

            indices = np.linspace(0, total - 1, N_FRAMES_INPUT, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                frames.append(transform(img))
            cap.release()

            if len(frames) < N_FRAMES_INPUT:
                # Pad by repeating last
                while len(frames) < N_FRAMES_INPUT:
                    frames.append(frames[-1])

            # (T, C, H, W) -> (1, T, C, H, W) for HF model
            video = torch.stack(frames, dim=0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(pixel_values_videos=video)
                hidden = output.last_hidden_state  # (1, T*spatial, D)
                n_tokens = hidden.shape[1]
                feat_dim = hidden.shape[2]
                n_temporal = n_tokens // N_SPATIAL
                if n_temporal < 1:
                    n_temporal = 1
                n_spatial = n_tokens // n_temporal
                hidden = hidden.view(1, n_temporal, n_spatial, feat_dim)
                pooled = hidden.mean(dim=2)  # (1, T_temporal, D)
                all_features.append(pooled.squeeze(0).cpu().half())
                obj_names.append(trial["obj"])
                mass_values.append(mass_dict.get(trial["obj"], 0.0))

        except Exception as e:
            print(f"    FAILED {trial['obj']}: {e}", flush=True)
            failed += 1
            continue

        if (i + 1) % 50 == 0:
            torch.mps.empty_cache()

    if not all_features:
        print(f"  ERROR: No features extracted for {scenario_name}", flush=True)
        return None, None, None

    features = torch.stack(all_features)  # (N, 8, 1024) float16
    mass_arr = np.array(mass_values)
    print(f"  Extracted: {features.shape}, failed: {failed}", flush=True)

    torch.save({
        "features": features,
        "obj_names": obj_names,
        "mass_values": mass_arr,
        "scenario": scenario_name,
    }, save_path)
    print(f"  Saved {save_path}", flush=True)
    return features, obj_names, mass_arr


def extract_dino_static_features(trials, scenario_name):
    """Extract DINOv2 single-frame features for static baseline."""
    save_path = RESULTS_DIR / f"phase87_phys101_{scenario_name}_static.pt"
    if save_path.exists():
        print(f"  Loading cached {save_path}", flush=True)
        data = torch.load(save_path, weights_only=False)
        return data["features"], data["obj_names"]

    print(f"  Extracting DINOv2 static features for {scenario_name}...", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino = dino.to(DEVICE).eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    all_features = []
    obj_names = []
    failed = 0

    for i, trial in enumerate(trials):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1}/{len(trials)}] DINOv2 {trial['obj']}", flush=True)

        # Read middle frame only
        cap = cv2.VideoCapture(trial["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            failed += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        img = (img - mean) / std

        with torch.no_grad():
            feat = dino(img)  # (1, 384)
            all_features.append(feat.cpu())
            obj_names.append(trial["obj"])

        if (i + 1) % 100 == 0:
            torch.mps.empty_cache()

    features = torch.cat(all_features, dim=0)  # (N, 384)
    print(f"  DINOv2 static: {features.shape}, failed: {failed}", flush=True)

    torch.save({"features": features, "obj_names": obj_names}, save_path)
    return features, obj_names


# ═══════════════════════════════════════════════════════════════
# Step 1.5: Probe gate
# ═══════════════════════════════════════════════════════════════

def run_probe(features, mass_values, obj_names, probe_name, n_seeds=5):
    """Train linear probe for mass pairwise comparison.
    Holdout: 20% of OBJECTS (not trials)."""
    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)

    all_aucs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
        train_idx = [i for i, o in enumerate(obj_names) if o not in holdout_objs]
        test_idx = [i for i, o in enumerate(obj_names) if o in holdout_objs]

        if len(test_idx) < 4:
            continue

        # Features
        if features.dim() == 3:
            # Temporal features -> mean pool
            train_feat = features[train_idx].float().mean(dim=1)
            test_feat = features[test_idx].float().mean(dim=1)
        elif features.dim() == 2:
            train_feat = features[train_idx].float()
            test_feat = features[test_idx].float()
        else:
            # Scalar (volume-only baseline)
            train_feat = features[train_idx].float().unsqueeze(1)
            test_feat = features[test_idx].float().unsqueeze(1)

        train_mass = mass_values[train_idx]
        test_mass = mass_values[test_idx]

        feat_dim = train_feat.shape[1]

        # Train pairwise comparison probe
        probe = nn.Sequential(
            nn.Linear(feat_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(DEVICE)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

        train_feat_d = train_feat.to(DEVICE)
        train_mass_d = torch.tensor(train_mass, dtype=torch.float32).to(DEVICE)

        for ep in range(200):
            probe.train()
            # Sample pairs
            ia = np.random.randint(0, len(train_idx), 64)
            ib = np.random.randint(0, len(train_idx), 64)
            same = ia == ib
            while same.any():
                ib[same] = np.random.randint(0, len(train_idx), same.sum())
                same = ia == ib

            fa = train_feat_d[ia]
            fb = train_feat_d[ib]
            label = (train_mass_d[ia] > train_mass_d[ib]).float()
            # Exclude near-ties
            diff = torch.abs(train_mass_d[ia] - train_mass_d[ib])
            keep = diff > 0.5  # at least 0.5g difference
            if keep.sum() < 4:
                continue

            pred = probe(torch.cat([fa[keep], fb[keep]], dim=-1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, label[keep])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate on holdout objects
        probe.eval()
        with torch.no_grad():
            test_feat_d = test_feat.to(DEVICE)
            test_mass_d = torch.tensor(test_mass, dtype=torch.float32).to(DEVICE)
            preds, labels = [], []
            for ia_idx in range(len(test_idx)):
                for ib_idx in range(ia_idx + 1, len(test_idx)):
                    if abs(test_mass[ia_idx] - test_mass[ib_idx]) < 0.5:
                        continue
                    fa = test_feat_d[ia_idx:ia_idx+1]
                    fb = test_feat_d[ib_idx:ib_idx+1]
                    pred = torch.sigmoid(probe(torch.cat([fa, fb], dim=-1))).item()
                    label = float(test_mass[ia_idx] > test_mass[ib_idx])
                    preds.append(pred)
                    labels.append(label)

            if len(preds) > 5:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(labels, preds)
                    all_aucs.append(auc)
                except ValueError:
                    pass

    if all_aucs:
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)
    else:
        mean_auc, std_auc = 0.5, 0.0

    print(f"    {probe_name}: AUC = {mean_auc:.3f} ± {std_auc:.3f} ({len(all_aucs)} seeds)", flush=True)
    return mean_auc, std_auc


# ═══════════════════════════════════════════════════════════════
# Step 2: Communication training (single property)
# ═══════════════════════════════════════════════════════════════

def train_communication_single(features, mass_values, obj_names,
                                n_agents, n_seeds=10, label_name="mass"):
    """Train multi-agent communication for single-property comparison."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

    # Object-level holdout
    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)

    all_results = []
    for seed in range(n_seeds):
        print(f"    Seed {seed}...", flush=True)
        rng = np.random.RandomState(seed * 1000 + 42)
        holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
        train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
        holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

        if len(holdout_ids) < 4:
            print(f"      Skipping: too few holdout trials ({len(holdout_ids)})", flush=True)
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        senders = [CompositionalSender(
            TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
            HIDDEN_DIM, VOCAB_SIZE, N_HEADS
        ) for _ in range(n_agents)]
        multi = MultiAgentSender(senders).to(DEVICE)
        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
        s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
        r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

        mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
        max_ent = math.log(VOCAB_SIZE)
        nb = max(1, len(train_ids) // BATCH_SIZE)
        best_acc = 0.0
        best_state = None
        nan_count = 0
        t0 = time.time()

        for ep in range(COMM_EPOCHS):
            if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
                for i in range(len(receivers)):
                    receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                    r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

            multi.train()
            for r in receivers:
                r.train()
            tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
            hard = ep >= SOFT_WARMUP

            for _ in range(nb):
                ia = rng.choice(train_ids, BATCH_SIZE)
                ib = rng.choice(train_ids, BATCH_SIZE)
                same = ia == ib
                while same.any():
                    ib[same] = rng.choice(train_ids, same.sum())
                    same = ia == ib

                # Skip near-ties in mass
                mass_diff = np.abs(mass_values[ia] - mass_values[ib])
                keep = mass_diff > 0.5
                if keep.sum() < 4:
                    continue
                ia, ib = ia[keep], ib[keep]

                va = [v[ia].to(DEVICE) for v in agent_views]
                vb = [v[ib].to(DEVICE) for v in agent_views]
                label = (mass_dev[ia] > mass_dev[ib]).float()
                ma, la = multi(va, tau=tau, hard=hard)
                mb, lb = multi(vb, tau=tau, hard=hard)

                total_loss = torch.tensor(0.0, device=DEVICE)
                for r in receivers:
                    pred = r(ma, mb)
                    total_loss = total_loss + F.binary_cross_entropy_with_logits(pred, label)
                loss = total_loss / len(receivers)

                # Entropy regularization
                for logits in la + lb:
                    lp = F.log_softmax(logits, dim=-1)
                    p = lp.exp().clamp(min=1e-8)
                    ent = -(p * lp).sum(dim=-1).mean()
                    if ent / max_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

                if torch.isnan(loss) or torch.isinf(loss):
                    s_opt.zero_grad()
                    for o in r_opts:
                        o.zero_grad()
                    nan_count += 1
                    continue

                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
                for r in receivers:
                    torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
                s_opt.step()
                for o in r_opts:
                    o.step()

            if ep % 50 == 0:
                torch.mps.empty_cache()

            if (ep + 1) % 50 == 0 or ep == 0:
                multi.eval()
                for r in receivers:
                    r.eval()
                with torch.no_grad():
                    correct = total = 0
                    er = np.random.RandomState(999)
                    for _ in range(30):
                        bs = min(BATCH_SIZE, len(holdout_ids))
                        ia_h = er.choice(holdout_ids, bs)
                        ib_h = er.choice(holdout_ids, bs)
                        same_h = ia_h == ib_h
                        while same_h.any():
                            ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                            same_h = ia_h == ib_h
                        md = np.abs(mass_values[ia_h] - mass_values[ib_h])
                        keep_h = md > 0.5
                        if keep_h.sum() < 2:
                            continue
                        ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                        va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                        vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                        la_h = mass_dev[ia_h] > mass_dev[ib_h]
                        ma_h, _ = multi(va_h)
                        mb_h, _ = multi(vb_h)
                        # Best receiver
                        for r in receivers:
                            pred_h = r(ma_h, mb_h) > 0
                            correct += (pred_h == la_h).sum().item()
                            total += len(la_h)
                    acc = correct / max(total, 1)
                    if acc > best_acc:
                        best_acc = acc
                        best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
                elapsed = time.time() - t0
                eta = elapsed / (ep + 1) * (COMM_EPOCHS - ep - 1)
                ns = f" NaN={nan_count}" if nan_count else ""
                print(f"      Ep {ep+1:3d}: holdout={acc:.1%} best={best_acc:.1%}{ns}  ETA {eta/60:.0f}min", flush=True)

        # Restore best
        if best_state:
            multi.load_state_dict(best_state)

        # Compute MI and TopSim
        multi.eval()
        all_tokens = []
        with torch.no_grad():
            for i in range(0, len(features), BATCH_SIZE):
                views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
                _, logits = multi(views)
                tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
                all_tokens.append(np.stack(tokens, axis=1))
        all_tokens = np.concatenate(all_tokens, axis=0)
        n_pos = all_tokens.shape[1]

        # MI: each position vs mass
        mi_values = []
        for p in range(n_pos):
            x = all_tokens[:, p]
            y = mass_values
            # Bin mass into 5 levels for MI
            y_bins = np.digitize(y, np.percentile(y, [20, 40, 60, 80]))
            xv, yv = np.unique(x), np.unique(y_bins)
            n = len(x)
            mi = 0.0
            for xval in xv:
                for yval in yv:
                    pxy = np.sum((x == xval) & (y_bins == yval)) / n
                    px = np.sum(x == xval) / n
                    py = np.sum(y_bins == yval) / n
                    if pxy > 0 and px > 0 and py > 0:
                        mi += pxy * np.log(pxy / (px * py))
            mi_values.append(float(mi))

        # TopSim
        rng_ts = np.random.RandomState(42)
        md_list, hd_list = [], []
        for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
            i, j = rng_ts.choice(len(features), 2, replace=False)
            md_list.append(abs(mass_values[i] - mass_values[j]))
            hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
        topsim, _ = stats.spearmanr(md_list, hd_list)
        if np.isnan(topsim):
            topsim = 0.0

        # Message entropy per position
        entropies = []
        for p in range(n_pos):
            counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(VOCAB_SIZE)
            entropies.append(float(ent))

        # Mass-symbol Spearman correlation (best position)
        best_rho = 0.0
        for p in range(n_pos):
            rho, _ = stats.spearmanr(all_tokens[:, p], mass_values)
            if abs(rho) > abs(best_rho):
                best_rho = rho

        dt = time.time() - t0
        result = {
            "seed": seed,
            "holdout_acc": float(best_acc),
            "mi_values": mi_values,
            "topsim": float(topsim),
            "entropies": entropies,
            "mass_symbol_rho": float(best_rho),
            "n_pos": n_pos,
            "time_sec": float(dt),
        }
        all_results.append(result)
        print(f"      -> acc={best_acc:.1%} TopSim={topsim:.3f} rho={best_rho:.3f} ({dt:.0f}s)", flush=True)

    return all_results


# ═══════════════════════════════════════════════════════════════
# Step 3: Fall restitution extraction
# ═══════════════════════════════════════════════════════════════

def extract_restitution_from_fall(trials):
    """Extract coefficient of restitution from fall videos using tracking."""
    print("  Extracting restitution from fall videos...", flush=True)

    results = []
    mass_dict, _, _ = parse_phys101_labels()

    for i, trial in enumerate(trials):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(trials)}]", flush=True)

        cap = cv2.VideoCapture(trial["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            continue

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        if len(frames) < 10:
            continue

        # Background: use first frame (object hasn't been dropped yet)
        bg = frames[0].astype(np.float32)

        # Track object centroid via frame differencing
        y_positions = []
        for f in frames:
            diff = np.abs(f.astype(np.float32) - bg)
            mask = (diff > 30).astype(np.uint8)
            # Find centroid of difference mask
            ys, xs = np.where(mask > 0)
            if len(ys) > 50:  # Minimum pixels
                y_positions.append(float(np.median(ys)))
            else:
                y_positions.append(np.nan)

        y_pos = np.array(y_positions)

        # Find valid trajectory: drop start, impact, bounce
        valid = ~np.isnan(y_pos)
        if valid.sum() < 5:
            continue

        # Smooth
        y_smooth = np.copy(y_pos)
        for j in range(1, len(y_smooth) - 1):
            if not np.isnan(y_smooth[j]):
                vals = [y_smooth[j]]
                if not np.isnan(y_smooth[j-1]):
                    vals.append(y_smooth[j-1])
                if not np.isnan(y_smooth[j+1]):
                    vals.append(y_smooth[j+1])
                y_smooth[j] = np.mean(vals)

        # Find impact: maximum y value (lowest point, since y increases downward)
        valid_idx = np.where(~np.isnan(y_smooth))[0]
        if len(valid_idx) < 5:
            continue

        impact_idx = valid_idx[np.argmax(y_smooth[valid_idx])]
        impact_y = y_smooth[impact_idx]

        # Find drop start: first valid position before impact where object appears
        pre_impact = valid_idx[valid_idx < impact_idx]
        if len(pre_impact) < 2:
            continue
        drop_start_y = y_smooth[pre_impact[0]]

        # Find bounce apex: minimum y after impact (highest point after bounce)
        post_impact = valid_idx[valid_idx > impact_idx]
        if len(post_impact) < 2:
            # No bounce detected
            results.append({
                "obj": trial["obj"],
                "sub_condition": trial["sub_condition"],
                "restitution": 0.0,
                "valid": False,
            })
            continue

        bounce_apex_idx = post_impact[np.argmin(y_smooth[post_impact])]
        bounce_apex_y = y_smooth[bounce_apex_idx]

        # Sanity checks
        # In image coords, y increases downward, so:
        # drop_start_y < impact_y (object falls down)
        # bounce_apex_y < impact_y (object bounces up)
        # bounce_apex_y >= drop_start_y (can't bounce higher than drop)
        if not (drop_start_y < impact_y and bounce_apex_y < impact_y):
            continue

        # Heights from surface (impact point)
        h_drop = impact_y - drop_start_y  # positive
        h_bounce = impact_y - bounce_apex_y  # positive

        if h_drop < 10:  # minimum pixel displacement
            continue

        e = math.sqrt(h_bounce / h_drop)
        e = min(e, 1.5)  # cap at reasonable max

        results.append({
            "obj": trial["obj"],
            "sub_condition": trial["sub_condition"],
            "path": trial["path"],
            "restitution": float(e),
            "h_drop": float(h_drop),
            "h_bounce": float(h_bounce),
            "valid": True,
            "mass": mass_dict.get(trial["obj"], 0.0),
        })

    valid_results = [r for r in results if r.get("valid", False) and r["restitution"] > 0.01]
    print(f"  Valid trials: {len(valid_results)}/{len(trials)}", flush=True)

    if valid_results:
        e_vals = [r["restitution"] for r in valid_results]
        print(f"  Restitution range: [{min(e_vals):.3f}, {max(e_vals):.3f}]", flush=True)
        print(f"  Restitution mean: {np.mean(e_vals):.3f} ± {np.std(e_vals):.3f}", flush=True)

        # By material
        by_mat = defaultdict(list)
        for r in valid_results:
            mat = r["obj"].rsplit("_", 1)[0]
            by_mat[mat].append(r["restitution"])
        print("  By material:", flush=True)
        for mat in sorted(by_mat.keys()):
            vals = by_mat[mat]
            print(f"    {mat:15s}: n={len(vals):3d}, e={np.mean(vals):.3f} ± {np.std(vals):.3f}", flush=True)

    return valid_results


# ═══════════════════════════════════════════════════════════════
# Step 4: Two-property compositionality
# ═══════════════════════════════════════════════════════════════

def train_two_property_communication(features, mass_values, rest_values, obj_names,
                                      n_agents, n_seeds=10):
    """Train multi-agent communication for two-property comparison."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)

    all_results = []
    for seed in range(n_seeds):
        print(f"    Seed {seed}...", flush=True)
        rng = np.random.RandomState(seed * 1000 + 42)
        holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
        train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
        holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

        if len(holdout_ids) < 4:
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        senders = [CompositionalSender(
            TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
            HIDDEN_DIM, VOCAB_SIZE, N_HEADS
        ) for _ in range(n_agents)]
        multi = MultiAgentSender(senders).to(DEVICE)
        receivers = [TwoPropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
        s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
        r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

        mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
        rest_dev = torch.tensor(rest_values, dtype=torch.float32).to(DEVICE)
        max_ent = math.log(VOCAB_SIZE)
        nb = max(1, len(train_ids) // BATCH_SIZE)
        best_both = 0.0
        best_state = None
        nan_count = 0
        t0 = time.time()

        for ep in range(COMM_EPOCHS):
            if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
                for i in range(len(receivers)):
                    receivers[i] = TwoPropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                    r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

            multi.train()
            for r in receivers:
                r.train()
            tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
            hard = ep >= SOFT_WARMUP

            for _ in range(nb):
                ia = rng.choice(train_ids, BATCH_SIZE)
                ib = rng.choice(train_ids, BATCH_SIZE)
                same = ia == ib
                while same.any():
                    ib[same] = rng.choice(train_ids, same.sum())
                    same = ia == ib

                va = [v[ia].to(DEVICE) for v in agent_views]
                vb = [v[ib].to(DEVICE) for v in agent_views]
                l_mass = (mass_dev[ia] > mass_dev[ib]).float()
                l_rest = (rest_dev[ia] > rest_dev[ib]).float()
                ma, la = multi(va, tau=tau, hard=hard)
                mb, lb = multi(vb, tau=tau, hard=hard)

                total_loss = torch.tensor(0.0, device=DEVICE)
                for r in receivers:
                    pm, pr = r(ma, mb)
                    total_loss = total_loss + F.binary_cross_entropy_with_logits(pm, l_mass)
                    total_loss = total_loss + F.binary_cross_entropy_with_logits(pr, l_rest)
                loss = total_loss / len(receivers)

                for logits in la + lb:
                    lp = F.log_softmax(logits, dim=-1)
                    p = lp.exp().clamp(min=1e-8)
                    ent = -(p * lp).sum(dim=-1).mean()
                    if ent / max_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

                if torch.isnan(loss) or torch.isinf(loss):
                    s_opt.zero_grad()
                    for o in r_opts:
                        o.zero_grad()
                    nan_count += 1
                    continue

                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
                for r in receivers:
                    torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
                s_opt.step()
                for o in r_opts:
                    o.step()

            if ep % 50 == 0:
                torch.mps.empty_cache()

            if (ep + 1) % 50 == 0 or ep == 0:
                multi.eval()
                for r in receivers:
                    r.eval()
                with torch.no_grad():
                    cm = cr = cb = tm = tr = tb = 0
                    er = np.random.RandomState(999)
                    for _ in range(30):
                        bs = min(BATCH_SIZE, len(holdout_ids))
                        ia_h = er.choice(holdout_ids, bs)
                        ib_h = er.choice(holdout_ids, bs)
                        same_h = ia_h == ib_h
                        while same_h.any():
                            ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                            same_h = ia_h == ib_h
                        va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                        vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                        ma_h, _ = multi(va_h)
                        mb_h, _ = multi(vb_h)
                        lm_h = mass_dev[ia_h] > mass_dev[ib_h]
                        lr_h = rest_dev[ia_h] > rest_dev[ib_h]
                        for r in receivers:
                            pm_h, pr_h = r(ma_h, mb_h)
                            m_diff = np.abs(mass_values[ia_h] - mass_values[ib_h]) > 0.5
                            r_diff = np.abs(rest_values[ia_h] - rest_values[ib_h]) > 0.02
                            m_diff_t = torch.tensor(m_diff, device=DEVICE)
                            r_diff_t = torch.tensor(r_diff, device=DEVICE)
                            if m_diff_t.sum() > 0:
                                cm += ((pm_h[m_diff_t] > 0) == lm_h[m_diff_t]).sum().item()
                                tm += m_diff_t.sum().item()
                            if r_diff_t.sum() > 0:
                                cr += ((pr_h[r_diff_t] > 0) == lr_h[r_diff_t]).sum().item()
                                tr += r_diff_t.sum().item()
                            both_d = m_diff_t & r_diff_t
                            if both_d.sum() > 0:
                                ok = ((pm_h[both_d] > 0) == lm_h[both_d]) & ((pr_h[both_d] > 0) == lr_h[both_d])
                                cb += ok.sum().item()
                                tb += both_d.sum().item()
                    hm = cm / max(tm, 1)
                    hr = cr / max(tr, 1)
                    hb = cb / max(tb, 1)
                    if hb > best_both:
                        best_both = hb
                        best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
                elapsed = time.time() - t0
                eta = elapsed / (ep + 1) * (COMM_EPOCHS - ep - 1)
                print(f"      Ep {ep+1:3d}: mass={hm:.1%} rest={hr:.1%} both={hb:.1%}  ETA {eta/60:.0f}min", flush=True)

        if best_state:
            multi.load_state_dict(best_state)

        # Compositionality metrics
        multi.eval()
        all_tokens = []
        with torch.no_grad():
            for i in range(0, len(features), BATCH_SIZE):
                views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
                _, logits = multi(views)
                tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
                all_tokens.append(np.stack(tokens, axis=1))
        all_tokens = np.concatenate(all_tokens, axis=0)
        n_pos = all_tokens.shape[1]

        # Bin mass and restitution into 5 levels each
        mass_bins = np.digitize(mass_values, np.percentile(mass_values, [20, 40, 60, 80]))
        rest_bins = np.digitize(rest_values, np.percentile(rest_values, [20, 40, 60, 80]))
        attrs = np.stack([mass_bins, rest_bins], axis=1)

        # MI matrix
        mi_matrix = np.zeros((n_pos, 2))
        for p in range(n_pos):
            for a in range(2):
                x, y = all_tokens[:, p], attrs[:, a]
                xv, yv = np.unique(x), np.unique(y)
                n = len(x)
                mi = 0.0
                for xval in xv:
                    for yval in yv:
                        pxy = np.sum((x == xval) & (y == yval)) / n
                        px = np.sum(x == xval) / n
                        py = np.sum(y == yval) / n
                        if pxy > 0 and px > 0 and py > 0:
                            mi += pxy * np.log(pxy / (px * py))
                mi_matrix[p, a] = mi

        # PosDis
        pos_dis = 0.0
        for p in range(n_pos):
            s = np.sort(mi_matrix[p])[::-1]
            if s[0] > 1e-10:
                pos_dis += (s[0] - s[1]) / s[0]
        pos_dis /= n_pos

        # TopSim
        rng_ts = np.random.RandomState(42)
        md_list, hd_list = [], []
        for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
            i, j = rng_ts.choice(len(features), 2, replace=False)
            md_list.append(abs(float(mass_bins[i]) - float(mass_bins[j])) +
                          abs(float(rest_bins[i]) - float(rest_bins[j])))
            hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
        topsim, _ = stats.spearmanr(md_list, hd_list)
        if np.isnan(topsim):
            topsim = 0.0

        # Entropies
        entropies = []
        for p in range(n_pos):
            counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(VOCAB_SIZE)
            entropies.append(float(ent))

        dt = time.time() - t0
        result = {
            "seed": seed,
            "holdout_mass": float(hm),
            "holdout_rest": float(hr),
            "holdout_both": float(best_both),
            "pos_dis": float(pos_dis),
            "topsim": float(topsim),
            "mi_matrix": mi_matrix.tolist(),
            "entropies": entropies,
            "time_sec": float(dt),
        }
        all_results.append(result)
        print(f"      -> both={best_both:.1%} PosDis={pos_dis:.3f} TopSim={topsim:.3f} ({dt:.0f}s)", flush=True)

    return all_results


# ═══════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70, flush=True)
    print("Phase 87: Physics 101 — Real-Video Emergent Communication", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    mass_dict, vol_dict, density_dict = parse_phys101_labels()
    results_all = {}

    # ═══ STEP 1: Feature extraction ═══
    print("\n" + "=" * 60, flush=True)
    print("STEP 1: Feature Extraction", flush=True)
    print("=" * 60, flush=True)

    # Inventory
    spring_trials = inventory_scenario("spring")
    fall_trials = inventory_scenario("fall")
    print(f"  Spring: {len(spring_trials)} trials", flush=True)
    print(f"  Fall: {len(fall_trials)} trials", flush=True)

    # Check if ramp is extracted
    ramp_dir = PHYS101_DIR / "scenarios" / "ramp"
    if ramp_dir.exists():
        ramp_trials = inventory_scenario("ramp")
        print(f"  Ramp: {len(ramp_trials)} trials", flush=True)
    else:
        ramp_trials = []
        print("  Ramp: NOT EXTRACTED (skipping)", flush=True)

    # V-JEPA 2 features — spring first (priority)
    print("\n  --- V-JEPA 2: Spring ---", flush=True)
    spring_feat, spring_objs, spring_mass = extract_vjepa2_features(spring_trials, "spring")
    print(f"  Spring features: {spring_feat.shape if spring_feat is not None else 'FAILED'}", flush=True)

    print("\n  --- V-JEPA 2: Fall ---", flush=True)
    fall_feat, fall_objs, fall_mass = extract_vjepa2_features(fall_trials, "fall")
    print(f"  Fall features: {fall_feat.shape if fall_feat is not None else 'FAILED'}", flush=True)

    if ramp_trials:
        print("\n  --- V-JEPA 2: Ramp ---", flush=True)
        ramp_feat, ramp_objs, ramp_mass = extract_vjepa2_features(ramp_trials, "ramp")
        print(f"  Ramp features: {ramp_feat.shape if ramp_feat is not None else 'FAILED'}", flush=True)
    else:
        ramp_feat, ramp_objs, ramp_mass = None, None, None

    # DINOv2 static features
    print("\n  --- DINOv2 Static: Spring ---", flush=True)
    spring_static, _ = extract_dino_static_features(spring_trials, "spring")

    print("\n  --- DINOv2 Static: Fall ---", flush=True)
    fall_static, _ = extract_dino_static_features(fall_trials, "fall")

    if ramp_trials:
        print("\n  --- DINOv2 Static: Ramp ---", flush=True)
        ramp_static, _ = extract_dino_static_features(ramp_trials, "ramp")
    else:
        ramp_static = None

    feat_time = time.time() - t_start
    print(f"\n  Feature extraction complete ({feat_time/60:.1f} min)", flush=True)

    # ═══ STEP 1.5: Probe gate ═══
    print("\n" + "=" * 60, flush=True)
    print("STEP 1.5: Probe Gate — Dynamics vs Appearance", flush=True)
    print("=" * 60, flush=True)

    probe_results = {}
    for scenario, feat_t, feat_s, objs, masses in [
        ("spring", spring_feat, spring_static, spring_objs, spring_mass),
        ("fall", fall_feat, fall_static, fall_objs, fall_mass),
    ]:
        if feat_t is None:
            continue
        print(f"\n  --- {scenario.upper()} ---", flush=True)

        # Volume-only baseline
        vol_arr = torch.tensor([vol_dict.get(o, 1.0) for o in objs], dtype=torch.float32)
        vol_auc, vol_std = run_probe(vol_arr, masses, objs, f"{scenario} Volume-only")

        # DINOv2 static
        dino_auc, dino_std = run_probe(feat_s, masses, objs, f"{scenario} DINOv2-static")

        # V-JEPA 2 temporal
        vjepa_auc, vjepa_std = run_probe(feat_t, masses, objs, f"{scenario} V-JEPA2-temporal")

        gap = vjepa_auc - dino_auc
        probe_results[scenario] = {
            "vjepa2_auc": float(vjepa_auc), "vjepa2_std": float(vjepa_std),
            "dino_auc": float(dino_auc), "dino_std": float(dino_std),
            "volume_auc": float(vol_auc), "volume_std": float(vol_std),
            "gap": float(gap),
        }
        print(f"    Gap (V-JEPA2 - DINOv2): {gap:+.3f}", flush=True)
        dynamics_label = "DYNAMICS CONTRIBUTE" if gap > 0.05 else ("APPEARANCE DOMINANT" if gap < -0.05 else "SIMILAR")
        print(f"    Verdict: {dynamics_label}", flush=True)

    # Also check ramp if available
    if ramp_feat is not None and ramp_objs is not None:
        print(f"\n  --- RAMP ---", flush=True)
        vol_arr = torch.tensor([vol_dict.get(o, 1.0) for o in ramp_objs], dtype=torch.float32)
        vol_auc, vol_std = run_probe(vol_arr, ramp_mass, ramp_objs, "ramp Volume-only")
        dino_auc, dino_std = run_probe(ramp_static, ramp_mass, ramp_objs, "ramp DINOv2-static")
        vjepa_auc, vjepa_std = run_probe(ramp_feat, ramp_mass, ramp_objs, "ramp V-JEPA2-temporal")
        gap = vjepa_auc - dino_auc
        probe_results["ramp"] = {
            "vjepa2_auc": float(vjepa_auc), "vjepa2_std": float(vjepa_std),
            "dino_auc": float(dino_auc), "dino_std": float(dino_std),
            "volume_auc": float(vol_auc), "volume_std": float(vol_std),
            "gap": float(gap),
        }
        print(f"    Gap: {gap:+.3f}", flush=True)

    results_all["probe_gate"] = probe_results

    # Print summary table
    print("\n  PROBE GATE SUMMARY:", flush=True)
    print(f"  {'Scenario':10s} {'V-JEPA2':>10s} {'DINOv2':>10s} {'Volume':>10s} {'Gap':>8s}", flush=True)
    for sc in ["spring", "fall", "ramp"]:
        if sc in probe_results:
            p = probe_results[sc]
            print(f"  {sc:10s} {p['vjepa2_auc']:10.3f} {p['dino_auc']:10.3f} {p['volume_auc']:10.3f} {p['gap']:+8.3f}", flush=True)

    # Gate check
    best_vjepa = max(p["vjepa2_auc"] for p in probe_results.values())
    if best_vjepa < 0.60:
        print("\n  *** ABORT: V-JEPA 2 AUC < 0.60 on all scenarios ***", flush=True)
        results_all["abort_reason"] = "V-JEPA2 AUC < 0.60 everywhere"
        with open(RESULTS_DIR / "phase87_phys101.json", "w") as f:
            json.dump(results_all, f, indent=2)
        sys.exit(0)

    print(f"\n  Gate PASSED (best V-JEPA2 AUC = {best_vjepa:.3f})", flush=True)

    # ═══ STEP 2: Spring Mass Communication ═══
    if spring_feat is not None and probe_results.get("spring", {}).get("vjepa2_auc", 0) >= 0.55:
        print("\n" + "=" * 60, flush=True)
        print("STEP 2: Spring Mass Communication", flush=True)
        print("=" * 60, flush=True)

        # Try 2 agents first (safer)
        for n_agents in [2, 4]:
            fpa = 8 // n_agents
            if fpa < 1:
                continue
            print(f"\n  --- {n_agents} agents × {fpa} frames ---", flush=True)
            comm_results = train_communication_single(
                spring_feat, spring_mass, spring_objs, n_agents, n_seeds=10
            )
            if comm_results:
                accs = [r["holdout_acc"] for r in comm_results]
                mean_acc = np.mean(accs)
                print(f"  {n_agents}-agent mean holdout: {mean_acc:.1%}", flush=True)
                results_all[f"spring_{n_agents}agent"] = {
                    "per_seed": comm_results,
                    "mean_acc": float(mean_acc),
                    "std_acc": float(np.std(accs)),
                }
                # If 2 agents work well, still try 4. If 4 doesn't converge, keep 2.
                if n_agents == 4 and mean_acc < 0.55:
                    print(f"  4-agent didn't converge. Using 2-agent results.", flush=True)
    else:
        print("\n  Skipping spring communication (insufficient probe AUC)", flush=True)

    # ═══ STEP 3: Fall Restitution Extraction ═══
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Fall Restitution Extraction", flush=True)
    print("=" * 60, flush=True)

    rest_data = extract_restitution_from_fall(fall_trials)
    n_valid = len(rest_data)
    results_all["restitution_extraction"] = {
        "n_valid": n_valid,
        "n_total": len(fall_trials),
    }

    if n_valid >= 100:
        # Save restitution data
        save_path = RESULTS_DIR / "phase87_phys101_restitution_labels.json"
        with open(save_path, "w") as f:
            json.dump(rest_data, f, indent=2)
        print(f"  Saved {save_path}", flush=True)

        # ═══ STEP 4: Two-Property Compositionality ═══
        if fall_feat is not None:
            print("\n" + "=" * 60, flush=True)
            print("STEP 4: Two-Property Compositionality (Fall)", flush=True)
            print("=" * 60, flush=True)

            # Match restitution data to fall features
            # Build index: obj -> list of restitution values
            obj_rest = defaultdict(list)
            for r in rest_data:
                obj_rest[r["obj"]].append(r["restitution"])
            # Mean restitution per object
            obj_rest_mean = {o: np.mean(v) for o, v in obj_rest.items() if v}

            # Match to fall_objs (which have features)
            matched_idx = []
            matched_mass = []
            matched_rest = []
            matched_objs = []
            for i, obj in enumerate(fall_objs):
                if obj in obj_rest_mean:
                    matched_idx.append(i)
                    matched_mass.append(fall_mass[i])
                    matched_rest.append(obj_rest_mean[obj])
                    matched_objs.append(obj)

            n_matched = len(matched_idx)
            print(f"  Matched trials (have both features + restitution): {n_matched}", flush=True)

            if n_matched >= 50:
                matched_feat = fall_feat[matched_idx]
                matched_mass_arr = np.array(matched_mass)
                matched_rest_arr = np.array(matched_rest)

                # Check mass-restitution independence
                rho, p = stats.spearmanr(matched_mass_arr, matched_rest_arr)
                print(f"  Mass-restitution Spearman rho = {rho:.3f} (p={p:.3f})", flush=True)

                for n_agents in [2, 4]:
                    fpa = 8 // n_agents
                    if fpa < 1:
                        continue
                    print(f"\n  --- {n_agents} agents × {fpa} frames ---", flush=True)
                    two_prop_results = train_two_property_communication(
                        matched_feat, matched_mass_arr, matched_rest_arr,
                        matched_objs, n_agents, n_seeds=10
                    )
                    if two_prop_results:
                        both_accs = [r["holdout_both"] for r in two_prop_results]
                        posdis = [r["pos_dis"] for r in two_prop_results]
                        topsims = [r["topsim"] for r in two_prop_results]
                        n_comp = sum(1 for r in two_prop_results if r["pos_dis"] > 0.4)
                        print(f"  {n_agents}-agent: both={np.mean(both_accs):.1%} PosDis={np.mean(posdis):.3f} "
                              f"comp={n_comp}/{len(two_prop_results)}", flush=True)
                        results_all[f"two_prop_{n_agents}agent"] = {
                            "per_seed": two_prop_results,
                            "mean_both": float(np.mean(both_accs)),
                            "mean_posdis": float(np.mean(posdis)),
                            "mean_topsim": float(np.mean(topsims)),
                            "n_compositional": n_comp,
                            "mass_rest_rho": float(rho),
                        }
                        if n_agents == 4 and np.mean(both_accs) < 0.30:
                            print(f"  4-agent didn't converge. Relying on 2-agent.", flush=True)
            else:
                print(f"  Too few matched trials ({n_matched}). Skipping Step 4.", flush=True)
    else:
        print(f"  Too few valid restitution trials ({n_valid}). Skipping Step 4.", flush=True)

    # ═══ STEP 5: Save Results ═══
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Save Results", flush=True)
    print("=" * 60, flush=True)

    total_time = time.time() - t_start
    results_all["total_time_min"] = float(total_time / 60)
    results_all["dataset"] = "Physics 101 (Wu et al. 2016)"

    save_path = RESULTS_DIR / "phase87_phys101.json"
    with open(save_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"Saved {save_path}", flush=True)

    # ═══ Final Summary ═══
    print("\n" + "=" * 70, flush=True)
    print("PHASE 87 RESULTS — Physics 101 Real Video", flush=True)
    print("=" * 70, flush=True)

    print("\nPROBE GATE:", flush=True)
    print(f"  {'Scenario':10s} {'V-JEPA2':>10s} {'DINOv2':>10s} {'Volume':>10s} {'Gap':>8s}", flush=True)
    for sc in ["spring", "fall", "ramp"]:
        if sc in probe_results:
            p = probe_results[sc]
            print(f"  {sc:10s} {p['vjepa2_auc']:10.3f} {p['dino_auc']:10.3f} {p['volume_auc']:10.3f} {p['gap']:+8.3f}", flush=True)

    if "spring_2agent" in results_all:
        r = results_all["spring_2agent"]
        print(f"\nSpring Mass (2-agent): {r['mean_acc']:.1%} ± {r['std_acc']:.1%}", flush=True)
    if "spring_4agent" in results_all:
        r = results_all["spring_4agent"]
        print(f"Spring Mass (4-agent): {r['mean_acc']:.1%} ± {r['std_acc']:.1%}", flush=True)

    if "restitution_extraction" in results_all:
        r = results_all["restitution_extraction"]
        print(f"\nFall Restitution: {r['n_valid']}/{r['n_total']} valid trials", flush=True)

    if "two_prop_2agent" in results_all:
        r = results_all["two_prop_2agent"]
        print(f"\nTwo-Property (2-agent): both={r['mean_both']:.1%} PosDis={r['mean_posdis']:.3f} "
              f"comp={r['n_compositional']}/10", flush=True)
    if "two_prop_4agent" in results_all:
        r = results_all["two_prop_4agent"]
        print(f"Two-Property (4-agent): both={r['mean_both']:.1%} PosDis={r['mean_posdis']:.3f} "
              f"comp={r['n_compositional']}/10", flush=True)

    print(f"\nTotal time: {total_time/60:.1f} min", flush=True)
    print("Phase 87 complete.", flush=True)

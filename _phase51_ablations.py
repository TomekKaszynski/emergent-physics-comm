"""
Phase 51 Ablations: Validating emergent communication claims
=============================================================
Runs 5 ablation experiments on the Phase 51 pixel-based elasticity system:

1. SUPERVISED BASELINE — Single CNN regresses restitution directly from video.
   Then use predictions for pairwise comparison. Upper bound without comm bottleneck.

2. RANDOM MESSAGES — Load trained sender, replace messages with random one-hot
   vectors at eval time. If accuracy → chance, messages carry real information.

3. SINGLE-FRAME — Retrain full comm system with only 1 frame (frame index 6,
   post-bounce). Tests if temporal dynamics are actually needed or if a single
   post-bounce snapshot suffices.

4. SHUFFLED MESSAGES — Load trained sender, but shuffle messages across pairs
   (break msg↔video correspondence). Tests if receiver exploits message content
   vs. some statistical artifact.

5. MUTUAL INFORMATION — Compute I(symbol; elasticity) from trained model.
   Quantifies how much physics the discrete channel actually transmits.

Run from ~/AI/:
  python _phase51_ablations.py
"""

import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


# ══════════════════════════════════════════════════════════════════
# Reuse Phase 51 architecture exactly
# ══════════════════════════════════════════════════════════════════

class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.conv(x).squeeze(-1).squeeze(-1)


class VideoEncoder(nn.Module):
    def __init__(self, hidden_dim, n_frames):
        super().__init__()
        self.frame_enc = FrameEncoder()
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, video):
        B, T = video.shape[:2]
        frames_flat = video.reshape(B * T, *video.shape[2:])
        frame_feats = self.frame_enc(frames_flat)
        frame_feats = frame_feats.reshape(B, T, 128)
        x = frame_feats.permute(0, 2, 1)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


class SingleFrameEncoder(nn.Module):
    """For single-frame ablation: no temporal conv, just CNN on one frame."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.frame_enc = FrameEncoder()
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, video):
        # video: (B, 1, 3, H, W) — single frame
        x = video[:, 0]  # (B, 3, H, W)
        return self.fc(self.frame_enc(x))


class PixelSender(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_frames, single_frame=False):
        super().__init__()
        self.vocab_size = vocab_size
        if single_frame:
            self.encoder = SingleFrameEncoder(hidden_dim)
        else:
            self.encoder = VideoEncoder(hidden_dim, n_frames)
        self.to_message = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video, tau=1.0, hard=True):
        h = self.encoder(video)
        logits = self.to_message(h)
        if self.training:
            message = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            idx = logits.argmax(dim=-1)
            message = F.one_hot(idx, self.vocab_size).float()
        return message, logits


class Receiver(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


class SupervisedPredictor(nn.Module):
    """Directly regresses restitution from video. No communication."""
    def __init__(self, hidden_dim, n_frames):
        super().__init__()
        self.encoder = VideoEncoder(hidden_dim, n_frames)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # restitution ∈ [0, 1]
        )

    def forward(self, video):
        h = self.encoder(video)
        return self.head(h).squeeze(-1)


class PixelOracle(nn.Module):
    def __init__(self, hidden_dim, n_frames):
        super().__init__()
        self.enc_a = VideoEncoder(hidden_dim, n_frames)
        self.enc_b = VideoEncoder(hidden_dim, n_frames)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, vid_a, vid_b):
        ha = self.enc_a(vid_a)
        hb = self.enc_b(vid_b)
        return self.head(torch.cat([ha, hb], dim=-1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data loading (shared)
# ══════════════════════════════════════════════════════════════════

def load_dataset(device):
    """Load Kubric dataset, return videos tensor, restitutions, train/val split."""
    dataset_dir = Path("kubric/output/elasticity_dataset")
    index_path = dataset_dir / "index.json"

    with open(index_path) as f:
        index = json.load(f)

    n_sample_frames = 8
    frame_indices = np.linspace(0, 47, n_sample_frames, dtype=int)

    all_videos = []
    restitutions = []

    for meta in index:
        sid = meta["scene_id"]
        scene_dir = dataset_dir / f"scene_{sid:04d}"
        if not (scene_dir / "rgba_00000.png").exists():
            continue

        frames = []
        skip = False
        for fi in frame_indices:
            fpath = scene_dir / f"rgba_{fi:05d}.png"
            if not fpath.exists():
                skip = True
                break
            img = Image.open(fpath).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            frames.append(img_np)

        if skip:
            continue

        video = np.stack(frames).transpose(0, 3, 1, 2)
        all_videos.append(video)
        restitutions.append(meta["restitution"])

    n_scenes = len(all_videos)
    all_videos = np.stack(all_videos)
    restitutions = np.array(restitutions)

    all_videos_t = torch.tensor(all_videos, dtype=torch.float32)
    all_rest_t = torch.tensor(restitutions, dtype=torch.float32)

    # ImageNet normalization
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    all_videos_t = (all_videos_t - img_mean) / img_std

    # Same split as Phase 51
    n_train = int(0.8 * n_scenes)
    perm = np.random.RandomState(42).permutation(n_scenes)
    train_ids = perm[:n_train]
    val_ids = perm[n_train:]

    print(f"│  Loaded {n_scenes} scenes, train={len(train_ids)}, val={len(val_ids)}")
    print(f"│  Restitution range: [{restitutions.min():.3f}, {restitutions.max():.3f}]")

    return all_videos_t, all_videos, all_rest_t, restitutions, train_ids, val_ids, n_scenes


def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


# ══════════════════════════════════════════════════════════════════
# ABLATION 1: Supervised Baseline
# ══════════════════════════════════════════════════════════════════

def ablation_supervised(all_videos_t, all_rest_t, restitutions, train_ids, val_ids, device):
    """Train CNN to directly predict restitution from video (MSE loss).
    Then evaluate pairwise comparison accuracy using predicted values."""
    print(f"\n{'=' * 60}")
    print(f"ABLATION 1: Supervised Baseline (direct restitution regression)")
    print(f"  Same CNN architecture, but trained with MSE on restitution.")
    print(f"  Pairwise accuracy = compare predicted values.")
    print(f"{'=' * 60}")
    t0 = time.time()

    hidden_dim = 128
    n_frames = 8
    n_epochs = 100
    lr = 1e-4
    batch_size = 32  # smaller batches, individual supervision

    model = SupervisedPredictor(hidden_dim, n_frames).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_rest_dev = all_rest_t.to(device)
    rng = np.random.RandomState(789)

    best_val_mae = float('inf')
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        # Shuffle training scenes
        train_perm = rng.permutation(train_ids)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_perm), batch_size):
            batch_ids = train_perm[i:i+batch_size]
            if len(batch_ids) < 4:
                continue
            vids = all_videos_t[batch_ids].to(device)
            targets = all_rest_dev[batch_ids]

            pred = model(vids)
            loss = F.mse_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_vids = all_videos_t[val_ids].to(device)
                val_targets = all_rest_dev[val_ids]

                # Process in chunks
                val_preds = []
                for i in range(0, len(val_ids), 64):
                    chunk = val_vids[i:i+64]
                    val_preds.append(model(chunk))
                val_pred = torch.cat(val_preds)

                val_mae = (val_pred - val_targets).abs().mean().item()
                val_mse = ((val_pred - val_targets) ** 2).mean().item()

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                print(f"│  Epoch {epoch:3d}/{n_epochs}: "
                      f"train_loss={epoch_loss/max(n_batches,1):.4f} "
                      f"val_MAE={val_mae:.4f} val_MSE={val_mse:.4f}", flush=True)

        if hasattr(torch.mps, 'empty_cache') and epoch % 20 == 0:
            torch.mps.empty_cache()

    # Restore best and evaluate pairwise accuracy
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        # Get predictions for all scenes
        all_preds = []
        for i in range(0, len(all_videos_t), 64):
            chunk = all_videos_t[i:i+64].to(device)
            all_preds.append(model(chunk).cpu())
        all_preds = torch.cat(all_preds).numpy()

        # Pairwise comparison on val set
        val_correct = 0
        val_total = 0
        for _ in range(50):
            vi_a, vi_b = sample_pairs(val_ids, 64, rng)
            pred_a = all_preds[vi_a]
            pred_b = all_preds[vi_b]
            true_a = restitutions[vi_a]
            true_b = restitutions[vi_b]

            pred_a_greater = pred_a > pred_b
            true_a_greater = true_a > true_b
            val_correct += (pred_a_greater == true_a_greater).sum()
            val_total += len(vi_a)

        pairwise_acc = val_correct / max(val_total, 1)

        # Also compute correlation
        from scipy.stats import spearmanr, pearsonr
        val_pred_vals = all_preds[val_ids]
        val_true_vals = restitutions[val_ids]
        spearman_r, _ = spearmanr(val_pred_vals, val_true_vals)
        pearson_r, _ = pearsonr(val_pred_vals, val_true_vals)

    print(f"│", flush=True)
    print(f"│  === SUPERVISED BASELINE RESULTS ===", flush=True)
    print(f"│  Pairwise accuracy:  {pairwise_acc*100:.1f}%", flush=True)
    print(f"│  Val MAE:            {best_val_mae:.4f}", flush=True)
    print(f"│  Spearman r:         {spearman_r:.3f}", flush=True)
    print(f"│  Pearson r:          {pearson_r:.3f}", flush=True)
    print(f"│  Phase 51 comm:      84.5%", flush=True)
    print(f"│  Phase 51 oracle:    89.1%", flush=True)
    print(f"│  Time: {time.time()-t0:.0f}s", flush=True)

    return {
        'pairwise_acc': float(pairwise_acc),
        'val_mae': best_val_mae,
        'spearman_r': float(spearman_r),
        'pearson_r': float(pearson_r),
    }


# ══════════════════════════════════════════════════════════════════
# ABLATION 2: Random Messages
# ══════════════════════════════════════════════════════════════════

def ablation_random_messages(all_videos_t, all_rest_t, restitutions, val_ids, device):
    """Load trained Phase 51 model, replace messages with random one-hot vectors."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"ABLATION 2: Random Messages", flush=True)
    print(f"  Load trained sender+receiver, but replace messages with random.", flush=True)
    print(f"  If acc → 50%, the learned messages carry real information.", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()

    vocab_size = 16
    hidden_dim = 128

    # Load trained model
    ckpt = torch.load("results/phase51_model.pt", map_location=device, weights_only=False)
    receiver = Receiver(vocab_size, hidden_dim).to(device)
    receiver.load_state_dict(ckpt['receiver'])
    receiver.eval()

    all_rest_dev = all_rest_t.to(device)
    rng = np.random.RandomState(999)

    with torch.no_grad():
        # Random messages
        correct_random = 0
        total = 0
        for _ in range(50):
            vi_a, vi_b = sample_pairs(val_ids, 64, rng)
            vr_a = all_rest_dev[vi_a]
            vr_b = all_rest_dev[vi_b]
            labels = (vr_a > vr_b).float()

            # Random one-hot messages
            rand_a = torch.randint(0, vocab_size, (len(vi_a),))
            rand_b = torch.randint(0, vocab_size, (len(vi_b),))
            msg_a = F.one_hot(rand_a, vocab_size).float().to(device)
            msg_b = F.one_hot(rand_b, vocab_size).float().to(device)

            pred = receiver(msg_a, msg_b)
            correct_random += ((pred > 0) == labels.bool()).sum().item()
            total += len(labels)

        acc_random = correct_random / max(total, 1)

    print(f"│", flush=True)
    print(f"│  === RANDOM MESSAGES RESULTS ===", flush=True)
    print(f"│  Random messages:    {acc_random*100:.1f}%", flush=True)
    print(f"│  Expected (chance):  50.0%", flush=True)
    print(f"│  Phase 51 learned:   84.5%", flush=True)
    print(f"│  Time: {time.time()-t0:.0f}s", flush=True)

    return {'acc_random': float(acc_random)}


# ══════════════════════════════════════════════════════════════════
# ABLATION 3: Single-Frame (no temporal info)
# ══════════════════════════════════════════════════════════════════

def ablation_single_frame(all_videos_t, all_rest_t, restitutions, train_ids, val_ids, device):
    """Retrain full comm system with only 1 frame (post-bounce).
    Tests whether temporal dynamics are needed or single snapshot suffices."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"ABLATION 3: Single-Frame Communication", flush=True)
    print(f"  Same architecture but only frame index 6 (post-bounce).", flush=True)
    print(f"  If accuracy stays high, temporal reasoning isn't needed.", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()

    vocab_size = 16
    hidden_dim = 128
    n_epochs = 150  # more epochs — harder task with less info
    lr = 1e-4
    batch_size = 64
    gumbel_tau_start = 3.0
    gumbel_tau_end = 1.5
    soft_warmup_epochs = 30
    oracle_pretrain_epochs = 40

    # Extract single frame (index 6 of 8 = ~frame 40 of 48, post-bounce)
    # all_videos_t shape: (N, 8, 3, 128, 128)
    single_frame_idx = 6
    single_videos = all_videos_t[:, single_frame_idx:single_frame_idx+1, :, :, :]  # (N, 1, 3, H, W)
    print(f"│  Using frame index {single_frame_idx} (of 0-7), shape: {single_videos.shape}", flush=True)

    all_rest_dev = all_rest_t.to(device)
    rng = np.random.RandomState(321)
    batches_per_epoch = max(1, len(train_ids) * 4 // batch_size)

    # --- Oracle pretrain (single frame) ---
    oracle_enc = SingleFrameEncoder(hidden_dim).to(device)
    oracle_head = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    oracle_params = list(oracle_enc.parameters()) + list(oracle_head.parameters())
    oracle_opt = torch.optim.Adam(oracle_params, lr=lr)

    print(f"│  Pre-training single-frame oracle ({oracle_pretrain_epochs} epochs)...", flush=True)
    for ep in range(1, oracle_pretrain_epochs + 1):
        oracle_enc.train(); oracle_head.train()
        for bi in range(batches_per_epoch):
            idx_a, idx_b = sample_pairs(train_ids, batch_size, rng)
            vid_a = single_videos[idx_a].to(device)
            vid_b = single_videos[idx_b].to(device)
            labels = (all_rest_dev[idx_a] > all_rest_dev[idx_b]).float()

            ha = oracle_enc(vid_a)
            hb = oracle_enc(vid_b)
            pred = oracle_head(torch.cat([ha, hb], dim=-1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, labels)

            oracle_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle_params, 1.0)
            oracle_opt.step()

        if ep % 10 == 0:
            oracle_enc.eval(); oracle_head.eval()
            with torch.no_grad():
                c, t = 0, 0
                for _ in range(10):
                    vi_a, vi_b = sample_pairs(val_ids, 64, rng)
                    va = single_videos[vi_a].to(device)
                    vb = single_videos[vi_b].to(device)
                    lb = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()
                    p = oracle_head(torch.cat([oracle_enc(va), oracle_enc(vb)], dim=-1)).squeeze(-1)
                    c += ((p > 0) == lb.bool()).sum().item()
                    t += len(lb)
                print(f"│    Oracle ep {ep}: val acc = {c/t:.3f}", flush=True)

    # Evaluate single-frame oracle
    oracle_enc.eval(); oracle_head.eval()
    with torch.no_grad():
        c, t = 0, 0
        for _ in range(50):
            vi_a, vi_b = sample_pairs(val_ids, 64, rng)
            va = single_videos[vi_a].to(device)
            vb = single_videos[vi_b].to(device)
            lb = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()
            p = oracle_head(torch.cat([oracle_enc(va), oracle_enc(vb)], dim=-1)).squeeze(-1)
            c += ((p > 0) == lb.bool()).sum().item()
            t += len(lb)
        sf_oracle_acc = c / t
    print(f"│  Single-frame oracle: {sf_oracle_acc*100:.1f}%", flush=True)

    # --- Communication training (single frame) ---
    torch.manual_seed(42)
    sender = PixelSender(hidden_dim, vocab_size, 1, single_frame=True).to(device)
    receiver = Receiver(vocab_size, hidden_dim).to(device)

    # Bootstrap from oracle
    sender_enc_state = sender.encoder.state_dict()
    oracle_enc_state = oracle_enc.state_dict()
    for key in sender_enc_state:
        if key in oracle_enc_state:
            sender_enc_state[key] = oracle_enc_state[key].clone()
    sender.encoder.load_state_dict(sender_enc_state)
    print(f"│  Bootstrapped sender from single-frame oracle", flush=True)

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=lr)
    rng = np.random.RandomState(654)

    best_val = 0.0
    best_sender_state = None
    best_receiver_state = None

    for epoch in range(1, n_epochs + 1):
        progress = min(epoch / (n_epochs * 0.7), 1.0)
        g_tau = gumbel_tau_start + (gumbel_tau_end - gumbel_tau_start) * progress
        use_hard = epoch > soft_warmup_epochs

        sender.train(); receiver.train()
        for bi in range(batches_per_epoch):
            idx_a, idx_b = sample_pairs(train_ids, batch_size, rng)
            vid_a = single_videos[idx_a].to(device)
            vid_b = single_videos[idx_b].to(device)
            labels = (all_rest_dev[idx_a] > all_rest_dev[idx_b]).float()

            msg_a, _ = sender(vid_a, tau=g_tau, hard=use_hard)
            msg_b, _ = sender(vid_b, tau=g_tau, hard=use_hard)
            pred = receiver(msg_a, msg_b)
            loss = F.binary_cross_entropy_with_logits(pred, labels)

            comm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            sender.eval(); receiver.eval()
            with torch.no_grad():
                c, t = 0, 0
                for _ in range(10):
                    vi_a, vi_b = sample_pairs(val_ids, 64, rng)
                    va = single_videos[vi_a].to(device)
                    vb = single_videos[vi_b].to(device)
                    lb = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()
                    ma, _ = sender(va)
                    mb, _ = sender(vb)
                    p = receiver(ma, mb)
                    c += ((p > 0) == lb.bool()).sum().item()
                    t += len(lb)
                val_acc = c / t

                # Entropy
                n_ent = min(200, len(single_videos))
                ent_msgs, _ = sender(single_videos[:n_ent].to(device))
                msg_ids = ent_msgs.argmax(dim=-1).cpu().numpy()
                counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)

                if val_acc > best_val:
                    best_val = val_acc
                    best_sender_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}
                    best_receiver_state = {k: v.cpu().clone() for k, v in receiver.state_dict().items()}

                print(f"│  Epoch {epoch:3d}/{n_epochs}: val={val_acc:.3f} ent={entropy:.3f} "
                      f"best={best_val:.3f}", flush=True)

        if hasattr(torch.mps, 'empty_cache') and epoch % 20 == 0:
            torch.mps.empty_cache()

    # Final eval with best model
    if best_sender_state is not None:
        sender.load_state_dict({k: v.to(device) for k, v in best_sender_state.items()})
        receiver.load_state_dict({k: v.to(device) for k, v in best_receiver_state.items()})

    sender.eval(); receiver.eval()
    with torch.no_grad():
        c, t = 0, 0
        for _ in range(50):
            vi_a, vi_b = sample_pairs(val_ids, 64, rng)
            va = single_videos[vi_a].to(device)
            vb = single_videos[vi_b].to(device)
            lb = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()
            ma, _ = sender(va)
            mb, _ = sender(vb)
            p = receiver(ma, mb)
            c += ((p > 0) == lb.bool()).sum().item()
            t += len(lb)
        final_sf_comm = c / t

    print(f"│", flush=True)
    print(f"│  === SINGLE-FRAME RESULTS ===", flush=True)
    print(f"│  Single-frame comm:   {final_sf_comm*100:.1f}%", flush=True)
    print(f"│  Single-frame oracle: {sf_oracle_acc*100:.1f}%", flush=True)
    print(f"│  Phase 51 comm (8f):  84.5%", flush=True)
    print(f"│  Phase 51 oracle (8f):89.1%", flush=True)
    print(f"│  If high → bounce height alone encodes elasticity", flush=True)
    print(f"│  If low  → temporal dynamics are essential", flush=True)
    print(f"│  Time: {time.time()-t0:.0f}s", flush=True)

    return {
        'sf_comm_acc': float(final_sf_comm),
        'sf_oracle_acc': float(sf_oracle_acc),
    }


# ══════════════════════════════════════════════════════════════════
# ABLATION 4: Shuffled Messages
# ══════════════════════════════════════════════════════════════════

def ablation_shuffled_messages(all_videos_t, all_rest_t, restitutions, val_ids, device):
    """Load trained model, shuffle messages across pairs (break correspondence)."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"ABLATION 4: Shuffled Messages", flush=True)
    print(f"  Trained sender encodes videos, but messages are shuffled", flush=True)
    print(f"  across pairs before receiver sees them.", flush=True)
    print(f"  Tests if receiver uses content vs. statistical artifact.", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()

    vocab_size = 16
    hidden_dim = 128

    ckpt = torch.load("results/phase51_model.pt", map_location=device, weights_only=False)
    sender = PixelSender(hidden_dim, vocab_size, 8).to(device)
    receiver = Receiver(vocab_size, hidden_dim).to(device)
    sender.load_state_dict(ckpt['sender'])
    receiver.load_state_dict(ckpt['receiver'])
    sender.eval(); receiver.eval()

    all_rest_dev = all_rest_t.to(device)
    rng = np.random.RandomState(777)

    with torch.no_grad():
        # Normal (sanity check)
        correct_normal = 0
        # Shuffled
        correct_shuffled = 0
        total = 0

        for _ in range(50):
            vi_a, vi_b = sample_pairs(val_ids, 64, rng)
            vv_a = all_videos_t[vi_a].to(device)
            vv_b = all_videos_t[vi_b].to(device)
            labels = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()

            msg_a, _ = sender(vv_a)
            msg_b, _ = sender(vv_b)

            # Normal
            pred_normal = receiver(msg_a, msg_b)
            correct_normal += ((pred_normal > 0) == labels.bool()).sum().item()

            # Shuffle: randomly permute msg_a and msg_b independently
            perm_a = torch.randperm(len(msg_a))
            perm_b = torch.randperm(len(msg_b))
            pred_shuffled = receiver(msg_a[perm_a], msg_b[perm_b])
            correct_shuffled += ((pred_shuffled > 0) == labels.bool()).sum().item()

            total += len(labels)

        acc_normal = correct_normal / max(total, 1)
        acc_shuffled = correct_shuffled / max(total, 1)

    print(f"│", flush=True)
    print(f"│  === SHUFFLED MESSAGES RESULTS ===", flush=True)
    print(f"│  Normal (sanity):    {acc_normal*100:.1f}%", flush=True)
    print(f"│  Shuffled messages:  {acc_shuffled*100:.1f}%", flush=True)
    print(f"│  Expected (chance):  50.0%", flush=True)
    print(f"│  Time: {time.time()-t0:.0f}s", flush=True)

    return {
        'acc_normal': float(acc_normal),
        'acc_shuffled': float(acc_shuffled),
    }


# ══════════════════════════════════════════════════════════════════
# ABLATION 5: Mutual Information Analysis
# ══════════════════════════════════════════════════════════════════

def ablation_mutual_information(all_videos_t, all_rest_t, restitutions, val_ids, n_scenes, device):
    """Compute I(symbol; elasticity) — how much physics info the channel transmits."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"ABLATION 5: Mutual Information I(symbol; elasticity)", flush=True)
    print(f"  Quantifies information content of discrete channel.", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()

    vocab_size = 16
    hidden_dim = 128

    ckpt = torch.load("results/phase51_model.pt", map_location=device, weights_only=False)
    sender = PixelSender(hidden_dim, vocab_size, 8).to(device)
    sender.load_state_dict(ckpt['sender'])
    sender.eval()

    with torch.no_grad():
        all_msg_ids = []
        for i in range(0, n_scenes, 100):
            vids = all_videos_t[i:i+100].to(device)
            msgs, _ = sender(vids)
            all_msg_ids.append(msgs.argmax(dim=-1).cpu().numpy())
        all_msg_ids = np.concatenate(all_msg_ids)

    # --- Discrete MI: bin restitution into K bins ---
    n_bins = 10
    rest_binned = np.digitize(restitutions, np.linspace(0, 1, n_bins + 1)[1:-1])

    # Joint distribution P(symbol, bin)
    joint = np.zeros((vocab_size, n_bins))
    for i in range(n_scenes):
        joint[all_msg_ids[i], rest_binned[i]] += 1
    joint /= joint.sum()

    # Marginals
    p_sym = joint.sum(axis=1)  # P(symbol)
    p_bin = joint.sum(axis=0)  # P(bin)

    # MI = sum P(s,b) * log(P(s,b) / (P(s)*P(b)))
    mi = 0.0
    for s in range(vocab_size):
        for b in range(n_bins):
            if joint[s, b] > 1e-10 and p_sym[s] > 1e-10 and p_bin[b] > 1e-10:
                mi += joint[s, b] * np.log2(joint[s, b] / (p_sym[s] * p_bin[b]))

    # Channel capacity = log2(vocab_size)
    channel_capacity = np.log2(vocab_size)

    # Entropy of symbols
    h_sym = -np.sum(p_sym[p_sym > 0] * np.log2(p_sym[p_sym > 0]))

    # Entropy of binned restitution
    h_bin = -np.sum(p_bin[p_bin > 0] * np.log2(p_bin[p_bin > 0]))

    # Normalized MI
    nmi = mi / min(h_sym, h_bin) if min(h_sym, h_bin) > 0 else 0

    # --- Per-symbol stats ---
    print(f"│", flush=True)
    print(f"│  === MUTUAL INFORMATION RESULTS ===", flush=True)
    print(f"│  I(symbol; elasticity):    {mi:.3f} bits", flush=True)
    print(f"│  Channel capacity:         {channel_capacity:.3f} bits (log2({vocab_size}))", flush=True)
    print(f"│  H(symbol):               {h_sym:.3f} bits", flush=True)
    print(f"│  H(elasticity_binned):     {h_bin:.3f} bits", flush=True)
    print(f"│  Normalized MI:            {nmi:.3f}", flush=True)
    print(f"│  MI / capacity:            {mi/channel_capacity:.3f}", flush=True)
    print(f"│", flush=True)

    # Variance reduction: how much does knowing the symbol reduce
    # variance in restitution prediction?
    total_var = np.var(restitutions)
    conditional_var = 0.0
    for s in range(vocab_size):
        mask = all_msg_ids == s
        if mask.sum() > 1:
            conditional_var += (mask.sum() / n_scenes) * np.var(restitutions[mask])

    var_reduction = 1.0 - conditional_var / total_var if total_var > 0 else 0

    print(f"│  Total var(restitution):   {total_var:.4f}", flush=True)
    print(f"│  Conditional var (given sym): {conditional_var:.4f}", flush=True)
    print(f"│  Variance reduction:       {var_reduction*100:.1f}%", flush=True)
    print(f"│  (% of restitution variance explained by symbol)", flush=True)
    print(f"│  Time: {time.time()-t0:.0f}s", flush=True)

    return {
        'mi_bits': float(mi),
        'channel_capacity': float(channel_capacity),
        'h_symbol': float(h_sym),
        'h_elasticity': float(h_bin),
        'nmi': float(nmi),
        'mi_over_capacity': float(mi / channel_capacity),
        'variance_reduction': float(var_reduction),
    }


# ══════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════

def make_summary_figure(results):
    """Create summary visualization of all ablations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Accuracy comparison bar chart
    ax = axes[0]
    conditions = []
    accuracies = []
    colors = []

    conditions.append('Phase 51\n(8-frame comm)')
    accuracies.append(84.5)
    colors.append('#2196F3')

    if 'supervised' in results:
        conditions.append(f"Supervised\nbaseline")
        accuracies.append(results['supervised']['pairwise_acc'] * 100)
        colors.append('#4CAF50')

    conditions.append('Oracle\n(full obs)')
    accuracies.append(89.1)
    colors.append('#8BC34A')

    if 'single_frame' in results:
        conditions.append('Single-frame\ncomm')
        accuracies.append(results['single_frame']['sf_comm_acc'] * 100)
        colors.append('#FF9800')

    if 'random_messages' in results:
        conditions.append('Random\nmessages')
        accuracies.append(results['random_messages']['acc_random'] * 100)
        colors.append('#F44336')

    if 'shuffled_messages' in results:
        conditions.append('Shuffled\nmessages')
        accuracies.append(results['shuffled_messages']['acc_shuffled'] * 100)
        colors.append('#E91E63')

    conditions.append('Chance')
    accuracies.append(50.0)
    colors.append('#9E9E9E')

    bars = ax.bar(range(len(conditions)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('Pairwise Accuracy (%)')
    ax.set_title('Ablation: Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(40, 100)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Panel 2: MI analysis
    ax = axes[1]
    if 'mutual_info' in results:
        mi = results['mutual_info']
        categories = ['MI\n(bits)', 'H(symbol)\n(bits)', 'H(elast)\n(bits)',
                       'Capacity\n(bits)']
        values = [mi['mi_bits'], mi['h_symbol'], mi['h_elasticity'],
                  mi['channel_capacity']]
        bar_colors = ['#9C27B0', '#7B1FA2', '#6A1B9A', '#4A148C']
        ax.bar(range(len(categories)), values, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel('Bits')
        ax.set_title(f'Information Analysis\nNMI={mi["nmi"]:.3f}, VarReduction={mi["variance_reduction"]*100:.0f}%',
                      fontsize=11, fontweight='bold')
        for i, v in enumerate(values):
            ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'MI not computed', transform=ax.transAxes, ha='center')
        ax.set_title('Information Analysis')

    # Panel 3: Summary text
    ax = axes[2]
    ax.axis('off')
    lines = ["Phase 51 Ablation Summary\n"]

    if 'supervised' in results:
        sup = results['supervised']
        lines.append(f"SUPERVISED BASELINE:")
        lines.append(f"  Pairwise acc: {sup['pairwise_acc']*100:.1f}%")
        lines.append(f"  Spearman r:   {sup['spearman_r']:.3f}")
        lines.append(f"  Val MAE:      {sup['val_mae']:.4f}")
        lines.append("")

    lines.append(f"COMMUNICATION SYSTEM:")
    lines.append(f"  Phase 51:     84.5%")
    lines.append(f"  Oracle:       89.1%")
    lines.append("")

    if 'random_messages' in results:
        lines.append(f"RANDOM MESSAGES: {results['random_messages']['acc_random']*100:.1f}%")

    if 'shuffled_messages' in results:
        lines.append(f"SHUFFLED MSGS:  {results['shuffled_messages']['acc_shuffled']*100:.1f}%")

    if 'single_frame' in results:
        sf = results['single_frame']
        lines.append(f"SINGLE-FRAME:")
        lines.append(f"  Comm:   {sf['sf_comm_acc']*100:.1f}%")
        lines.append(f"  Oracle: {sf['sf_oracle_acc']*100:.1f}%")

    if 'mutual_info' in results:
        mi = results['mutual_info']
        lines.append(f"\nMUTUAL INFORMATION:")
        lines.append(f"  I(sym;e): {mi['mi_bits']:.3f} bits")
        lines.append(f"  NMI:      {mi['nmi']:.3f}")
        lines.append(f"  VarRed:   {mi['variance_reduction']*100:.0f}%")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top')

    fig.suptitle('Phase 51 Ablation Suite', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/phase51_ablations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results/phase51_ablations.png", flush=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("PHASE 51 ABLATION SUITE", flush=True)
    print("  5 experiments to validate emergent communication claims", flush=True)
    print("=" * 70, flush=True)
    t_total = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # Load data once
    all_videos_t, all_videos_raw, all_rest_t, restitutions, train_ids, val_ids, n_scenes = load_dataset(device)

    results = {}

    # Ablation 1: Supervised baseline
    try:
        results['supervised'] = ablation_supervised(
            all_videos_t, all_rest_t, restitutions, train_ids, val_ids, device)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

    # Ablation 2: Random messages
    try:
        results['random_messages'] = ablation_random_messages(
            all_videos_t, all_rest_t, restitutions, val_ids, device)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

    # Ablation 3: Single-frame (this takes longest — trains from scratch)
    try:
        results['single_frame'] = ablation_single_frame(
            all_videos_t, all_rest_t, restitutions, train_ids, val_ids, device)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

    # Ablation 4: Shuffled messages
    try:
        results['shuffled_messages'] = ablation_shuffled_messages(
            all_videos_t, all_rest_t, restitutions, val_ids, device)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

    # Ablation 5: Mutual information
    try:
        results['mutual_info'] = ablation_mutual_information(
            all_videos_t, all_rest_t, restitutions, val_ids, n_scenes, device)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

    # Save all results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_json = json.loads(json.dumps(results, default=convert))
    with open("results/phase51_ablations.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results/phase51_ablations.json", flush=True)

    # Summary visualization
    try:
        make_summary_figure(results)
    except Exception as e:
        print(f"  Viz failed: {e}", flush=True)

    # Final summary
    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}", flush=True)
    print(f"ABLATION SUITE COMPLETE [{elapsed:.0f}s]", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n  Phase 51 communication: 84.5%", flush=True)
    print(f"  Phase 51 oracle:        89.1%", flush=True)
    print(f"  Chance:                 50.0%", flush=True)
    print(flush=True)

    if 'supervised' in results:
        print(f"  Supervised baseline:    {results['supervised']['pairwise_acc']*100:.1f}%", flush=True)
    if 'random_messages' in results:
        print(f"  Random messages:        {results['random_messages']['acc_random']*100:.1f}%", flush=True)
    if 'shuffled_messages' in results:
        print(f"  Shuffled messages:      {results['shuffled_messages']['acc_shuffled']*100:.1f}%", flush=True)
    if 'single_frame' in results:
        print(f"  Single-frame comm:      {results['single_frame']['sf_comm_acc']*100:.1f}%", flush=True)
        print(f"  Single-frame oracle:    {results['single_frame']['sf_oracle_acc']*100:.1f}%", flush=True)
    if 'mutual_info' in results:
        mi = results['mutual_info']
        print(f"  MI(sym;elast):          {mi['mi_bits']:.3f} bits", flush=True)
        print(f"  NMI:                    {mi['nmi']:.3f}", flush=True)
        print(f"  Variance reduction:     {mi['variance_reduction']*100:.0f}%", flush=True)

    print(f"\n  INTERPRETATION GUIDE:", flush=True)
    print(f"  ─────────────────────", flush=True)
    print(f"  Supervised ≈ Phase 51  → comm bottleneck doesn't lose much", flush=True)
    print(f"  Supervised >> Phase 51 → comm bottleneck is costly", flush=True)
    print(f"  Random/Shuffled ≈ 50%  → messages carry real information ✓", flush=True)
    print(f"  Single-frame ≈ Phase 51 → temporal info NOT needed (weaker claim)", flush=True)
    print(f"  Single-frame << Phase 51 → temporal dynamics essential ✓", flush=True)
    print(f"  High MI, high VarRed   → symbols faithfully encode physics ✓", flush=True)


if __name__ == "__main__":
    main()

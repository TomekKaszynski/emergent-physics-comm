"""
Phase 51: Pixel-Based Elasticity Communication
================================================
Task: Same as Phase 50 — two agents each see one ball-drop, exchange one
discrete token each, predict which ball is bouncier. But now agents see
RENDERED RGB VIDEO instead of GT trajectories.

Communication loss is the ONLY supervision — perception must emerge
from the need to communicate.

Architecture:
  - Vision Encoder: small CNN per frame + temporal 1D conv
  - Shared Sender: vision_encoder → Linear → Gumbel-Softmax (vocab=16)
  - Receiver: two one-hot messages → binary prediction
  - Oracle: two separate vision encoders → concat → binary prediction

Dataset: 1000 Kubric ball-drop scenes with rendered RGB frames (128x128)
  - 8 evenly-spaced frames per video
  - Restitution ∈ [0.05, 0.95], appearance decorrelated

Run from ~/AI/:
  python _phase51_pixel_elasticity_communication.py
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


def run_phase51():
    print("=" * 70, flush=True)
    print("PHASE 51: Pixel-Based Elasticity Communication", flush=True)
    print("  Rendered RGB video — perception emerges from communication", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"│  Device: {device}", flush=True)

    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 0: Load Kubric dataset (rendered frames)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load rendered video frames", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    dataset_dir = Path("kubric/output/elasticity_dataset")
    index_path = dataset_dir / "index.json"

    with open(index_path) as f:
        index = json.load(f)

    # Check how many scenes have rendered frames
    n_total = len(index)
    n_rendered = 0
    for meta in index:
        sid = meta["scene_id"]
        scene_dir = dataset_dir / f"scene_{sid:04d}"
        if (scene_dir / "rgba_00000.png").exists():
            n_rendered += 1

    print(f"│  Total scenes: {n_total}, Rendered: {n_rendered}", flush=True)

    if n_rendered < 100:
        print(f"│  ERROR: Need at least 100 rendered scenes, have {n_rendered}", flush=True)
        print(f"│  Run: cd kubric && docker run ... generate_elasticity_dataset.py --render", flush=True)
        return {'verdict': 'BLOCKED'}

    # Sample 8 evenly-spaced frames from 48-frame videos (0-47)
    n_sample_frames = 8
    frame_indices = np.linspace(0, 47, n_sample_frames, dtype=int)
    print(f"│  Sampling frames: {frame_indices.tolist()}", flush=True)

    # Load all videos into memory
    all_videos = []  # (N, 8, 3, 128, 128)
    restitutions = []
    valid_ids = []

    for meta in index:
        sid = meta["scene_id"]
        scene_dir = dataset_dir / f"scene_{sid:04d}"

        # Skip scenes without rendered frames
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
            img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            frames.append(img_np)

        if skip:
            continue

        video = np.stack(frames)  # (8, H, W, 3)
        video = video.transpose(0, 3, 1, 2)  # (8, 3, H, W)
        all_videos.append(video)
        restitutions.append(meta["restitution"])
        valid_ids.append(sid)

    n_scenes = len(all_videos)
    all_videos = np.stack(all_videos)  # (N, 8, 3, 128, 128)
    restitutions = np.array(restitutions)

    # Convert to tensors
    all_videos_t = torch.tensor(all_videos, dtype=torch.float32)  # (N, 8, 3, 128, 128)
    all_rest_t = torch.tensor(restitutions, dtype=torch.float32)

    # Normalize: ImageNet-style (approximate, good enough for from-scratch CNN)
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    all_videos_t = (all_videos_t - img_mean) / img_std

    # Train/val split: 80/20 by scene
    n_train = int(0.8 * n_scenes)
    perm = np.random.RandomState(42).permutation(n_scenes)
    train_ids = perm[:n_train]
    val_ids = perm[n_train:]

    video_mem_mb = all_videos_t.nelement() * 4 / 1e6
    print(f"│  Loaded {n_scenes} videos: {all_videos_t.shape}", flush=True)
    print(f"│  Memory: {video_mem_mb:.0f} MB", flush=True)
    print(f"│  Train: {len(train_ids)}, Val: {len(val_ids)}", flush=True)
    print(f"│  Restitution range: [{restitutions.min():.3f}, {restitutions.max():.3f}]", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: Define models
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Define pixel-based communication architecture", flush=True)
    print(f"{'=' * 60}", flush=True)

    vocab_size = 16
    hidden_dim = 128
    n_epochs = 200
    oracle_pretrain_epochs = 50   # pre-train oracle, then bootstrap sender
    lr = 1e-4         # lower than Phase 50 — vision is harder
    batch_size = 64    # smaller — images use more memory
    gumbel_tau_start = 3.0    # higher than Phase 50 — keep channel diverse early
    gumbel_tau_end = 1.5
    soft_warmup_epochs = 30   # use soft Gumbel (no hard=True) for first N epochs
    n_frames_per_video = n_sample_frames  # 8

    class FrameEncoder(nn.Module):
        """Small CNN to encode a single 128x128 RGB frame."""
        def __init__(self):
            super().__init__()
            # 128x128 → 64 → 32 → 16 → 8
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),   # → 64x64
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # → 32x32
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1), # → 16x16
                nn.ReLU(),
                nn.Conv2d(128, 128, 4, stride=2, padding=1), # → 8x8
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),                     # → 1x1
            )

        def forward(self, x):
            # x: (B, 3, 128, 128)
            return self.conv(x).squeeze(-1).squeeze(-1)  # (B, 128)

    class VideoEncoder(nn.Module):
        """Encode a video (sequence of frames) into a fixed representation.
        Per-frame CNN + temporal 1D conv to capture bounce dynamics."""
        def __init__(self, hidden_dim, n_frames):
            super().__init__()
            self.frame_enc = FrameEncoder()
            # Temporal conv over frame features to capture bounce patterns
            self.temporal = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # → (B, 128, 1)
            )
            self.fc = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, video):
            # video: (B, T, 3, H, W)
            B, T = video.shape[:2]
            # Encode each frame
            frames_flat = video.reshape(B * T, *video.shape[2:])  # (B*T, 3, H, W)
            frame_feats = self.frame_enc(frames_flat)  # (B*T, 128)
            frame_feats = frame_feats.reshape(B, T, 128)  # (B, T, 128)
            # Temporal processing
            x = frame_feats.permute(0, 2, 1)  # (B, 128, T)
            x = self.temporal(x).squeeze(-1)   # (B, 128)
            return self.fc(x)  # (B, hidden_dim)

    class PixelSender(nn.Module):
        """Encodes video → discrete message via Gumbel-Softmax."""
        def __init__(self, hidden_dim, vocab_size, n_frames):
            super().__init__()
            self.vocab_size = vocab_size
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
        """Takes two messages → predicts which ball is bouncier."""
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

    class PixelOracle(nn.Module):
        """Sees both raw videos — upper bound."""
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

    torch.manual_seed(42)
    sender = PixelSender(hidden_dim, vocab_size, n_frames_per_video).to(device)
    receiver = Receiver(vocab_size, hidden_dim).to(device)
    oracle = PixelOracle(hidden_dim, n_frames_per_video).to(device)

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=lr)
    oracle_optimizer = torch.optim.Adam(oracle.parameters(), lr=lr)

    n_sender_params = sum(p.numel() for p in sender.parameters())
    n_receiver_params = sum(p.numel() for p in receiver.parameters())
    n_oracle_params = sum(p.numel() for p in oracle.parameters())

    print(f"│  Sender:   {n_sender_params:,} params", flush=True)
    print(f"│  Receiver: {n_receiver_params:,} params", flush=True)
    print(f"│  Oracle:   {n_oracle_params:,} params", flush=True)
    print(f"│  Vocab: {vocab_size}, Hidden: {hidden_dim}", flush=True)
    print(f"│  Epochs: {n_epochs}, Batch: {batch_size} pairs", flush=True)
    print(f"│  LR: {lr}, Gumbel tau: {gumbel_tau_start} → {gumbel_tau_end}", flush=True)
    print(f"│  Frames per video: {n_frames_per_video}", flush=True)
    print(f"│  Task: binary — which ball is bouncier?", flush=True)
    print(f"│  Chance baseline: 50%", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1b: Pre-train oracle (bootstrap vision encoder)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1b: Pre-train oracle ({oracle_pretrain_epochs} epochs)", flush=True)
    print(f"  This teaches the vision encoder to extract bounce patterns.", flush=True)
    print(f"  Then we copy the encoder to the sender to break the", flush=True)
    print(f"  chicken-and-egg problem.", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1b = time.time()

    all_rest_dev = all_rest_t.to(device)

    def sample_pairs(scene_ids, batch_size, rng):
        """Sample random pairs of different scenes."""
        idx_a = rng.choice(scene_ids, size=batch_size)
        idx_b = rng.choice(scene_ids, size=batch_size)
        same = idx_a == idx_b
        while same.any():
            idx_b[same] = rng.choice(scene_ids, size=same.sum())
            same = idx_a == idx_b
        return idx_a, idx_b

    rng = np.random.RandomState(123)
    batches_per_epoch = max(1, len(train_ids) * 4 // batch_size)

    oracle.train()
    for ep in range(1, oracle_pretrain_epochs + 1):
        ep_correct = 0
        ep_total = 0
        for bi in range(batches_per_epoch):
            idx_a, idx_b = sample_pairs(train_ids, batch_size, rng)
            vid_a = all_videos_t[idx_a].to(device)
            vid_b = all_videos_t[idx_b].to(device)
            rest_a = all_rest_dev[idx_a]
            rest_b = all_rest_dev[idx_b]
            labels = (rest_a > rest_b).float()

            pred = oracle(vid_a, vid_b)
            loss = F.binary_cross_entropy_with_logits(pred, labels)

            oracle_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            oracle_optimizer.step()

            with torch.no_grad():
                ep_correct += ((pred > 0) == labels.bool()).sum().item()
                ep_total += len(labels)

        if ep % 10 == 0 or ep == 1:
            acc = ep_correct / max(ep_total, 1)
            print(f"│  Oracle pretrain epoch {ep:3d}/{oracle_pretrain_epochs}: "
                  f"acc={acc:.3f} [{time.time()-ts1b:.0f}s]", flush=True)

        if hasattr(torch.mps, 'empty_cache') and ep % 20 == 0:
            torch.mps.empty_cache()

    # Evaluate oracle after pre-training
    oracle.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for _ in range(20):
            vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
            vv_a = all_videos_t[vi_a].to(device)
            vv_b = all_videos_t[vi_b].to(device)
            vr_a = all_rest_dev[vi_a]
            vr_b = all_rest_dev[vi_b]
            vlabels = (vr_a > vr_b).float()
            vp = oracle(vv_a, vv_b)
            val_correct += ((vp > 0) == vlabels.bool()).sum().item()
            val_total += len(vlabels)
        oracle_pretrain_acc = val_correct / max(val_total, 1)
    print(f"│  Oracle pretrain val accuracy: {oracle_pretrain_acc:.3f}", flush=True)

    # Bootstrap: copy oracle.enc_a → sender.encoder
    sender_enc_state = sender.encoder.state_dict()
    oracle_enc_state = oracle.enc_a.state_dict()
    for key in sender_enc_state:
        if key in oracle_enc_state:
            sender_enc_state[key] = oracle_enc_state[key].clone()
    sender.encoder.load_state_dict(sender_enc_state)
    print(f"│  Copied oracle.enc_a → sender.encoder", flush=True)
    print(f"└─ Stage 1b done [{time.time()-ts1b:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Training
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Train pixel-based communication agents", flush=True)
    print(f"  Sender encoder bootstrapped from oracle.", flush=True)
    print(f"  Soft Gumbel for first {soft_warmup_epochs} epochs, then hard.", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    # Reset comm optimizer (sender has new weights from bootstrap)
    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=lr)

    # Reset rng for reproducible training
    rng = np.random.RandomState(456)

    history = {
        'epoch': [], 'train_comm': [], 'val_comm': [],
        'train_oracle': [], 'val_oracle': [],
        'msg_entropy': [], 'gumbel_tau': [],
    }

    best_val_comm = 0.0
    best_sender_state = None
    best_receiver_state = None
    best_epoch = 0
    collapse_patience = 0

    epoch_times = []
    for epoch in range(1, n_epochs + 1):
        ep_start = time.time()
        progress = min(epoch / (n_epochs * 0.7), 1.0)
        g_tau = gumbel_tau_start + (gumbel_tau_end - gumbel_tau_start) * progress

        epoch_comm_correct = 0
        epoch_oracle_correct = 0
        epoch_total = 0

        sender.train(); receiver.train(); oracle.train()

        for bi in range(batches_per_epoch):
            idx_a, idx_b = sample_pairs(train_ids, batch_size, rng)

            # Move batch videos to device
            vid_a = all_videos_t[idx_a].to(device)  # (B, 8, 3, 128, 128)
            vid_b = all_videos_t[idx_b].to(device)
            rest_a = all_rest_dev[idx_a]
            rest_b = all_rest_dev[idx_b]

            labels = (rest_a > rest_b).float()

            # --- Communication path ---
            use_hard = epoch > soft_warmup_epochs
            msg_a, _ = sender(vid_a, tau=g_tau, hard=use_hard)
            msg_b, _ = sender(vid_b, tau=g_tau, hard=use_hard)
            pred_comm = receiver(msg_a, msg_b)
            comm_loss = F.binary_cross_entropy_with_logits(pred_comm, labels)

            comm_optimizer.zero_grad()
            comm_loss.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

            # --- Oracle path ---
            pred_oracle = oracle(vid_a, vid_b)
            oracle_loss = F.binary_cross_entropy_with_logits(pred_oracle, labels)

            oracle_optimizer.zero_grad()
            oracle_loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            oracle_optimizer.step()

            with torch.no_grad():
                epoch_comm_correct += ((pred_comm > 0) == labels.bool()).sum().item()
                epoch_oracle_correct += ((pred_oracle > 0) == labels.bool()).sum().item()
                epoch_total += len(labels)

        train_comm_acc = epoch_comm_correct / max(epoch_total, 1)
        train_oracle_acc = epoch_oracle_correct / max(epoch_total, 1)

        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)

        # Clear MPS cache periodically
        if hasattr(torch.mps, 'empty_cache') and epoch % 20 == 0:
            torch.mps.empty_cache()

        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            sender.eval(); receiver.eval(); oracle.eval()

            with torch.no_grad():
                val_correct_comm = 0
                val_correct_oracle = 0
                val_total = 0

                for _ in range(10):  # 10 × 64 = 640 pairs
                    vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
                    vv_a = all_videos_t[vi_a].to(device)
                    vv_b = all_videos_t[vi_b].to(device)
                    vr_a = all_rest_dev[vi_a]
                    vr_b = all_rest_dev[vi_b]
                    vlabels = (vr_a > vr_b).float()

                    vm_a, _ = sender(vv_a)
                    vm_b, _ = sender(vv_b)
                    vp_comm = receiver(vm_a, vm_b)
                    vp_oracle = oracle(vv_a, vv_b)

                    val_correct_comm += ((vp_comm > 0) == vlabels.bool()).sum().item()
                    val_correct_oracle += ((vp_oracle > 0) == vlabels.bool()).sum().item()
                    val_total += len(vlabels)

                val_comm_acc = val_correct_comm / max(val_total, 1)
                val_oracle_acc = val_correct_oracle / max(val_total, 1)

                # Message entropy (use first 200 scenes to save time)
                n_ent = min(200, n_scenes)
                ent_vids = all_videos_t[:n_ent].to(device)
                all_msg, _ = sender(ent_vids)
                msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
                counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
                probs = counts / counts.sum()
                msg_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
                n_used = (probs > 0.01).sum()

            history['epoch'].append(epoch)
            history['train_comm'].append(train_comm_acc)
            history['val_comm'].append(val_comm_acc)
            history['train_oracle'].append(train_oracle_acc)
            history['val_oracle'].append(val_oracle_acc)
            history['msg_entropy'].append(float(msg_entropy))
            history['gumbel_tau'].append(g_tau)

            avg_time = np.mean(epoch_times[-20:])
            remaining = (n_epochs - epoch) * avg_time
            eta_str = f"{remaining/60:.0f}m" if remaining > 60 else f"{remaining:.0f}s"

            # Save best model
            if val_comm_acc > best_val_comm:
                best_val_comm = val_comm_acc
                best_sender_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}
                best_receiver_state = {k: v.cpu().clone() for k, v in receiver.state_dict().items()}
                best_epoch = epoch

            # Collapse detection
            if msg_entropy < 0.05 and best_val_comm > 0.6:
                collapse_patience += 1
            else:
                collapse_patience = 0

            if epoch % 20 == 0 or epoch == 1 or epoch == soft_warmup_epochs + 1:
                mode_str = "soft" if epoch <= soft_warmup_epochs else "hard"
                best_str = f" *best={best_val_comm:.3f}@{best_epoch}" if best_val_comm > 0.55 else ""
                print(f"│  Epoch {epoch:4d}/{n_epochs} [{mode_str}]: "
                      f"comm={train_comm_acc:.3f}/{val_comm_acc:.3f} "
                      f"oracle={train_oracle_acc:.3f}/{val_oracle_acc:.3f} "
                      f"ent={msg_entropy:.3f}({n_used}/{vocab_size}) "
                      f"τ={g_tau:.2f} [{ep_time:.1f}s] eta={eta_str}{best_str}", flush=True)

            # Early stop on collapse
            if collapse_patience >= 3:
                print(f"│  *** COLLAPSE DETECTED at epoch {epoch} ***", flush=True)
                print(f"│  Restoring best model from epoch {best_epoch} "
                      f"(val={best_val_comm:.3f})", flush=True)
                break

    # Restore best checkpoint
    if best_sender_state is not None and best_val_comm > 0.55:
        sender.load_state_dict({k: v.to(device) for k, v in best_sender_state.items()})
        receiver.load_state_dict({k: v.to(device) for k, v in best_receiver_state.items()})
        print(f"│  Restored best model from epoch {best_epoch} "
              f"(val comm={best_val_comm:.3f})", flush=True)

    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: Final evaluation
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Final evaluation", flush=True)
    print(f"{'=' * 60}", flush=True)

    sender.eval(); receiver.eval(); oracle.eval()

    with torch.no_grad():
        val_correct_comm = 0
        val_correct_oracle = 0
        val_total = 0

        for _ in range(50):  # 50 × 64 = 3200 pairs
            vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
            vv_a = all_videos_t[vi_a].to(device)
            vv_b = all_videos_t[vi_b].to(device)
            vr_a = all_rest_dev[vi_a]
            vr_b = all_rest_dev[vi_b]
            vlabels = (vr_a > vr_b).float()

            vm_a, _ = sender(vv_a)
            vm_b, _ = sender(vv_b)
            vp_comm = receiver(vm_a, vm_b)
            vp_oracle = oracle(vv_a, vv_b)

            val_correct_comm += ((vp_comm > 0) == vlabels.bool()).sum().item()
            val_correct_oracle += ((vp_oracle > 0) == vlabels.bool()).sum().item()
            val_total += len(vlabels)

        final_val_comm = val_correct_comm / max(val_total, 1)
        final_val_oracle = val_correct_oracle / max(val_total, 1)

        # Message analysis on all scenes
        all_msg_ids_list = []
        chunk = 100
        for i in range(0, n_scenes, chunk):
            vids = all_videos_t[i:i+chunk].to(device)
            msgs, _ = sender(vids)
            all_msg_ids_list.append(msgs.argmax(dim=-1).cpu().numpy())
        all_msg_ids = np.concatenate(all_msg_ids_list)

        # Per-symbol mean restitution
        symbol_means = {}
        for i in range(n_scenes):
            sym = int(all_msg_ids[i])
            if sym not in symbol_means:
                symbol_means[sym] = []
            symbol_means[sym].append(restitutions[i])

        print(f"│", flush=True)
        print(f"│  === VALIDATION RESULTS ({val_total} pairs) ===", flush=True)
        print(f"│  With communication:  {final_val_comm*100:.1f}%", flush=True)
        print(f"│  Oracle (full obs):   {final_val_oracle*100:.1f}%", flush=True)
        print(f"│  Chance baseline:     50.0%", flush=True)
        print(f"│  Phase 50 (GT traj):  91.6%", flush=True)
        print(f"│  Comm gain over chance: {(final_val_comm-0.5)*100:+.1f}pp", flush=True)
        print(f"│", flush=True)

        # Message semantics
        counts = np.bincount(all_msg_ids, minlength=vocab_size).astype(float)
        probs = counts / counts.sum()
        final_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
        n_symbols_used = (probs > 0.01).sum()

        print(f"│  === MESSAGE ANALYSIS ===", flush=True)
        print(f"│  Entropy: {final_entropy:.3f} (normalized)", flush=True)
        print(f"│  Symbols used: {n_symbols_used}/{vocab_size}", flush=True)
        print(f"│", flush=True)
        print(f"│  Symbol → Mean restitution:", flush=True)

        sym_stats = []
        for sym in sorted(symbol_means.keys()):
            vals = symbol_means[sym]
            m = np.mean(vals)
            s = np.std(vals)
            n = len(vals)
            sym_stats.append((sym, m, s, n))
            if n >= 5:
                print(f"│    Symbol {sym:2d}: mean_e={m:.3f} ± {s:.3f} (n={n:3d})", flush=True)

        # Symbol ordering
        if len(sym_stats) >= 2:
            used_stats = [(s, m, std, n) for s, m, std, n in sym_stats if n >= 5]
            if len(used_stats) >= 2:
                sym_order = sorted(used_stats, key=lambda x: x[1])
                ordering = [s for s, _, _, _ in sym_order]
                print(f"│", flush=True)
                print(f"│  Symbol order (low→high elasticity): {ordering}", flush=True)

                symbol_to_rank = {s: i for i, (s, _, _, _) in enumerate(sym_order)}
                rank_correct = 0
                rank_total = 0
                for _ in range(50):
                    vi_a, vi_b = sample_pairs(val_ids, min(64, len(val_ids)), rng)
                    ma = all_msg_ids[vi_a]
                    mb = all_msg_ids[vi_b]
                    ra = restitutions[vi_a]
                    rb = restitutions[vi_b]
                    for j in range(len(vi_a)):
                        rank_a = symbol_to_rank.get(int(ma[j]), -1)
                        rank_b = symbol_to_rank.get(int(mb[j]), -1)
                        if rank_a < 0 or rank_b < 0 or rank_a == rank_b:
                            continue
                        if (rank_a > rank_b) == (ra[j] > rb[j]):
                            rank_correct += 1
                        rank_total += 1

                if rank_total > 0:
                    rank_acc = rank_correct / rank_total
                    print(f"│  Ordinal accuracy: {rank_acc*100:.1f}% "
                          f"({rank_total} valid pairs)", flush=True)

        # Accuracy by difficulty
        print(f"│", flush=True)
        print(f"│  === ACCURACY BY DIFFICULTY ===", flush=True)
        gap_bins = [(0.0, 0.1, "tiny"), (0.1, 0.3, "small"),
                    (0.3, 0.5, "medium"), (0.5, 1.0, "large")]
        for gap_lo, gap_hi, name in gap_bins:
            correct = 0
            total = 0
            for _ in range(30):
                vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
                vr_a_np = restitutions[vi_a]
                vr_b_np = restitutions[vi_b]
                gaps = np.abs(vr_a_np - vr_b_np)
                mask = (gaps >= gap_lo) & (gaps < gap_hi)
                if mask.sum() == 0:
                    continue

                vv_a = all_videos_t[vi_a[mask]].to(device)
                vv_b = all_videos_t[vi_b[mask]].to(device)
                vlabels = (all_rest_dev[vi_a[mask]] > all_rest_dev[vi_b[mask]]).float()

                vm_a, _ = sender(vv_a)
                vm_b, _ = sender(vv_b)
                vp = receiver(vm_a, vm_b)

                correct += ((vp > 0) == vlabels.bool()).sum().item()
                total += len(vlabels)

            if total > 0:
                acc = correct / total
                print(f"│    Δe ∈ [{gap_lo:.1f}, {gap_hi:.1f}) ({name:6s}): "
                      f"{acc*100:.1f}% (n={total})", flush=True)

    # Save model
    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'oracle': oracle.state_dict(),
        'history': history,
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
    }, str(OUTPUT_DIR / "phase51_model.pt"))
    print(f"│  Saved results/phase51_model.pt", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 4: Visualization
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Visualization", flush=True)
    print(f"{'=' * 60}", flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Training curves
    ax = axes[0, 0]
    epochs = history['epoch']
    ax.plot(epochs, history['val_comm'], 'b-', linewidth=2,
            label=f'Comm val ({final_val_comm*100:.0f}%)')
    ax.plot(epochs, history['train_comm'], 'b--', alpha=0.5, label='Comm train')
    ax.plot(epochs, history['val_oracle'], 'g-', linewidth=2,
            label=f'Oracle val ({final_val_oracle*100:.0f}%)')
    ax.plot(epochs, history['train_oracle'], 'g--', alpha=0.5, label='Oracle train')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (50%)')
    ax.axhline(y=0.916, color='red', linestyle=':', alpha=0.5, label='Phase 50 GT (91.6%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Which Ball is Bouncier? (From Pixels)', fontsize=11)
    ax.legend(fontsize=7)
    ax.set_ylim(0.4, 1.05)

    # Panel 2: Message entropy
    ax = axes[0, 1]
    ax.plot(epochs, history['msg_entropy'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Message Diversity', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(epochs, history['gumbel_tau'], 'orange', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Gumbel τ', color='orange')

    # Panel 3: Symbol → restitution scatter
    ax = axes[0, 2]
    for sym in range(vocab_size):
        mask = all_msg_ids == sym
        if mask.sum() > 0:
            ax.scatter(np.full(mask.sum(), sym) + np.random.randn(mask.sum()) * 0.15,
                       restitutions[mask], alpha=0.3, s=10,
                       c=restitutions[mask], cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('Message Symbol')
    ax.set_ylabel('True Restitution')
    ax.set_title('What Each Symbol Encodes', fontsize=11)
    ax.set_xticks(range(vocab_size))

    # Panel 4: Symbol frequency
    ax = axes[1, 0]
    ax.bar(range(vocab_size), probs, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Symbol Usage (entropy={final_entropy:.3f})', fontsize=11)
    ax.set_xticks(range(vocab_size))

    # Panel 5: Example frames from different symbol clusters
    ax = axes[1, 1]
    # Show frame 24 (mid-bounce) from 5 scenes per symbol, for 3 symbols
    used_syms = sorted([(n, s) for s, _, _, n in sym_stats if n >= 10], reverse=True)[:3]
    n_show_per_sym = 4
    grid_imgs = []
    grid_labels = []
    for _, sym in used_syms:
        sym_mask = np.where(all_msg_ids == sym)[0][:n_show_per_sym]
        for idx in sym_mask:
            # Show frame 4 (mid-video, index into 8-frame sample)
            frame = all_videos[idx, 4]  # (3, 128, 128)
            frame = frame.transpose(1, 2, 0)  # (H, W, 3)
            grid_imgs.append(frame)
            grid_labels.append(f"S{sym} e={restitutions[idx]:.2f}")

    if grid_imgs:
        n_imgs = len(grid_imgs)
        n_cols = min(n_show_per_sym, 4)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        combined = np.ones((n_rows * 130, n_cols * 130, 3))
        for i, (img, lbl) in enumerate(zip(grid_imgs, grid_labels)):
            r, c = i // n_cols, i % n_cols
            combined[r*130+1:r*130+129, c*130+1:c*130+129] = np.clip(img, 0, 1)
        ax.imshow(combined)
        ax.set_title('Example Frames by Symbol', fontsize=11)
    ax.axis('off')

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = time.time() - t0

    if final_val_comm > 0.85:
        verdict = "STRONG SUCCESS"
    elif final_val_comm > 0.70:
        verdict = "SUCCESS"
    elif final_val_comm > 0.60:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 51: Pixel Elasticity Comm\n"
        f"  (rendered RGB video)\n\n"
        f"Task: which ball is bouncier?\n"
        f"  From 8 RGB frames, binary\n"
        f"Channel: Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Data: {n_scenes} Kubric scenes\n"
        f"  128x128 RGB, 8 frames/video\n"
        f"  Paired online\n\n"
        f"=== VALIDATION ===\n"
        f"Communication: {final_val_comm*100:.1f}%\n"
        f"Oracle:        {final_val_oracle*100:.1f}%\n"
        f"Chance:        50.0%\n"
        f"Phase 50 (GT): 91.6%\n\n"
        f"Entropy: {final_entropy:.3f}\n"
        f"Symbols: {n_symbols_used}/{vocab_size}\n\n"
        f"Time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 51: Pixel-Based Elasticity Communication\n'
                 f'val comm={final_val_comm*100:.0f}% '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} '
                 f'symbols={n_symbols_used}/{vocab_size} | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase51_pixel_elasticity_comm.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase51_pixel_elasticity_comm.png", flush=True)

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val communication: {final_val_comm*100:.1f}% (target >70%)", flush=True)
    print(f"  Val oracle:        {final_val_oracle*100:.1f}% (target >85%)", flush=True)
    print(f"  Phase 50 (GT):     91.6%", flush=True)
    print(f"  Chance baseline:   50.0%", flush=True)
    print(f"  Message entropy:   {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"  Symbols used:      {n_symbols_used}/{vocab_size}", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

    return {
        'val_comm': final_val_comm,
        'val_oracle': final_val_oracle,
        'entropy': final_entropy,
        'symbols_used': int(n_symbols_used),
        'verdict': verdict,
    }


if __name__ == "__main__":
    run_phase51()

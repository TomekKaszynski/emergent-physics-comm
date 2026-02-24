"""
Phase 50: Emergent Communication about Elasticity (GT Trajectories)
====================================================================
Task: Two agents each watch a different ball drop. They exchange one
discrete token each. A receiver uses both messages to predict which
ball is bouncier. Neither agent can answer alone — communication is
NECESSARY, not just helpful.

This is the GT trajectory version (like Phase 48d proved the architecture
before adding perception). Phase 51 will swap in pixels.

Architecture:
  - Shared Sender: trajectory → discrete token (Gumbel-Softmax)
  - Receiver: two tokens → binary prediction (which is bouncier?)
  - Oracle: sees both raw trajectories → upper bound
  - No-comm: receiver guesses without messages → 50% baseline

Dataset: 1000 Kubric ball-drop scenes with varied restitution
  - Each scene: 48 frames of 3D positions
  - Restitution ∈ [0.05, 0.95], appearance decorrelated
  - Pairs sampled online: N scenes → O(N²) pairs

Run from ~/AI/:
  python _phase50_elasticity_communication.py
"""

import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def run_phase50():
    print("=" * 70, flush=True)
    print("PHASE 50: Emergent Communication about Elasticity", flush=True)
    print("  GT trajectories — proving the task before adding perception", flush=True)
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
    # STAGE 0: Load Kubric dataset
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load Kubric elasticity dataset", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    dataset_dir = Path("kubric/output/elasticity_dataset")
    index_path = dataset_dir / "index.json"

    with open(index_path) as f:
        index = json.load(f)

    n_scenes = len(index)
    print(f"│  Loaded {n_scenes} scenes", flush=True)

    # Load all trajectories
    trajectories = []  # list of (48, 3) arrays
    restitutions = []
    bounce_ratios = []

    for meta in index:
        sid = meta["scene_id"]
        scene_dir = dataset_dir / f"scene_{sid:04d}"
        pos = np.load(scene_dir / "positions.npy")  # (48, 3)
        trajectories.append(pos)
        restitutions.append(meta["restitution"])
        bounce_ratios.append(meta["bounce_ratio"])

    restitutions = np.array(restitutions)
    bounce_ratios = np.array(bounce_ratios)

    # Extract features from trajectories
    # Key insight: z-position over time IS the signal for elasticity
    # Also include velocity (dz) and acceleration (ddz)
    def extract_features(pos_array):
        """Extract physics features from (T, 3) position trajectory.

        Returns: (T, 5) — z, dz, ddz, |v|, height_from_ground
        """
        z = pos_array[:, 2]  # vertical position
        dz = np.diff(z, prepend=z[0])  # vertical velocity
        ddz = np.diff(dz, prepend=dz[0])  # vertical acceleration
        # Full 3D velocity magnitude
        vel = np.diff(pos_array, axis=0, prepend=pos_array[:1])
        speed = np.linalg.norm(vel, axis=1)
        # Height from approximate ground (min z in trajectory)
        ground = z.min()
        height = z - ground

        features = np.stack([z, dz, ddz, speed, height], axis=1)  # (T, 5)
        return features.astype(np.float32)

    all_features = []
    for pos in trajectories:
        feat = extract_features(pos)
        all_features.append(feat)

    all_features = np.stack(all_features)  # (N, 48, 5)
    n_frames, n_feat = all_features.shape[1], all_features.shape[2]

    # Normalize features globally
    feat_mean = all_features.reshape(-1, n_feat).mean(axis=0)
    feat_std = all_features.reshape(-1, n_feat).std(axis=0) + 1e-8
    all_features = (all_features - feat_mean) / feat_std

    # Convert to tensors
    all_features_t = torch.tensor(all_features, dtype=torch.float32)  # (N, 48, 5)
    all_restitutions_t = torch.tensor(restitutions, dtype=torch.float32)

    # Train/val split: 80/20 by scene
    n_train = int(0.8 * n_scenes)
    perm = np.random.RandomState(42).permutation(n_scenes)
    train_ids = perm[:n_train]
    val_ids = perm[n_train:]

    print(f"│  Features per frame: {n_feat} (z, dz, ddz, speed, height)", flush=True)
    print(f"│  Frames per scene: {n_frames}", flush=True)
    print(f"│  Feature shape: {all_features_t.shape}", flush=True)
    print(f"│  Train: {len(train_ids)} scenes, Val: {len(val_ids)} scenes", flush=True)
    print(f"│  Restitution range: [{restitutions.min():.3f}, {restitutions.max():.3f}]", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: Define models
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Define communication architecture", flush=True)
    print(f"{'=' * 60}", flush=True)

    vocab_size = 16   # discrete symbols available
    hidden_dim = 128
    n_epochs = 200    # converged by ~80 in v1, 200 is plenty
    lr = 3e-4
    batch_size = 256  # number of PAIRS per batch
    gumbel_tau_start = 2.0
    gumbel_tau_end = 1.2  # stop ABOVE collapse point (1.03 triggered collapse)
    entropy_weight = 0.0  # no entropy reg — it hurt in all runs with it
    input_dim = n_frames * n_feat  # 48 * 5 = 240

    class TrajectoryEncoder(nn.Module):
        """Encode a trajectory into a fixed-size representation.
        Uses a small 1D CNN to capture temporal patterns in the bounce."""
        def __init__(self, n_frames, n_feat, hidden_dim):
            super().__init__()
            # 1D conv over time: captures bounce patterns
            self.conv = nn.Sequential(
                nn.Conv1d(n_feat, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # (B, 64, 1)
            )
            self.fc = nn.Sequential(
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            # x: (B, T, F)
            x = x.permute(0, 2, 1)  # (B, F, T) for Conv1d
            x = self.conv(x).squeeze(-1)  # (B, 64)
            return self.fc(x)  # (B, hidden_dim)

    class Sender(nn.Module):
        """Encodes trajectory → discrete message via Gumbel-Softmax."""
        def __init__(self, n_frames, n_feat, hidden_dim, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.encoder = TrajectoryEncoder(n_frames, n_feat, hidden_dim)
            self.to_message = nn.Linear(hidden_dim, vocab_size)

        def forward(self, trajectory, tau=1.0):
            h = self.encoder(trajectory)
            logits = self.to_message(h)
            if self.training:
                message = F.gumbel_softmax(logits, tau=tau, hard=True)
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
                nn.Linear(hidden_dim // 2, 1),  # logit: P(ball_a bouncier)
            )

        def forward(self, msg_a, msg_b):
            return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)

    class Oracle(nn.Module):
        """Sees both raw trajectories — upper bound."""
        def __init__(self, n_frames, n_feat, hidden_dim):
            super().__init__()
            self.enc_a = TrajectoryEncoder(n_frames, n_feat, hidden_dim)
            self.enc_b = TrajectoryEncoder(n_frames, n_feat, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, traj_a, traj_b):
            ha = self.enc_a(traj_a)
            hb = self.enc_b(traj_b)
            return self.head(torch.cat([ha, hb], dim=-1)).squeeze(-1)

    torch.manual_seed(42)
    sender = Sender(n_frames, n_feat, hidden_dim, vocab_size).to(device)
    receiver = Receiver(vocab_size, hidden_dim).to(device)
    oracle = Oracle(n_frames, n_feat, hidden_dim).to(device)

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
    print(f"│  Gumbel tau: {gumbel_tau_start} → {gumbel_tau_end} (stop above collapse)", flush=True)
    print(f"│  Task: binary — which ball is bouncier?", flush=True)
    print(f"│  Chance baseline: 50%", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Training
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Train communication agents", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    # Move all features to device
    all_features_dev = all_features_t.to(device)
    all_rest_dev = all_restitutions_t.to(device)

    def sample_pairs(scene_ids, batch_size, rng):
        """Sample random pairs of different scenes. Returns (idx_a, idx_b, labels).
        Label = 1 if scene_a has higher restitution, 0 otherwise."""
        idx_a = rng.choice(scene_ids, size=batch_size)
        idx_b = rng.choice(scene_ids, size=batch_size)
        # Ensure different scenes
        same = idx_a == idx_b
        while same.any():
            idx_b[same] = rng.choice(scene_ids, size=same.sum())
            same = idx_a == idx_b
        return idx_a, idx_b

    rng = np.random.RandomState(123)
    batches_per_epoch = max(1, len(train_ids) * 4 // batch_size)  # ~4x coverage

    history = {
        'epoch': [], 'train_comm': [], 'val_comm': [],
        'train_oracle': [], 'val_oracle': [],
        'msg_entropy': [], 'gumbel_tau': [],
    }

    best_val_comm = 0.0
    best_sender_state = None
    best_receiver_state = None
    best_epoch = 0
    collapse_patience = 0  # count epochs with entropy < 0.05

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
            traj_a = all_features_dev[idx_a]  # (B, T, F)
            traj_b = all_features_dev[idx_b]
            rest_a = all_rest_dev[idx_a]
            rest_b = all_rest_dev[idx_b]

            # Label: 1 if ball_a is bouncier
            labels = (rest_a > rest_b).float()

            # --- Communication path ---
            msg_a, _ = sender(traj_a, tau=g_tau)
            msg_b, _ = sender(traj_b, tau=g_tau)
            pred_comm = receiver(msg_a, msg_b)
            comm_loss = F.binary_cross_entropy_with_logits(pred_comm, labels)

            comm_optimizer.zero_grad()
            comm_loss.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

            # --- Oracle path ---
            pred_oracle = oracle(traj_a, traj_b)
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

        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            sender.eval(); receiver.eval(); oracle.eval()

            with torch.no_grad():
                # Sample many val pairs for stable estimate
                val_correct_comm = 0
                val_correct_oracle = 0
                val_total = 0

                for _ in range(20):  # 20 batches × 256 = 5120 pairs
                    vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
                    vt_a = all_features_dev[vi_a]
                    vt_b = all_features_dev[vi_b]
                    vr_a = all_rest_dev[vi_a]
                    vr_b = all_rest_dev[vi_b]
                    vlabels = (vr_a > vr_b).float()

                    vm_a, _ = sender(vt_a)
                    vm_b, _ = sender(vt_b)
                    vp_comm = receiver(vm_a, vm_b)
                    vp_oracle = oracle(vt_a, vt_b)

                    val_correct_comm += ((vp_comm > 0) == vlabels.bool()).sum().item()
                    val_correct_oracle += ((vp_oracle > 0) == vlabels.bool()).sum().item()
                    val_total += len(vlabels)

                val_comm_acc = val_correct_comm / max(val_total, 1)
                val_oracle_acc = val_correct_oracle / max(val_total, 1)

                # Message entropy
                all_msg, _ = sender(all_features_dev[:500])
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

            # Save best model by val comm accuracy
            if val_comm_acc > best_val_comm:
                best_val_comm = val_comm_acc
                best_sender_state = {k: v.clone() for k, v in sender.state_dict().items()}
                best_receiver_state = {k: v.clone() for k, v in receiver.state_dict().items()}
                best_epoch = epoch

            # Collapse detection: entropy near zero after model had learned
            if msg_entropy < 0.05 and best_val_comm > 0.6:
                collapse_patience += 1
            else:
                collapse_patience = 0

            if epoch % 20 == 0 or epoch == 1:
                best_str = f" *best={best_val_comm:.3f}@{best_epoch}" if best_val_comm > 0.55 else ""
                print(f"│  Epoch {epoch:4d}/{n_epochs}: "
                      f"comm={train_comm_acc:.3f}/{val_comm_acc:.3f} "
                      f"oracle={train_oracle_acc:.3f}/{val_oracle_acc:.3f} "
                      f"ent={msg_entropy:.3f}({n_used}/{vocab_size}) "
                      f"τ={g_tau:.2f} eta={eta_str}{best_str}", flush=True)

            # Early stop on confirmed collapse
            if collapse_patience >= 3:
                print(f"│  *** COLLAPSE DETECTED at epoch {epoch} "
                      f"(entropy={msg_entropy:.3f}) ***", flush=True)
                print(f"│  Restoring best model from epoch {best_epoch} "
                      f"(val={best_val_comm:.3f})", flush=True)
                break

    # Restore best checkpoint if we have one
    if best_sender_state is not None and best_val_comm > 0.55:
        sender.load_state_dict(best_sender_state)
        receiver.load_state_dict(best_receiver_state)
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
        # Large-scale val evaluation
        val_correct_comm = 0
        val_correct_oracle = 0
        val_total = 0

        for _ in range(100):  # 100 × 256 = 25,600 pairs
            vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
            vt_a = all_features_dev[vi_a]
            vt_b = all_features_dev[vi_b]
            vr_a = all_rest_dev[vi_a]
            vr_b = all_rest_dev[vi_b]
            vlabels = (vr_a > vr_b).float()

            vm_a, _ = sender(vt_a)
            vm_b, _ = sender(vt_b)
            vp_comm = receiver(vm_a, vm_b)
            vp_oracle = oracle(vt_a, vt_b)

            val_correct_comm += ((vp_comm > 0) == vlabels.bool()).sum().item()
            val_correct_oracle += ((vp_oracle > 0) == vlabels.bool()).sum().item()
            val_total += len(vlabels)

        final_val_comm = val_correct_comm / max(val_total, 1)
        final_val_oracle = val_correct_oracle / max(val_total, 1)

        # Message analysis: what does each symbol mean?
        all_msg, _ = sender(all_features_dev)
        all_msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
        all_rest_np = restitutions

        # Per-symbol mean restitution
        symbol_means = {}
        symbol_counts = {}
        for i in range(n_scenes):
            sym = int(all_msg_ids[i])
            if sym not in symbol_means:
                symbol_means[sym] = []
            symbol_means[sym].append(all_rest_np[i])

        print(f"│", flush=True)
        print(f"│  === VALIDATION RESULTS (25,600 pairs) ===", flush=True)
        print(f"│  With communication:  {final_val_comm*100:.1f}%", flush=True)
        print(f"│  Oracle (full obs):   {final_val_oracle*100:.1f}%", flush=True)
        print(f"│  Chance baseline:     50.0%", flush=True)
        print(f"│  Comm gain over chance: {(final_val_comm-0.5)*100:+.1f}pp", flush=True)
        print(f"│", flush=True)

        # Message semantics
        counts = np.bincount(all_msg_ids, minlength=vocab_size).astype(float)
        probs = counts / counts.sum()
        final_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
        n_symbols_used = (probs > 0.01).sum()

        print(f"│  === MESSAGE ANALYSIS ===", flush=True)
        print(f"│  Entropy: {final_entropy:.3f} (normalized, 1.0 = uniform)", flush=True)
        print(f"│  Symbols used: {n_symbols_used}/{vocab_size}", flush=True)
        print(f"│", flush=True)
        print(f"│  Symbol → Mean restitution (what each symbol 'means'):", flush=True)

        # Sort symbols by mean restitution to check if language is ordered
        sym_stats = []
        for sym in sorted(symbol_means.keys()):
            vals = symbol_means[sym]
            m = np.mean(vals)
            s = np.std(vals)
            n = len(vals)
            sym_stats.append((sym, m, s, n))
            if n >= 5:
                print(f"│    Symbol {sym:2d}: mean_e={m:.3f} ± {s:.3f} (n={n:3d})", flush=True)

        # Check if symbols form an ordered "language"
        if len(sym_stats) >= 2:
            used_stats = [(s, m, std, n) for s, m, std, n in sym_stats if n >= 5]
            if len(used_stats) >= 2:
                means_sorted = sorted([m for _, m, _, _ in used_stats])
                # Spearman correlation between symbol rank and mean restitution
                sym_order = sorted(used_stats, key=lambda x: x[1])
                ordering = [s for s, _, _, _ in sym_order]
                print(f"│", flush=True)
                print(f"│  Symbol order (low→high elasticity): {ordering}", flush=True)

                # Test: does the ordering produce correct comparisons?
                # If symbol A > symbol B in the ordering, ball A should be bouncier
                symbol_to_rank = {s: i for i, (s, _, _, _) in enumerate(sym_order)}
                rank_correct = 0
                rank_total = 0
                for _ in range(50):
                    vi_a, vi_b = sample_pairs(val_ids, 256, rng)
                    ma = all_msg_ids[vi_a]
                    mb = all_msg_ids[vi_b]
                    ra = restitutions[vi_a]
                    rb = restitutions[vi_b]
                    for j in range(256):
                        rank_a = symbol_to_rank.get(int(ma[j]), -1)
                        rank_b = symbol_to_rank.get(int(mb[j]), -1)
                        if rank_a < 0 or rank_b < 0 or rank_a == rank_b:
                            continue
                        pred_a_bouncier = rank_a > rank_b
                        true_a_bouncier = ra[j] > rb[j]
                        if pred_a_bouncier == true_a_bouncier:
                            rank_correct += 1
                        rank_total += 1

                if rank_total > 0:
                    rank_acc = rank_correct / rank_total
                    print(f"│  Ordinal accuracy (rank comparison): {rank_acc*100:.1f}% "
                          f"({rank_total} valid pairs)", flush=True)

        # Accuracy by restitution gap (harder when similar)
        print(f"│", flush=True)
        print(f"│  === ACCURACY BY DIFFICULTY ===", flush=True)
        gap_bins = [(0.0, 0.1, "tiny"), (0.1, 0.3, "small"),
                    (0.3, 0.5, "medium"), (0.5, 1.0, "large")]
        for gap_lo, gap_hi, name in gap_bins:
            correct = 0
            total = 0
            for _ in range(50):
                vi_a, vi_b = sample_pairs(val_ids, 256, rng)
                vr_a_np = restitutions[vi_a]
                vr_b_np = restitutions[vi_b]
                gaps = np.abs(vr_a_np - vr_b_np)
                mask = (gaps >= gap_lo) & (gaps < gap_hi)
                if mask.sum() == 0:
                    continue

                vt_a = all_features_dev[vi_a[mask]]
                vt_b = all_features_dev[vi_b[mask]]
                vlabels = (all_rest_dev[vi_a[mask]] > all_rest_dev[vi_b[mask]]).float()

                vm_a, _ = sender(vt_a)
                vm_b, _ = sender(vt_b)
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
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
    }, str(OUTPUT_DIR / "phase50_model.pt"))
    print(f"│  Saved results/phase50_model.pt", flush=True)

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
    ax.plot(epochs, history['train_comm'], 'b--', alpha=0.5,
            label=f'Comm train')
    ax.plot(epochs, history['val_oracle'], 'g-', linewidth=2,
            label=f'Oracle val ({final_val_oracle*100:.0f}%)')
    ax.plot(epochs, history['train_oracle'], 'g--', alpha=0.5,
            label=f'Oracle train')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (50%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Which Ball is Bouncier? (Binary)', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.05)

    # Panel 2: Message entropy over training
    ax = axes[0, 1]
    ax.plot(epochs, history['msg_entropy'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Message Diversity', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(epochs, history['gumbel_tau'], 'orange', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Gumbel τ', color='orange')

    # Panel 3: Symbol → restitution mapping
    ax = axes[0, 2]
    scatter_colors = plt.cm.viridis(restitutions / restitutions.max())
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

    # Panel 5: Example trajectories colored by symbol
    ax = axes[1, 1]
    n_examples = min(50, n_scenes)
    cmap = plt.cm.tab10
    for i in range(n_examples):
        sym = all_msg_ids[i]
        z_traj = all_features[i, :, 0] * feat_std[0] + feat_mean[0]  # un-normalize z
        ax.plot(range(n_frames), z_traj, color=cmap(sym % 10), alpha=0.4, linewidth=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Z position (height)')
    ax.set_title('Trajectories Colored by Message', fontsize=11)

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
        f"Phase 50: Elasticity Communication\n"
        f"  (GT trajectories)\n\n"
        f"Task: which ball is bouncier?\n"
        f"  Binary, chance = 50%\n"
        f"Channel: Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Data: {n_scenes} Kubric scenes\n"
        f"  1000 drop trajectories\n"
        f"  Paired online → O(N²) pairs\n\n"
        f"=== VALIDATION ===\n"
        f"Communication: {final_val_comm*100:.1f}%\n"
        f"Oracle:        {final_val_oracle*100:.1f}%\n"
        f"Chance:        50.0%\n\n"
        f"Message entropy: {final_entropy:.3f}\n"
        f"Symbols used: {n_symbols_used}/{vocab_size}\n\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 50: Emergent Communication about Elasticity\n'
                 f'val comm={final_val_comm*100:.0f}% '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} '
                 f'symbols={n_symbols_used}/{vocab_size} | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase50_elasticity_comm.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase50_elasticity_comm.png", flush=True)

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val communication: {final_val_comm*100:.1f}% (target >70%)", flush=True)
    print(f"  Val oracle:        {final_val_oracle*100:.1f}% (target >90%)", flush=True)
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
    run_phase50()

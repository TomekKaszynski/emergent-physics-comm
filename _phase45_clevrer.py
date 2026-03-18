def run_phase45_clevrer_perception():
    """Phase 45: DINOv2 + Slot Attention on CLEVRER videos — real perception.

    Test whether DINOv2 → slot attention → tracking works on photorealistic
    rendered video. Pure perception evaluation, no planning or communication.
    """
    import time
    import json
    import os
    import copy
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from pathlib import Path

    print("=" * 70, flush=True)
    print("PHASE 45: DINOv2 + Slot Attention on CLEVRER Videos", flush=True)
    print("  Real perception pipeline — no planning, no communication", flush=True)
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
    data_dir = "clevrer_data"
    video_ids = list(range(10000, 10020))
    n_videos = len(video_ids)
    n_frames = 128

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations + verify data
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load annotations + verify data", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    annotations = {}
    for vid_id in video_ids:
        ann_path = f"{data_dir}/annotation_{vid_id}.json"
        with open(ann_path) as f:
            annotations[vid_id] = json.load(f)
        n_obj = len(annotations[vid_id]['object_property'])
        print(f"│  Video {vid_id}: {n_obj} objects, "
              f"{len(annotations[vid_id]['motion_trajectory'])} frames", flush=True)

    # 3D → 2D projection (fitted from empirical correspondences)
    # Affine mapping from 3D world (x, y) to normalized image coords (0-1, 0-1)
    # Fitted from 21 color-blob correspondences across 5 videos, mean residual ~2%
    PROJ_U = np.array([0.0589, 0.2286, 0.4850])  # u = a*x + b*y + c
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])   # v = a*x + b*y + c

    def project_3d_to_2d(x, y):
        """Project 3D world (x, y) to normalized image coords (u, v) in [0, 1]."""
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    # Verify projection on a sample
    sample_ann = annotations[10000]
    frame_objs = sample_ann['motion_trajectory'][64]['objects']
    in_view = [o for o in frame_objs if o['inside_camera_view']]
    print(f"│  Projection check (video 10000, frame 64):", flush=True)
    for o in in_view:
        u, v = project_3d_to_2d(o['location'][0], o['location'][1])
        props = {p['object_id']: p for p in sample_ann['object_property']}
        color = props[o['object_id']]['color']
        print(f"│    obj {o['object_id']} ({color}): 3D=({o['location'][0]:.2f}, "
              f"{o['location'][1]:.2f}) → 2D=({u:.3f}, {v:.3f})", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Frame extraction
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Extract frames from {n_videos} videos", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    all_frames = []  # list of [128, 3, 224, 224] arrays
    for vi, vid_id in enumerate(video_ids):
        cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
        frames = []
        for fi in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)  # HWC → CHW
            frames.append(frame)
        cap.release()
        all_frames.append(np.stack(frames))  # [128, 3, 224, 224]
        if (vi + 1) % 5 == 0:
            print(f"│  Extracted {vi+1}/{n_videos} videos "
                  f"({len(frames)} frames each)", flush=True)

    all_frames_tensor = torch.tensor(
        np.concatenate(all_frames, axis=0), dtype=torch.float32)
    # [2560, 3, 224, 224]
    print(f"│  Total frames: {all_frames_tensor.shape[0]} "
          f"({all_frames_tensor.shape})", flush=True)
    print(f"│  Memory: {all_frames_tensor.numel() * 4 / 1e9:.2f} GB", flush=True)
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: DINOv2 feature extraction
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: DINOv2 feature extraction (frozen vits14)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    cache_path = str(OUTPUT_DIR / "phase45_dino_features.pt")
    if os.path.exists(cache_path):
        print(f"│  Loading cached features from {cache_path}", flush=True)
        all_features = torch.load(cache_path, weights_only=True)
        print(f"│  Features shape: {all_features.shape}", flush=True)
    else:
        print(f"│  Loading DINOv2-small (vits14)...", flush=True)
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                              pretrained=True)
        dino.eval().to(device)
        for p in dino.parameters():
            p.requires_grad = False
        n_dino_params = sum(p.numel() for p in dino.parameters())
        print(f"│  DINOv2 params: {n_dino_params:,} (all frozen)", flush=True)

        # ImageNet normalization
        dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        # Extract features in batches
        batch_size_dino = 16
        n_total = all_frames_tensor.shape[0]
        feature_list = []
        for start in range(0, n_total, batch_size_dino):
            end = min(start + batch_size_dino, n_total)
            batch = all_frames_tensor[start:end].to(device)
            batch = (batch - dino_mean) / dino_std
            with torch.no_grad():
                features = dino.forward_features(batch)
                patch_tokens = features['x_norm_patchtokens']  # [B, 256, 384]
            feature_list.append(patch_tokens.cpu())
            if (start // batch_size_dino + 1) % 20 == 0:
                print(f"│    Processed {end}/{n_total} frames", flush=True)
            if device.type == 'mps' and (start // batch_size_dino + 1) % 50 == 0:
                torch.mps.empty_cache()

        all_features = torch.cat(feature_list, dim=0)  # [2560, 256, 384]
        print(f"│  Features shape: {all_features.shape}", flush=True)
        print(f"│  Features memory: {all_features.numel() * 4 / 1e9:.2f} GB", flush=True)

        # Cache to disk
        torch.save(all_features, cache_path)
        print(f"│  Cached features to {cache_path}", flush=True)

        # Free DINOv2 model
        del dino, dino_mean, dino_std
        if device.type == 'mps':
            torch.mps.empty_cache()

    # Free raw frames (no longer needed)
    del all_frames_tensor, all_frames
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Slot Attention training on DINOv2 features
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Train Slot Attention (feature reconstruction)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    dino_dim = 384
    n_patches = 256  # 16×16
    P = 16
    n_slots = 7  # up to 6 objects + background
    slot_dim = 64
    n_sa_iters = 5
    sa_epochs = 100
    sa_batch = 32
    sa_lr = 1e-4

    # ── Slot Attention Module (clean implementation) ──
    class SlotAttention(nn.Module):
        def __init__(self, n_slots, slot_dim, n_iters, feature_dim, epsilon=1e-8):
            super().__init__()
            self.n_slots = n_slots
            self.slot_dim = slot_dim
            self.n_iters = n_iters
            self.epsilon = epsilon

            # Per-slot learnable init (BO-QSA style)
            self.slot_init = nn.Parameter(
                torch.randn(1, n_slots, slot_dim) * 0.02)

            self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
            self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
            self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)

            self.gru = nn.GRUCell(slot_dim, slot_dim)
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(),
                nn.Linear(slot_dim * 2, slot_dim))

            self.norm_inputs = nn.LayerNorm(feature_dim)
            self.norm_slots = nn.LayerNorm(slot_dim)
            self.norm_mlp = nn.LayerNorm(slot_dim)

        def forward(self, inputs):
            """inputs: [B, N, D_in] → slots: [B, K, D_slot], attn: [B, K, N]"""
            B, N, _ = inputs.shape
            inputs = self.norm_inputs(inputs)
            k = self.project_k(inputs)
            v = self.project_v(inputs)
            slots = self.slot_init.expand(B, -1, -1)
            scale = self.slot_dim ** 0.5

            attn_weights = None
            for _ in range(self.n_iters):
                slots_prev = slots
                slots = self.norm_slots(slots)
                q = self.project_q(slots)
                attn_logits = torch.bmm(k, q.transpose(1, 2)) / scale  # [B,N,K]
                attn = F.softmax(attn_logits, dim=-1) + self.epsilon
                attn = attn / attn.sum(dim=1, keepdim=True)
                updates = torch.bmm(attn.transpose(1, 2), v)  # [B,K,D]
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                ).reshape(B, self.n_slots, self.slot_dim)
                slots = slots + self.mlp(self.norm_mlp(slots))
                attn_weights = attn.transpose(1, 2)  # [B,K,N]

            return slots, attn_weights

    # ── Encoder MLP (process DINOv2 features before SA) ──
    class EncoderMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.norm(self.mlp(x))

    # ── Spatial Broadcast Decoder ──
    class SpatialBroadcastDecoder(nn.Module):
        def __init__(self, slot_dim, output_dim, n_patches_side=16):
            super().__init__()
            self.n_patches_side = n_patches_side
            self.decoder = nn.Sequential(
                nn.Linear(slot_dim + 2, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, output_dim + 1))  # features + alpha

            # Position grid for 16×16 patches
            xs = torch.linspace(-1, 1, n_patches_side)
            ys = torch.linspace(-1, 1, n_patches_side)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            self.register_buffer(
                'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

        def forward(self, slots):
            """slots: [B, K, D] → recon: [B, N, D_out], alpha: [B, K, N]"""
            B, K, D = slots.shape
            N = self.grid.shape[0]

            slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
            grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
            dec_in = torch.cat([slots_bc, grid], dim=-1)

            decoded = self.decoder(dec_in)  # [B, K, N, D_out+1]
            features = decoded[:, :, :, :-1]
            alpha_logits = decoded[:, :, :, -1:]

            alpha = F.softmax(alpha_logits, dim=1)  # per-patch slot competition
            recon = (alpha * features).sum(dim=1)  # [B, N, D_out]
            alpha = alpha.squeeze(-1)  # [B, K, N]
            return recon, alpha

    # Build model
    torch.manual_seed(42)
    encoder = EncoderMLP(dino_dim).to(device)
    slot_attn = SlotAttention(
        n_slots=n_slots, slot_dim=slot_dim, n_iters=n_sa_iters,
        feature_dim=dino_dim).to(device)
    decoder = SpatialBroadcastDecoder(
        slot_dim=slot_dim, output_dim=dino_dim).to(device)

    total_params = (sum(p.numel() for p in encoder.parameters()) +
                    sum(p.numel() for p in slot_attn.parameters()) +
                    sum(p.numel() for p in decoder.parameters()))
    print(f"│  SA model: {total_params:,} trainable params", flush=True)
    print(f"│  Config: {n_slots} slots, {slot_dim}-dim, {n_sa_iters} iters, "
          f"{sa_epochs} epochs", flush=True)

    all_params = (list(encoder.parameters()) +
                  list(slot_attn.parameters()) +
                  list(decoder.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=sa_lr)

    n_total_frames = all_features.shape[0]
    n_train = int(0.8 * n_total_frames)

    # Warm-up LR schedule
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, sa_epochs + 1):
        encoder.train()
        slot_attn.train()
        decoder.train()

        perm = torch.randperm(n_train)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, sa_batch):
            idx = perm[start:start + sa_batch]
            features_batch = all_features[idx].to(device)  # [B, 256, 384]

            enc_feat = encoder(features_batch)
            slots, attn = slot_attn(enc_feat)
            recon, alpha = decoder(slots)

            loss = F.mse_loss(recon, features_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

            if device.type == 'mps' and n_batches % 50 == 0:
                torch.mps.empty_cache()

        scheduler.step()
        avg_loss = ep_loss / n_batches

        # Print + validate every 10 epochs or first epoch
        if epoch % 10 == 0 or epoch == 1:
            encoder.eval()
            slot_attn.eval()
            decoder.eval()

            with torch.no_grad():
                val_features = all_features[n_train:n_train + 256].to(device)
                enc_val = encoder(val_features)
                slots_val, attn_val = slot_attn(enc_val)
                recon_val, alpha_val = decoder(slots_val)
                val_loss = F.mse_loss(recon_val, val_features).item()

                # Entropy diagnostic
                ownership = alpha_val.argmax(dim=1)  # [B, N]
                B_d = ownership.shape[0]
                slot_counts = torch.zeros(B_d, n_slots, device=device)
                for s in range(n_slots):
                    slot_counts[:, s] = (ownership == s).float().sum(dim=1)
                mean_fracs = (slot_counts / n_patches).mean(dim=0)
                active = int((mean_fracs > 0.01).sum().item())
                max_cov = mean_fracs.max().item() * 100
                ent = -(alpha_val * (alpha_val + 1e-8).log()).sum(dim=1).mean()
                norm_ent = ent.item() / np.log(n_slots)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'encoder': {k: v.cpu().clone()
                                for k, v in encoder.state_dict().items()},
                    'slot_attn': {k: v.cpu().clone()
                                  for k, v in slot_attn.state_dict().items()},
                    'decoder': {k: v.cpu().clone()
                                for k, v in decoder.state_dict().items()},
                }

            elapsed = time.time() - t0
            print(f"│  Epoch {epoch:3d}/{sa_epochs}: "
                  f"train={avg_loss:.5f} val={val_loss:.5f} "
                  f"active={active}/{n_slots} max_cov={max_cov:.0f}% "
                  f"entropy={norm_ent:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    # Restore best model
    encoder.load_state_dict(best_state['encoder'])
    slot_attn.load_state_dict(best_state['slot_attn'])
    decoder.load_state_dict(best_state['decoder'])
    encoder.to(device).eval()
    slot_attn.to(device).eval()
    decoder.to(device).eval()
    print(f"│  Best val loss: {best_val_loss:.5f}", flush=True)
    print(f"└─ Stage 3 done [{time.time()-ts3:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 4: Object tracking via slot consistency
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Object tracking via slot consistency", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts4 = time.time()

    # Patch grid positions for centroid computation
    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2)
    # grid_positions[i] = (x_norm, y_norm) for patch i

    # Run SA on all frames, collect attention weights and centroids
    all_centroids = []  # [n_videos][n_frames, n_slots, 2]
    all_attns = []  # [n_videos][n_frames, n_slots, n_patches]

    for vi in range(n_videos):
        vid_start = vi * n_frames
        vid_end = vid_start + n_frames
        vid_centroids = []
        vid_attns = []

        for start in range(vid_start, vid_end, sa_batch):
            end = min(start + sa_batch, vid_end)
            batch_feat = all_features[start:end].to(device)

            with torch.no_grad():
                enc_feat = encoder(batch_feat)
                slots, attn = slot_attn(enc_feat)  # attn: [B, K, N]

            # Compute centroids from attention weights
            attn_np = attn.cpu().numpy()  # [B, K, N]
            for bi in range(attn_np.shape[0]):
                frame_centroids = np.zeros((n_slots, 2))
                for si in range(n_slots):
                    weights = attn_np[bi, si, :]  # [N]
                    w_sum = weights.sum()
                    if w_sum > 1e-6:
                        cx = (weights * grid_positions[:, 0].numpy()).sum() / w_sum
                        cy = (weights * grid_positions[:, 1].numpy()).sum() / w_sum
                        frame_centroids[si] = [cx, cy]
                    else:
                        frame_centroids[si] = [0.5, 0.5]
                vid_centroids.append(frame_centroids)

            vid_attns.append(attn.cpu().numpy())

        all_centroids.append(np.stack(vid_centroids))  # [128, 7, 2]
        all_attns.append(np.concatenate(vid_attns, axis=0))  # [128, 7, 256]

        if (vi + 1) % 5 == 0:
            print(f"│  Processed {vi+1}/{n_videos} videos", flush=True)

    # Hungarian matching across frames for temporal consistency
    # For each video: match slots between consecutive frames
    all_slot_assignments = []  # [n_videos][n_frames] array of permutations

    for vi in range(n_videos):
        centroids = all_centroids[vi]  # [128, 7, 2]
        assignments = [np.arange(n_slots)]  # frame 0: identity

        for fi in range(1, n_frames):
            prev_c = centroids[fi - 1][assignments[-1]]  # reordered previous
            curr_c = centroids[fi]

            # Cost: L2 distance between slots
            cost = np.zeros((n_slots, n_slots))
            for si in range(n_slots):
                for sj in range(n_slots):
                    cost[si, sj] = np.linalg.norm(prev_c[si] - curr_c[sj])

            row_ind, col_ind = linear_sum_assignment(cost)
            # col_ind[i] = which current slot matches previous slot i
            assignment = np.zeros(n_slots, dtype=int)
            for r, c in zip(row_ind, col_ind):
                assignment[r] = c
            assignments.append(assignment)

        all_slot_assignments.append(assignments)

    print(f"│  Hungarian matching done for {n_videos} videos", flush=True)
    print(f"└─ Stage 4 done [{time.time()-ts4:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 5: Evaluation against GT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 5: Evaluation against GT annotations", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts5 = time.time()

    # For each video, for each frame:
    # 1. Get GT 2D positions (project 3D annotations)
    # 2. Get slot centroids (reordered by temporal assignment)
    # 3. Match slots to GT objects (Hungarian on first frame where all in view)
    # 4. Track assignments, measure error

    tracking_errors = []  # per-object per-frame errors (normalized)
    slot_consistency_per_video = []
    binding_accuracy_per_video = []
    all_gt_positions = []  # for visualization
    all_pred_positions = []

    for vi in range(n_videos):
        vid_id = video_ids[vi]
        ann = annotations[vid_id]
        obj_props = {p['object_id']: p for p in ann['object_property']}
        n_obj = len(ann['object_property'])
        traj = ann['motion_trajectory']

        centroids = all_centroids[vi]  # [128, 7, 2]
        assignments = all_slot_assignments[vi]

        # Apply temporal reordering to centroids
        reordered_centroids = np.zeros_like(centroids)
        for fi in range(n_frames):
            perm = assignments[fi]
            # Create inverse permutation
            inv_perm = np.zeros(n_slots, dtype=int)
            for s_from, s_to in enumerate(perm):
                inv_perm[s_to] = s_from
            # Reorder: slot inv_perm[s] in frame fi maps to canonical slot s
            for s in range(n_slots):
                reordered_centroids[fi, s] = centroids[fi, perm[s]]

        # Find a reference frame where all objects are in view (or most)
        best_frame = 0
        best_in_view = 0
        for fi in range(n_frames):
            n_in = sum(1 for o in traj[fi]['objects']
                       if o['inside_camera_view'])
            if n_in > best_in_view:
                best_in_view = n_in
                best_frame = fi

        # Get GT 2D positions at reference frame
        ref_objs = [o for o in traj[best_frame]['objects']
                     if o['inside_camera_view']]
        gt_positions_ref = np.zeros((len(ref_objs), 2))
        gt_obj_ids = []
        for oi, o in enumerate(ref_objs):
            u, v = project_3d_to_2d(o['location'][0], o['location'][1])
            gt_positions_ref[oi] = [u, v]
            gt_obj_ids.append(o['object_id'])

        # Match slots to GT objects at reference frame
        # Only consider slots with significant coverage (>2% of patches)
        slot_coverage = np.zeros(n_slots)
        attn_ref = all_attns[vi][best_frame]  # [7, 256]
        ownership_ref = attn_ref.argmax(axis=0)  # [256]
        for s in range(n_slots):
            slot_coverage[s] = (ownership_ref == s).sum() / n_patches

        active_slots = [s for s in range(n_slots) if slot_coverage[s] > 0.02]

        pred_centroids_ref = reordered_centroids[best_frame]  # [7, 2]
        # Only use active slots for matching
        if len(active_slots) >= len(ref_objs):
            cost_mat = np.zeros((len(ref_objs), len(active_slots)))
            for gi in range(len(ref_objs)):
                for si_idx, si in enumerate(active_slots):
                    cost_mat[gi, si_idx] = np.linalg.norm(
                        gt_positions_ref[gi] - pred_centroids_ref[si])
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            gt_to_slot = {}
            for r, c in zip(row_ind, col_ind):
                gt_to_slot[gt_obj_ids[r]] = active_slots[c]
        else:
            # Fewer active slots than objects — do best effort
            cost_mat = np.zeros((len(ref_objs), n_slots))
            for gi in range(len(ref_objs)):
                for si in range(n_slots):
                    cost_mat[gi, si] = np.linalg.norm(
                        gt_positions_ref[gi] - pred_centroids_ref[si])
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            gt_to_slot = {}
            for r, c in zip(row_ind, col_ind):
                gt_to_slot[gt_obj_ids[r]] = c

        # Evaluate tracking across all frames
        vid_errors = []
        consistent_frames = 0
        bound_frames = 0
        total_eval_frames = 0

        vid_gt_pos = []
        vid_pred_pos = []

        for fi in range(n_frames):
            frame_objs = [o for o in traj[fi]['objects']
                          if o['inside_camera_view']
                          and o['object_id'] in gt_to_slot]
            if len(frame_objs) == 0:
                continue

            total_eval_frames += 1
            frame_errors = []
            used_slots = set()
            all_match = True

            gt_pos_frame = []
            pred_pos_frame = []

            for o in frame_objs:
                oid = o['object_id']
                assigned_slot = gt_to_slot[oid]
                gt_u, gt_v = project_3d_to_2d(
                    o['location'][0], o['location'][1])
                pred_uv = reordered_centroids[fi, assigned_slot]

                err = np.linalg.norm(
                    np.array([gt_u, gt_v]) - pred_uv)
                frame_errors.append(err)
                tracking_errors.append(err)

                gt_pos_frame.append([gt_u, gt_v])
                pred_pos_frame.append(pred_uv.tolist())

                # Check binding: slot not already used by another object
                if assigned_slot in used_slots:
                    all_match = False
                used_slots.add(assigned_slot)

                # Check consistency: is this slot still closest to this GT?
                closest_slot = None
                min_dist = float('inf')
                for s in range(n_slots):
                    d = np.linalg.norm(
                        np.array([gt_u, gt_v]) - reordered_centroids[fi, s])
                    if d < min_dist:
                        min_dist = d
                        closest_slot = s
                if closest_slot != assigned_slot:
                    all_match = False

            vid_errors.extend(frame_errors)
            if all_match:
                consistent_frames += 1
            bound_frames += 1 if len(used_slots) == len(frame_objs) else 0

            vid_gt_pos.append(gt_pos_frame)
            vid_pred_pos.append(pred_pos_frame)

        consistency = consistent_frames / max(total_eval_frames, 1)
        binding = bound_frames / max(total_eval_frames, 1)
        mean_err = np.mean(vid_errors) if vid_errors else 1.0

        slot_consistency_per_video.append(consistency)
        binding_accuracy_per_video.append(binding)

        all_gt_positions.append(vid_gt_pos)
        all_pred_positions.append(vid_pred_pos)

        if (vi + 1) % 5 == 0:
            print(f"│  Video {vid_id}: err={mean_err:.4f}, "
                  f"consistency={consistency*100:.0f}%, "
                  f"binding={binding*100:.0f}%", flush=True)

    # Aggregate metrics
    mean_tracking_error = np.mean(tracking_errors) if tracking_errors else 1.0
    median_tracking_error = np.median(tracking_errors) if tracking_errors else 1.0
    mean_consistency = np.mean(slot_consistency_per_video) * 100
    mean_binding = np.mean(binding_accuracy_per_video) * 100

    print(f"\n│  === RESULTS ===", flush=True)
    print(f"│  Tracking error (mean):   {mean_tracking_error:.4f} "
          f"({mean_tracking_error*100:.1f}% of frame)", flush=True)
    print(f"│  Tracking error (median): {median_tracking_error:.4f} "
          f"({median_tracking_error*100:.1f}% of frame)", flush=True)
    print(f"│  Slot consistency:        {mean_consistency:.1f}%", flush=True)
    print(f"│  Binding accuracy:        {mean_binding:.1f}%", flush=True)
    print(f"│  Targets: err<10%, consistency>80%, binding>85%", flush=True)
    print(f"└─ Stage 5 done [{time.time()-ts5:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 6: Visualization
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 6: Visualization", flush=True)
    print(f"{'=' * 60}", flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    slot_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                   '#911eb4', '#42d4f4', '#f032e6']

    # Panel 1: Sample frames with slot masks from 3 videos
    ax = axes[0, 0]
    # Show 3 videos, 1 frame each, with slot masks overlaid
    sample_vids = [0, 5, 10]
    sample_frame = 64
    n_show = min(3, len(sample_vids))
    composite_img = np.zeros((P * n_show, P * 2, 3))

    for si, vi in enumerate(sample_vids[:n_show]):
        # Load the original frame for display
        vid_id = video_ids[vi]
        cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (P, P))
            composite_img[si*P:(si+1)*P, :P, :] = frame_small / 255.0

        # Slot mask
        attn_frame = all_attns[vi][sample_frame]  # [7, 256]
        masks = attn_frame.reshape(n_slots, P, P)
        mask_composite = np.zeros((P, P, 3))
        for s in range(n_slots):
            color = plt.cm.Set1(s / n_slots)[:3]
            for c in range(3):
                mask_composite[:, :, c] += masks[s] * color[c]
        composite_img[si*P:(si+1)*P, P:2*P, :] = np.clip(mask_composite, 0, 1)

    ax.imshow(composite_img, interpolation='nearest')
    ax.set_title(f'Slot Masks (3 videos, frame {sample_frame})', fontsize=10)
    ax.set_ylabel('Videos 0, 5, 10')
    ax.set_xticks([P//2, P + P//2])
    ax.set_xticklabels(['Original (16x16)', 'Slot Masks'])
    ax.set_yticks([])

    # Panel 2: Slot centroids vs GT positions for one video
    ax = axes[0, 1]
    plot_vid = 0
    vid_id = video_ids[plot_vid]
    ann = annotations[vid_id]
    traj = ann['motion_trajectory']
    obj_props = {p['object_id']: p for p in ann['object_property']}

    # Plot GT trajectories
    obj_ids = [p['object_id'] for p in ann['object_property']]
    gt_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for oi, oid in enumerate(obj_ids):
        gt_us, gt_vs = [], []
        for fi in range(n_frames):
            o = [ob for ob in traj[fi]['objects'] if ob['object_id'] == oid][0]
            if o['inside_camera_view']:
                u, v = project_3d_to_2d(o['location'][0], o['location'][1])
                gt_us.append(u)
                gt_vs.append(v)
            else:
                gt_us.append(np.nan)
                gt_vs.append(np.nan)
        color = gt_colors[oi % len(gt_colors)]
        props = obj_props[oid]
        ax.plot(gt_us, gt_vs, '-', color=color, alpha=0.5, linewidth=1,
                label=f'GT {props["color"]} {props["shape"]}')

    # Plot slot centroids (reordered)
    assignments = all_slot_assignments[plot_vid]
    reordered = np.zeros_like(all_centroids[plot_vid])
    for fi in range(n_frames):
        perm = assignments[fi]
        for s in range(n_slots):
            reordered[fi, s] = all_centroids[plot_vid][fi, perm[s]]

    for s in range(n_slots):
        # Only plot active slots
        coverage = np.mean([(all_attns[plot_vid][fi].argmax(0) == s).sum() / n_patches
                            for fi in range(0, n_frames, 16)])
        if coverage > 0.03:
            ax.plot(reordered[:, s, 0], reordered[:, s, 1], '--',
                    color=slot_colors[s], alpha=0.7, linewidth=1,
                    label=f'Slot {s}')

    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_title(f'Trajectories: GT vs Slots (video {vid_id})', fontsize=10)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()

    # Panel 3: Tracking error distribution
    ax = axes[0, 2]
    errors_pct = [e * 100 for e in tracking_errors]
    ax.hist(errors_pct, bins=40, color='#2196F3', edgecolor='black',
            linewidth=0.3, alpha=0.8)
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2,
               label='10% threshold')
    ax.axvline(x=mean_tracking_error * 100, color='orange', linestyle='-',
               linewidth=2, label=f'Mean={mean_tracking_error*100:.1f}%')
    ax.set_xlabel('Tracking Error (% of frame)')
    ax.set_ylabel('Count')
    ax.set_title('Tracking Error Distribution', fontsize=10)
    ax.legend(fontsize=8)

    # Panel 4: Slot consistency per video
    ax = axes[1, 0]
    x_vids = np.arange(n_videos)
    bars = ax.bar(x_vids, [c * 100 for c in slot_consistency_per_video],
                  color='#4CAF50', edgecolor='black', linewidth=0.3)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% target')
    ax.axhline(y=mean_consistency, color='orange', linestyle='-',
               label=f'Mean={mean_consistency:.0f}%')
    ax.set_xlabel('Video Index')
    ax.set_ylabel('Slot Consistency (%)')
    ax.set_title('Slot Consistency per Video', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # Panel 5: DINOv2 feature PCA visualization
    ax = axes[1, 1]
    # PCA on DINOv2 features of one frame
    pca_vid = 0
    pca_frame = 64
    pca_idx = pca_vid * n_frames + pca_frame
    feat_pca = all_features[pca_idx].numpy()  # [256, 384]
    # Center and compute PCA
    feat_centered = feat_pca - feat_pca.mean(axis=0)
    U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
    pca_3 = feat_centered @ Vt[:3].T  # [256, 3]
    # Normalize to [0, 1] for RGB display
    pca_3 = pca_3 - pca_3.min(axis=0)
    pca_3 = pca_3 / (pca_3.max(axis=0) + 1e-8)
    pca_img = pca_3.reshape(P, P, 3)
    ax.imshow(pca_img, interpolation='nearest')
    ax.set_title(f'DINOv2 PCA (video {video_ids[pca_vid]}, frame {pca_frame})',
                 fontsize=10)
    ax.axis('off')

    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = time.time() - t0

    # Determine verdict
    err_ok = mean_tracking_error < 0.10
    cons_ok = mean_consistency > 80
    bind_ok = mean_binding > 85
    if err_ok and cons_ok and bind_ok:
        verdict = "SUCCESS"
    elif (err_ok or cons_ok) and bind_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 45: CLEVRER Perception\n\n"
        f"Data: {n_videos} videos × {n_frames} frames\n"
        f"Model: DINOv2-small → SA ({n_slots} slots)\n"
        f"       {total_params:,} trainable params\n\n"
        f"Tracking error:   {mean_tracking_error*100:.1f}% "
        f"({'OK' if err_ok else 'MISS'}, target <10%)\n"
        f"  median:         {median_tracking_error*100:.1f}%\n"
        f"Slot consistency: {mean_consistency:.1f}% "
        f"({'OK' if cons_ok else 'MISS'}, target >80%)\n"
        f"Binding accuracy: {mean_binding:.1f}% "
        f"({'OK' if bind_ok else 'MISS'}, target >85%)\n\n"
        f"Best val loss:    {best_val_loss:.5f}\n"
        f"Total time:       {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 45: DINOv2 + Slot Attention on CLEVRER\n'
                 f'err={mean_tracking_error*100:.1f}% '
                 f'consistency={mean_consistency:.0f}% '
                 f'binding={mean_binding:.0f}% | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase45_clevrer_perception.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase45_clevrer_perception.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Tracking error:   {mean_tracking_error*100:.1f}% "
          f"(target <10%)", flush=True)
    print(f"  Slot consistency: {mean_consistency:.1f}% "
          f"(target >80%)", flush=True)
    print(f"  Binding accuracy: {mean_binding:.1f}% "
          f"(target >85%)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

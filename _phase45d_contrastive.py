def run_phase45d_contrastive_perception():
    """Phase 45d: Contrastive Slot Attention — softer temperature + entropy reg.

    Fix Phase 45c's collapse (1/7 active by epoch 20). Three changes:
    1. Softer contrastive: τ=0.5 (was 0.1), α=0.3 (was 1.0)
    2. Attention entropy regularization (β=0.1) to prevent slot collapse
    3. Same InfoNCE + cross-video batching + SAVi propagation
    """
    import time
    import json
    import os
    import numpy as np
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict
    from pathlib import Path

    print("=" * 70, flush=True)
    print("PHASE 45d: Contrastive Slot Attention on CLEVRER", flush=True)
    print("  Softer τ=0.5, α=0.3, entropy reg β=0.1", flush=True)
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
    video_ids = list(range(10000, 10100))
    n_frames = 128

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations + verify data
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load annotations + verify data", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    available_ids = []
    for vid_id in video_ids:
        ann_path = f"{data_dir}/annotation_{vid_id}.json"
        vid_path = f"{data_dir}/videos/video_{vid_id}.mp4"
        if os.path.exists(ann_path) and os.path.exists(vid_path):
            available_ids.append(vid_id)
    video_ids = available_ids
    n_videos = len(video_ids)
    print(f"│  Found {n_videos} videos with annotations", flush=True)

    annotations = {}
    obj_counts = []
    for vid_id in video_ids:
        with open(f"{data_dir}/annotation_{vid_id}.json") as f:
            annotations[vid_id] = json.load(f)
        obj_counts.append(len(annotations[vid_id]['object_property']))
    print(f"│  Objects per video: {min(obj_counts)}-{max(obj_counts)} "
          f"(mean={np.mean(obj_counts):.1f})", flush=True)

    PROJ_U = np.array([0.0589, 0.2286, 0.4850])
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])

    def project_3d_to_2d(x, y):
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Frame extraction
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Extract frames from {n_videos} videos", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    all_frames = []
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
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
        cap.release()
        all_frames.append(np.stack(frames))
        if (vi + 1) % 20 == 0:
            print(f"│  Extracted {vi+1}/{n_videos} videos", flush=True)

    total_frames = sum(f.shape[0] for f in all_frames)
    print(f"│  Total frames: {total_frames}", flush=True)
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: DINOv2 feature extraction
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: DINOv2 feature extraction", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    cache_path = str(OUTPUT_DIR / f"phase45b_dino_features_{n_videos}v.pt")
    old_cache = str(OUTPUT_DIR / "phase45_dino_features.pt")

    if os.path.exists(cache_path):
        print(f"│  Loading cached features from {cache_path}", flush=True)
        all_features = torch.load(cache_path, weights_only=True)
        print(f"│  Features shape: {all_features.shape}", flush=True)
    else:
        cached_20 = None
        if os.path.exists(old_cache):
            cached_20 = torch.load(old_cache, weights_only=True)
            print(f"│  Loaded Phase 45 cache: {cached_20.shape} "
                  f"(first 20 videos)", flush=True)

        print(f"│  Loading DINOv2-small (vits14)...", flush=True)
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                              pretrained=True)
        dino.eval().to(device)
        for p in dino.parameters():
            p.requires_grad = False
        print(f"│  DINOv2 loaded", flush=True)

        dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        batch_size_dino = 16
        feature_parts = []

        for vi, vid_id in enumerate(video_ids):
            if cached_20 is not None and vi < 20 and vid_id == 10000 + vi:
                start_idx = vi * n_frames
                end_idx = start_idx + n_frames
                feature_parts.append(cached_20[start_idx:end_idx])
                if (vi + 1) % 20 == 0:
                    print(f"│    Reused cache for videos 0-{vi}", flush=True)
                continue

            vid_frames = all_frames[vi]
            vid_features = []
            for start in range(0, len(vid_frames), batch_size_dino):
                end = min(start + batch_size_dino, len(vid_frames))
                batch = torch.tensor(vid_frames[start:end]).to(device)
                batch = (batch - dino_mean) / dino_std
                with torch.no_grad():
                    features = dino.forward_features(batch)
                    patch_tokens = features['x_norm_patchtokens']
                vid_features.append(patch_tokens.cpu())
            feature_parts.append(torch.cat(vid_features, dim=0))

            if (vi + 1) % 20 == 0:
                print(f"│    Extracted {vi+1}/{n_videos} videos", flush=True)
                if device.type == 'mps':
                    torch.mps.empty_cache()

        all_features = torch.cat(feature_parts, dim=0)
        print(f"│  Features shape: {all_features.shape}", flush=True)
        print(f"│  Memory: {all_features.numel() * 4 / 1e9:.2f} GB", flush=True)

        torch.save(all_features, cache_path)
        print(f"│  Cached to {cache_path}", flush=True)

        del dino, dino_mean, dino_std, cached_20
        if device.type == 'mps':
            torch.mps.empty_cache()

    del all_frames
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Train Contrastive Slot Attention
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Train Contrastive Slot Attention", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    dino_dim = 384
    n_patches = 256
    P = 16
    n_slots = 7
    slot_dim = 64
    n_sa_iters = 5
    sa_epochs = 100
    sa_batch = 32
    sa_lr = 1e-4
    alpha_contrastive = 0.3
    alpha_warmup_epochs = 10
    temperature = 0.5
    beta_entropy = 0.1
    pair_stride = 4

    # ── InfoNCE contrastive loss ──
    def info_nce_slot_loss(slots_t, slots_tp1, tau=temperature):
        """InfoNCE: slot i at t ↔ slot i at t+1 (positive), all others negative."""
        B, S, D = slots_t.shape
        s_t = F.normalize(slots_t.reshape(B * S, D), dim=-1)      # [B*S, D]
        s_tp1 = F.normalize(slots_tp1.reshape(B * S, D), dim=-1)  # [B*S, D]
        logits = torch.mm(s_t, s_tp1.T) / tau                     # [B*S, B*S]
        labels = torch.arange(B * S, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            pos_sim = (s_t * s_tp1).sum(dim=-1).mean()
        return loss, pos_sim

    # ── Slot Attention Module ──
    class SlotAttention(nn.Module):
        def __init__(self, n_slots, slot_dim, n_iters, feature_dim, epsilon=1e-8):
            super().__init__()
            self.n_slots = n_slots
            self.slot_dim = slot_dim
            self.n_iters = n_iters
            self.epsilon = epsilon
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

        def forward(self, inputs, init_slots=None):
            """Run slot attention. If init_slots provided, use as initial state
            instead of learnable init (SAVi-style slot propagation)."""
            B, N, _ = inputs.shape
            inputs = self.norm_inputs(inputs)
            k = self.project_k(inputs)
            v = self.project_v(inputs)
            if init_slots is not None:
                slots = init_slots
            else:
                slots = self.slot_init.expand(B, -1, -1)
            scale = self.slot_dim ** 0.5
            attn_weights = None
            for _ in range(self.n_iters):
                slots_prev = slots
                slots = self.norm_slots(slots)
                q = self.project_q(slots)
                attn_logits = torch.bmm(k, q.transpose(1, 2)) / scale
                attn = F.softmax(attn_logits, dim=-1) + self.epsilon
                attn = attn / attn.sum(dim=1, keepdim=True)
                updates = torch.bmm(attn.transpose(1, 2), v)
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                ).reshape(B, self.n_slots, self.slot_dim)
                slots = slots + self.mlp(self.norm_mlp(slots))
                attn_weights = attn.transpose(1, 2)
            return slots, attn_weights

    class EncoderMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.norm = nn.LayerNorm(dim)
        def forward(self, x):
            return self.norm(self.mlp(x))

    class SpatialBroadcastDecoder(nn.Module):
        def __init__(self, slot_dim, output_dim, n_patches_side=16):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(slot_dim + 2, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, output_dim + 1))
            xs = torch.linspace(-1, 1, n_patches_side)
            ys = torch.linspace(-1, 1, n_patches_side)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            self.register_buffer(
                'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))
        def forward(self, slots):
            B, K, D = slots.shape
            N = self.grid.shape[0]
            slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
            grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
            dec_in = torch.cat([slots_bc, grid], dim=-1)
            decoded = self.decoder(dec_in)
            features = decoded[:, :, :, :-1]
            alpha_logits = decoded[:, :, :, -1:]
            alpha = F.softmax(alpha_logits, dim=1)
            recon = (alpha * features).sum(dim=1)
            alpha = alpha.squeeze(-1)
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
    print(f"│  Contrastive: α={alpha_contrastive} (warmup {alpha_warmup_epochs}ep), "
          f"τ={temperature}, β_ent={beta_entropy}, stride={pair_stride}", flush=True)

    all_params = (list(encoder.parameters()) +
                  list(slot_attn.parameters()) +
                  list(decoder.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=sa_lr)

    # Cosine LR schedule with warmup
    lr_warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < lr_warmup_epochs:
            return (epoch + 1) / lr_warmup_epochs
        progress = (epoch - lr_warmup_epochs) / (sa_epochs - lr_warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Build frame pair indices with stride, grouped by video for cross-video batching
    n_train_vids = int(0.8 * n_videos)

    pairs_by_video = {}
    for vi in range(n_train_vids):
        vid_start = vi * n_frames
        pairs_by_video[vi] = [(vid_start + fi, vid_start + fi + pair_stride)
                              for fi in range(0, n_frames - pair_stride, pair_stride)]

    val_pairs = []
    for vi in range(n_train_vids, n_videos):
        vid_start = vi * n_frames
        for fi in range(0, n_frames - pair_stride, pair_stride):
            val_pairs.append((vid_start + fi, vid_start + fi + pair_stride))
    val_pairs = np.array(val_pairs)

    total_train_pairs = sum(len(ps) for ps in pairs_by_video.values())

    def make_epoch_batches(pairs_by_video, batch_size):
        """Round-robin across shuffled videos for cross-video batch diversity."""
        vid_ids = list(pairs_by_video.keys())
        np.random.shuffle(vid_ids)
        # Shuffle pair order within each video
        vid_pair_idx = {v: list(np.random.permutation(len(ps)))
                        for v, ps in pairs_by_video.items()}
        vid_cursors = {v: 0 for v in vid_ids}
        all_pairs = []
        active = list(vid_ids)
        while active:
            for v in list(active):
                if vid_cursors[v] < len(vid_pair_idx[v]):
                    idx = vid_pair_idx[v][vid_cursors[v]]
                    all_pairs.append(pairs_by_video[v][idx])
                    vid_cursors[v] += 1
                else:
                    active.remove(v)
        # Split into batches
        batches = []
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            if len(batch) >= batch_size // 2:  # drop tiny last batch
                batches.append(np.array(batch))
        return batches

    print(f"│  Train pairs: {total_train_pairs}, Val pairs: {len(val_pairs)}", flush=True)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, sa_epochs + 1):
        encoder.train()
        slot_attn.train()
        decoder.train()

        batches = make_epoch_batches(pairs_by_video, sa_batch)
        ep_recon_loss = 0.0
        ep_ctr_loss = 0.0
        ep_ent_loss = 0.0
        ep_pos_sim = 0.0
        ep_total_loss = 0.0
        n_batches = 0

        for pairs in batches:
            idx_t = pairs[:, 0]
            idx_tp1 = pairs[:, 1]

            feat_t = all_features[idx_t].to(device)
            feat_tp1 = all_features[idx_tp1].to(device)

            # Forward frame t (learnable init)
            enc_t = encoder(feat_t)
            slots_t, attn_t = slot_attn(enc_t)
            recon_t, alpha_t = decoder(slots_t)

            # Forward frame t+1 (SAVi-style: init from frame t's slots)
            enc_tp1 = encoder(feat_tp1)
            slots_tp1, attn_tp1 = slot_attn(enc_tp1, init_slots=slots_t.detach())
            recon_tp1, alpha_tp1 = decoder(slots_tp1)

            # Reconstruction loss (both frames)
            recon_loss = (F.mse_loss(recon_t, feat_t) +
                          F.mse_loss(recon_tp1, feat_tp1)) / 2

            # InfoNCE contrastive loss
            ctr_loss, pos_sim = info_nce_slot_loss(slots_t, slots_tp1)

            # Attention entropy regularization — prevent slot collapse
            # Maximize entropy of attention weights over patches per slot
            H_t = -(attn_t * (attn_t + 1e-8).log()).sum(dim=-1)    # [B, n_slots]
            H_tp1 = -(attn_tp1 * (attn_tp1 + 1e-8).log()).sum(dim=-1)
            ent_loss = -(H_t.mean() + H_tp1.mean()) / 2  # negative = maximize entropy

            # Alpha warmup: 0→alpha_contrastive over first alpha_warmup_epochs
            if epoch <= alpha_warmup_epochs:
                alpha_eff = alpha_contrastive * (epoch / alpha_warmup_epochs)
            else:
                alpha_eff = alpha_contrastive

            total_loss = recon_loss + alpha_eff * ctr_loss + beta_entropy * ent_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            ep_recon_loss += recon_loss.item()
            ep_ctr_loss += ctr_loss.item()
            ep_ent_loss += ent_loss.item()
            ep_pos_sim += pos_sim.item()
            ep_total_loss += total_loss.item()
            n_batches += 1

            if device.type == 'mps' and n_batches % 100 == 0:
                torch.mps.empty_cache()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            encoder.eval()
            slot_attn.eval()
            decoder.eval()

            with torch.no_grad():
                val_sub = val_pairs[:256]
                vf_t = all_features[val_sub[:, 0]].to(device)
                vf_tp1 = all_features[val_sub[:, 1]].to(device)

                ve_t = encoder(vf_t)
                vs_t, va_t = slot_attn(ve_t)
                vr_t, valpha_t = decoder(vs_t)

                # SAVi-style: propagate slots from t to t+1
                ve_tp1 = encoder(vf_tp1)
                vs_tp1, va_tp1 = slot_attn(ve_tp1, init_slots=vs_t)
                vr_tp1, valpha_tp1 = decoder(vs_tp1)

                val_recon = (F.mse_loss(vr_t, vf_t) +
                             F.mse_loss(vr_tp1, vf_tp1)).item() / 2

                val_ctr, val_pos_sim = info_nce_slot_loss(vs_t, vs_tp1)
                val_ctr = val_ctr.item()

                # Entropy diagnostic (on frame t only)
                ownership = valpha_t.argmax(dim=1)
                B_d = ownership.shape[0]
                slot_counts = torch.zeros(B_d, n_slots, device=device)
                for s in range(n_slots):
                    slot_counts[:, s] = (ownership == s).float().sum(dim=1)
                mean_fracs = (slot_counts / n_patches).mean(dim=0)
                active = int((mean_fracs > 0.01).sum().item())
                ent = -(valpha_t * (valpha_t + 1e-8).log()).sum(dim=1).mean()
                norm_ent = ent.item() / np.log(n_slots)

            val_total = val_recon + alpha_contrastive * val_ctr
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_state = {
                    'encoder': {k: v.cpu().clone()
                                for k, v in encoder.state_dict().items()},
                    'slot_attn': {k: v.cpu().clone()
                                  for k, v in slot_attn.state_dict().items()},
                    'decoder': {k: v.cpu().clone()
                                for k, v in decoder.state_dict().items()},
                }

            elapsed = time.time() - t0
            epoch_time = elapsed / epoch
            eta_s = (sa_epochs - epoch) * epoch_time
            eta_m = eta_s / 60

            print(f"│  Epoch {epoch:3d}/{sa_epochs}: "
                  f"recon={ep_recon_loss/n_batches:.5f} "
                  f"ctr={ep_ctr_loss/n_batches:.4f} α={alpha_eff:.3f} "
                  f"ent_l={ep_ent_loss/n_batches:.3f} "
                  f"sim={ep_pos_sim/n_batches:.3f} "
                  f"val_r={val_recon:.5f} val_c={val_ctr:.4f} "
                  f"active={active}/{n_slots} ent={norm_ent:.3f} "
                  f"eta={eta_m:.0f}m [{elapsed:.0f}s]", flush=True)

    # Restore best
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

    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2)

    all_centroids = []
    all_attns = []

    for vi in range(n_videos):
        vid_start = vi * n_frames
        vid_centroids = []
        vid_attns = []
        prev_slots = None

        for fi in range(n_frames):
            feat = all_features[vid_start + fi:vid_start + fi + 1].to(device)
            with torch.no_grad():
                enc_feat = encoder(feat)
                slots, attn = slot_attn(enc_feat, init_slots=prev_slots)
                prev_slots = slots

            attn_np = attn[0].cpu().numpy()
            frame_centroids = np.zeros((n_slots, 2))
            for si in range(n_slots):
                weights = attn_np[si, :]
                w_sum = weights.sum()
                if w_sum > 1e-6:
                    cx = (weights * grid_positions[:, 0].numpy()).sum() / w_sum
                    cy = (weights * grid_positions[:, 1].numpy()).sum() / w_sum
                    frame_centroids[si] = [cx, cy]
                else:
                    frame_centroids[si] = [0.5, 0.5]
            vid_centroids.append(frame_centroids)
            vid_attns.append(attn_np)

        all_centroids.append(np.stack(vid_centroids))
        all_attns.append(np.stack(vid_attns))

        if (vi + 1) % 20 == 0:
            print(f"│  Processed {vi+1}/{n_videos} videos", flush=True)

    all_slot_assignments = []
    for vi in range(n_videos):
        assignments = [np.arange(n_slots)] * n_frames
        all_slot_assignments.append(assignments)

    print(f"│  Slot propagation done (no Hungarian matching needed)", flush=True)
    print(f"└─ Stage 4 done [{time.time()-ts4:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 5: Evaluation against GT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 5: Evaluation against GT", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts5 = time.time()

    tracking_errors = []
    slot_consistency_per_video = []
    binding_accuracy_per_video = []

    for vi in range(n_videos):
        vid_id = video_ids[vi]
        ann = annotations[vid_id]
        obj_props = {p['object_id']: p for p in ann['object_property']}
        traj = ann['motion_trajectory']

        centroids = all_centroids[vi]
        assignments = all_slot_assignments[vi]

        reordered_centroids = np.zeros_like(centroids)
        for fi in range(n_frames):
            perm = assignments[fi]
            for s in range(n_slots):
                reordered_centroids[fi, s] = centroids[fi, perm[s]]

        best_frame = 0
        best_in_view = 0
        for fi in range(n_frames):
            n_in = sum(1 for o in traj[fi]['objects']
                       if o['inside_camera_view'])
            if n_in > best_in_view:
                best_in_view = n_in
                best_frame = fi

        ref_objs = [o for o in traj[best_frame]['objects']
                     if o['inside_camera_view']]
        gt_positions_ref = np.zeros((len(ref_objs), 2))
        gt_obj_ids = []
        for oi, o in enumerate(ref_objs):
            u, v = project_3d_to_2d(o['location'][0], o['location'][1])
            gt_positions_ref[oi] = [u, v]
            gt_obj_ids.append(o['object_id'])

        slot_coverage = np.zeros(n_slots)
        attn_ref = all_attns[vi][best_frame]
        ownership_ref = attn_ref.argmax(axis=0)
        for s in range(n_slots):
            slot_coverage[s] = (ownership_ref == s).sum() / n_patches
        active_slots = [s for s in range(n_slots) if slot_coverage[s] > 0.02]

        pred_centroids_ref = reordered_centroids[best_frame]
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
            cost_mat = np.zeros((len(ref_objs), n_slots))
            for gi in range(len(ref_objs)):
                for si in range(n_slots):
                    cost_mat[gi, si] = np.linalg.norm(
                        gt_positions_ref[gi] - pred_centroids_ref[si])
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            gt_to_slot = {}
            for r, c in zip(row_ind, col_ind):
                gt_to_slot[gt_obj_ids[r]] = c

        vid_errors = []
        consistent_frames = 0
        bound_frames = 0
        total_eval_frames = 0

        for fi in range(n_frames):
            frame_objs = [o for o in traj[fi]['objects']
                          if o['inside_camera_view']
                          and o['object_id'] in gt_to_slot]
            if len(frame_objs) == 0:
                continue
            total_eval_frames += 1
            used_slots = set()
            all_match = True

            for o in frame_objs:
                oid = o['object_id']
                assigned_slot = gt_to_slot[oid]
                gt_u, gt_v = project_3d_to_2d(
                    o['location'][0], o['location'][1])
                pred_uv = reordered_centroids[fi, assigned_slot]
                err = np.linalg.norm(np.array([gt_u, gt_v]) - pred_uv)
                vid_errors.append(err)
                tracking_errors.append(err)

                if assigned_slot in used_slots:
                    all_match = False
                used_slots.add(assigned_slot)

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

            if all_match:
                consistent_frames += 1
            bound_frames += 1 if len(used_slots) == len(frame_objs) else 0

        consistency = consistent_frames / max(total_eval_frames, 1)
        binding = bound_frames / max(total_eval_frames, 1)

        slot_consistency_per_video.append(consistency)
        binding_accuracy_per_video.append(binding)

        if (vi + 1) % 20 == 0:
            print(f"│  Videos 1-{vi+1}: err={np.mean(tracking_errors)*100:.1f}%, "
                  f"consistency={np.mean(slot_consistency_per_video)*100:.0f}%, "
                  f"binding={np.mean(binding_accuracy_per_video)*100:.0f}%", flush=True)

    mean_tracking_error = np.mean(tracking_errors)
    median_tracking_error = np.median(tracking_errors)
    mean_consistency = np.mean(slot_consistency_per_video) * 100
    mean_binding = np.mean(binding_accuracy_per_video) * 100

    p45_err = 28.8
    p45_cons = 3.4
    p45_bind = 100.0
    p45b_err = 31.1
    p45b_cons = 0.9

    print(f"\n│  === RESULTS ===", flush=True)
    print(f"│  Tracking error (mean):   {mean_tracking_error*100:.1f}% "
          f"(Phase 45: {p45_err:.1f}%, 45b: {p45b_err:.1f}%)", flush=True)
    print(f"│  Tracking error (median): {median_tracking_error*100:.1f}%", flush=True)
    print(f"│  Slot consistency:        {mean_consistency:.1f}% "
          f"(Phase 45: {p45_cons:.1f}%, 45b: {p45b_cons:.1f}%)", flush=True)
    print(f"│  Binding accuracy:        {mean_binding:.1f}% "
          f"(Phase 45: {p45_bind:.1f}%)", flush=True)
    print(f"│  Targets: err<20%, consistency>30%, binding>85%", flush=True)
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

    # Panel 1: Sample frames with slot masks
    ax = axes[0, 0]
    sample_vids = [0, min(5, n_videos-1), min(10, n_videos-1)]
    sample_frame = 64
    n_show = min(3, len(sample_vids))
    composite_img = np.zeros((P * n_show, P * 2, 3))
    for si, vi in enumerate(sample_vids[:n_show]):
        vid_id = video_ids[vi]
        cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (P, P))
            composite_img[si*P:(si+1)*P, :P, :] = frame_small / 255.0
        attn_frame = all_attns[vi][sample_frame]
        masks = attn_frame.reshape(n_slots, P, P)
        mask_composite = np.zeros((P, P, 3))
        for s in range(n_slots):
            color = plt.cm.Set1(s / n_slots)[:3]
            for c in range(3):
                mask_composite[:, :, c] += masks[s] * color[c]
        composite_img[si*P:(si+1)*P, P:2*P, :] = np.clip(mask_composite, 0, 1)
    ax.imshow(composite_img, interpolation='nearest')
    ax.set_title(f'Slot Masks (frame {sample_frame})', fontsize=10)
    ax.set_xticks([P//2, P + P//2])
    ax.set_xticklabels(['Original', 'Slot Masks'])
    ax.set_yticks([])

    # Panel 2: Trajectories for one video
    ax = axes[0, 1]
    plot_vid = 0
    vid_id = video_ids[plot_vid]
    ann = annotations[vid_id]
    traj = ann['motion_trajectory']
    obj_props = {p['object_id']: p for p in ann['object_property']}
    obj_ids = [p['object_id'] for p in ann['object_property']]
    gt_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    slot_colors_plot = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                        '#911eb4', '#42d4f4', '#f032e6']

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
        props = obj_props[oid]
        ax.plot(gt_us, gt_vs, '-', color=gt_colors[oi % len(gt_colors)],
                alpha=0.5, linewidth=1,
                label=f'GT {props["color"]} {props["shape"]}')

    assignments = all_slot_assignments[plot_vid]
    reordered = np.zeros_like(all_centroids[plot_vid])
    for fi in range(n_frames):
        perm = assignments[fi]
        for s in range(n_slots):
            reordered[fi, s] = all_centroids[plot_vid][fi, perm[s]]
    for s in range(n_slots):
        coverage = np.mean([(all_attns[plot_vid][fi].argmax(0) == s).sum() / n_patches
                            for fi in range(0, n_frames, 16)])
        if coverage > 0.03:
            ax.plot(reordered[:, s, 0], reordered[:, s, 1], '--',
                    color=slot_colors_plot[s], alpha=0.7, linewidth=1,
                    label=f'Slot {s}')
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_title(f'Trajectories: GT vs Slots (video {vid_id})', fontsize=10)
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()

    # Panel 3: Error distribution
    ax = axes[0, 2]
    errors_pct = [e * 100 for e in tracking_errors]
    ax.hist(errors_pct, bins=40, color='#2196F3', edgecolor='black',
            linewidth=0.3, alpha=0.8)
    ax.axvline(x=20, color='red', linestyle='--', linewidth=2,
               label='20% threshold')
    ax.axvline(x=mean_tracking_error * 100, color='orange', linestyle='-',
               linewidth=2, label=f'Mean={mean_tracking_error*100:.1f}%')
    ax.set_xlabel('Tracking Error (% of frame)')
    ax.set_ylabel('Count')
    ax.set_title('Tracking Error Distribution', fontsize=10)
    ax.legend(fontsize=8)

    # Panel 4: Slot consistency per video
    ax = axes[1, 0]
    x_vids = np.arange(n_videos)
    ax.bar(x_vids, [c * 100 for c in slot_consistency_per_video],
           color='#4CAF50', edgecolor='black', linewidth=0.3)
    ax.axhline(y=30, color='red', linestyle='--', alpha=0.7,
               label='30% target')
    ax.axhline(y=mean_consistency, color='orange', linestyle='-',
               label=f'Mean={mean_consistency:.0f}%')
    ax.set_xlabel('Video Index')
    ax.set_ylabel('Slot Consistency (%)')
    ax.set_title('Slot Consistency per Video', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # Panel 5: DINOv2 PCA
    ax = axes[1, 1]
    pca_idx = 0 * n_frames + 64
    feat_pca = all_features[pca_idx].numpy()
    feat_centered = feat_pca - feat_pca.mean(axis=0)
    U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
    pca_3 = feat_centered @ Vt[:3].T
    pca_3 = pca_3 - pca_3.min(axis=0)
    pca_3 = pca_3 / (pca_3.max(axis=0) + 1e-8)
    ax.imshow(pca_3.reshape(P, P, 3), interpolation='nearest')
    ax.set_title(f'DINOv2 PCA (video {video_ids[0]}, frame 64)', fontsize=10)
    ax.axis('off')

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = time.time() - t0

    err_ok = mean_tracking_error * 100 < 20
    cons_ok = mean_consistency > 30
    bind_ok = mean_binding > 85
    if err_ok and cons_ok and bind_ok:
        verdict = "SUCCESS"
    elif (err_ok or cons_ok) and bind_ok:
        verdict = "PARTIAL"
    elif mean_tracking_error * 100 < p45_err and mean_consistency > p45_cons:
        verdict = "PARTIAL (improved over 45)"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 45d: Contrastive Slot Attention\n\n"
        f"Data: {n_videos} videos × {n_frames} frames\n"
        f"Model: DINOv2 → SA (7 slots, InfoNCE)\n"
        f"       α={alpha_contrastive}, τ={temperature}, β={beta_entropy}\n\n"
        f"                     45d     45    45b\n"
        f"Tracking error:  {mean_tracking_error*100:5.1f}%  {p45_err:5.1f}% {p45b_err:5.1f}%\n"
        f"Slot consistency:{mean_consistency:5.1f}%  {p45_cons:5.1f}%  {p45b_cons:5.1f}%\n"
        f"Binding accuracy:{mean_binding:5.1f}% {p45_bind:5.1f}%    —\n\n"
        f"Best val loss: {best_val_loss:.5f}\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 45d: Contrastive Slot Attention on CLEVRER\n'
                 f'err={mean_tracking_error*100:.1f}% '
                 f'consistency={mean_consistency:.0f}% '
                 f'binding={mean_binding:.0f}% | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase45d_contrastive_perception.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase45d_contrastive_perception.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Tracking error:   {mean_tracking_error*100:.1f}% "
          f"(target <20%, Phase 45: {p45_err}%, 45b: {p45b_err}%)", flush=True)
    print(f"  Slot consistency: {mean_consistency:.1f}% "
          f"(target >30%, Phase 45: {p45_cons}%, 45b: {p45b_cons}%)", flush=True)
    print(f"  Binding accuracy: {mean_binding:.1f}% "
          f"(target >85%, Phase 45: {p45_bind}%)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

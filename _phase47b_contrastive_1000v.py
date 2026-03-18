def run_phase47b_contrastive_1000v():
    """Phase 47b: Temperature-annealed contrastive SA on 1000 CLEVRER videos.

    Fix for 47's collapse: soft τ early, sharpen late.
    - τ: 1.0 → 0.3 cosine anneal over 100 epochs, then fixed 0.3
    - α=0.5 (warmup 20ep), β_ent=0.5 (5× stronger entropy reg)
    - 1000 videos, 200 epochs, reuse phase47 feature cache
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
    from pathlib import Path

    print("=" * 70, flush=True)
    print("PHASE 47b: Temperature-Annealed Contrastive SA — 1000 Videos", flush=True)
    print("  τ: 1.0→0.3 cosine, α=0.5, β_ent=0.5, 200 epochs", flush=True)
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

    # ── Config ──
    n_extract = 16         # frames per video for training
    n_eval_frames = 128    # frames per video for evaluation
    n_eval_videos = 20     # test videos for evaluation

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations + verify data
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load annotations + verify data", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    video_ids_all = list(range(10000, 11000))
    available_ids = []
    for vid_id in video_ids_all:
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
    # STAGE 1: DINOv2 feature extraction (reuse phase47 cache)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: DINOv2 features ({n_videos} videos × {n_extract} frames)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    cache_path = str(OUTPUT_DIR / "phase47_dino_features_1000v.pt")
    frame_indices = np.linspace(0, 127, n_extract).astype(int)
    print(f"│  Frame indices: {frame_indices.tolist()}", flush=True)

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
        print(f"│  DINOv2 loaded", flush=True)

        dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        feature_parts = []
        for vi, vid_id in enumerate(video_ids):
            cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
            frames = []
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
            cap.release()

            batch = torch.tensor(np.stack(frames)).to(device)
            batch = (batch - dino_mean) / dino_std
            with torch.no_grad():
                features = dino.forward_features(batch)
                patch_tokens = features['x_norm_patchtokens']
            feature_parts.append(patch_tokens.cpu())

            if (vi + 1) % 100 == 0:
                elapsed_ext = time.time() - ts1
                eta_ext = elapsed_ext / (vi + 1) * (n_videos - vi - 1)
                print(f"│    Extracted {vi+1}/{n_videos} videos "
                      f"[{elapsed_ext:.0f}s, eta {eta_ext:.0f}s]", flush=True)
                if device.type == 'mps':
                    torch.mps.empty_cache()

        all_features = torch.cat(feature_parts, dim=0)
        print(f"│  Features shape: {all_features.shape}", flush=True)
        print(f"│  Memory: {all_features.numel() * 4 / 1e9:.2f} GB", flush=True)

        torch.save(all_features, cache_path)
        print(f"│  Cached to {cache_path}", flush=True)

        del dino, dino_mean, dino_std, feature_parts
        if device.type == 'mps':
            torch.mps.empty_cache()

    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Train Contrastive Slot Attention (τ-annealed)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Train Contrastive Slot Attention (τ-annealed)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    dino_dim = 384
    n_patches = 256
    P = 16
    n_slots = 7
    slot_dim = 64
    n_sa_iters = 5
    sa_epochs = 200
    sa_batch = 32
    sa_lr = 1e-4
    alpha_contrastive = 0.5      # 47b: halved from 1.0
    alpha_warmup_epochs = 20     # 47b: doubled from 10
    tau_start = 1.0              # 47b: soft start
    tau_end = 0.3                # 47b: anneal target
    tau_anneal_epochs = 100      # 47b: cosine anneal over 100 epochs
    beta_entropy = 0.5           # 47b: 5× stronger than 47
    pair_stride = 4

    def get_temperature(epoch):
        """τ: 1.0 → 0.3 cosine anneal over first 100 epochs, then fixed 0.3."""
        if epoch >= tau_anneal_epochs:
            return tau_end
        progress = epoch / tau_anneal_epochs
        return tau_end + (tau_start - tau_end) * 0.5 * (1.0 + np.cos(np.pi * progress))

    # ── InfoNCE contrastive loss ──
    def info_nce_slot_loss(slots_t, slots_tp1, tau):
        B, S, D = slots_t.shape
        s_t = F.normalize(slots_t.reshape(B * S, D), dim=-1)
        s_tp1 = F.normalize(slots_tp1.reshape(B * S, D), dim=-1)
        logits = torch.mm(s_t, s_tp1.T) / tau
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
          f"τ={tau_start}→{tau_end} (anneal {tau_anneal_epochs}ep), "
          f"β_ent={beta_entropy}", flush=True)

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

    # Build frame pair indices: 3 pairs per video from 16-frame indices
    n_train_vids = 800

    pairs_by_video = {}
    for vi in range(n_train_vids):
        vid_start = vi * n_extract
        pairs_by_video[vi] = [(vid_start + fi, vid_start + fi + pair_stride)
                              for fi in range(0, n_extract - pair_stride, pair_stride)]

    val_pairs = []
    for vi in range(n_train_vids, n_videos):
        vid_start = vi * n_extract
        for fi in range(0, n_extract - pair_stride, pair_stride):
            val_pairs.append((vid_start + fi, vid_start + fi + pair_stride))
    val_pairs = np.array(val_pairs)

    total_train_pairs = sum(len(ps) for ps in pairs_by_video.values())

    def make_epoch_batches(pairs_by_video, batch_size):
        """Round-robin across shuffled videos for cross-video batch diversity."""
        vid_ids = list(pairs_by_video.keys())
        np.random.shuffle(vid_ids)
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
        batches = []
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            if len(batch) >= batch_size // 2:
                batches.append(np.array(batch))
        return batches

    print(f"│  Train: {n_train_vids} videos, {total_train_pairs} pairs", flush=True)
    print(f"│  Val: {n_videos - n_train_vids} videos, {len(val_pairs)} pairs", flush=True)

    best_val_loss = float('inf')
    best_state = None
    epoch_times = []  # for rolling ETA

    for epoch in range(1, sa_epochs + 1):
        epoch_t0 = time.time()
        encoder.train()
        slot_attn.train()
        decoder.train()

        # Temperature schedule
        tau = get_temperature(epoch)

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

            # InfoNCE contrastive loss (with current τ)
            ctr_loss, pos_sim = info_nce_slot_loss(slots_t, slots_tp1, tau)

            # Attention entropy regularization
            H_t = -(attn_t * (attn_t + 1e-8).log()).sum(dim=-1)
            H_tp1 = -(attn_tp1 * (attn_tp1 + 1e-8).log()).sum(dim=-1)
            ent_loss = -(H_t.mean() + H_tp1.mean()) / 2

            # Alpha warmup
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

        if epoch % 10 != 0 and epoch != 1:
            epoch_times.append(time.time() - epoch_t0)

        if epoch % 10 == 0 or epoch == 1:
            encoder.eval()
            slot_attn.eval()
            decoder.eval()

            with torch.no_grad():
                val_sub = val_pairs[:min(256, len(val_pairs))]
                vf_t = all_features[val_sub[:, 0]].to(device)
                vf_tp1 = all_features[val_sub[:, 1]].to(device)

                ve_t = encoder(vf_t)
                vs_t, va_t = slot_attn(ve_t)
                vr_t, valpha_t = decoder(vs_t)

                ve_tp1 = encoder(vf_tp1)
                vs_tp1, va_tp1 = slot_attn(ve_tp1, init_slots=vs_t)
                vr_tp1, valpha_tp1 = decoder(vs_tp1)

                val_recon = (F.mse_loss(vr_t, vf_t) +
                             F.mse_loss(vr_tp1, vf_tp1)).item() / 2

                val_ctr, val_pos_sim = info_nce_slot_loss(vs_t, vs_tp1, tau)
                val_ctr = val_ctr.item()

                # Entropy diagnostic
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
            epoch_times.append(time.time() - epoch_t0)
            recent = epoch_times[-10:]
            avg_epoch_time = np.mean(recent)
            eta_s = (sa_epochs - epoch) * avg_epoch_time
            eta_m = eta_s / 60

            print(f"│  Epoch {epoch:3d}/{sa_epochs}: "
                  f"recon={ep_recon_loss/n_batches:.5f} "
                  f"ctr={ep_ctr_loss/n_batches:.4f} α={alpha_eff:.3f} "
                  f"τ={tau:.3f} "
                  f"ent_l={ep_ent_loss/n_batches:.3f} "
                  f"sim={ep_pos_sim/n_batches:.3f} "
                  f"val_r={val_recon:.5f} val_c={val_ctr:.4f} "
                  f"active={active}/{n_slots} ent={norm_ent:.3f} "
                  f"eta={eta_m:.0f}m [{elapsed:.0f}s]", flush=True)

            # Collapse warning
            if active < 4 and epoch <= 50:
                print(f"│  WARNING: Only {active}/{n_slots} slots active "
                      f"(soft tau may recover)", flush=True)

    # Restore best model
    encoder.load_state_dict(best_state['encoder'])
    slot_attn.load_state_dict(best_state['slot_attn'])
    decoder.load_state_dict(best_state['decoder'])
    encoder.to(device).eval()
    slot_attn.to(device).eval()
    decoder.to(device).eval()

    # Save model checkpoint
    torch.save({
        'encoder': best_state['encoder'],
        'slot_attn': best_state['slot_attn'],
        'decoder': best_state['decoder'],
    }, str(OUTPUT_DIR / "phase47b_model.pt"))

    print(f"│  Best val loss: {best_val_loss:.5f}", flush=True)
    print(f"│  Saved model to results/phase47b_model.pt", flush=True)
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Load eval features (20 videos × 128 frames)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Load evaluation features", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    del all_features
    if device.type == 'mps':
        torch.mps.empty_cache()

    eval_video_ids = list(range(10000, 10000 + n_eval_videos))
    eval_cache = str(OUTPUT_DIR / "phase45_dino_features.pt")

    if os.path.exists(eval_cache):
        print(f"│  Loading eval cache: {eval_cache}", flush=True)
        eval_features = torch.load(eval_cache, weights_only=True)
        print(f"│  Eval features shape: {eval_features.shape}", flush=True)
    else:
        print(f"│  Extracting eval features ({n_eval_videos} videos × "
              f"{n_eval_frames} frames)...", flush=True)
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                              pretrained=True)
        dino.eval().to(device)
        for p in dino.parameters():
            p.requires_grad = False

        dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        feature_parts = []
        batch_size_dino = 16
        for vi, vid_id in enumerate(eval_video_ids):
            cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
            frames = []
            for fi in range(n_eval_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
            cap.release()

            vid_features = []
            vid_frames_np = np.stack(frames)
            for start in range(0, len(frames), batch_size_dino):
                end = min(start + batch_size_dino, len(frames))
                batch = torch.tensor(vid_frames_np[start:end]).to(device)
                batch = (batch - dino_mean) / dino_std
                with torch.no_grad():
                    features = dino.forward_features(batch)
                    patch_tokens = features['x_norm_patchtokens']
                vid_features.append(patch_tokens.cpu())
            feature_parts.append(torch.cat(vid_features, dim=0))
            print(f"│    Eval video {vi+1}/{n_eval_videos}", flush=True)

        eval_features = torch.cat(feature_parts, dim=0)
        torch.save(eval_features, eval_cache)
        del dino, dino_mean, dino_std
        if device.type == 'mps':
            torch.mps.empty_cache()

    print(f"└─ Stage 3 done [{time.time()-ts3:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 4: Object tracking via SAVi propagation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Object tracking ({n_eval_videos} videos × "
          f"{n_eval_frames} frames)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts4 = time.time()

    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2)

    all_centroids = []
    all_attns = []

    for vi in range(n_eval_videos):
        vid_start = vi * n_eval_frames
        vid_centroids = []
        vid_attns = []
        prev_slots = None

        for fi in range(n_eval_frames):
            feat = eval_features[vid_start + fi:vid_start + fi + 1].to(device)
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

        if (vi + 1) % 5 == 0:
            print(f"│  Processed {vi+1}/{n_eval_videos} eval videos", flush=True)

    all_slot_assignments = []
    for vi in range(n_eval_videos):
        assignments = [np.arange(n_slots)] * n_eval_frames
        all_slot_assignments.append(assignments)

    print(f"└─ Stage 4 done [{time.time()-ts4:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 5: Evaluation against GT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 5: Evaluation against GT", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts5 = time.time()

    tracking_errors = []
    per_frame_errors = [[] for _ in range(n_eval_frames)]
    slot_consistency_per_video = []
    binding_accuracy_per_video = []

    for vi in range(n_eval_videos):
        vid_id = eval_video_ids[vi]
        ann = annotations[vid_id]
        traj = ann['motion_trajectory']

        centroids = all_centroids[vi]
        assignments = all_slot_assignments[vi]

        reordered_centroids = np.zeros_like(centroids)
        for fi in range(n_eval_frames):
            perm = assignments[fi]
            for s in range(n_slots):
                reordered_centroids[fi, s] = centroids[fi, perm[s]]

        best_frame = 0
        best_in_view = 0
        for fi in range(n_eval_frames):
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

        for fi in range(n_eval_frames):
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
                per_frame_errors[fi].append(err)

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

    mean_tracking_error = np.mean(tracking_errors)
    median_tracking_error = np.median(tracking_errors)
    mean_consistency = np.mean(slot_consistency_per_video) * 100
    mean_binding = np.mean(binding_accuracy_per_video) * 100

    # Temporal degradation
    early_errors = []
    late_errors = []
    for fi in range(n_eval_frames):
        if fi < 16:
            early_errors.extend(per_frame_errors[fi])
        elif fi >= 64:
            late_errors.extend(per_frame_errors[fi])
    early_err = np.mean(early_errors) if early_errors else 0
    late_err = np.mean(late_errors) if late_errors else 0

    # Attention entropy on eval videos
    all_entropies = []
    for vi in range(n_eval_videos):
        for fi in range(n_eval_frames):
            attn_fi = all_attns[vi][fi]
            ownership = attn_fi.argmax(axis=0)
            counts = np.bincount(ownership, minlength=n_slots)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-8)) / np.log(n_slots)
            all_entropies.append(ent)
    mean_entropy = np.mean(all_entropies)
    n_active_avg = np.mean([
        np.sum(np.bincount(all_attns[vi][64].argmax(0), minlength=n_slots) > n_patches * 0.02)
        for vi in range(n_eval_videos)
    ])

    # Baselines
    p45_err = 28.8
    p45_cons = 3.4
    p47_err = 31.4
    p47_cons = 0.4
    p46_err = 19.2
    p46_cons = 6.4

    print(f"\n│  === RESULTS ===", flush=True)
    print(f"│  Tracking error (mean):   {mean_tracking_error*100:.1f}% "
          f"(47: {p47_err}%, 45: {p45_err}%, 46: {p46_err}%)", flush=True)
    print(f"│  Tracking error (median): {median_tracking_error*100:.1f}%", flush=True)
    print(f"│  Early (0-15):   {early_err*100:.1f}%", flush=True)
    print(f"│  Late (64-127):  {late_err*100:.1f}%", flush=True)
    print(f"│  Slot consistency:        {mean_consistency:.1f}% "
          f"(47: {p47_cons}%, 45: {p45_cons}%, 46: {p46_cons}%)", flush=True)
    print(f"│  Binding accuracy:        {mean_binding:.1f}%", flush=True)
    print(f"│  Attention entropy:        {mean_entropy:.3f}", flush=True)
    print(f"│  Active slots (eval):      {n_active_avg:.1f}/{n_slots}", flush=True)
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
    slot_colors_plot = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                        '#911eb4', '#42d4f4', '#f032e6']

    # Panel 1: Sample frames with slot masks
    ax = axes[0, 0]
    sample_vids = [0, min(5, n_eval_videos-1), min(10, n_eval_videos-1)]
    sample_frame = 64
    n_show = min(3, n_eval_videos)
    composite_img = np.zeros((P * n_show, P * 2, 3))
    for si, vi in enumerate(sample_vids[:n_show]):
        vid_id = eval_video_ids[vi]
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
    vid_id = eval_video_ids[plot_vid]
    ann = annotations[vid_id]
    traj = ann['motion_trajectory']
    obj_props = {p['object_id']: p for p in ann['object_property']}
    obj_ids = [p['object_id'] for p in ann['object_property']]
    gt_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    for oi, oid in enumerate(obj_ids):
        gt_us, gt_vs = [], []
        for fi in range(n_eval_frames):
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

    reordered = all_centroids[plot_vid]
    for s in range(n_slots):
        coverage = np.mean([(all_attns[plot_vid][fi].argmax(0) == s).sum() / n_patches
                            for fi in range(0, n_eval_frames, 16)])
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

    # Panel 4: Temporal degradation
    ax = axes[1, 0]
    frame_bins = list(range(0, n_eval_frames, 8))
    bin_errors = []
    for b in frame_bins:
        bin_errs = []
        for fi in range(b, min(b + 8, n_eval_frames)):
            bin_errs.extend(per_frame_errors[fi])
        bin_errors.append(np.mean(bin_errs) * 100 if bin_errs else 0)
    ax.plot(frame_bins, bin_errors, 'o-', color='#FF5722', linewidth=2, markersize=4)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Tracking Error (%)')
    ax.set_title('Temporal Degradation (8-frame bins)', fontsize=10)
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% target')
    ax.axhline(y=early_err * 100, color='green', linestyle=':', alpha=0.7,
               label=f'Early (0-15): {early_err * 100:.1f}%')
    ax.axhline(y=late_err * 100, color='purple', linestyle=':', alpha=0.7,
               label=f'Late (64-127): {late_err * 100:.1f}%')
    ax.legend(fontsize=7)

    # Panel 5: DINOv2 PCA
    ax = axes[1, 1]
    pca_idx = 0 * n_eval_frames + 64
    feat_pca = eval_features[pca_idx].numpy()
    feat_centered = feat_pca - feat_pca.mean(axis=0)
    U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
    pca_3 = feat_centered @ Vt[:3].T
    pca_3 = pca_3 - pca_3.min(axis=0)
    pca_3 = pca_3 / (pca_3.max(axis=0) + 1e-8)
    ax.imshow(pca_3.reshape(P, P, 3), interpolation='nearest')
    ax.set_title(f'DINOv2 PCA (video {eval_video_ids[0]}, frame 64)', fontsize=10)
    ax.axis('off')

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = time.time() - t0

    err_ok = mean_tracking_error * 100 < 20
    cons_ok = mean_consistency > 15
    bind_ok = mean_binding > 85
    if err_ok and cons_ok and bind_ok:
        verdict = "SUCCESS"
    elif err_ok or cons_ok:
        verdict = "PARTIAL"
    elif mean_tracking_error * 100 < p45_err:
        verdict = "PARTIAL (improved over 45)"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 47b: Tau-Annealed Contrastive SA\n\n"
        f"Data: 1000 train videos x {n_extract} frames\n"
        f"Eval: {n_eval_videos} videos x {n_eval_frames} frames\n"
        f"Model: DINOv2 -> SA (7 slots, InfoNCE)\n"
        f"  alpha={alpha_contrastive}, tau={tau_start}->{tau_end}\n"
        f"  beta_ent={beta_entropy}\n\n"
        f"                     47b    47    46    45\n"
        f"Tracking error:  {mean_tracking_error*100:5.1f}% {p47_err:5.1f}% "
        f"{p46_err:5.1f}% {p45_err:5.1f}%\n"
        f"  early (0-15):  {early_err*100:5.1f}%\n"
        f"  late (64-127): {late_err*100:5.1f}%\n"
        f"Consistency:     {mean_consistency:5.1f}% {p47_cons:5.1f}%  "
        f"{p46_cons:5.1f}%  {p45_cons:5.1f}%\n"
        f"Binding:         {mean_binding:5.1f}%\n"
        f"Entropy:         {mean_entropy:.3f}\n"
        f"Active slots:    {n_active_avg:.1f}/{n_slots}\n\n"
        f"Best val loss: {best_val_loss:.5f}\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 47b: Tau-Annealed Contrastive SA (1000 Videos)\n'
                 f'err={mean_tracking_error*100:.1f}% '
                 f'consistency={mean_consistency:.0f}% '
                 f'binding={mean_binding:.0f}% '
                 f'entropy={mean_entropy:.3f} | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase47b_contrastive_1000v.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase47b_contrastive_1000v.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Tracking error:   {mean_tracking_error*100:.1f}% "
          f"(target <20%, 47: {p47_err}%, 46: {p46_err}%, 45: {p45_err}%)",
          flush=True)
    print(f"  Early (0-15):     {early_err*100:.1f}%", flush=True)
    print(f"  Late (64-127):    {late_err*100:.1f}%", flush=True)
    print(f"  Slot consistency: {mean_consistency:.1f}% "
          f"(target >15%, 47: {p47_cons}%, 46: {p46_cons}%, 45: {p45_cons}%)",
          flush=True)
    print(f"  Binding accuracy: {mean_binding:.1f}% (target >85%)", flush=True)
    print(f"  Entropy:          {mean_entropy:.3f}", flush=True)
    print(f"  Active slots:     {n_active_avg:.1f}/{n_slots}", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

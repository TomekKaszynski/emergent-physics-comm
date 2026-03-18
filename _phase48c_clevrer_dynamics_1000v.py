def run_phase48c_clevrer_dynamics_1000v():
    """Phase 48c: Scale dynamics communication to 1000 CLEVRER videos.

    Same task as 48b (predict post-collision direction), but with ~3000+ examples.
    Pipeline:
    1. Train SA on phase47 DINOv2 features (1000 vids × 16 frames, 50 epochs)
    2. For each collision in 1000 videos: extract frames, DINOv2, SA → slot features
    3. Train communication agents (200 epochs, proper train/val by video)
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
    print("PHASE 48c: Dynamics Communication — 1000 CLEVRER Videos", flush=True)
    print("  Predict post-collision direction via communication", flush=True)
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

    n_videos = 1000
    n_eval_frames = 128
    video_ids = list(range(10000, 10000 + n_videos))
    collision_window = 8
    n_direction_bins = 8
    dir_names = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
    n_train_vids = 800  # videos 10000-10799

    PROJ_U = np.array([0.0589, 0.2286, 0.4850])
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])

    def project_3d_to_2d(x, y):
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load all annotations + extract collision metadata
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load annotations + extract collision metadata", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    annotations = {}
    for vid_id in video_ids:
        ann_path = f"{data_dir}/annotation_{vid_id}.json"
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                annotations[vid_id] = json.load(f)

    def get_obj_pos_2d(traj, frame_idx, obj_id):
        for o in traj[frame_idx]['objects']:
            if o['object_id'] == obj_id:
                return np.array(project_3d_to_2d(
                    o['location'][0], o['location'][1]))
        return None

    def velocity_direction_bin(pos_start, pos_end, n_frames):
        v = (pos_end - pos_start) / max(n_frames, 1)
        speed = np.linalg.norm(v)
        if speed < 0.001:
            return -1, speed
        angle = np.arctan2(v[1], v[0])
        bin_idx = int((angle + np.pi) / (2 * np.pi / n_direction_bins)) % n_direction_bins
        return bin_idx, speed

    collision_events = []
    for vid_id in video_ids:
        if vid_id not in annotations:
            continue
        ann = annotations[vid_id]
        traj = ann['motion_trajectory']
        props = {p['object_id']: p for p in ann['object_property']}
        vi = vid_id - 10000  # video index 0-999

        for coll in ann.get('collision', []):
            cf = coll['frame_id']
            oid_a, oid_b = coll['object_ids']

            post_end = min(n_eval_frames - 1, cf + 4)
            pos_cf = get_obj_pos_2d(traj, cf, oid_a)
            pos_post = get_obj_pos_2d(traj, post_end, oid_a)
            if pos_cf is None or pos_post is None:
                continue

            dir_bin, speed = velocity_direction_bin(pos_cf, pos_post, post_end - cf)
            if dir_bin < 0:
                continue

            # GT positions at collision frame for slot matching
            gt_a = pos_cf
            gt_b = get_obj_pos_2d(traj, cf, oid_b)
            if gt_b is None:
                continue

            collision_events.append({
                'video_idx': vi,
                'video_id': vid_id,
                'collision_frame': cf,
                'obj_a_id': oid_a,
                'obj_b_id': oid_b,
                'direction_bin': dir_bin,
                'gt_a': gt_a,
                'gt_b': gt_b,
                'is_train': vi < n_train_vids,
            })

    train_events = [c for c in collision_events if c['is_train']]
    val_events = [c for c in collision_events if not c['is_train']]

    dir_bins_all = [c['direction_bin'] for c in collision_events]
    bin_counts = np.bincount(dir_bins_all, minlength=n_direction_bins)

    print(f"│  Total collisions: {len(collision_events)}", flush=True)
    print(f"│  Train: {len(train_events)} (videos 10000-10799)", flush=True)
    print(f"│  Val: {len(val_events)} (videos 10800-10999)", flush=True)
    print(f"│  Direction bins: {bin_counts.tolist()}", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Train Slot Attention on 1000 videos
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Train SA on 1000 videos (reconstruction only)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    dino_dim = 384
    n_patches = 256
    P = 16
    n_slots = 7
    slot_dim = 64
    n_sa_iters = 5

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

    # Load phase47 features for SA training
    sa_cache = str(OUTPUT_DIR / "phase48c_sa_model.pt")
    phase47_feat_path = str(OUTPUT_DIR / "phase47_dino_features_1000v.pt")

    torch.manual_seed(42)
    encoder = EncoderMLP(dino_dim).to(device)
    slot_attn = SlotAttention(
        n_slots=n_slots, slot_dim=slot_dim, n_iters=n_sa_iters,
        feature_dim=dino_dim).to(device)
    sa_decoder = SpatialBroadcastDecoder(
        slot_dim=slot_dim, output_dim=dino_dim).to(device)

    if os.path.exists(sa_cache):
        print(f"│  Loading SA model from {sa_cache}", flush=True)
        ckpt = torch.load(sa_cache, weights_only=True)
        encoder.load_state_dict(ckpt['encoder'])
        slot_attn.load_state_dict(ckpt['slot_attn'])
        sa_decoder.load_state_dict(ckpt['decoder'])
    else:
        print(f"│  Loading phase47 features for SA training...", flush=True)
        sa_features = torch.load(phase47_feat_path, weights_only=True)
        print(f"│  Features: {sa_features.shape}", flush=True)

        sa_params = (list(encoder.parameters()) +
                     list(slot_attn.parameters()) +
                     list(sa_decoder.parameters()))
        sa_optimizer = torch.optim.Adam(sa_params, lr=1e-4)

        n_total = sa_features.shape[0]
        sa_batch = 32
        sa_epochs = 50
        indices = np.arange(n_total)

        print(f"│  Training SA: {sa_epochs} epochs on {n_total} frames", flush=True)

        for epoch in range(1, sa_epochs + 1):
            encoder.train(); slot_attn.train(); sa_decoder.train()
            np.random.shuffle(indices)
            ep_loss, n_b = 0.0, 0

            for i in range(0, n_total, sa_batch):
                bidx = indices[i:i + sa_batch]
                feat = sa_features[bidx].to(device)
                enc = encoder(feat)
                slots, attn = slot_attn(enc)
                recon, alpha = sa_decoder(slots)
                loss = F.mse_loss(recon, feat)
                sa_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sa_params, 1.0)
                sa_optimizer.step()
                ep_loss += loss.item(); n_b += 1

                if device.type == 'mps' and n_b % 100 == 0:
                    torch.mps.empty_cache()

            if epoch % 10 == 0 or epoch == 1:
                print(f"│    SA Epoch {epoch:2d}/{sa_epochs}: "
                      f"recon={ep_loss/n_b:.5f}", flush=True)

        # Save SA model
        torch.save({
            'encoder': encoder.state_dict(),
            'slot_attn': slot_attn.state_dict(),
            'decoder': sa_decoder.state_dict(),
        }, sa_cache)
        print(f"│  Saved SA model to {sa_cache}", flush=True)

        del sa_features
        if device.type == 'mps':
            torch.mps.empty_cache()

    # Freeze perception
    encoder.eval()
    slot_attn.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in slot_attn.parameters():
        p.requires_grad = False

    print(f"│  SA perception frozen", flush=True)
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Extract slot features at collision windows
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Extract slot features at collision windows", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    slot_cache_path = str(OUTPUT_DIR / "phase48c_slot_features.pt")

    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2)
    grid_np = grid_positions.numpy()

    window_len = collision_window + 1  # 9 frames each half

    def pad_or_truncate(seq, target_len):
        if len(seq) >= target_len:
            return seq[:target_len]
        pad = np.zeros((target_len - len(seq), seq.shape[1]))
        return np.concatenate([pad, seq], axis=0)

    if os.path.exists(slot_cache_path):
        print(f"│  Loading slot features from {slot_cache_path}", flush=True)
        slot_cache = torch.load(slot_cache_path, weights_only=True)
        agent_a_inputs = slot_cache['agent_a']
        agent_b_inputs = slot_cache['agent_b']
        labels = slot_cache['labels']
        is_train = slot_cache['is_train']
        print(f"│  Loaded: {len(labels)} examples", flush=True)
    else:
        # Load DINOv2 for feature extraction
        print(f"│  Loading DINOv2...", flush=True)
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                              pretrained=True)
        dino.eval().to(device)
        for p in dino.parameters():
            p.requires_grad = False

        dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        agent_a_list = []
        agent_b_list = []
        label_list = []
        is_train_list = []

        # Group collisions by video for efficient processing
        collisions_by_video = {}
        for ci, coll in enumerate(collision_events):
            vid_id = coll['video_id']
            if vid_id not in collisions_by_video:
                collisions_by_video[vid_id] = []
            collisions_by_video[vid_id].append((ci, coll))

        n_processed = 0
        n_videos_with_colls = len(collisions_by_video)

        for vi_count, (vid_id, vid_colls) in enumerate(collisions_by_video.items()):
            # Collect all unique frames needed for this video
            frames_needed = set()
            for ci, coll in vid_colls:
                cf = coll['collision_frame']
                start = max(0, cf - collision_window)
                end = min(n_eval_frames, cf + collision_window + 1)
                for fi in range(start, end):
                    frames_needed.add(fi)
            frames_sorted = sorted(frames_needed)

            # Extract video frames
            cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
            frame_data = {}
            for fi in frames_sorted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frame = frame.transpose(2, 0, 1)
                    frame_data[fi] = frame
            cap.release()

            # DINOv2 forward on all needed frames (batch)
            if not frame_data:
                continue
            frame_indices_list = sorted(frame_data.keys())
            batch_np = np.stack([frame_data[fi] for fi in frame_indices_list])
            batch_tensor = torch.tensor(batch_np).to(device)
            batch_tensor = (batch_tensor - dino_mean) / dino_std

            # Process in chunks of 16 to avoid OOM
            dino_features = {}
            for chunk_start in range(0, len(frame_indices_list), 16):
                chunk_end = min(chunk_start + 16, len(frame_indices_list))
                chunk = batch_tensor[chunk_start:chunk_end]
                with torch.no_grad():
                    feats = dino.forward_features(chunk)
                    patch_tokens = feats['x_norm_patchtokens']  # [B, 256, 384]
                for j, fi in enumerate(frame_indices_list[chunk_start:chunk_end]):
                    dino_features[fi] = patch_tokens[j:j+1]  # [1, 256, 384]

            del batch_tensor, batch_np

            # For each collision in this video, run SA with SAVi propagation
            for ci, coll in vid_colls:
                cf = coll['collision_frame']
                start = max(0, cf - collision_window)
                end = min(n_eval_frames, cf + collision_window + 1)
                mid_idx = cf - start

                # SAVi propagation through collision window
                prev_slots = None
                slot_seq = []
                centroid_seq = []

                for fi in range(start, end):
                    if fi not in dino_features:
                        break
                    feat = dino_features[fi].to(device)
                    with torch.no_grad():
                        enc = encoder(feat)
                        slots, attn = slot_attn(enc, init_slots=prev_slots)
                        prev_slots = slots

                    slots_np = slots[0].cpu().numpy()
                    attn_np = attn[0].cpu().numpy()
                    centroids = np.zeros((n_slots, 2))
                    for si in range(n_slots):
                        w = attn_np[si]
                        w_sum = w.sum()
                        if w_sum > 1e-6:
                            cx = (w * grid_np[:, 0]).sum() / w_sum
                            cy = (w * grid_np[:, 1]).sum() / w_sum
                            centroids[si] = [cx, cy]
                        else:
                            centroids[si] = [0.5, 0.5]
                    slot_seq.append(slots_np)
                    centroid_seq.append(centroids)

                if len(slot_seq) < mid_idx + 1:
                    continue

                slot_seq = np.stack(slot_seq)
                centroid_seq = np.stack(centroid_seq)

                # Match slots to objects at collision frame
                gt_a = coll['gt_a']
                gt_b = coll['gt_b']
                dists_a = np.linalg.norm(centroid_seq[mid_idx] - gt_a, axis=1)
                dists_b = np.linalg.norm(centroid_seq[mid_idx] - gt_b, axis=1)
                slot_a = np.argmin(dists_a)
                slot_b = np.argmin(dists_b)

                # Extract per-object slot sequences
                pre_a = pad_or_truncate(slot_seq[:mid_idx + 1, slot_a, :], window_len)
                pre_b = pad_or_truncate(slot_seq[:mid_idx + 1, slot_b, :], window_len)
                post_a = pad_or_truncate(slot_seq[mid_idx:, slot_a, :], window_len)
                post_b = pad_or_truncate(slot_seq[mid_idx:, slot_b, :], window_len)

                # Agent A: full collision (pre+post for both objects)
                a_in = np.concatenate([pre_a, post_a, pre_b, post_b], axis=0)
                agent_a_list.append(a_in)

                # Agent B: pre-collision only
                b_in = np.concatenate([pre_a, pre_b], axis=0)
                agent_b_list.append(b_in)

                label_list.append(coll['direction_bin'])
                is_train_list.append(coll['is_train'])
                n_processed += 1

            # Clean up
            del dino_features
            if device.type == 'mps' and (vi_count + 1) % 50 == 0:
                torch.mps.empty_cache()

            if (vi_count + 1) % 100 == 0:
                elapsed_ext = time.time() - ts2
                eta = elapsed_ext / (vi_count + 1) * (n_videos_with_colls - vi_count - 1)
                print(f"│    Processed {vi_count+1}/{n_videos_with_colls} videos, "
                      f"{n_processed} collisions [{elapsed_ext:.0f}s, eta {eta:.0f}s]",
                      flush=True)

        # Clean up DINOv2
        del dino, dino_mean, dino_std
        if device.type == 'mps':
            torch.mps.empty_cache()

        agent_a_inputs = torch.tensor(np.stack(agent_a_list), dtype=torch.float32)
        agent_b_inputs = torch.tensor(np.stack(agent_b_list), dtype=torch.float32)
        labels = torch.tensor(label_list, dtype=torch.long)
        is_train = torch.tensor(is_train_list, dtype=torch.bool)

        # Cache
        torch.save({
            'agent_a': agent_a_inputs,
            'agent_b': agent_b_inputs,
            'labels': labels,
            'is_train': is_train,
        }, slot_cache_path)
        print(f"│  Cached to {slot_cache_path}", flush=True)

    n_examples = len(labels)
    n_train = is_train.sum().item()
    n_val = n_examples - n_train
    train_idx = torch.where(is_train)[0]
    val_idx = torch.where(~is_train)[0]

    print(f"│  Total examples: {n_examples}", flush=True)
    print(f"│  Train: {n_train}, Val: {n_val}", flush=True)
    print(f"│  Agent A input: {agent_a_inputs.shape}", flush=True)
    print(f"│  Agent B input: {agent_b_inputs.shape}", flush=True)
    print(f"│  Label dist: {np.bincount(labels.numpy(), minlength=n_direction_bins).tolist()}", flush=True)
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Train communication agents
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Train communication agents", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    vocab_size = 8
    hidden_dim = 128
    comm_epochs = 200
    comm_lr = 3e-4
    comm_batch = 64
    gumbel_tau_start = 2.0
    gumbel_tau_end = 0.5

    a_input_dim = agent_a_inputs.shape[1] * slot_dim
    b_input_dim = agent_b_inputs.shape[1] * slot_dim

    class SenderAgent(nn.Module):
        def __init__(self, input_dim, hidden_dim, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size),
            )
        def forward(self, x, tau=1.0):
            logits = self.encoder(x)
            if self.training:
                message = F.gumbel_softmax(logits, tau=tau, hard=True)
            else:
                idx = logits.argmax(dim=-1)
                message = F.one_hot(idx, self.vocab_size).float()
            return message, logits

    class ReceiverAgent(nn.Module):
        def __init__(self, obs_dim, vocab_size, hidden_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + vocab_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
        def forward(self, obs, message):
            return self.net(torch.cat([obs, message], dim=-1))

    class ReceiverNoComm(nn.Module):
        def __init__(self, obs_dim, hidden_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
        def forward(self, obs):
            return self.net(obs)

    class OraclePredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(42)
    sender = SenderAgent(a_input_dim, hidden_dim, vocab_size).to(device)
    receiver = ReceiverAgent(b_input_dim, vocab_size, hidden_dim, n_direction_bins).to(device)
    receiver_nocomm = ReceiverNoComm(b_input_dim, hidden_dim, n_direction_bins).to(device)
    oracle = OraclePredictor(a_input_dim, hidden_dim, n_direction_bins).to(device)

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=comm_lr)
    nocomm_optimizer = torch.optim.Adam(receiver_nocomm.parameters(), lr=comm_lr)
    oracle_optimizer = torch.optim.Adam(oracle.parameters(), lr=comm_lr)

    print(f"│  Task: {n_direction_bins}-class direction "
          f"(chance={100/n_direction_bins:.1f}%)", flush=True)
    print(f"│  Vocab: {vocab_size}, Epochs: {comm_epochs}, "
          f"Batch: {comm_batch}", flush=True)
    print(f"│  Train: {n_train}, Val: {n_val}", flush=True)

    a_flat = agent_a_inputs.reshape(n_examples, -1).to(device)
    b_flat = agent_b_inputs.reshape(n_examples, -1).to(device)
    labels_dev = labels.to(device)

    history = {
        'val_comm_acc': [], 'val_nocomm_acc': [], 'val_oracle_acc': [],
        'train_comm_acc': [], 'train_nocomm_acc': [],
        'msg_entropy': [], 'gumbel_tau': [],
    }
    epoch_times = []

    for epoch in range(1, comm_epochs + 1):
        epoch_t0 = time.time()
        progress = min(epoch / (comm_epochs * 0.7), 1.0)
        g_tau = gumbel_tau_start + (gumbel_tau_end - gumbel_tau_start) * progress

        # Mini-batch training
        perm = train_idx[torch.randperm(n_train)]

        sender.train(); receiver.train()
        receiver_nocomm.train(); oracle.train()

        ep_comm_correct = 0
        ep_nocomm_correct = 0
        ep_total = 0

        for bi in range(0, n_train, comm_batch):
            batch = perm[bi:bi + comm_batch]

            # Communication
            msg, logits = sender(a_flat[batch], tau=g_tau)
            pred = receiver(b_flat[batch], msg)
            loss_comm = F.cross_entropy(pred, labels_dev[batch])
            comm_optimizer.zero_grad()
            loss_comm.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

            # No-comm
            pred_nc = receiver_nocomm(b_flat[batch])
            loss_nc = F.cross_entropy(pred_nc, labels_dev[batch])
            nocomm_optimizer.zero_grad()
            loss_nc.backward()
            torch.nn.utils.clip_grad_norm_(receiver_nocomm.parameters(), 1.0)
            nocomm_optimizer.step()

            # Oracle
            pred_or = oracle(a_flat[batch])
            loss_or = F.cross_entropy(pred_or, labels_dev[batch])
            oracle_optimizer.zero_grad()
            loss_or.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            oracle_optimizer.step()

            with torch.no_grad():
                ep_comm_correct += (pred.argmax(1) == labels_dev[batch]).sum().item()
                ep_nocomm_correct += (pred_nc.argmax(1) == labels_dev[batch]).sum().item()
                ep_total += len(batch)

        train_comm_acc = ep_comm_correct / ep_total
        train_nocomm_acc = ep_nocomm_correct / ep_total

        epoch_times.append(time.time() - epoch_t0)

        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            sender.eval(); receiver.eval()
            receiver_nocomm.eval(); oracle.eval()

            with torch.no_grad():
                val_msg, _ = sender(a_flat[val_idx])
                val_pred = receiver(b_flat[val_idx], val_msg)
                val_comm_acc = (val_pred.argmax(1) == labels_dev[val_idx]).float().mean().item()

                val_nc = receiver_nocomm(b_flat[val_idx])
                val_nocomm_acc = (val_nc.argmax(1) == labels_dev[val_idx]).float().mean().item()

                val_or = oracle(a_flat[val_idx])
                val_oracle_acc = (val_or.argmax(1) == labels_dev[val_idx]).float().mean().item()

                all_msg, _ = sender(a_flat)
                msg_ids_all = all_msg.argmax(dim=-1).cpu().numpy()
                counts = np.bincount(msg_ids_all, minlength=vocab_size).astype(float)
                probs = counts / counts.sum()
                msg_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)

            history['val_comm_acc'].append(val_comm_acc)
            history['val_nocomm_acc'].append(val_nocomm_acc)
            history['val_oracle_acc'].append(val_oracle_acc)
            history['train_comm_acc'].append(train_comm_acc)
            history['train_nocomm_acc'].append(train_nocomm_acc)
            history['msg_entropy'].append(msg_entropy)
            history['gumbel_tau'].append(g_tau)

            if epoch % 20 == 0 or epoch == 1:
                recent = epoch_times[-10:]
                avg_ep = np.mean(recent)
                eta_m = (comm_epochs - epoch) * avg_ep / 60

                print(f"│  Epoch {epoch:3d}/{comm_epochs}: "
                      f"comm={train_comm_acc:.2f}/{val_comm_acc:.2f} "
                      f"nocomm={train_nocomm_acc:.2f}/{val_nocomm_acc:.2f} "
                      f"oracle=?/{val_oracle_acc:.2f} "
                      f"ent={msg_entropy:.3f} τ={g_tau:.2f} "
                      f"eta={eta_m:.1f}m", flush=True)

    # Final evaluation
    sender.eval(); receiver.eval()
    receiver_nocomm.eval(); oracle.eval()

    with torch.no_grad():
        # Val metrics
        val_msg, _ = sender(a_flat[val_idx])
        val_pred = receiver(b_flat[val_idx], val_msg)
        final_val_comm = (val_pred.argmax(1) == labels_dev[val_idx]).float().mean().item()

        val_nc = receiver_nocomm(b_flat[val_idx])
        final_val_nocomm = (val_nc.argmax(1) == labels_dev[val_idx]).float().mean().item()

        val_or = oracle(a_flat[val_idx])
        final_val_oracle = (val_or.argmax(1) == labels_dev[val_idx]).float().mean().item()

        # Message stats
        all_msg, _ = sender(a_flat)
        msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
        counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
        msg_probs = counts / counts.sum()
        final_entropy = -np.sum(msg_probs * np.log(msg_probs + 1e-8)) / np.log(vocab_size)

    comm_gain = final_val_comm - final_val_nocomm

    # Save model
    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'receiver_nocomm': receiver_nocomm.state_dict(),
        'oracle': oracle.state_dict(),
    }, str(OUTPUT_DIR / "phase48c_model.pt"))

    print(f"\n│  === RESULTS (validation) ===", flush=True)
    print(f"│  With communication:    {final_val_comm*100:.1f}%", flush=True)
    print(f"│  Without communication: {final_val_nocomm*100:.1f}%", flush=True)
    print(f"│  Oracle (A sees all):   {final_val_oracle*100:.1f}%", flush=True)
    print(f"│  Communication gain:    {comm_gain*100:+.1f}pp", flush=True)
    print(f"│  Chance baseline:       {100/n_direction_bins:.1f}%", flush=True)
    print(f"│  Message entropy:       {final_entropy:.3f}", flush=True)
    print(f"│  Messages used:         {(msg_probs > 0.01).sum()}/{vocab_size}", flush=True)
    print(f"└─ Stage 3 done [{time.time()-ts3:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 4: Message analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Message analysis", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts4 = time.time()

    msg_by_dir = {d: [] for d in range(n_direction_bins)}
    labels_np = labels.numpy()
    for i in range(n_examples):
        msg_by_dir[labels_np[i]].append(msg_ids[i])

    print(f"│  Message distribution by direction:", flush=True)
    for d in range(n_direction_bins):
        if msg_by_dir[d]:
            c = np.bincount(msg_by_dir[d], minlength=vocab_size)
            dominant = c.argmax()
            cons = c[dominant] / c.sum()
            print(f"│    {dir_names[d]:2s}: dominant=msg{dominant} ({cons:.0%}), "
                  f"counts={c.tolist()}", flush=True)

    consistency_scores = []
    for d in range(n_direction_bins):
        if len(msg_by_dir[d]) >= 2:
            c = np.bincount(msg_by_dir[d], minlength=vocab_size)
            cons = c.max() / c.sum()
            consistency_scores.append(cons)
    mean_consistency = np.mean(consistency_scores) if consistency_scores else 0

    print(f"│  Mean consistency: {mean_consistency:.2f}", flush=True)
    print(f"└─ Stage 4 done [{time.time()-ts4:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 5: Visualization
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 5: Visualization", flush=True)
    print(f"{'=' * 60}", flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Accuracy over training (val only)
    ax = axes[0, 0]
    epochs_logged = [1] + list(range(10, comm_epochs + 1, 10))
    ax.plot(epochs_logged, history['val_comm_acc'], 'b-', linewidth=2,
            label=f'With comm ({final_val_comm*100:.0f}%)')
    ax.plot(epochs_logged, history['val_nocomm_acc'], 'r--', linewidth=2,
            label=f'No comm ({final_val_nocomm*100:.0f}%)')
    ax.plot(epochs_logged, history['val_oracle_acc'], 'g:', linewidth=2,
            label=f'Oracle ({final_val_oracle*100:.0f}%)')
    ax.axhline(y=1/n_direction_bins, color='gray', linestyle=':', alpha=0.5,
               label=f'Chance ({100/n_direction_bins:.0f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Post-Collision Direction Prediction (Val)', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.5, max(
        max(history['val_comm_acc']),
        max(history['val_nocomm_acc']),
        max(history['val_oracle_acc'])) + 0.1))

    # Panel 2: Message-direction heatmap
    ax = axes[0, 1]
    msg_matrix = np.zeros((n_direction_bins, vocab_size))
    for d in range(n_direction_bins):
        if msg_by_dir[d]:
            c = np.bincount(msg_by_dir[d], minlength=vocab_size).astype(float)
            msg_matrix[d] = c / max(c.sum(), 1)
    im = ax.imshow(msg_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Message Symbol')
    ax.set_ylabel('True Direction')
    ax.set_yticks(range(n_direction_bins))
    ax.set_yticklabels(dir_names)
    ax.set_xticks(range(vocab_size))
    ax.set_title('Message Usage by Direction', fontsize=11)
    plt.colorbar(im, ax=ax, label='Frequency')
    for d in range(n_direction_bins):
        for m in range(vocab_size):
            if msg_matrix[d, m] > 0.1:
                ax.text(m, d, f'{msg_matrix[d, m]:.1f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if msg_matrix[d, m] > 0.5 else 'black')

    # Panel 3: Confusion matrix (val, with comm)
    ax = axes[1, 0]
    with torch.no_grad():
        val_pred_labels = val_pred.argmax(1).cpu().numpy()
    val_true_labels = labels[val_idx].numpy()
    conf_matrix = np.zeros((n_direction_bins, n_direction_bins))
    for t, p in zip(val_true_labels, val_pred_labels):
        conf_matrix[t, p] += 1
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_norm = conf_matrix / np.maximum(row_sums, 1)
    im2 = ax.imshow(conf_norm, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Predicted Direction')
    ax.set_ylabel('True Direction')
    ax.set_xticks(range(n_direction_bins))
    ax.set_xticklabels(dir_names, fontsize=8)
    ax.set_yticks(range(n_direction_bins))
    ax.set_yticklabels(dir_names, fontsize=8)
    ax.set_title('Val Confusion (with comm)', fontsize=11)
    plt.colorbar(im2, ax=ax)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    elapsed = time.time() - t0

    if comm_gain >= 0.10 and final_val_comm > 0.30:
        verdict = "SUCCESS"
    elif comm_gain >= 0.05 or final_val_comm > 0.30:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 48c: Dynamics Comm (1000 vids)\n\n"
        f"Task: post-collision direction\n"
        f"  ({n_direction_bins} bins, chance={100/n_direction_bins:.0f}%)\n"
        f"Data: {n_examples} collisions\n"
        f"  Train: {n_train}, Val: {n_val}\n"
        f"Channel: Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Val accuracy:\n"
        f"  With comm:    {final_val_comm*100:.1f}%\n"
        f"  Without comm: {final_val_nocomm*100:.1f}%\n"
        f"  Oracle:       {final_val_oracle*100:.1f}%\n"
        f"  Gain:         {comm_gain*100:+.1f}pp\n\n"
        f"Message entropy: {final_entropy:.3f}\n"
        f"Symbols used: {(msg_probs > 0.01).sum()}/{vocab_size}\n"
        f"Msg consistency: {mean_consistency:.2f}\n\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 48c: CLEVRER Dynamics Communication (1000 Videos)\n'
                 f'val: comm={final_val_comm*100:.0f}% nocomm={final_val_nocomm*100:.0f}% '
                 f'(+{comm_gain*100:.0f}pp) '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} | {verdict}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase48c_clevrer_dynamics.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase48c_clevrer_dynamics.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val with communication:    {final_val_comm*100:.1f}% (target >30%)", flush=True)
    print(f"  Val without communication: {final_val_nocomm*100:.1f}%", flush=True)
    print(f"  Val oracle:                {final_val_oracle*100:.1f}%", flush=True)
    print(f"  Communication gain:        {comm_gain*100:+.1f}pp (target >10pp)", flush=True)
    print(f"  Message entropy:           {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

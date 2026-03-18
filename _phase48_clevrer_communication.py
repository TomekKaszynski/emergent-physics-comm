def run_phase48_clevrer_communication():
    """Phase 48: CLEVRER communication pipeline with slot attention perception.

    Agents observe real collision videos through learned slot representations
    and communicate about material (metal vs rubber = mass proxy).
    - Agent A sees pre+post collision slot features for 2 objects
    - Agent A sends discrete message (Gumbel-Softmax, vocab=8)
    - Agent B sees pre-collision only + message → predicts which is heavier (metal)
    - Perception: retrain Phase 45 SA (frozen during comm training)
    """
    import time
    import json
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from pathlib import Path

    print("=" * 70, flush=True)
    print("PHASE 48: CLEVRER Communication — Slot Perception + Language", flush=True)
    print("  Agents communicate about mass from collision dynamics", flush=True)
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

    n_eval_videos = 20
    n_eval_frames = 128
    eval_video_ids = list(range(10000, 10000 + n_eval_videos))
    collision_window = 8  # frames before/after collision

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations + extract collision events
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load annotations + extract collisions", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    annotations = {}
    for vid_id in eval_video_ids:
        with open(f"{data_dir}/annotation_{vid_id}.json") as f:
            annotations[vid_id] = json.load(f)

    PROJ_U = np.array([0.0589, 0.2286, 0.4850])
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])

    def project_3d_to_2d(x, y):
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    # Extract all collisions with metadata
    collision_events = []
    for vi, vid_id in enumerate(eval_video_ids):
        ann = annotations[vid_id]
        props = {p['object_id']: p for p in ann['object_property']}
        collisions = ann.get('collision', [])

        for ci, coll in enumerate(collisions):
            oid_a, oid_b = coll['object_ids']
            frame = coll['frame_id']
            mat_a = props[oid_a]['material']
            mat_b = props[oid_b]['material']

            # Label: 1 if object A is heavier (metal), 0 otherwise
            # metal > rubber; if same material, label = 0 (no mass difference)
            if mat_a == mat_b:
                label = -1  # same material — skip for binary task
            elif mat_a == 'metal':
                label = 1  # A is heavier
            else:
                label = 0  # B is heavier

            collision_events.append({
                'video_idx': vi,
                'video_id': vid_id,
                'collision_frame': frame,
                'obj_a_id': oid_a,
                'obj_b_id': oid_b,
                'material_a': mat_a,
                'material_b': mat_b,
                'label': label,
            })

    all_collisions = [c for c in collision_events]
    mixed_collisions = [c for c in collision_events if c['label'] >= 0]
    same_collisions = [c for c in collision_events if c['label'] < 0]

    print(f"│  Total collisions: {len(all_collisions)}", flush=True)
    print(f"│  Mixed material (metal vs rubber): {len(mixed_collisions)}", flush=True)
    print(f"│  Same material (skip for binary): {len(same_collisions)}", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Load DINOv2 features + retrain Slot Attention
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Slot Attention perception (retrain from scratch)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    # Load cached DINOv2 features
    eval_cache = str(OUTPUT_DIR / "phase45_dino_features.pt")
    print(f"│  Loading eval features: {eval_cache}", flush=True)
    eval_features = torch.load(eval_cache, weights_only=True)
    print(f"│  Features shape: {eval_features.shape}", flush=True)

    dino_dim = 384
    n_patches = 256
    P = 16
    n_slots = 7
    slot_dim = 64
    n_sa_iters = 5

    # ── Slot Attention Module (same as Phase 45) ──
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

    # Train slot attention on 20 eval videos (reconstruction only)
    sa_epochs = 50
    sa_batch = 32
    sa_lr = 1e-4

    torch.manual_seed(42)
    encoder = EncoderMLP(dino_dim).to(device)
    slot_attn = SlotAttention(
        n_slots=n_slots, slot_dim=slot_dim, n_iters=n_sa_iters,
        feature_dim=dino_dim).to(device)
    sa_decoder = SpatialBroadcastDecoder(
        slot_dim=slot_dim, output_dim=dino_dim).to(device)

    sa_params = (list(encoder.parameters()) +
                 list(slot_attn.parameters()) +
                 list(sa_decoder.parameters()))
    sa_optimizer = torch.optim.Adam(sa_params, lr=sa_lr)

    n_total_frames = eval_features.shape[0]  # 2560
    train_indices = np.arange(n_total_frames)

    print(f"│  Training SA: {sa_epochs} epochs on {n_total_frames} frames", flush=True)

    for epoch in range(1, sa_epochs + 1):
        encoder.train()
        slot_attn.train()
        sa_decoder.train()

        np.random.shuffle(train_indices)
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, n_total_frames, sa_batch):
            batch_idx = train_indices[i:i + sa_batch]
            feat = eval_features[batch_idx].to(device)

            enc = encoder(feat)
            slots, attn = slot_attn(enc)
            recon, alpha = sa_decoder(slots)

            loss = F.mse_loss(recon, feat)
            sa_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sa_params, 1.0)
            sa_optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"│    SA Epoch {epoch:3d}/{sa_epochs}: "
                  f"recon={ep_loss/n_batches:.5f}", flush=True)

    # Freeze perception
    encoder.eval()
    slot_attn.eval()
    sa_decoder.eval()
    for p in sa_params:
        p.requires_grad = False

    print(f"│  SA training done, perception frozen", flush=True)
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Extract slot features at collision windows
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Extract slot features at collision windows", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    # Grid for computing centroids
    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2)

    def get_slot_features_and_centroids(video_idx, frame_idx):
        """Get slot features and centroids for one frame using SAVi propagation."""
        feat_global_idx = video_idx * n_eval_frames + frame_idx
        feat = eval_features[feat_global_idx:feat_global_idx + 1].to(device)
        with torch.no_grad():
            enc = encoder(feat)
            slots, attn = slot_attn(enc)
        slots_np = slots[0].cpu().numpy()  # [n_slots, slot_dim]
        attn_np = attn[0].cpu().numpy()    # [n_slots, n_patches]

        centroids = np.zeros((n_slots, 2))
        for si in range(n_slots):
            w = attn_np[si]
            w_sum = w.sum()
            if w_sum > 1e-6:
                cx = (w * grid_positions[:, 0].numpy()).sum() / w_sum
                cy = (w * grid_positions[:, 1].numpy()).sum() / w_sum
                centroids[si] = [cx, cy]
            else:
                centroids[si] = [0.5, 0.5]
        return slots_np, centroids, attn_np

    def get_savi_slot_sequence(video_idx, start_frame, end_frame):
        """Get slot features via SAVi propagation over a frame range."""
        all_slots = []
        all_centroids = []
        prev_slots_tensor = None

        for fi in range(start_frame, end_frame):
            feat_global_idx = video_idx * n_eval_frames + fi
            feat = eval_features[feat_global_idx:feat_global_idx + 1].to(device)
            with torch.no_grad():
                enc = encoder(feat)
                slots, attn = slot_attn(enc, init_slots=prev_slots_tensor)
                prev_slots_tensor = slots

            slots_np = slots[0].cpu().numpy()
            attn_np = attn[0].cpu().numpy()

            centroids = np.zeros((n_slots, 2))
            for si in range(n_slots):
                w = attn_np[si]
                w_sum = w.sum()
                if w_sum > 1e-6:
                    cx = (w * grid_positions[:, 0].numpy()).sum() / w_sum
                    cy = (w * grid_positions[:, 1].numpy()).sum() / w_sum
                    centroids[si] = [cx, cy]
                else:
                    centroids[si] = [0.5, 0.5]

            all_slots.append(slots_np)
            all_centroids.append(centroids)

        return np.stack(all_slots), np.stack(all_centroids)  # [T, S, D], [T, S, 2]

    def match_slot_to_object(centroids_at_frame, obj_gt_pos):
        """Find which slot is closest to a GT object position."""
        dists = np.linalg.norm(centroids_at_frame - obj_gt_pos, axis=1)
        return np.argmin(dists)

    # For each collision, extract slot feature sequences and match to objects
    collision_data = []

    for ci, coll in enumerate(mixed_collisions):
        vi = coll['video_idx']
        vid_id = coll['video_id']
        cf = coll['collision_frame']
        oid_a = coll['obj_a_id']
        oid_b = coll['obj_b_id']

        ann = annotations[vid_id]
        traj = ann['motion_trajectory']

        # Get GT positions at collision frame
        frame_objs = {o['object_id']: o for o in traj[cf]['objects']}
        if oid_a not in frame_objs or oid_b not in frame_objs:
            continue
        obj_a = frame_objs[oid_a]
        obj_b = frame_objs[oid_b]
        if not obj_a['inside_camera_view'] or not obj_b['inside_camera_view']:
            continue

        gt_a = np.array(project_3d_to_2d(obj_a['location'][0], obj_a['location'][1]))
        gt_b = np.array(project_3d_to_2d(obj_b['location'][0], obj_b['location'][1]))

        # SAVi propagation over collision window
        start_frame = max(0, cf - collision_window)
        end_frame = min(n_eval_frames, cf + collision_window + 1)
        mid_idx = cf - start_frame  # index of collision frame within window

        slot_seq, centroid_seq = get_savi_slot_sequence(vi, start_frame, end_frame)
        # slot_seq: [window_len, 7, 64], centroid_seq: [window_len, 7, 2]

        # Match slots to objects at collision frame
        slot_a = match_slot_to_object(centroid_seq[mid_idx], gt_a)
        slot_b = match_slot_to_object(centroid_seq[mid_idx], gt_b)

        # Extract per-object slot feature sequences
        # Pre-collision: frames before collision within window
        pre_slots_a = slot_seq[:mid_idx + 1, slot_a, :]  # [<=9, 64]
        pre_slots_b = slot_seq[:mid_idx + 1, slot_b, :]
        # Post-collision: frames after collision within window
        post_slots_a = slot_seq[mid_idx:, slot_a, :]  # [<=9, 64]
        post_slots_b = slot_seq[mid_idx:, slot_b, :]

        collision_data.append({
            'pre_a': pre_slots_a,
            'pre_b': pre_slots_b,
            'post_a': post_slots_a,
            'post_b': post_slots_b,
            'label': coll['label'],
            'material_a': coll['material_a'],
            'material_b': coll['material_b'],
            'video_id': vid_id,
            'collision_frame': cf,
            'slot_a': slot_a,
            'slot_b': slot_b,
        })

    print(f"│  Extracted {len(collision_data)} collision examples "
          f"(from {len(mixed_collisions)} mixed-material)", flush=True)

    # Pad/truncate to fixed length
    window_len = collision_window + 1  # 9 frames (including collision frame)

    def pad_or_truncate(seq, target_len):
        """Pad with zeros or truncate to target_len."""
        if len(seq) >= target_len:
            return seq[:target_len]
        pad = np.zeros((target_len - len(seq), seq.shape[1]))
        return np.concatenate([pad, seq], axis=0)

    # Build tensors: Agent A sees pre+post, Agent B sees pre only
    agent_a_inputs = []  # [N, 2*window_len*2, slot_dim] = [N, 36, 64]
    agent_b_inputs = []  # [N, 2*window_len, slot_dim] = [N, 18, 64]
    labels = []

    for cd in collision_data:
        pre_a = pad_or_truncate(cd['pre_a'], window_len)   # [9, 64]
        pre_b = pad_or_truncate(cd['pre_b'], window_len)
        post_a = pad_or_truncate(cd['post_a'], window_len)
        post_b = pad_or_truncate(cd['post_b'], window_len)

        # Agent A: full collision (pre+post for both objects)
        a_in = np.concatenate([pre_a, post_a, pre_b, post_b], axis=0)  # [36, 64]
        agent_a_inputs.append(a_in)

        # Agent B: pre-collision only (both objects)
        b_in = np.concatenate([pre_a, pre_b], axis=0)  # [18, 64]
        agent_b_inputs.append(b_in)

        labels.append(cd['label'])

    agent_a_inputs = torch.tensor(np.stack(agent_a_inputs), dtype=torch.float32)
    agent_b_inputs = torch.tensor(np.stack(agent_b_inputs), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    print(f"│  Agent A input: {agent_a_inputs.shape}", flush=True)
    print(f"│  Agent B input: {agent_b_inputs.shape}", flush=True)
    print(f"│  Labels: {labels.shape} (positive={labels.sum().item()}, "
          f"negative={(labels == 0).sum().item()})", flush=True)
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Train communication agents
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Train communication agents", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    n_examples = len(labels)
    vocab_size = 8
    msg_dim = 32  # hidden dim for message processing
    comm_epochs = 500
    comm_lr = 1e-3
    gumbel_tau_start = 2.0
    gumbel_tau_end = 0.5

    a_input_dim = agent_a_inputs.shape[1] * slot_dim  # 36*64 = 2304
    b_input_dim = agent_b_inputs.shape[1] * slot_dim  # 18*64 = 1152

    class SenderAgent(nn.Module):
        """Agent A: sees full collision, sends discrete message."""
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
            # x: [B, input_dim]
            logits = self.encoder(x)  # [B, vocab_size]
            if self.training:
                message = F.gumbel_softmax(logits, tau=tau, hard=True)
            else:
                idx = logits.argmax(dim=-1)
                message = F.one_hot(idx, self.vocab_size).float()
            return message, logits

    class ReceiverAgent(nn.Module):
        """Agent B: sees pre-collision + message, predicts mass ordering."""
        def __init__(self, obs_dim, vocab_size, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + vocab_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # binary: A heavier or B heavier
            )

        def forward(self, obs, message):
            # obs: [B, obs_dim], message: [B, vocab_size]
            combined = torch.cat([obs, message], dim=-1)
            return self.net(combined)

    class ReceiverNoComm(nn.Module):
        """Baseline: Agent B without message (observation only)."""
        def __init__(self, obs_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, obs):
            return self.net(obs)

    torch.manual_seed(42)
    sender = SenderAgent(a_input_dim, 128, vocab_size).to(device)
    receiver = ReceiverAgent(b_input_dim, vocab_size, 128).to(device)
    receiver_nocomm = ReceiverNoComm(b_input_dim, 128).to(device)

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=comm_lr)

    nocomm_optimizer = torch.optim.Adam(receiver_nocomm.parameters(), lr=comm_lr)

    print(f"│  Sender: {sum(p.numel() for p in sender.parameters()):,} params", flush=True)
    print(f"│  Receiver: {sum(p.numel() for p in receiver.parameters()):,} params", flush=True)
    print(f"│  No-comm baseline: "
          f"{sum(p.numel() for p in receiver_nocomm.parameters()):,} params", flush=True)
    print(f"│  Vocab: {vocab_size}, Epochs: {comm_epochs}, "
          f"Examples: {n_examples}", flush=True)

    # Move data to device
    a_flat = agent_a_inputs.reshape(n_examples, -1).to(device)  # [N, 2304]
    b_flat = agent_b_inputs.reshape(n_examples, -1).to(device)  # [N, 1152]
    labels_dev = labels.to(device)

    # Train/val split (leave-one-out per video would be ideal but dataset is tiny)
    # Use 80/20 split
    n_train = int(0.8 * n_examples)
    perm = torch.randperm(n_examples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    history = {
        'comm_acc': [], 'nocomm_acc': [],
        'comm_loss': [], 'nocomm_loss': [],
        'val_comm_acc': [], 'val_nocomm_acc': [],
        'msg_entropy': [], 'gumbel_tau': [],
    }

    for epoch in range(1, comm_epochs + 1):
        # Gumbel temperature annealing
        progress = min(epoch / (comm_epochs * 0.7), 1.0)
        g_tau = gumbel_tau_start + (gumbel_tau_end - gumbel_tau_start) * progress

        # ── Train with communication ──
        sender.train()
        receiver.train()

        message, logits = sender(a_flat[train_idx], tau=g_tau)
        pred = receiver(b_flat[train_idx], message)
        comm_loss = F.cross_entropy(pred, labels_dev[train_idx])

        comm_optimizer.zero_grad()
        comm_loss.backward()
        torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
        comm_optimizer.step()

        with torch.no_grad():
            comm_acc = (pred.argmax(1) == labels_dev[train_idx]).float().mean().item()

        # ── Train without communication (baseline) ──
        receiver_nocomm.train()
        pred_nc = receiver_nocomm(b_flat[train_idx])
        nc_loss = F.cross_entropy(pred_nc, labels_dev[train_idx])

        nocomm_optimizer.zero_grad()
        nc_loss.backward()
        torch.nn.utils.clip_grad_norm_(receiver_nocomm.parameters(), 1.0)
        nocomm_optimizer.step()

        with torch.no_grad():
            nocomm_acc = (pred_nc.argmax(1) == labels_dev[train_idx]).float().mean().item()

        # ── Validation ──
        if epoch % 10 == 0 or epoch == 1:
            sender.eval()
            receiver.eval()
            receiver_nocomm.eval()

            with torch.no_grad():
                # Comm validation
                val_msg, val_logits = sender(a_flat[val_idx], tau=g_tau)
                val_pred = receiver(b_flat[val_idx], val_msg)
                val_comm_acc = (val_pred.argmax(1) == labels_dev[val_idx]).float().mean().item()

                # No-comm validation
                val_pred_nc = receiver_nocomm(b_flat[val_idx])
                val_nocomm_acc = (val_pred_nc.argmax(1) == labels_dev[val_idx]).float().mean().item()

                # Message entropy
                all_msg, all_logits = sender(a_flat, tau=g_tau)
                msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
                counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
                probs = counts / counts.sum()
                msg_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)

            history['comm_acc'].append(comm_acc)
            history['nocomm_acc'].append(nocomm_acc)
            history['comm_loss'].append(comm_loss.item())
            history['nocomm_loss'].append(nc_loss.item())
            history['val_comm_acc'].append(val_comm_acc)
            history['val_nocomm_acc'].append(val_nocomm_acc)
            history['msg_entropy'].append(msg_entropy)
            history['gumbel_tau'].append(g_tau)

            if epoch % 50 == 0 or epoch == 1:
                print(f"│  Epoch {epoch:3d}/{comm_epochs}: "
                      f"comm={comm_acc:.2f}/{val_comm_acc:.2f} "
                      f"nocomm={nocomm_acc:.2f}/{val_nocomm_acc:.2f} "
                      f"ent={msg_entropy:.3f} τ_g={g_tau:.2f}", flush=True)

    # Final evaluation
    sender.eval()
    receiver.eval()
    receiver_nocomm.eval()

    with torch.no_grad():
        final_msg, final_logits = sender(a_flat)
        final_pred = receiver(b_flat, final_msg)
        final_comm_acc = (final_pred.argmax(1) == labels_dev).float().mean().item()

        final_pred_nc = receiver_nocomm(b_flat)
        final_nocomm_acc = (final_pred_nc.argmax(1) == labels_dev).float().mean().item()

        msg_ids = final_msg.argmax(dim=-1).cpu().numpy()
        counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
        msg_probs = counts / counts.sum()
        final_entropy = -np.sum(msg_probs * np.log(msg_probs + 1e-8)) / np.log(vocab_size)

    # Save model
    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'receiver_nocomm': receiver_nocomm.state_dict(),
        'encoder': encoder.state_dict(),
        'slot_attn': slot_attn.state_dict(),
    }, str(OUTPUT_DIR / "phase48_model.pt"))

    print(f"\n│  === COMMUNICATION RESULTS ===", flush=True)
    print(f"│  With communication:    {final_comm_acc*100:.1f}%", flush=True)
    print(f"│  Without communication: {final_nocomm_acc*100:.1f}%", flush=True)
    print(f"│  Communication gain:    {(final_comm_acc - final_nocomm_acc)*100:+.1f}%", flush=True)
    print(f"│  Message entropy:       {final_entropy:.3f}", flush=True)
    print(f"│  Messages used:         {(msg_probs > 0.01).sum()}/{vocab_size}", flush=True)
    print(f"└─ Stage 3 done [{time.time()-ts3:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 4: Analysis — message-material correspondence
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Message analysis", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts4 = time.time()

    # What message does the sender use for each material pair?
    msg_by_label = {0: [], 1: []}
    for i, cd in enumerate(collision_data):
        msg_by_label[cd['label']].append(msg_ids[i])

    print(f"│  Messages when A=rubber, B=metal (label=0):", flush=True)
    if msg_by_label[0]:
        c0 = np.bincount(msg_by_label[0], minlength=vocab_size)
        print(f"│    {c0.tolist()}", flush=True)
    print(f"│  Messages when A=metal, B=rubber (label=1):", flush=True)
    if msg_by_label[1]:
        c1 = np.bincount(msg_by_label[1], minlength=vocab_size)
        print(f"│    {c1.tolist()}", flush=True)

    # Message consistency: does same label always get same message?
    consistency_scores = []
    for lbl in [0, 1]:
        msgs = msg_by_label[lbl]
        if len(msgs) >= 2:
            most_common = np.bincount(msgs, minlength=vocab_size).argmax()
            cons = np.mean([m == most_common for m in msgs])
            consistency_scores.append(cons)
            print(f"│  Label {lbl} consistency: {cons:.2f} "
                  f"(most common msg: {most_common})", flush=True)
    mean_consistency = np.mean(consistency_scores) if consistency_scores else 0

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

    # Panel 1: Accuracy over training
    ax = axes[0, 0]
    epochs_logged = list(range(1, comm_epochs + 1, 10))
    # Fix: epoch 1 is logged separately
    epochs_logged = [1] + list(range(10, comm_epochs + 1, 10))
    ax.plot(epochs_logged, history['val_comm_acc'], 'b-', linewidth=2,
            label=f'With comm (final={final_comm_acc*100:.0f}%)')
    ax.plot(epochs_logged, history['val_nocomm_acc'], 'r--', linewidth=2,
            label=f'No comm (final={final_nocomm_acc*100:.0f}%)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Mass Prediction Accuracy', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel 2: Message usage heatmap
    ax = axes[0, 1]
    # Create a 2D heatmap: rows = label, columns = message
    msg_matrix = np.zeros((2, vocab_size))
    for lbl in [0, 1]:
        if msg_by_label[lbl]:
            c = np.bincount(msg_by_label[lbl], minlength=vocab_size).astype(float)
            msg_matrix[lbl] = c / max(c.sum(), 1)
    im = ax.imshow(msg_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Message Symbol')
    ax.set_ylabel('Label')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['A=rubber\nB=metal', 'A=metal\nB=rubber'])
    ax.set_xticks(range(vocab_size))
    ax.set_title('Message Usage by Label', fontsize=11)
    plt.colorbar(im, ax=ax, label='Frequency')
    # Add counts as text
    for lbl in range(2):
        for msg in range(vocab_size):
            if msg_matrix[lbl, msg] > 0.05:
                ax.text(msg, lbl, f'{msg_matrix[lbl, msg]:.2f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if msg_matrix[lbl, msg] > 0.5 else 'black')

    # Panel 3: Message entropy over training
    ax = axes[1, 0]
    ax.plot(epochs_logged, history['msg_entropy'], 'g-', linewidth=2,
            label='Message entropy')
    ax.plot(epochs_logged, history['gumbel_tau'], 'orange', linewidth=1,
            linestyle='--', label='Gumbel τ')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Message Entropy & Gumbel Temperature', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 2.5)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    elapsed = time.time() - t0

    comm_helps = final_comm_acc > final_nocomm_acc
    acc_ok = final_comm_acc > 0.70
    ent_ok = final_entropy > 0.3

    if acc_ok and comm_helps and ent_ok:
        verdict = "SUCCESS"
    elif acc_ok or (comm_helps and final_comm_acc > 0.60):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 48: CLEVRER Communication\n\n"
        f"Perception: DINOv2 + SA (7 slots, frozen)\n"
        f"Task: predict heavier object from collision\n"
        f"Channel: Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Examples: {n_examples} mixed-material collisions\n"
        f"  (from {n_eval_videos} videos)\n\n"
        f"With communication:    {final_comm_acc*100:.1f}%\n"
        f"Without communication: {final_nocomm_acc*100:.1f}%\n"
        f"Communication gain:    {(final_comm_acc-final_nocomm_acc)*100:+.1f}%\n\n"
        f"Message entropy: {final_entropy:.3f}\n"
        f"Symbols used: {(msg_probs > 0.01).sum()}/{vocab_size}\n"
        f"Msg consistency: {mean_consistency:.2f}\n\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 48: CLEVRER Communication Pipeline\n'
                 f'comm={final_comm_acc*100:.0f}% vs nocomm={final_nocomm_acc*100:.0f}% '
                 f'(+{(final_comm_acc-final_nocomm_acc)*100:.0f}%) '
                 f'entropy={final_entropy:.3f} | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase48_clevrer_communication.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase48_clevrer_communication.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  With communication:    {final_comm_acc*100:.1f}% (target >70%)", flush=True)
    print(f"  Without communication: {final_nocomm_acc*100:.1f}%", flush=True)
    print(f"  Communication gain:    {(final_comm_acc-final_nocomm_acc)*100:+.1f}%", flush=True)
    print(f"  Message entropy:       {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"  Message consistency:   {mean_consistency:.2f}", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

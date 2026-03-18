def run_phase48b_clevrer_dynamics():
    """Phase 48b: CLEVRER communication with genuine information asymmetry.

    Predict post-collision velocity direction of object A (8 angular bins).
    - Agent A sees full collision window (pre+post, both objects)
    - Agent B sees pre-collision only → needs message to know outcome
    - This requires communication: post-collision state is NOT visible to B
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
    print("PHASE 48b: CLEVRER Dynamics Communication", flush=True)
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

    n_eval_videos = 20
    n_eval_frames = 128
    eval_video_ids = list(range(10000, 10000 + n_eval_videos))
    collision_window = 8
    n_direction_bins = 8
    dir_names = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations + extract collision events with velocity
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Extract collisions + velocity labels", flush=True)
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

    def get_obj_pos_2d(traj, frame_idx, obj_id):
        for o in traj[frame_idx]['objects']:
            if o['object_id'] == obj_id:
                return np.array(project_3d_to_2d(
                    o['location'][0], o['location'][1]))
        return None

    def velocity_direction_bin(pos_start, pos_end, n_frames):
        """Compute post-collision velocity direction bin."""
        v = (pos_end - pos_start) / max(n_frames, 1)
        speed = np.linalg.norm(v)
        if speed < 0.001:
            return -1, speed  # stationary
        angle = np.arctan2(v[1], v[0])
        bin_idx = int((angle + np.pi) / (2 * np.pi / n_direction_bins)) % n_direction_bins
        return bin_idx, speed

    collision_events = []
    for vi, vid_id in enumerate(eval_video_ids):
        ann = annotations[vid_id]
        traj = ann['motion_trajectory']
        props = {p['object_id']: p for p in ann['object_property']}

        for coll in ann.get('collision', []):
            cf = coll['frame_id']
            oid_a, oid_b = coll['object_ids']

            # Post-collision velocity direction for object A
            post_end = min(n_eval_frames - 1, cf + 4)
            pos_cf = get_obj_pos_2d(traj, cf, oid_a)
            pos_post = get_obj_pos_2d(traj, post_end, oid_a)
            if pos_cf is None or pos_post is None:
                continue

            dir_bin, speed = velocity_direction_bin(pos_cf, pos_post, post_end - cf)
            if dir_bin < 0:
                continue  # skip stationary

            collision_events.append({
                'video_idx': vi,
                'video_id': vid_id,
                'collision_frame': cf,
                'obj_a_id': oid_a,
                'obj_b_id': oid_b,
                'direction_bin': dir_bin,
                'speed': speed,
                'material_a': props[oid_a]['material'],
                'material_b': props[oid_b]['material'],
            })

    # Direction bin distribution
    dir_bins = [c['direction_bin'] for c in collision_events]
    bin_counts = np.bincount(dir_bins, minlength=n_direction_bins)

    print(f"│  Total collision examples: {len(collision_events)}", flush=True)
    print(f"│  Direction bin distribution:", flush=True)
    for i, name in enumerate(dir_names):
        print(f"│    {name}: {bin_counts[i]}", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Retrain Slot Attention (or reuse from Phase 48)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Slot Attention perception", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

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

    # Check for Phase 48 model first
    phase48_model_path = str(OUTPUT_DIR / "phase48_model.pt")
    if os.path.exists(phase48_model_path):
        print(f"│  Loading SA weights from Phase 48 model", flush=True)
        torch.manual_seed(42)
        encoder = EncoderMLP(dino_dim).to(device)
        slot_attn = SlotAttention(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=n_sa_iters,
            feature_dim=dino_dim).to(device)
        sa_decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim, output_dim=dino_dim).to(device)

        ckpt = torch.load(phase48_model_path, weights_only=True)
        encoder.load_state_dict(ckpt['encoder'])
        slot_attn.load_state_dict(ckpt['slot_attn'])
        print(f"│  Loaded encoder + slot_attn from Phase 48", flush=True)
    else:
        print(f"│  No Phase 48 model, retraining SA...", flush=True)
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
        sa_optimizer = torch.optim.Adam(sa_params, lr=1e-4)

        n_total = eval_features.shape[0]
        train_idx = np.arange(n_total)

        for epoch in range(1, 51):
            encoder.train(); slot_attn.train(); sa_decoder.train()
            np.random.shuffle(train_idx)
            ep_loss, n_b = 0.0, 0
            for i in range(0, n_total, 32):
                bidx = train_idx[i:i+32]
                feat = eval_features[bidx].to(device)
                enc = encoder(feat)
                slots, attn = slot_attn(enc)
                recon, alpha = sa_decoder(slots)
                loss = F.mse_loss(recon, feat)
                sa_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sa_params, 1.0)
                sa_optimizer.step()
                ep_loss += loss.item(); n_b += 1
            if epoch % 10 == 0:
                print(f"│    SA Epoch {epoch}/50: recon={ep_loss/n_b:.5f}", flush=True)

    # Freeze perception
    encoder.eval()
    slot_attn.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in slot_attn.parameters():
        p.requires_grad = False

    print(f"│  Perception frozen", flush=True)
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Extract slot features at collision windows
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Extract slot features at collision windows", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2)

    def get_savi_slot_sequence(video_idx, start_frame, end_frame):
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

        return np.stack(all_slots), np.stack(all_centroids)

    def match_slot_to_object(centroids_at_frame, obj_gt_pos):
        dists = np.linalg.norm(centroids_at_frame - obj_gt_pos, axis=1)
        return np.argmin(dists)

    window_len = collision_window + 1  # 9 frames each half

    def pad_or_truncate(seq, target_len):
        if len(seq) >= target_len:
            return seq[:target_len]
        pad = np.zeros((target_len - len(seq), seq.shape[1]))
        return np.concatenate([pad, seq], axis=0)

    collision_data = []
    for ci, coll in enumerate(collision_events):
        vi = coll['video_idx']
        vid_id = coll['video_id']
        cf = coll['collision_frame']
        oid_a = coll['obj_a_id']
        oid_b = coll['obj_b_id']

        ann = annotations[vid_id]
        traj = ann['motion_trajectory']

        frame_objs = {o['object_id']: o for o in traj[cf]['objects']}
        if oid_a not in frame_objs or oid_b not in frame_objs:
            continue
        obj_a = frame_objs[oid_a]
        obj_b = frame_objs[oid_b]
        if not obj_a['inside_camera_view'] or not obj_b['inside_camera_view']:
            continue

        gt_a = np.array(project_3d_to_2d(obj_a['location'][0], obj_a['location'][1]))
        gt_b = np.array(project_3d_to_2d(obj_b['location'][0], obj_b['location'][1]))

        start_frame = max(0, cf - collision_window)
        end_frame = min(n_eval_frames, cf + collision_window + 1)
        mid_idx = cf - start_frame

        slot_seq, centroid_seq = get_savi_slot_sequence(vi, start_frame, end_frame)

        slot_a = match_slot_to_object(centroid_seq[mid_idx], gt_a)
        slot_b = match_slot_to_object(centroid_seq[mid_idx], gt_b)

        pre_slots_a = slot_seq[:mid_idx + 1, slot_a, :]
        pre_slots_b = slot_seq[:mid_idx + 1, slot_b, :]
        post_slots_a = slot_seq[mid_idx:, slot_a, :]
        post_slots_b = slot_seq[mid_idx:, slot_b, :]

        collision_data.append({
            'pre_a': pre_slots_a, 'pre_b': pre_slots_b,
            'post_a': post_slots_a, 'post_b': post_slots_b,
            'direction_bin': coll['direction_bin'],
            'video_id': vid_id, 'collision_frame': cf,
        })

    # Build tensors
    agent_a_inputs = []  # Full collision: pre+post for both objects
    agent_b_inputs = []  # Pre-collision only for both objects
    labels = []

    for cd in collision_data:
        pre_a = pad_or_truncate(cd['pre_a'], window_len)
        pre_b = pad_or_truncate(cd['pre_b'], window_len)
        post_a = pad_or_truncate(cd['post_a'], window_len)
        post_b = pad_or_truncate(cd['post_b'], window_len)

        a_in = np.concatenate([pre_a, post_a, pre_b, post_b], axis=0)
        agent_a_inputs.append(a_in)

        b_in = np.concatenate([pre_a, pre_b], axis=0)
        agent_b_inputs.append(b_in)

        labels.append(cd['direction_bin'])

    agent_a_inputs = torch.tensor(np.stack(agent_a_inputs), dtype=torch.float32)
    agent_b_inputs = torch.tensor(np.stack(agent_b_inputs), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    n_examples = len(labels)
    print(f"│  Collision examples: {n_examples}", flush=True)
    print(f"│  Agent A input: {agent_a_inputs.shape}", flush=True)
    print(f"│  Agent B input: {agent_b_inputs.shape}", flush=True)
    print(f"│  Label distribution: {np.bincount(labels.numpy(), minlength=n_direction_bins).tolist()}", flush=True)
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
    comm_epochs = 1000  # more epochs for harder 8-class task
    comm_lr = 3e-4
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

    # Also train an "oracle" — Agent A directly predicts (upper bound)
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

    print(f"│  Sender: {sum(p.numel() for p in sender.parameters()):,} params", flush=True)
    print(f"│  Receiver: {sum(p.numel() for p in receiver.parameters()):,} params", flush=True)
    print(f"│  Task: {n_direction_bins}-class direction prediction "
          f"(chance={100/n_direction_bins:.1f}%)", flush=True)
    print(f"│  Vocab: {vocab_size}, Epochs: {comm_epochs}, "
          f"Examples: {n_examples}", flush=True)

    a_flat = agent_a_inputs.reshape(n_examples, -1).to(device)
    b_flat = agent_b_inputs.reshape(n_examples, -1).to(device)
    labels_dev = labels.to(device)

    # Leave-2-out cross-validation style: 80/20 split
    n_train = int(0.8 * n_examples)
    perm = torch.randperm(n_examples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    history = {
        'comm_acc': [], 'nocomm_acc': [], 'oracle_acc': [],
        'val_comm_acc': [], 'val_nocomm_acc': [], 'val_oracle_acc': [],
        'msg_entropy': [], 'gumbel_tau': [],
        'comm_loss': [], 'nocomm_loss': [],
    }

    for epoch in range(1, comm_epochs + 1):
        progress = min(epoch / (comm_epochs * 0.7), 1.0)
        g_tau = gumbel_tau_start + (gumbel_tau_end - gumbel_tau_start) * progress

        # ── Train with communication ──
        sender.train(); receiver.train()
        message, logits = sender(a_flat[train_idx], tau=g_tau)
        pred = receiver(b_flat[train_idx], message)
        comm_loss = F.cross_entropy(pred, labels_dev[train_idx])
        comm_optimizer.zero_grad()
        comm_loss.backward()
        torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
        comm_optimizer.step()

        with torch.no_grad():
            comm_acc = (pred.argmax(1) == labels_dev[train_idx]).float().mean().item()

        # ── Train without communication ──
        receiver_nocomm.train()
        pred_nc = receiver_nocomm(b_flat[train_idx])
        nc_loss = F.cross_entropy(pred_nc, labels_dev[train_idx])
        nocomm_optimizer.zero_grad()
        nc_loss.backward()
        torch.nn.utils.clip_grad_norm_(receiver_nocomm.parameters(), 1.0)
        nocomm_optimizer.step()

        with torch.no_grad():
            nocomm_acc = (pred_nc.argmax(1) == labels_dev[train_idx]).float().mean().item()

        # ── Train oracle ──
        oracle.train()
        pred_or = oracle(a_flat[train_idx])
        or_loss = F.cross_entropy(pred_or, labels_dev[train_idx])
        oracle_optimizer.zero_grad()
        or_loss.backward()
        torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
        oracle_optimizer.step()

        with torch.no_grad():
            oracle_acc = (pred_or.argmax(1) == labels_dev[train_idx]).float().mean().item()

        # ── Validation ──
        if epoch % 20 == 0 or epoch == 1:
            sender.eval(); receiver.eval()
            receiver_nocomm.eval(); oracle.eval()

            with torch.no_grad():
                val_msg, _ = sender(a_flat[val_idx], tau=g_tau)
                val_pred = receiver(b_flat[val_idx], val_msg)
                val_comm_acc = (val_pred.argmax(1) == labels_dev[val_idx]).float().mean().item()

                val_nc = receiver_nocomm(b_flat[val_idx])
                val_nocomm_acc = (val_nc.argmax(1) == labels_dev[val_idx]).float().mean().item()

                val_or = oracle(a_flat[val_idx])
                val_oracle_acc = (val_or.argmax(1) == labels_dev[val_idx]).float().mean().item()

                all_msg, _ = sender(a_flat, tau=g_tau)
                msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
                counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
                probs = counts / counts.sum()
                msg_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)

            history['comm_acc'].append(comm_acc)
            history['nocomm_acc'].append(nocomm_acc)
            history['oracle_acc'].append(oracle_acc)
            history['val_comm_acc'].append(val_comm_acc)
            history['val_nocomm_acc'].append(val_nocomm_acc)
            history['val_oracle_acc'].append(val_oracle_acc)
            history['msg_entropy'].append(msg_entropy)
            history['gumbel_tau'].append(g_tau)
            history['comm_loss'].append(comm_loss.item())
            history['nocomm_loss'].append(nc_loss.item())

            if epoch % 100 == 0 or epoch == 1:
                print(f"│  Epoch {epoch:4d}/{comm_epochs}: "
                      f"comm={comm_acc:.2f}/{val_comm_acc:.2f} "
                      f"nocomm={nocomm_acc:.2f}/{val_nocomm_acc:.2f} "
                      f"oracle={oracle_acc:.2f}/{val_oracle_acc:.2f} "
                      f"ent={msg_entropy:.3f} τ_g={g_tau:.2f}", flush=True)

    # Final eval on ALL data
    sender.eval(); receiver.eval()
    receiver_nocomm.eval(); oracle.eval()

    with torch.no_grad():
        final_msg, _ = sender(a_flat)
        final_pred = receiver(b_flat, final_msg)
        final_comm_acc = (final_pred.argmax(1) == labels_dev).float().mean().item()

        final_nc = receiver_nocomm(b_flat)
        final_nocomm_acc = (final_nc.argmax(1) == labels_dev).float().mean().item()

        final_or = oracle(a_flat)
        final_oracle_acc = (final_or.argmax(1) == labels_dev).float().mean().item()

        msg_ids = final_msg.argmax(dim=-1).cpu().numpy()
        counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
        msg_probs = counts / counts.sum()
        final_entropy = -np.sum(msg_probs * np.log(msg_probs + 1e-8)) / np.log(vocab_size)

    # Save model
    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'receiver_nocomm': receiver_nocomm.state_dict(),
        'oracle': oracle.state_dict(),
    }, str(OUTPUT_DIR / "phase48b_model.pt"))

    comm_gain = final_comm_acc - final_nocomm_acc

    print(f"\n│  === RESULTS ===", flush=True)
    print(f"│  With communication:    {final_comm_acc*100:.1f}%", flush=True)
    print(f"│  Without communication: {final_nocomm_acc*100:.1f}%", flush=True)
    print(f"│  Oracle (A sees all):   {final_oracle_acc*100:.1f}%", flush=True)
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

    # Message → direction mapping
    msg_by_dir = {d: [] for d in range(n_direction_bins)}
    for i, cd in enumerate(collision_data):
        msg_by_dir[cd['direction_bin']].append(msg_ids[i])

    print(f"│  Message distribution by direction:", flush=True)
    for d in range(n_direction_bins):
        if msg_by_dir[d]:
            c = np.bincount(msg_by_dir[d], minlength=vocab_size)
            dominant = c.argmax()
            cons = c[dominant] / c.sum()
            print(f"│    {dir_names[d]:2s}: msg counts={c.tolist()} "
                  f"dominant={dominant} ({cons:.0%})", flush=True)

    # Overall message consistency
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

    # Panel 1: Accuracy over training
    ax = axes[0, 0]
    epochs_logged = [1] + list(range(20, comm_epochs + 1, 20))
    ax.plot(epochs_logged, history['val_comm_acc'], 'b-', linewidth=2,
            label=f'With comm ({final_comm_acc*100:.0f}%)')
    ax.plot(epochs_logged, history['val_nocomm_acc'], 'r--', linewidth=2,
            label=f'No comm ({final_nocomm_acc*100:.0f}%)')
    ax.plot(epochs_logged, history['val_oracle_acc'], 'g:', linewidth=2,
            label=f'Oracle ({final_oracle_acc*100:.0f}%)')
    ax.axhline(y=1/n_direction_bins, color='gray', linestyle=':', alpha=0.5,
               label=f'Chance ({100/n_direction_bins:.0f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Post-Collision Direction Prediction', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

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

    # Panel 3: Confusion matrix (with comm)
    ax = axes[1, 0]
    pred_labels = final_pred.argmax(1).cpu().numpy()
    true_labels = labels.numpy()
    conf_matrix = np.zeros((n_direction_bins, n_direction_bins))
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1
    # Normalize per row
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_norm = conf_matrix / np.maximum(row_sums, 1)
    im2 = ax.imshow(conf_norm, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Predicted Direction')
    ax.set_ylabel('True Direction')
    ax.set_xticks(range(n_direction_bins))
    ax.set_xticklabels(dir_names, fontsize=8)
    ax.set_yticks(range(n_direction_bins))
    ax.set_yticklabels(dir_names, fontsize=8)
    ax.set_title('Confusion Matrix (with comm)', fontsize=11)
    plt.colorbar(im2, ax=ax)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    elapsed = time.time() - t0

    if comm_gain >= 0.10 and final_comm_acc > 0.30:
        verdict = "SUCCESS"
    elif comm_gain >= 0.05 or final_comm_acc > 0.30:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 48b: Dynamics Communication\n\n"
        f"Task: predict post-collision direction\n"
        f"  ({n_direction_bins} bins, chance={100/n_direction_bins:.0f}%)\n"
        f"Perception: DINOv2 + SA (frozen)\n"
        f"Channel: Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Examples: {n_examples} collisions\n"
        f"  (from {n_eval_videos} videos)\n\n"
        f"With communication:    {final_comm_acc*100:.1f}%\n"
        f"Without communication: {final_nocomm_acc*100:.1f}%\n"
        f"Oracle (full obs):     {final_oracle_acc*100:.1f}%\n"
        f"Communication gain:    {comm_gain*100:+.1f}pp\n\n"
        f"Message entropy: {final_entropy:.3f}\n"
        f"Symbols used: {(msg_probs > 0.01).sum()}/{vocab_size}\n"
        f"Msg consistency: {mean_consistency:.2f}\n\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 48b: CLEVRER Dynamics Communication\n'
                 f'comm={final_comm_acc*100:.0f}% vs nocomm={final_nocomm_acc*100:.0f}% '
                 f'(+{comm_gain*100:.0f}pp) '
                 f'oracle={final_oracle_acc*100:.0f}% '
                 f'ent={final_entropy:.3f} | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase48b_clevrer_dynamics.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase48b_clevrer_dynamics.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  With communication:    {final_comm_acc*100:.1f}% (target >30%)", flush=True)
    print(f"  Without communication: {final_nocomm_acc*100:.1f}%", flush=True)
    print(f"  Oracle:                {final_oracle_acc*100:.1f}%", flush=True)
    print(f"  Communication gain:    {comm_gain*100:+.1f}pp (target >10pp)", flush=True)
    print(f"  Message entropy:       {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

def run_phase49b_physics_features():
    """Phase 49b: Mass inference with engineered physics features.

    Same task as 49 (two agents, two collisions, which object is heavier).
    Change: instead of raw 180-dim trajectories, each agent gets 11 engineered
    physics features per collision:
      target: speed_pre, speed_post, speed_ratio, delta_v, deflection_angle (5)
      partner: speed_pre, speed_post, speed_ratio, delta_v, deflection_angle (5)
      relative_speed (1)
    Total: 11 dims per agent. Oracle: 22 dims.
    """
    import time
    import json
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pathlib import Path

    print("=" * 70, flush=True)
    print("PHASE 49b: Mass Communication — Engineered Physics Features", flush=True)
    print("  11 dims/agent: speed_pre/post/ratio, delta_v, deflection, rel_speed", flush=True)
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
    collision_window = 4  # frames before/after for speed averaging
    n_train_vids = 800

    PROJ_U = np.array([0.0589, 0.2286, 0.4850])
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])

    def project_3d_to_2d(x, y):
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load annotations", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    annotations = {}
    for vid_id in video_ids:
        ann_path = f"{data_dir}/annotation_{vid_id}.json"
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                annotations[vid_id] = json.load(f)

    print(f"│  Loaded {len(annotations)} annotations", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Build mass-comparison dataset with physics features
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Build dataset with engineered physics features", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    eps = 1e-6

    def get_obj_pos_2d(traj, frame_idx, obj_id):
        for o in traj[frame_idx]['objects']:
            if o['object_id'] == obj_id:
                return np.array(project_3d_to_2d(
                    o['location'][0], o['location'][1]))
        return None

    def compute_physics_features(traj, cf, target_id, partner_id):
        """Compute 11 engineered physics features for a collision.

        Per object (5 each):
          speed_pre: avg speed over frames [cf-4, cf)
          speed_post: avg speed over frames (cf, cf+4]
          speed_ratio: speed_post / (speed_pre + eps)
          delta_v: |velocity_post - velocity_pre| at collision frame
          deflection_angle: angle change in motion direction (radians)
        Plus (1):
          relative_speed: approach speed between objects pre-collision
        """
        # Gather positions for both objects around collision
        frames_pre = list(range(max(0, cf - collision_window), cf + 1))
        frames_post = list(range(cf, min(n_eval_frames, cf + collision_window + 1)))

        def get_positions(obj_id, frame_list):
            positions = []
            for fi in frame_list:
                p = get_obj_pos_2d(traj, fi, obj_id)
                if p is None:
                    return None
                positions.append(p)
            return np.array(positions)

        pos_pre_t = get_positions(target_id, frames_pre)
        pos_post_t = get_positions(target_id, frames_post)
        pos_pre_p = get_positions(partner_id, frames_pre)
        pos_post_p = get_positions(partner_id, frames_post)

        if any(x is None for x in [pos_pre_t, pos_post_t, pos_pre_p, pos_post_p]):
            return None

        def obj_features(pos_pre, pos_post):
            """Compute 5 physics features for one object."""
            # Velocities: displacement between consecutive frames
            vel_pre = np.diff(pos_pre, axis=0)  # [T_pre-1, 2]
            vel_post = np.diff(pos_post, axis=0)  # [T_post-1, 2]

            # Speeds
            speeds_pre = np.linalg.norm(vel_pre, axis=1)  # [T_pre-1]
            speeds_post = np.linalg.norm(vel_post, axis=1)  # [T_post-1]

            speed_pre = speeds_pre.mean() if len(speeds_pre) > 0 else 0.0
            speed_post = speeds_post.mean() if len(speeds_post) > 0 else 0.0
            speed_ratio = speed_post / (speed_pre + eps)

            # Velocity change at collision
            v_before = vel_pre[-1] if len(vel_pre) > 0 else np.zeros(2)
            v_after = vel_post[0] if len(vel_post) > 0 else np.zeros(2)
            delta_v = np.linalg.norm(v_after - v_before)

            # Deflection angle
            angle_before = np.arctan2(v_before[1], v_before[0])
            angle_after = np.arctan2(v_after[1], v_after[0])
            deflection = abs(angle_after - angle_before)
            if deflection > np.pi:
                deflection = 2 * np.pi - deflection

            return np.array([speed_pre, speed_post, speed_ratio, delta_v, deflection])

        target_feats = obj_features(pos_pre_t, pos_post_t)
        partner_feats = obj_features(pos_pre_p, pos_post_p)

        # Relative approach speed
        vel_pre_t = np.diff(pos_pre_t, axis=0)
        vel_pre_p = np.diff(pos_pre_p, axis=0)
        if len(vel_pre_t) > 0 and len(vel_pre_p) > 0:
            rel_vel = vel_pre_t[-1] - vel_pre_p[-1]
            relative_speed = np.linalg.norm(rel_vel)
        else:
            relative_speed = 0.0

        # Stack: [target(5) + partner(5) + relative_speed(1)] = 11
        return np.concatenate([target_feats, partner_feats, [relative_speed]])

    # For each video: find object materials and collisions per object
    agent_a_list = []
    agent_b_list = []
    label_list = []
    is_train_list = []

    n_pairs_total = 0
    n_videos_with_pairs = 0

    for vid_id in video_ids:
        if vid_id not in annotations:
            continue
        ann = annotations[vid_id]
        traj = ann['motion_trajectory']
        vi = vid_id - 10000

        # Get materials for all objects from object_property
        obj_materials = {}
        for o in ann.get('object_property', []):
            obj_materials[o['object_id']] = o.get('material', 'unknown')

        # Find collisions per object: obj_id → list of (collision_frame, partner_id)
        obj_collisions = {}
        for coll in ann.get('collision', []):
            cf = coll['frame_id']
            oid_a, oid_b = coll['object_ids']
            if oid_a not in obj_collisions:
                obj_collisions[oid_a] = []
            if oid_b not in obj_collisions:
                obj_collisions[oid_b] = []
            obj_collisions[oid_a].append((cf, oid_b))
            obj_collisions[oid_b].append((cf, oid_a))

        # Find pairs of objects with different materials
        obj_ids = list(obj_materials.keys())
        video_had_pair = False

        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                oid1, oid2 = obj_ids[i], obj_ids[j]
                mat1, mat2 = obj_materials[oid1], obj_materials[oid2]

                if mat1 == mat2 or mat1 == 'unknown' or mat2 == 'unknown':
                    continue
                if oid1 not in obj_collisions or oid2 not in obj_collisions:
                    continue

                if mat1 == 'metal' and mat2 == 'rubber':
                    heavy_id, light_id = oid1, oid2
                elif mat2 == 'metal' and mat1 == 'rubber':
                    heavy_id, light_id = oid2, oid1
                else:
                    continue

                heavy_cf, heavy_partner = obj_collisions[heavy_id][0]
                light_cf, light_partner = obj_collisions[light_id][0]

                heavy_feats = compute_physics_features(
                    traj, heavy_cf, heavy_id, heavy_partner)
                light_feats = compute_physics_features(
                    traj, light_cf, light_id, light_partner)

                if heavy_feats is None or light_feats is None:
                    continue

                video_had_pair = True

                # Version 1: A=heavy, B=light → label=0 (A is heavier)
                agent_a_list.append(heavy_feats)
                agent_b_list.append(light_feats)
                label_list.append(0)
                is_train_list.append(vi < n_train_vids)

                # Version 2: A=light, B=heavy → label=1 (B is heavier)
                agent_a_list.append(light_feats)
                agent_b_list.append(heavy_feats)
                label_list.append(1)
                is_train_list.append(vi < n_train_vids)

                n_pairs_total += 1

        if video_had_pair:
            n_videos_with_pairs += 1

    agent_a_inputs = torch.tensor(np.stack(agent_a_list), dtype=torch.float32)
    agent_b_inputs = torch.tensor(np.stack(agent_b_list), dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.long)
    is_train = torch.tensor(is_train_list, dtype=torch.bool)

    n_examples = len(labels)
    n_train = is_train.sum().item()
    n_val = n_examples - n_train
    train_idx = torch.where(is_train)[0]
    val_idx = torch.where(~is_train)[0]

    label_counts = np.bincount(labels.numpy(), minlength=2)
    input_dim = agent_a_inputs.shape[1]  # 11

    # Print feature statistics
    print(f"│  Videos with valid pairs: {n_videos_with_pairs}/{n_videos}", flush=True)
    print(f"│  Object pairs: {n_pairs_total}", flush=True)
    print(f"│  Total examples: {n_examples}", flush=True)
    print(f"│  Train: {n_train}, Val: {n_val}", flush=True)
    print(f"│  Label balance: A-heavier={label_counts[0]}, B-heavier={label_counts[1]}", flush=True)
    print(f"│  Feature dim per agent: {input_dim}", flush=True)

    # Feature distribution for heavy vs light objects
    heavy_feats = agent_a_inputs[labels == 0].numpy()  # A=heavy
    light_feats = agent_a_inputs[labels == 1].numpy()  # A=light
    feat_names = ['spd_pre', 'spd_post', 'spd_ratio', 'delta_v', 'deflect',
                  'p_spd_pre', 'p_spd_post', 'p_spd_ratio', 'p_delta_v', 'p_deflect',
                  'rel_spd']
    print(f"│  Feature means (heavy vs light target object):", flush=True)
    for fi, name in enumerate(feat_names[:5]):  # target features only
        h_mean = heavy_feats[:, fi].mean()
        l_mean = light_feats[:, fi].mean()
        print(f"│    {name:10s}: heavy={h_mean:.4f}  light={l_mean:.4f}  ratio={h_mean/(l_mean+eps):.2f}", flush=True)

    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Train communication agents
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Train communication agents", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    vocab_size = 8
    hidden_dim = 128
    comm_epochs = 200
    comm_lr = 3e-4
    comm_batch = 64
    gumbel_tau_start = 2.0
    gumbel_tau_end = 0.5
    n_classes = 2

    print(f"│  Input dim per agent: {input_dim}", flush=True)
    print(f"│  Task: binary (A heavier vs B heavier), chance=50%", flush=True)
    print(f"│  Vocab: {vocab_size}, Epochs: {comm_epochs}", flush=True)

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
    sender = SenderAgent(input_dim, hidden_dim, vocab_size).to(device)
    receiver = ReceiverAgent(input_dim, vocab_size, hidden_dim, n_classes).to(device)
    receiver_nocomm = ReceiverNoComm(input_dim, hidden_dim, n_classes).to(device)
    oracle = OraclePredictor(input_dim * 2, hidden_dim, n_classes).to(device)

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=comm_lr)
    nocomm_optimizer = torch.optim.Adam(receiver_nocomm.parameters(), lr=comm_lr)
    oracle_optimizer = torch.optim.Adam(oracle.parameters(), lr=comm_lr)

    a_flat = agent_a_inputs.to(device)  # already [N, 11]
    b_flat = agent_b_inputs.to(device)
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

        perm = train_idx[torch.randperm(n_train)]

        sender.train(); receiver.train()
        receiver_nocomm.train(); oracle.train()

        ep_comm_correct = 0
        ep_nocomm_correct = 0
        ep_total = 0

        for bi in range(0, n_train, comm_batch):
            batch = perm[bi:bi + comm_batch]

            # Communication: A sends message, B receives + predicts
            msg, logits = sender(a_flat[batch], tau=g_tau)
            pred = receiver(b_flat[batch], msg)
            loss_comm = F.cross_entropy(pred, labels_dev[batch])
            comm_optimizer.zero_grad()
            loss_comm.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

            # No-comm: B predicts from own observation only
            pred_nc = receiver_nocomm(b_flat[batch])
            loss_nc = F.cross_entropy(pred_nc, labels_dev[batch])
            nocomm_optimizer.zero_grad()
            loss_nc.backward()
            torch.nn.utils.clip_grad_norm_(receiver_nocomm.parameters(), 1.0)
            nocomm_optimizer.step()

            # Oracle: sees both
            oracle_in = torch.cat([a_flat[batch], b_flat[batch]], dim=-1)
            pred_or = oracle(oracle_in)
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

        if epoch % 10 == 0 or epoch == 1:
            sender.eval(); receiver.eval()
            receiver_nocomm.eval(); oracle.eval()

            with torch.no_grad():
                val_msg, _ = sender(a_flat[val_idx])
                val_pred = receiver(b_flat[val_idx], val_msg)
                val_comm_acc = (val_pred.argmax(1) == labels_dev[val_idx]).float().mean().item()

                val_nc = receiver_nocomm(b_flat[val_idx])
                val_nocomm_acc = (val_nc.argmax(1) == labels_dev[val_idx]).float().mean().item()

                oracle_in = torch.cat([a_flat[val_idx], b_flat[val_idx]], dim=-1)
                val_or = oracle(oracle_in)
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
        val_msg, _ = sender(a_flat[val_idx])
        val_pred = receiver(b_flat[val_idx], val_msg)
        final_val_comm = (val_pred.argmax(1) == labels_dev[val_idx]).float().mean().item()

        val_nc = receiver_nocomm(b_flat[val_idx])
        final_val_nocomm = (val_nc.argmax(1) == labels_dev[val_idx]).float().mean().item()

        oracle_in = torch.cat([a_flat[val_idx], b_flat[val_idx]], dim=-1)
        val_or = oracle(oracle_in)
        final_val_oracle = (val_or.argmax(1) == labels_dev[val_idx]).float().mean().item()

        all_msg, _ = sender(a_flat)
        msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
        counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
        msg_probs = counts / counts.sum()
        final_entropy = -np.sum(msg_probs * np.log(msg_probs + 1e-8)) / np.log(vocab_size)

    comm_gain = final_val_comm - final_val_nocomm

    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'receiver_nocomm': receiver_nocomm.state_dict(),
        'oracle': oracle.state_dict(),
    }, str(OUTPUT_DIR / "phase49b_model.pt"))

    print(f"\n│  === RESULTS (validation) ===", flush=True)
    print(f"│  With communication:    {final_val_comm*100:.1f}%", flush=True)
    print(f"│  Without communication: {final_val_nocomm*100:.1f}%", flush=True)
    print(f"│  Oracle (sees both):    {final_val_oracle*100:.1f}%", flush=True)
    print(f"│  Communication gain:    {comm_gain*100:+.1f}pp", flush=True)
    print(f"│  Chance baseline:       50.0%", flush=True)
    print(f"│  Message entropy:       {final_entropy:.3f}", flush=True)
    print(f"│  Messages used:         {(msg_probs > 0.01).sum()}/{vocab_size}", flush=True)
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Message analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Message analysis", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    labels_np = labels.numpy()

    for lab in range(2):
        mask = labels_np == lab
        if mask.sum() == 0:
            continue
        c = np.bincount(msg_ids[mask], minlength=vocab_size)
        dominant = c.argmax()
        cons = c[dominant] / c.sum()
        lab_name = "A heavier" if lab == 0 else "B heavier"
        print(f"│  {lab_name}: dominant=msg{dominant} ({cons:.0%}), "
              f"dist={c.tolist()}", flush=True)

    a_heavy_msgs = msg_ids[labels_np == 0]
    a_light_msgs = msg_ids[labels_np == 1]
    print(f"│  A watches METAL:  msg dist={np.bincount(a_heavy_msgs, minlength=vocab_size).tolist()}", flush=True)
    print(f"│  A watches RUBBER: msg dist={np.bincount(a_light_msgs, minlength=vocab_size).tolist()}", flush=True)

    heavy_dominant = np.bincount(a_heavy_msgs, minlength=vocab_size).argmax()
    light_dominant = np.bincount(a_light_msgs, minlength=vocab_size).argmax()
    if heavy_dominant != light_dominant:
        print(f"│  ** Messages differentiate mass: metal→msg{heavy_dominant}, rubber→msg{light_dominant} **", flush=True)
    else:
        print(f"│  Messages NOT differentiating mass (both→msg{heavy_dominant})", flush=True)

    consistency_scores = []
    for lab in range(2):
        mask = labels_np == lab
        if mask.sum() >= 2:
            c = np.bincount(msg_ids[mask], minlength=vocab_size)
            cons = c.max() / c.sum()
            consistency_scores.append(cons)
    mean_consistency = np.mean(consistency_scores) if consistency_scores else 0

    print(f"│  Mean consistency: {mean_consistency:.2f}", flush=True)
    print(f"└─ Stage 3 done [{time.time()-ts3:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 4: Visualization
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Visualization", flush=True)
    print(f"{'=' * 60}", flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Accuracy over training
    ax = axes[0, 0]
    epochs_logged = [1] + list(range(10, comm_epochs + 1, 10))
    ax.plot(epochs_logged, history['val_comm_acc'], 'b-', linewidth=2,
            label=f'With comm ({final_val_comm*100:.0f}%)')
    ax.plot(epochs_logged, history['val_nocomm_acc'], 'r--', linewidth=2,
            label=f'No comm ({final_val_nocomm*100:.0f}%)')
    ax.plot(epochs_logged, history['val_oracle_acc'], 'g:', linewidth=2,
            label=f'Oracle ({final_val_oracle*100:.0f}%)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5,
               label='Chance (50%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Mass Comparison: Physics Features', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 1.05)

    # Panel 2: Message distribution by mass
    ax = axes[0, 1]
    bar_width = 0.35
    x = np.arange(vocab_size)
    c_heavy = np.bincount(a_heavy_msgs, minlength=vocab_size).astype(float)
    c_light = np.bincount(a_light_msgs, minlength=vocab_size).astype(float)
    c_heavy = c_heavy / max(c_heavy.sum(), 1)
    c_light = c_light / max(c_light.sum(), 1)
    ax.bar(x - bar_width/2, c_heavy, bar_width, label='A=metal (heavy)',
           color='steelblue', alpha=0.8)
    ax.bar(x + bar_width/2, c_light, bar_width, label='A=rubber (light)',
           color='coral', alpha=0.8)
    ax.set_xlabel('Message Symbol')
    ax.set_ylabel('Frequency')
    ax.set_title('Messages by Agent A\'s Object Mass', fontsize=11)
    ax.set_xticks(range(vocab_size))
    ax.legend(fontsize=8)

    # Panel 3: Feature distributions (heavy vs light)
    ax = axes[1, 0]
    target_feat_names = ['spd_pre', 'spd_post', 'spd_ratio', 'delta_v', 'deflect']
    heavy_means = [heavy_feats[:, i].mean() for i in range(5)]
    light_means = [light_feats[:, i].mean() for i in range(5)]
    x_feat = np.arange(5)
    ax.bar(x_feat - 0.2, heavy_means, 0.4, label='Metal (heavy)', color='steelblue')
    ax.bar(x_feat + 0.2, light_means, 0.4, label='Rubber (light)', color='coral')
    ax.set_xticks(x_feat)
    ax.set_xticklabels(target_feat_names, fontsize=8)
    ax.set_ylabel('Mean Value')
    ax.set_title('Physics Features: Heavy vs Light Objects', fontsize=11)
    ax.legend(fontsize=8)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    elapsed = time.time() - t0

    if comm_gain >= 0.15 and final_val_comm > 0.65:
        verdict = "SUCCESS"
    elif comm_gain >= 0.05 or final_val_comm > 0.60:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 49b: Physics Features\n\n"
        f"Task: which object is heavier?\n"
        f"  Agent A watches collision of obj 1\n"
        f"  Agent B watches collision of obj 2\n"
        f"  11 physics features per collision\n\n"
        f"Data: {n_examples} examples ({n_pairs_total} pairs)\n"
        f"  Train: {n_train}, Val: {n_val}\n"
        f"  From {n_videos_with_pairs} videos\n\n"
        f"Channel: 1x Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Val accuracy:\n"
        f"  With comm:    {final_val_comm*100:.1f}%\n"
        f"  Without comm: {final_val_nocomm*100:.1f}%\n"
        f"  Oracle:       {final_val_oracle*100:.1f}%\n"
        f"  Gain:         {comm_gain*100:+.1f}pp\n\n"
        f"Message entropy: {final_entropy:.3f}\n"
        f"Symbols used: {(msg_probs > 0.01).sum()}/{vocab_size}\n"
        f"Consistency: {mean_consistency:.2f}\n\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 49b: Mass Communication — Physics Features\n'
                 f'val: comm={final_val_comm*100:.0f}% nocomm={final_val_nocomm*100:.0f}% '
                 f'(+{comm_gain*100:.0f}pp) '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} | {verdict}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase49b_physics_features.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase49b_physics_features.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val with communication:    {final_val_comm*100:.1f}% (target >65%)", flush=True)
    print(f"  Val without communication: {final_val_nocomm*100:.1f}% (target ~50%)", flush=True)
    print(f"  Val oracle:                {final_val_oracle*100:.1f}% (target >80%)", flush=True)
    print(f"  Communication gain:        {comm_gain*100:+.1f}pp (target >15pp)", flush=True)
    print(f"  Message entropy:           {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

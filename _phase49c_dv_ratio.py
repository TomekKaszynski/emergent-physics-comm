def run_phase49c_dv_ratio():
    """Phase 49c: Mass inference via delta-v ratio.

    Same task as 49/49b. Key change: features focus on within-collision
    deflection asymmetry. dv_ratio = |Δv_target| / |Δv_partner| is THE
    mass signal. Heavy → <1, Light → >1.

    Prefer cross-material collisions (metal hitting rubber) where signal
    is strongest.

    7 features per agent: dv_ratio, speed_change_target, speed_change_partner,
    deflection_target, deflection_partner, dv_target, dv_partner.
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
    print("PHASE 49c: Mass Communication — Delta-V Ratio", flush=True)
    print("  dv_ratio = |Δv_target| / |Δv_partner| — heavy<1, light>1", flush=True)
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
    eps = 1e-4

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
    # STAGE 1: Build dataset with delta-v ratio features
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Build dataset with delta-v ratio features", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    def get_obj_pos_2d(traj, frame_idx, obj_id):
        for o in traj[frame_idx]['objects']:
            if o['object_id'] == obj_id:
                return np.array(project_3d_to_2d(
                    o['location'][0], o['location'][1]))
        return None

    def compute_dv_features(traj, cf, target_id, partner_id):
        """Compute 7 delta-v ratio features for a collision.

        Returns [7]: dv_ratio, speed_change_target, speed_change_partner,
                     deflection_target, deflection_partner, dv_target, dv_partner
        """
        frames_pre = list(range(max(0, cf - collision_window), cf + 1))
        frames_post = list(range(cf, min(n_eval_frames, cf + collision_window + 1)))

        if len(frames_pre) < 2 or len(frames_post) < 2:
            return None

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

        def obj_dv_features(pos_pre, pos_post):
            """Compute per-object collision features."""
            vel_pre = np.diff(pos_pre, axis=0)   # [T-1, 2]
            vel_post = np.diff(pos_post, axis=0)  # [T-1, 2]

            speeds_pre = np.linalg.norm(vel_pre, axis=1)
            speeds_post = np.linalg.norm(vel_post, axis=1)

            speed_pre = speeds_pre.mean()
            speed_post = speeds_post.mean()
            speed_change = speed_post / (speed_pre + eps)

            # Velocity just before and just after collision
            v_before = vel_pre[-1]
            v_after = vel_post[0]
            dv = np.linalg.norm(v_after - v_before)

            # Deflection angle
            angle_before = np.arctan2(v_before[1], v_before[0])
            angle_after = np.arctan2(v_after[1], v_after[0])
            deflection = abs(angle_after - angle_before)
            if deflection > np.pi:
                deflection = 2 * np.pi - deflection

            return dv, speed_change, deflection

        dv_t, sc_t, defl_t = obj_dv_features(pos_pre_t, pos_post_t)
        dv_p, sc_p, defl_p = obj_dv_features(pos_pre_p, pos_post_p)

        # THE mass signal: heavy target deflects less → ratio < 1
        dv_ratio = dv_t / (dv_p + eps)

        return np.array([dv_ratio, sc_t, sc_p, defl_t, defl_p, dv_t, dv_p])

    # Build collision index with material info
    agent_a_list = []
    agent_b_list = []
    label_list = []
    is_train_list = []

    n_pairs_total = 0
    n_videos_with_pairs = 0
    n_cross_material_a = 0
    n_same_material_a = 0
    n_cross_material_b = 0
    n_same_material_b = 0

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

        # Find collisions per object with partner material info
        # obj_id → list of (collision_frame, partner_id, is_cross_material)
        obj_collisions = {}
        for coll in ann.get('collision', []):
            cf = coll['frame_id']
            oid_a, oid_b = coll['object_ids']
            mat_a = obj_materials.get(oid_a, 'unknown')
            mat_b = obj_materials.get(oid_b, 'unknown')
            is_cross = (mat_a != mat_b and mat_a != 'unknown' and mat_b != 'unknown')

            if oid_a not in obj_collisions:
                obj_collisions[oid_a] = []
            if oid_b not in obj_collisions:
                obj_collisions[oid_b] = []
            obj_collisions[oid_a].append((cf, oid_b, is_cross))
            obj_collisions[oid_b].append((cf, oid_a, is_cross))

        def best_collision(obj_id):
            """Pick best collision: prefer cross-material, then first available."""
            if obj_id not in obj_collisions:
                return None
            colls = obj_collisions[obj_id]
            # Sort: cross-material first
            cross = [c for c in colls if c[2]]
            if cross:
                return cross[0][0], cross[0][1], True
            return colls[0][0], colls[0][1], False

        # Find pairs of objects with different materials
        obj_ids = list(obj_materials.keys())
        video_had_pair = False

        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                oid1, oid2 = obj_ids[i], obj_ids[j]
                mat1, mat2 = obj_materials[oid1], obj_materials[oid2]

                if mat1 == mat2 or mat1 == 'unknown' or mat2 == 'unknown':
                    continue

                bc1 = best_collision(oid1)
                bc2 = best_collision(oid2)
                if bc1 is None or bc2 is None:
                    continue

                if mat1 == 'metal' and mat2 == 'rubber':
                    heavy_id, light_id = oid1, oid2
                    heavy_cf, heavy_partner, heavy_cross = bc1
                    light_cf, light_partner, light_cross = bc2
                elif mat2 == 'metal' and mat1 == 'rubber':
                    heavy_id, light_id = oid2, oid1
                    heavy_cf, heavy_partner, heavy_cross = bc2
                    light_cf, light_partner, light_cross = bc1
                else:
                    continue

                heavy_feats = compute_dv_features(traj, heavy_cf, heavy_id, heavy_partner)
                light_feats = compute_dv_features(traj, light_cf, light_id, light_partner)

                if heavy_feats is None or light_feats is None:
                    continue

                video_had_pair = True

                # Version 1: A=heavy, B=light → label=0
                agent_a_list.append(heavy_feats)
                agent_b_list.append(light_feats)
                label_list.append(0)
                is_train_list.append(vi < n_train_vids)

                # Version 2: A=light, B=heavy → label=1
                agent_a_list.append(light_feats)
                agent_b_list.append(heavy_feats)
                label_list.append(1)
                is_train_list.append(vi < n_train_vids)

                n_pairs_total += 1
                if heavy_cross:
                    n_cross_material_a += 1
                else:
                    n_same_material_a += 1
                if light_cross:
                    n_cross_material_b += 1
                else:
                    n_same_material_b += 1

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
    input_dim = agent_a_inputs.shape[1]  # 7

    print(f"│  Videos with valid pairs: {n_videos_with_pairs}/{n_videos}", flush=True)
    print(f"│  Object pairs: {n_pairs_total}", flush=True)
    print(f"│  Total examples: {n_examples}", flush=True)
    print(f"│  Train: {n_train}, Val: {n_val}", flush=True)
    print(f"│  Label balance: A-heavier={label_counts[0]}, B-heavier={label_counts[1]}", flush=True)
    print(f"│  Feature dim per agent: {input_dim}", flush=True)
    print(f"│  Heavy obj collisions: {n_cross_material_a} cross-material, {n_same_material_a} same-material", flush=True)
    print(f"│  Light obj collisions: {n_cross_material_b} cross-material, {n_same_material_b} same-material", flush=True)

    # THE CRITICAL DIAGNOSTIC: dv_ratio for heavy vs light targets
    heavy_feats = agent_a_inputs[labels == 0].numpy()  # A=heavy
    light_feats = agent_a_inputs[labels == 1].numpy()  # A=light
    feat_names = ['dv_ratio', 'sc_target', 'sc_partner', 'defl_target',
                  'defl_partner', 'dv_target', 'dv_partner']

    print(f"│", flush=True)
    print(f"│  === MASS SIGNAL DIAGNOSTIC ===", flush=True)
    for fi, name in enumerate(feat_names):
        h_mean = heavy_feats[:, fi].mean()
        h_med = np.median(heavy_feats[:, fi])
        l_mean = light_feats[:, fi].mean()
        l_med = np.median(light_feats[:, fi])
        print(f"│    {name:14s}: heavy mean={h_mean:.4f} med={h_med:.4f}  "
              f"light mean={l_mean:.4f} med={l_med:.4f}", flush=True)

    h_dv_ratio_mean = heavy_feats[:, 0].mean()
    l_dv_ratio_mean = light_feats[:, 0].mean()
    if h_dv_ratio_mean < l_dv_ratio_mean * 0.9:
        print(f"│  ** SIGNAL EXISTS: heavy dv_ratio ({h_dv_ratio_mean:.3f}) < light dv_ratio ({l_dv_ratio_mean:.3f}) **", flush=True)
    else:
        print(f"│  WARNING: No clear signal. heavy dv_ratio ({h_dv_ratio_mean:.3f}) ≈ light dv_ratio ({l_dv_ratio_mean:.3f})", flush=True)

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

    a_flat = agent_a_inputs.to(device)  # [N, 7]
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

            msg, logits = sender(a_flat[batch], tau=g_tau)
            pred = receiver(b_flat[batch], msg)
            loss_comm = F.cross_entropy(pred, labels_dev[batch])
            comm_optimizer.zero_grad()
            loss_comm.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

            pred_nc = receiver_nocomm(b_flat[batch])
            loss_nc = F.cross_entropy(pred_nc, labels_dev[batch])
            nocomm_optimizer.zero_grad()
            loss_nc.backward()
            torch.nn.utils.clip_grad_norm_(receiver_nocomm.parameters(), 1.0)
            nocomm_optimizer.step()

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
    }, str(OUTPUT_DIR / "phase49c_model.pt"))

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

    # dv_ratio by message symbol
    print(f"│  Mean dv_ratio by message symbol:", flush=True)
    a_feats_np = agent_a_inputs.numpy()
    for m in range(vocab_size):
        mask_m = msg_ids == m
        if mask_m.sum() > 10:
            mean_dvr = a_feats_np[mask_m, 0].mean()
            print(f"│    msg{m}: dv_ratio={mean_dvr:.3f} (n={mask_m.sum()})", flush=True)

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
    ax.set_title('Mass Comparison: Delta-V Ratio Features', fontsize=11)
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

    # Panel 3: dv_ratio histogram by material
    ax = axes[1, 0]
    # Clip for visualization
    h_dvr = np.clip(heavy_feats[:, 0], 0, 5)
    l_dvr = np.clip(light_feats[:, 0], 0, 5)
    ax.hist(h_dvr, bins=50, alpha=0.6, label=f'Metal (heavy) mean={heavy_feats[:, 0].mean():.2f}',
            color='steelblue', density=True)
    ax.hist(l_dvr, bins=50, alpha=0.6, label=f'Rubber (light) mean={light_feats[:, 0].mean():.2f}',
            color='coral', density=True)
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='ratio=1.0')
    ax.set_xlabel('dv_ratio (|Δv_target| / |Δv_partner|)')
    ax.set_ylabel('Density')
    ax.set_title('Delta-V Ratio: Heavy vs Light Objects', fontsize=11)
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
        f"Phase 49c: Delta-V Ratio\n\n"
        f"Task: which object is heavier?\n"
        f"  dv_ratio = |Δv_t|/|Δv_p|\n"
        f"  heavy→<1, light→>1\n\n"
        f"Data: {n_examples} examples ({n_pairs_total} pairs)\n"
        f"  Train: {n_train}, Val: {n_val}\n"
        f"  Cross-mat colls: heavy={n_cross_material_a},\n"
        f"    light={n_cross_material_b}\n\n"
        f"dv_ratio signal:\n"
        f"  Heavy mean: {heavy_feats[:, 0].mean():.3f}\n"
        f"  Light mean: {light_feats[:, 0].mean():.3f}\n\n"
        f"Channel: 1x Gumbel-Softmax, vocab={vocab_size}\n\n"
        f"Val accuracy:\n"
        f"  With comm:    {final_val_comm*100:.1f}%\n"
        f"  Without comm: {final_val_nocomm*100:.1f}%\n"
        f"  Oracle:       {final_val_oracle*100:.1f}%\n"
        f"  Gain:         {comm_gain*100:+.1f}pp\n\n"
        f"Entropy: {final_entropy:.3f}\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 49c: Mass Communication — Delta-V Ratio\n'
                 f'val: comm={final_val_comm*100:.0f}% nocomm={final_val_nocomm*100:.0f}% '
                 f'(+{comm_gain*100:.0f}pp) '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} | {verdict}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase49c_dv_ratio.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase49c_dv_ratio.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val with communication:    {final_val_comm*100:.1f}% (target >65%)", flush=True)
    print(f"  Val without communication: {final_val_nocomm*100:.1f}% (target ~50%)", flush=True)
    print(f"  Val oracle:                {final_val_oracle*100:.1f}% (target >80%)", flush=True)
    print(f"  Communication gain:        {comm_gain*100:+.1f}pp (target >15pp)", flush=True)
    print(f"  Message entropy:           {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

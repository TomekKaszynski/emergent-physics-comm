def run_phase49_mass_communication():
    """Phase 49: Mass inference from collision dynamics.

    Two agents each watch a different collision involving a different object.
    One object is metal (heavy), one is rubber (light). Neither agent can
    see the other's collision. They communicate to agree on which is heavier.
    Mass is invisible — inferred only from how objects deflect in collisions.
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
    print("PHASE 49: Mass Communication — Who Is Heavier?", flush=True)
    print("  Two agents, two collisions, invisible mass, communication required", flush=True)
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
    # STAGE 1: Build mass-comparison dataset
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Build mass-comparison dataset", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    window_len = collision_window + 1  # 9 frames each half
    total_frames = 2 * window_len - 1  # 17 frames (pre: 9, post: 8, overlap at collision frame)
    feat_per_frame = 5  # x, y, dx, dy, speed

    def get_obj_pos_2d(traj, frame_idx, obj_id):
        for o in traj[frame_idx]['objects']:
            if o['object_id'] == obj_id:
                return np.array(project_3d_to_2d(
                    o['location'][0], o['location'][1]))
        return None

    def get_obj_material(ann, obj_id):
        """Get material from object_property list."""
        for o in ann.get('object_property', []):
            if o['object_id'] == obj_id:
                return o.get('material', None)
        return None

    def compute_trajectory_features(pos_seq):
        """Convert position sequence [T, 2] to [T, 5] with velocity/speed."""
        T = len(pos_seq)
        features = np.zeros((T, feat_per_frame))
        features[:, :2] = pos_seq
        for t in range(T - 1):
            dx = pos_seq[t + 1, 0] - pos_seq[t, 0]
            dy = pos_seq[t + 1, 1] - pos_seq[t, 1]
            features[t, 2] = dx
            features[t, 3] = dy
            features[t, 4] = np.sqrt(dx**2 + dy**2)
        if T > 1:
            features[-1, 2:] = features[-2, 2:]
        return features

    def pad_or_truncate(seq, target_len):
        if len(seq) >= target_len:
            return seq[:target_len]
        pad = np.zeros((target_len - len(seq), seq.shape[1]))
        return np.concatenate([pad, seq], axis=0)

    def extract_collision_features(traj, collision_frame, target_obj_id, partner_obj_id):
        """Extract trajectory features for a collision.

        Returns [34, 5]: target (17 frames) + partner (17 frames).
        Target object listed first.
        """
        cf = collision_frame
        start = max(0, cf - collision_window)
        end = min(n_eval_frames, cf + collision_window + 1)
        mid_idx = cf - start

        pos_target = []
        pos_partner = []
        for fi in range(start, end):
            pt = get_obj_pos_2d(traj, fi, target_obj_id)
            pp = get_obj_pos_2d(traj, fi, partner_obj_id)
            if pt is None or pp is None:
                return None
            pos_target.append(pt)
            pos_partner.append(pp)

        if len(pos_target) < mid_idx + 1:
            return None

        pos_target = np.stack(pos_target)
        pos_partner = np.stack(pos_partner)

        feat_target = compute_trajectory_features(pos_target)
        feat_partner = compute_trajectory_features(pos_partner)

        # Pre + post for target
        pre_t = pad_or_truncate(feat_target[:mid_idx + 1], window_len)   # [9, 5]
        post_t = pad_or_truncate(feat_target[mid_idx:], window_len)      # [9, 5]
        # Pre + post for partner
        pre_p = pad_or_truncate(feat_partner[:mid_idx + 1], window_len)  # [9, 5]
        post_p = pad_or_truncate(feat_partner[mid_idx:], window_len)     # [9, 5]

        # Stack: target pre+post, then partner pre+post → [36, 5]
        return np.concatenate([pre_t, post_t, pre_p, post_p], axis=0)

    # For each video: find object materials and collisions per object
    agent_a_list = []
    agent_b_list = []
    label_list = []
    is_train_list = []
    meta_list = []

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

                # Need different materials
                if mat1 == mat2 or mat1 == 'unknown' or mat2 == 'unknown':
                    continue

                # Both must have at least one collision
                if oid1 not in obj_collisions or oid2 not in obj_collisions:
                    continue

                # Determine which is heavier (metal > rubber)
                if mat1 == 'metal' and mat2 == 'rubber':
                    heavy_id, light_id = oid1, oid2
                elif mat2 == 'metal' and mat1 == 'rubber':
                    heavy_id, light_id = oid2, oid1
                else:
                    continue

                # Pick first collision for each object
                heavy_cf, heavy_partner = obj_collisions[heavy_id][0]
                light_cf, light_partner = obj_collisions[light_id][0]

                # Extract features
                heavy_feats = extract_collision_features(
                    traj, heavy_cf, heavy_id, heavy_partner)
                light_feats = extract_collision_features(
                    traj, light_cf, light_id, light_partner)

                if heavy_feats is None or light_feats is None:
                    continue

                video_had_pair = True

                # Version 1: A=heavy, B=light → label=0 (A is heavier)
                agent_a_list.append(heavy_feats)
                agent_b_list.append(light_feats)
                label_list.append(0)
                is_train_list.append(vi < n_train_vids)
                meta_list.append({
                    'vid_id': vid_id, 'heavy_id': heavy_id,
                    'light_id': light_id, 'version': 'A=heavy'
                })

                # Version 2: A=light, B=heavy → label=1 (B is heavier)
                agent_a_list.append(light_feats)
                agent_b_list.append(heavy_feats)
                label_list.append(1)
                is_train_list.append(vi < n_train_vids)
                meta_list.append({
                    'vid_id': vid_id, 'heavy_id': heavy_id,
                    'light_id': light_id, 'version': 'A=light'
                })

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

    print(f"│  Videos with valid pairs: {n_videos_with_pairs}/{n_videos}", flush=True)
    print(f"│  Object pairs: {n_pairs_total}", flush=True)
    print(f"│  Total examples: {n_examples} (×2 for both orderings)", flush=True)
    print(f"│  Train: {n_train}, Val: {n_val}", flush=True)
    print(f"│  Label balance: A-heavier={label_counts[0]}, B-heavier={label_counts[1]}", flush=True)
    print(f"│  Agent A input: {agent_a_inputs.shape}", flush=True)
    print(f"│  Agent B input: {agent_b_inputs.shape}", flush=True)
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
    n_classes = 2  # binary: A heavier or B heavier

    input_dim = agent_a_inputs.shape[1] * agent_a_inputs.shape[2]  # 36 * 5 = 180

    print(f"│  Input dim per agent: {input_dim} ({agent_a_inputs.shape[1]}x{agent_a_inputs.shape[2]})", flush=True)
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

        perm = train_idx[torch.randperm(n_train)]

        sender.train(); receiver.train()
        receiver_nocomm.train(); oracle.train()

        ep_comm_correct = 0
        ep_nocomm_correct = 0
        ep_oracle_correct = 0
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

            # Oracle: sees both collisions
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
                ep_oracle_correct += (pred_or.argmax(1) == labels_dev[batch]).sum().item()
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
    }, str(OUTPUT_DIR / "phase49_model.pt"))

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

    # Message distribution by label
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

    # Message by material of Agent A's target
    # When label=0 (A heavier), A watches heavy (metal) object
    # When label=1 (B heavier), A watches light (rubber) object
    a_heavy_msgs = msg_ids[labels_np == 0]  # A has metal
    a_light_msgs = msg_ids[labels_np == 1]  # A has rubber
    print(f"│  A watches METAL:  msg dist={np.bincount(a_heavy_msgs, minlength=vocab_size).tolist()}", flush=True)
    print(f"│  A watches RUBBER: msg dist={np.bincount(a_light_msgs, minlength=vocab_size).tolist()}", flush=True)

    # Are messages encoding mass?
    # If sender learned: "my object is heavy" → msg X, "my object is light" → msg Y
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
    ax.set_title('Mass Comparison: Who Is Heavier?', fontsize=11)
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

    # Panel 3: Entropy + tau over training
    ax = axes[1, 0]
    ax.plot(epochs_logged, history['msg_entropy'], 'purple', linewidth=2,
            label='Message entropy')
    ax2 = ax.twinx()
    ax2.plot(epochs_logged, history['gumbel_tau'], 'orange', linewidth=1,
             alpha=0.5, label='Gumbel τ')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy (normalized)', color='purple')
    ax2.set_ylabel('Gumbel τ', color='orange')
    ax.set_title('Channel Health', fontsize=11)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

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
        f"Phase 49: Mass Communication\n\n"
        f"Task: which object is heavier?\n"
        f"  Agent A watches collision of obj 1\n"
        f"  Agent B watches collision of obj 2\n"
        f"  Mass is invisible (metal vs rubber)\n\n"
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

    fig.suptitle(f'Phase 49: Mass Communication — Who Is Heavier?\n'
                 f'val: comm={final_val_comm*100:.0f}% nocomm={final_val_nocomm*100:.0f}% '
                 f'(+{comm_gain*100:.0f}pp) '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} | {verdict}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase49_mass_communication.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase49_mass_communication.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val with communication:    {final_val_comm*100:.1f}% (target >65%)", flush=True)
    print(f"  Val without communication: {final_val_nocomm*100:.1f}% (target ~50%)", flush=True)
    print(f"  Val oracle:                {final_val_oracle*100:.1f}% (target >80%)", flush=True)
    print(f"  Communication gain:        {comm_gain*100:+.1f}pp (target >15pp)", flush=True)
    print(f"  Message entropy:           {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

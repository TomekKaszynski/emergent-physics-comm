def run_phase48g_entropy_reg():
    """Phase 48g: Multi-token messages with entropy regularization.

    Identical to 48f with two changes:
    1. gumbel_tau_end = 1.0 (was 0.5) — prevent temperature collapse
    2. loss = task_loss - 0.1 * msg_entropy — keep channel alive
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
    print("PHASE 48g: Multi-Token + Entropy Reg — 1000 CLEVRER Videos", flush=True)
    print("  tau_end=1.0, loss = task - 0.1*entropy", flush=True)
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
    n_train_vids = 800

    PROJ_U = np.array([0.0589, 0.2286, 0.4850])
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])

    def project_3d_to_2d(x, y):
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load annotations + extract collision metadata
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
        vi = vid_id - 10000

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
    # STAGE 1: Extract GT trajectory features at collision windows
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Extract GT trajectory features", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    window_len = collision_window + 1  # 9 frames each half
    feat_per_frame = 5  # x, y, dx, dy, speed

    def compute_trajectory_features(pos_seq):
        """Convert position sequence [T, 2] to [T, 5] with velocity/speed."""
        T = len(pos_seq)
        features = np.zeros((T, feat_per_frame))
        features[:, :2] = pos_seq  # x, y

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

    agent_a_list = []
    agent_b_list = []
    label_list = []
    is_train_list = []

    for coll in collision_events:
        vid_id = coll['video_id']
        ann = annotations[vid_id]
        traj = ann['motion_trajectory']
        cf = coll['collision_frame']
        oid_a = coll['obj_a_id']
        oid_b = coll['obj_b_id']

        start = max(0, cf - collision_window)
        end = min(n_eval_frames, cf + collision_window + 1)
        mid_idx = cf - start

        # Extract GT positions for both objects across window
        pos_a_seq = []
        pos_b_seq = []
        valid = True
        for fi in range(start, end):
            pa = get_obj_pos_2d(traj, fi, oid_a)
            pb = get_obj_pos_2d(traj, fi, oid_b)
            if pa is None or pb is None:
                valid = False
                break
            pos_a_seq.append(pa)
            pos_b_seq.append(pb)

        if not valid or len(pos_a_seq) < mid_idx + 1:
            continue

        pos_a_seq = np.stack(pos_a_seq)  # [T, 2]
        pos_b_seq = np.stack(pos_b_seq)  # [T, 2]

        # Compute trajectory features (pos + vel + speed)
        feat_a = compute_trajectory_features(pos_a_seq)  # [T, 5]
        feat_b = compute_trajectory_features(pos_b_seq)  # [T, 5]

        # Split pre/post
        pre_a = pad_or_truncate(feat_a[:mid_idx + 1], window_len)   # [9, 5]
        pre_b = pad_or_truncate(feat_b[:mid_idx + 1], window_len)   # [9, 5]
        post_a = pad_or_truncate(feat_a[mid_idx:], window_len)      # [9, 5]
        post_b = pad_or_truncate(feat_b[mid_idx:], window_len)      # [9, 5]

        # Agent A: full collision (pre+post for both objects) → [36, 5]
        a_in = np.concatenate([pre_a, post_a, pre_b, post_b], axis=0)
        agent_a_list.append(a_in)

        # Agent B: object A pre-collision only → [9, 5]
        b_in = pre_a
        agent_b_list.append(b_in)

        label_list.append(coll['direction_bin'])
        is_train_list.append(coll['is_train'])

    agent_a_inputs = torch.tensor(np.stack(agent_a_list), dtype=torch.float32)
    agent_b_inputs = torch.tensor(np.stack(agent_b_list), dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.long)
    is_train = torch.tensor(is_train_list, dtype=torch.bool)

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
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Train communication agents
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Train communication agents", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    vocab_size = 16
    msg_len = 4
    hidden_dim = 128
    comm_epochs = 200
    comm_lr = 3e-4
    comm_batch = 64
    gumbel_tau_start = 2.0
    gumbel_tau_end = 1.0  # CHANGE 1: was 0.5 in 48f
    entropy_reg_weight = 0.1  # CHANGE 2: entropy regularization
    msg_dim = msg_len * vocab_size  # 4 * 16 = 64

    a_input_dim = agent_a_inputs.shape[1] * agent_a_inputs.shape[2]  # 36 * 5 = 180
    b_input_dim = agent_b_inputs.shape[1] * agent_b_inputs.shape[2]  # 9 * 5 = 45

    print(f"│  Agent A input dim: {a_input_dim} ({agent_a_inputs.shape[1]}x{agent_a_inputs.shape[2]})", flush=True)
    print(f"│  Agent B input dim: {b_input_dim} ({agent_b_inputs.shape[1]}x{agent_b_inputs.shape[2]})", flush=True)
    print(f"│  Message: {msg_len} tokens x vocab {vocab_size} = {vocab_size**msg_len} possible", flush=True)
    print(f"│  Gumbel tau: {gumbel_tau_start} → {gumbel_tau_end}", flush=True)
    print(f"│  Entropy reg: {entropy_reg_weight}", flush=True)

    class SenderAgent(nn.Module):
        def __init__(self, input_dim, hidden_dim, vocab_size, msg_len):
            super().__init__()
            self.vocab_size = vocab_size
            self.msg_len = msg_len
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, msg_len * vocab_size),
            )
        def forward(self, x, tau=1.0):
            logits = self.encoder(x)  # [B, msg_len * vocab_size]
            logits = logits.reshape(-1, self.msg_len, self.vocab_size)  # [B, msg_len, vocab_size]
            if self.training:
                tokens = []
                for t in range(self.msg_len):
                    tokens.append(F.gumbel_softmax(logits[:, t, :], tau=tau, hard=True))
                message = torch.cat(tokens, dim=-1)  # [B, msg_len * vocab_size]
            else:
                idx = logits.argmax(dim=-1)  # [B, msg_len]
                tokens = F.one_hot(idx, self.vocab_size).float()  # [B, msg_len, vocab_size]
                message = tokens.reshape(-1, self.msg_len * self.vocab_size)
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
    sender = SenderAgent(a_input_dim, hidden_dim, vocab_size, msg_len).to(device)
    receiver = ReceiverAgent(b_input_dim, msg_dim, hidden_dim, n_direction_bins).to(device)
    receiver_nocomm = ReceiverNoComm(b_input_dim, hidden_dim, n_direction_bins).to(device)
    oracle = OraclePredictor(a_input_dim, hidden_dim, n_direction_bins).to(device)

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=comm_lr)
    nocomm_optimizer = torch.optim.Adam(receiver_nocomm.parameters(), lr=comm_lr)
    oracle_optimizer = torch.optim.Adam(oracle.parameters(), lr=comm_lr)

    print(f"│  Task: {n_direction_bins}-class direction "
          f"(chance={100/n_direction_bins:.1f}%)", flush=True)
    print(f"│  Vocab: {vocab_size}, MsgLen: {msg_len}, Epochs: {comm_epochs}, "
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
            loss_task = F.cross_entropy(pred, labels_dev[batch])

            # CHANGE 2: Compute per-token entropy from logits and subtract from loss
            # logits shape: [B, msg_len, vocab_size]
            token_probs = F.softmax(logits, dim=-1)  # [B, msg_len, vocab_size]
            token_entropy = -(token_probs * torch.log(token_probs + 1e-8)).sum(dim=-1)  # [B, msg_len]
            # Normalize by log(vocab_size) so entropy is in [0, 1]
            token_entropy = token_entropy / np.log(vocab_size)
            batch_entropy = token_entropy.mean()  # scalar

            loss_comm = loss_task - entropy_reg_weight * batch_entropy

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
                # Per-token entropy (average across tokens)
                all_msg_reshaped = all_msg.reshape(-1, msg_len, vocab_size)
                token_ids = all_msg_reshaped.argmax(dim=-1).cpu().numpy()  # [N, msg_len]
                per_token_ent = []
                for t in range(msg_len):
                    counts = np.bincount(token_ids[:, t], minlength=vocab_size).astype(float)
                    probs = counts / counts.sum()
                    ent = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
                    per_token_ent.append(ent)
                msg_entropy = np.mean(per_token_ent)

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

        val_or = oracle(a_flat[val_idx])
        final_val_oracle = (val_or.argmax(1) == labels_dev[val_idx]).float().mean().item()

        all_msg, _ = sender(a_flat)
        all_msg_reshaped = all_msg.reshape(-1, msg_len, vocab_size)
        token_ids = all_msg_reshaped.argmax(dim=-1).cpu().numpy()  # [N, msg_len]

        # Per-token entropy
        per_token_ent = []
        per_token_used = []
        for t in range(msg_len):
            counts = np.bincount(token_ids[:, t], minlength=vocab_size).astype(float)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
            per_token_ent.append(ent)
            per_token_used.append((probs > 0.01).sum())
        final_entropy = np.mean(per_token_ent)

        # Unique messages
        msg_tuples = [tuple(row) for row in token_ids]
        n_unique = len(set(msg_tuples))

        # For message analysis: encode as single int
        msg_ids = np.zeros(len(token_ids), dtype=int)
        for t in range(msg_len):
            msg_ids += token_ids[:, t] * (vocab_size ** t)

    comm_gain = final_val_comm - final_val_nocomm

    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'receiver_nocomm': receiver_nocomm.state_dict(),
        'oracle': oracle.state_dict(),
    }, str(OUTPUT_DIR / "phase48g_model.pt"))

    print(f"\n│  === RESULTS (validation) ===", flush=True)
    print(f"│  With communication:    {final_val_comm*100:.1f}%", flush=True)
    print(f"│  Without communication: {final_val_nocomm*100:.1f}%", flush=True)
    print(f"│  Oracle (A sees all):   {final_val_oracle*100:.1f}%", flush=True)
    print(f"│  Communication gain:    {comm_gain*100:+.1f}pp", flush=True)
    print(f"│  Chance baseline:       {100/n_direction_bins:.1f}%", flush=True)
    print(f"│  Per-token entropy:     {[f'{e:.3f}' for e in per_token_ent]}", flush=True)
    print(f"│  Mean entropy:          {final_entropy:.3f}", flush=True)
    print(f"│  Per-token symbols used: {per_token_used}", flush=True)
    print(f"│  Unique messages:       {n_unique}/{vocab_size**msg_len}", flush=True)
    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Message analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Message analysis", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    labels_np = labels.numpy()

    # Per-token analysis by direction
    print(f"│  Per-token dominant symbols by direction:", flush=True)
    for d in range(n_direction_bins):
        mask = labels_np == d
        if mask.sum() < 2:
            continue
        token_summary = []
        for t in range(msg_len):
            c = np.bincount(token_ids[mask, t], minlength=vocab_size)
            dominant = c.argmax()
            cons = c[dominant] / c.sum()
            token_summary.append(f"t{t}={dominant}({cons:.0%})")
        print(f"│    {dir_names[d]:2s}: {' '.join(token_summary)}", flush=True)

    # Unique messages per direction
    msg_by_dir = {d: set() for d in range(n_direction_bins)}
    for i in range(n_examples):
        msg_by_dir[labels_np[i]].add(tuple(token_ids[i]))

    print(f"│  Unique messages per direction:", flush=True)
    for d in range(n_direction_bins):
        print(f"│    {dir_names[d]:2s}: {len(msg_by_dir[d])} unique", flush=True)

    # Consistency: fraction of examples with the most common message per direction
    consistency_scores = []
    for d in range(n_direction_bins):
        mask = labels_np == d
        if mask.sum() < 2:
            continue
        msg_strs = [tuple(token_ids[i]) for i in range(n_examples) if mask[i]]
        from collections import Counter
        mc = Counter(msg_strs).most_common(1)[0][1]
        cons = mc / len(msg_strs)
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
    ax.set_title('Post-Collision Direction (Entropy Reg)', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.5, max(
        max(history['val_comm_acc']),
        max(history['val_nocomm_acc']),
        max(history['val_oracle_acc'])) + 0.1))

    # Panel 2: Token 0 direction heatmap (most informative token)
    ax = axes[0, 1]
    msg_matrix = np.zeros((n_direction_bins, vocab_size))
    for d in range(n_direction_bins):
        mask = labels_np == d
        if mask.sum() > 0:
            c = np.bincount(token_ids[mask, 0], minlength=vocab_size).astype(float)
            msg_matrix[d] = c / c.sum()
    im = ax.imshow(msg_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Token 0 Symbol')
    ax.set_ylabel('True Direction')
    ax.set_yticks(range(n_direction_bins))
    ax.set_yticklabels(dir_names)
    ax.set_xticks(range(vocab_size))
    ax.set_title('Token 0 Usage by Direction', fontsize=11)
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
        f"Phase 48g: Entropy Reg Comm\n\n"
        f"Input: GT traj, 4-token msg (vocab=16)\n"
        f"  Agent A: {agent_a_inputs.shape[1]}x{agent_a_inputs.shape[2]} = {a_input_dim}d\n"
        f"  Agent B: {agent_b_inputs.shape[1]}x{agent_b_inputs.shape[2]} = {b_input_dim}d\n\n"
        f"Data: {n_examples} collisions\n"
        f"  Train: {n_train}, Val: {n_val}\n"
        f"Channel: {msg_len}x Gumbel-Softmax, vocab={vocab_size}\n"
        f"  tau: {gumbel_tau_start} → {gumbel_tau_end}\n"
        f"  entropy_reg: {entropy_reg_weight}\n\n"
        f"Val accuracy:\n"
        f"  With comm:    {final_val_comm*100:.1f}%\n"
        f"  Without comm: {final_val_nocomm*100:.1f}%\n"
        f"  Oracle:       {final_val_oracle*100:.1f}%\n"
        f"  Gain:         {comm_gain*100:+.1f}pp\n\n"
        f"Mean token entropy: {final_entropy:.3f}\n"
        f"Unique messages: {n_unique}/{vocab_size**msg_len}\n"
        f"Msg consistency: {mean_consistency:.2f}\n\n"
        f"Total time: {elapsed:.0f}s\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 48g: Entropy Reg Communication (1000 Videos)\n'
                 f'val: comm={final_val_comm*100:.0f}% nocomm={final_val_nocomm*100:.0f}% '
                 f'(+{comm_gain*100:.0f}pp) '
                 f'oracle={final_val_oracle*100:.0f}% '
                 f'ent={final_entropy:.3f} | {verdict}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase48g_entropy_reg.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase48g_entropy_reg.png", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\n  Val with communication:    {final_val_comm*100:.1f}% (target >30%)", flush=True)
    print(f"  Val without communication: {final_val_nocomm*100:.1f}%", flush=True)
    print(f"  Val oracle:                {final_val_oracle*100:.1f}% (target >50%)", flush=True)
    print(f"  Communication gain:        {comm_gain*100:+.1f}pp (target >10pp)", flush=True)
    print(f"  Message entropy:           {final_entropy:.3f} (target >0.3)", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

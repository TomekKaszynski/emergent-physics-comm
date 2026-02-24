def run_phase46_slotcontrast_clevrer():
    """Phase 46: SlotContrast pretrained inference on CLEVRER.

    Zero-shot transfer: Load MOVi-C pretrained SlotContrast model,
    run inference on CLEVRER videos, evaluate tracking.
    No training — just load and evaluate.

    Config:
      - Model: SlotContrast MOVi-C (11 slots, 64-dim, ViT-S/14 DINOv2)
      - Input: 336×336 (matches training), MOVi normalization (mean=0.5, std=0.5)
      - Data: 20 CLEVRER videos × 128 frames
      - Evaluation: tracking error, slot consistency, binding accuracy
    """
    import sys
    import os
    import time
    import json
    import cv2
    import numpy as np
    import torch
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment

    # Add slotcontrast to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'slotcontrast'))
    sys.path.insert(0, 'slotcontrast')

    from slotcontrast import configuration, models
    from slotcontrast.data.transforms import build_inference_transform
    from torchvision import transforms as tvt
    from omegaconf import OmegaConf

    t0 = time.time()
    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ─── Config ───
    n_videos_target = 20
    n_frames = 128
    n_slots = 11
    slot_dim = 64
    patch_size = 14
    img_size = 336  # Match training resolution (decoder has n_patches=576 = 24×24)
    P = img_size // patch_size  # 24
    n_patches = P * P  # 576

    data_dir = "clevrer_data"
    checkpoint_path = "slotcontrast/checkpoints/movi_c.ckpt"
    config_path = "slotcontrast/configs/slotcontrast/movi_c.yaml"

    device = torch.device("cpu")  # CPU for compatibility

    print(f"╔══════════════════════════════════════════════════════════╗", flush=True)
    print(f"║  Phase 46: SlotContrast Pretrained → CLEVRER            ║", flush=True)
    print(f"╚══════════════════════════════════════════════════════════╝", flush=True)
    print(f"│  {n_videos_target} videos × {n_frames} frames", flush=True)
    print(f"│  {n_slots} slots, {slot_dim}-dim, {img_size}px (match training)", flush=True)
    print(f"│  Device: {device}", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 0: Load model
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load SlotContrast model", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts0 = time.time()

    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model loaded: {total_params:,} params", flush=True)
    print(f"│  Backbone: DINOv2 ViT-S/14 (frozen)", flush=True)
    print(f"│  Encoder → SlotAttention (2 iters) → MLP Decoder", flush=True)
    print(f"│  ScanOverTime + TransformerEncoder predictor", flush=True)
    print(f"└─ Stage 0 done [{time.time()-ts0:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Load data + annotations
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Load CLEVRER data", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1 = time.time()

    video_ids_all = list(range(10000, 10100))
    available_ids = []
    for vid_id in video_ids_all:
        ann_path = f"{data_dir}/annotation_{vid_id}.json"
        vid_path = f"{data_dir}/videos/video_{vid_id}.mp4"
        if os.path.exists(ann_path) and os.path.exists(vid_path):
            available_ids.append(vid_id)
    video_ids = available_ids[:n_videos_target]
    n_videos = len(video_ids)
    print(f"│  Found {len(available_ids)} videos, using {n_videos}", flush=True)

    annotations = {}
    for vid_id in video_ids:
        with open(f"{data_dir}/annotation_{vid_id}.json") as f:
            annotations[vid_id] = json.load(f)

    PROJ_U = np.array([0.0589, 0.2286, 0.4850])
    PROJ_V = np.array([0.1562, 0.0105, 0.4506])

    def project_3d_to_2d(x, y):
        u = PROJ_U[0] * x + PROJ_U[1] * y + PROJ_U[2]
        v = PROJ_V[0] * x + PROJ_V[1] * y + PROJ_V[2]
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    # Build inference transform (MOVi normalization: mean=0.5, std=0.5)
    transform_config = OmegaConf.create({
        "use_movi_normalization": True,
        "dataset_type": "video",
        "input_size": img_size,
    })
    tfs = build_inference_transform(transform_config)

    print(f"│  Annotations loaded for {n_videos} videos", flush=True)
    print(f"│  Using MOVi normalization (mean=0.5, std=0.5)", flush=True)
    print(f"└─ Stage 1 done [{time.time()-ts1:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Run inference per video
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Inference ({n_videos} videos × {n_frames} frames)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    # Patch grid for centroid computation
    xs = torch.linspace(0, 1, P)
    ys = torch.linspace(0, 1, P)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(n_patches, 2).numpy()

    all_centroids = []  # [n_videos][n_frames, n_slots, 2]
    all_attns = []      # [n_videos][n_frames, n_slots, n_patches]
    all_slots = []      # [n_videos][n_frames, n_slots, slot_dim]

    for vi, vid_id in enumerate(video_ids):
        vt0 = time.time()

        # Load video frames
        cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
        frames = []
        for fi in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)  # HWC → CHW
            frames.append(frame)
        cap.release()
        actual_frames = len(frames)

        # Stack: [T, 3, H_orig, W_orig]
        video_tensor = torch.tensor(np.stack(frames))  # [T, 3, H, W]

        # SlotContrast expects [3, T, H, W] for video transforms
        video_ct = video_tensor.permute(1, 0, 2, 3)  # [3, T, H, W]
        video_normalized = tfs(video_ct)  # [3, T, 224, 224]
        video_input = video_normalized.permute(1, 0, 2, 3)  # [T, 3, 224, 224]

        # Visualization version (just resized, not normalized)
        video_vis = tvt.Resize((img_size, img_size))(video_tensor)  # [T, 3, 224, 224]
        video_vis_ct = video_vis.permute(1, 0, 2, 3)  # [3, T, 224, 224]

        inputs = {
            "video": video_input.unsqueeze(0).to(device),          # [1, T, 3, 224, 224]
            "video_visualization": video_vis_ct.unsqueeze(0).to(device),  # [1, 3, T, 224, 224]
        }

        with torch.no_grad():
            outputs = model(inputs)

        # Extract slots: [1, T, 11, 64] → [T, 11, 64]
        slots = outputs["processor"]["state"][0].cpu().numpy()

        # Extract attention maps from slot attention: [1, T, 11, n_patches]
        attn = outputs["processor"]["corrector"]["masks"][0].cpu().numpy()

        # Compute centroids from attention weights
        vid_centroids = np.zeros((actual_frames, n_slots, 2))
        for fi in range(actual_frames):
            for si in range(n_slots):
                weights = attn[fi, si, :]  # [n_patches]
                w_sum = weights.sum()
                if w_sum > 1e-6:
                    cx = (weights * grid_positions[:, 0]).sum() / w_sum
                    cy = (weights * grid_positions[:, 1]).sum() / w_sum
                    vid_centroids[fi, si] = [cx, cy]
                else:
                    vid_centroids[fi, si] = [0.5, 0.5]

        all_centroids.append(vid_centroids)
        all_attns.append(attn)
        all_slots.append(slots)

        vt1 = time.time()
        # Quick entropy check
        ownership = attn[64].argmax(axis=0)  # [n_patches] at frame 64
        counts = np.bincount(ownership, minlength=n_slots)
        n_active = (counts > n_patches * 0.02).sum()
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs + 1e-8)) / np.log(n_slots)

        print(f"│  Video {vi+1:2d}/{n_videos} (id={vid_id}): "
              f"{vt1-vt0:.1f}s, {n_active}/{n_slots} active, ent={ent:.3f}",
              flush=True)

        # Memory cleanup
        del inputs, outputs, video_tensor, video_input, video_vis, video_vis_ct
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Hungarian matching + evaluation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Evaluation", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts3 = time.time()

    # --- Temporal Hungarian matching (frame-to-frame) ---
    all_slot_assignments = []
    for vi in range(n_videos):
        centroids = all_centroids[vi]
        assignments = [np.arange(n_slots)]  # frame 0: identity
        for fi in range(1, n_frames):
            prev_c = centroids[fi - 1][assignments[-1]]
            curr_c = centroids[fi]
            cost = np.zeros((n_slots, n_slots))
            for si in range(n_slots):
                for sj in range(n_slots):
                    cost[si, sj] = np.linalg.norm(prev_c[si] - curr_c[sj])
            _, col_ind = linear_sum_assignment(cost)
            assignment = np.zeros(n_slots, dtype=int)
            for r, c in enumerate(col_ind):
                assignment[r] = c
            assignments.append(assignment)
        all_slot_assignments.append(assignments)

    # Apply assignments → reordered centroids
    all_reordered = []
    for vi in range(n_videos):
        reordered = np.zeros_like(all_centroids[vi])
        for fi in range(n_frames):
            perm = all_slot_assignments[vi][fi]
            for s in range(n_slots):
                reordered[fi, s] = all_centroids[vi][fi, perm[s]]
        all_reordered.append(reordered)

    # --- Slot-to-GT assignment at reference frame ---
    tracking_errors = []
    per_frame_errors = [[] for _ in range(n_frames)]  # for temporal degradation
    slot_consistency_per_video = []
    binding_per_video = []
    gt_to_slot_per_video = []

    for vi in range(n_videos):
        vid_id = video_ids[vi]
        ann = annotations[vid_id]
        traj = ann['motion_trajectory']

        # Find reference frame (most objects in view)
        best_frame = 0
        best_in_view = 0
        for fi in range(n_frames):
            n_in = sum(1 for o in traj[fi]['objects'] if o['inside_camera_view'])
            if n_in > best_in_view:
                best_in_view = n_in
                best_frame = fi

        ref_objs = [o for o in traj[best_frame]['objects'] if o['inside_camera_view']]
        gt_positions_ref = np.zeros((len(ref_objs), 2))
        gt_obj_ids = []
        for oi, o in enumerate(ref_objs):
            u, v = project_3d_to_2d(o['location'][0], o['location'][1])
            gt_positions_ref[oi] = [u, v]
            gt_obj_ids.append(o['object_id'])

        # Active slots (coverage > 2% at reference frame)
        perm_ref = all_slot_assignments[vi][best_frame]
        attn_ref = all_attns[vi][best_frame][perm_ref]  # reordered [11, n_patches]
        ownership_ref = attn_ref.argmax(axis=0)
        slot_coverage = np.zeros(n_slots)
        for s in range(n_slots):
            slot_coverage[s] = (ownership_ref == s).sum() / n_patches
        active_slots = [s for s in range(n_slots) if slot_coverage[s] > 0.02]

        # Hungarian matching: GT objects → active slots
        pred_centroids_ref = all_reordered[vi][best_frame]

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

        gt_to_slot_per_video.append(gt_to_slot)

        # --- Evaluate per frame ---
        vid_consistent = 0
        vid_bound = 0
        vid_total = 0

        for fi in range(n_frames):
            frame_objs = [o for o in traj[fi]['objects'] if o['inside_camera_view']]
            if len(frame_objs) == 0:
                continue
            vid_total += 1
            all_match = True
            used_slots = set()

            for o in frame_objs:
                oid = o['object_id']
                if oid not in gt_to_slot:
                    continue
                assigned_slot = gt_to_slot[oid]
                gt_u, gt_v = project_3d_to_2d(o['location'][0], o['location'][1])
                pred_uv = all_reordered[vi][fi, assigned_slot]

                err = np.linalg.norm(np.array([gt_u, gt_v]) - pred_uv)
                tracking_errors.append(err)
                per_frame_errors[fi].append(err)

                # Consistency check
                closest_slot = None
                min_dist = float('inf')
                for s in range(n_slots):
                    d = np.linalg.norm(np.array([gt_u, gt_v]) - all_reordered[vi][fi, s])
                    if d < min_dist:
                        min_dist = d
                        closest_slot = s
                if closest_slot != assigned_slot:
                    all_match = False

                if assigned_slot in used_slots:
                    pass
                used_slots.add(assigned_slot)

            if all_match:
                vid_consistent += 1
            if len(used_slots) == len([o for o in frame_objs if o['object_id'] in gt_to_slot]):
                vid_bound += 1

        cons = vid_consistent / max(vid_total, 1)
        bind = vid_bound / max(vid_total, 1)
        slot_consistency_per_video.append(cons)
        binding_per_video.append(bind)

    # --- Aggregate metrics ---
    mean_tracking_error = np.mean(tracking_errors)
    median_tracking_error = np.median(tracking_errors)
    mean_consistency = np.mean(slot_consistency_per_video) * 100
    mean_binding = np.mean(binding_per_video) * 100

    # Temporal degradation: early vs late
    early_errors = []
    late_errors = []
    for fi in range(n_frames):
        if fi < 16:
            early_errors.extend(per_frame_errors[fi])
        elif fi >= 64:
            late_errors.extend(per_frame_errors[fi])
    early_err = np.mean(early_errors) if early_errors else 0
    late_err = np.mean(late_errors) if late_errors else 0

    # Attention entropy
    all_entropies = []
    for vi in range(n_videos):
        for fi in range(n_frames):
            attn_fi = all_attns[vi][fi]
            ownership = attn_fi.argmax(axis=0)
            counts = np.bincount(ownership, minlength=n_slots)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-8)) / np.log(n_slots)
            all_entropies.append(ent)
    mean_entropy = np.mean(all_entropies)
    n_active_avg = np.mean([
        np.sum(np.bincount(all_attns[vi][64].argmax(0), minlength=n_slots) > n_patches * 0.02)
        for vi in range(n_videos)
    ])

    elapsed = time.time() - t0
    print(f"│  Tracking error: {mean_tracking_error*100:.1f}% "
          f"(median {median_tracking_error*100:.1f}%)", flush=True)
    print(f"│  Early (0-15):   {early_err*100:.1f}%", flush=True)
    print(f"│  Late (64-127):  {late_err*100:.1f}%", flush=True)
    print(f"│  Slot consistency: {mean_consistency:.1f}%", flush=True)
    print(f"│  Binding accuracy: {mean_binding:.1f}%", flush=True)
    print(f"│  Attention entropy: {mean_entropy:.3f}", flush=True)
    print(f"│  Active slots: {n_active_avg:.1f}/{n_slots}", flush=True)
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    slot_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                   '#911eb4', '#42d4f4', '#f032e6', '#bcf60c',
                   '#fabebe', '#008080', '#e6beff']

    # ─── Panel 1: Slot masks (3 videos) ───
    ax = axes[0, 0]
    sample_vids = [0, min(5, n_videos - 1), min(10, n_videos - 1)]
    sample_frame = 64
    n_show = min(3, n_videos)
    composite_img = np.zeros((P * n_show, P * 2, 3))

    for si_idx, vi in enumerate(sample_vids[:n_show]):
        vid_id = video_ids[vi]
        cap = cv2.VideoCapture(f"{data_dir}/videos/video_{vid_id}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (P, P))
            composite_img[si_idx * P:(si_idx + 1) * P, :P, :] = frame_small / 255.0

        # Apply temporal reordering to attention
        perm = all_slot_assignments[vi][sample_frame]
        attn_frame = all_attns[vi][sample_frame][perm]
        masks = attn_frame.reshape(n_slots, P, P)

        mask_composite = np.zeros((P, P, 3))
        for s in range(n_slots):
            color = plt.cm.tab20(s / n_slots)[:3]
            for c in range(3):
                mask_composite[:, :, c] += masks[s] * color[c]
        composite_img[si_idx * P:(si_idx + 1) * P, P:2 * P, :] = np.clip(mask_composite, 0, 1)

    ax.imshow(composite_img, interpolation='nearest')
    ax.set_title(f'Slot Masks (frame {sample_frame})', fontsize=10)
    ax.axis('off')

    # ─── Panel 2: Trajectories (GT vs Slots) ───
    ax = axes[0, 1]
    plot_vid = 0
    vid_id = video_ids[plot_vid]
    ann = annotations[vid_id]
    traj = ann['motion_trajectory']
    obj_props = {p['object_id']: p for p in ann['object_property']}

    gt_colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    obj_ids_plot = [p['object_id'] for p in ann['object_property']]
    for oi, oid in enumerate(obj_ids_plot):
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
        ax.plot(gt_us, gt_vs, '-', color=gt_colors_list[oi % 6], alpha=0.5,
                linewidth=1, label=f'GT {props["color"]} {props["shape"]}')

    for s in range(n_slots):
        coverage = np.mean([
            (all_attns[plot_vid][fi].argmax(0) == s).sum() / n_patches
            for fi in range(0, n_frames, 16)
        ])
        if coverage > 0.03:
            ax.plot(all_reordered[plot_vid][:, s, 0],
                    all_reordered[plot_vid][:, s, 1],
                    '--', color=slot_colors[s % len(slot_colors)], alpha=0.7,
                    linewidth=1, label=f'Slot {s}')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_title(f'Trajectories (video {vid_id})', fontsize=10)
    ax.legend(fontsize=6, loc='upper right', ncol=2)

    # ─── Panel 3: Tracking error histogram ───
    ax = axes[0, 2]
    errors_pct = [e * 100 for e in tracking_errors]
    ax.hist(errors_pct, bins=40, color='#2196F3', edgecolor='black',
            linewidth=0.3, alpha=0.8)
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2,
               label='10% threshold')
    ax.axvline(x=mean_tracking_error * 100, color='orange', linestyle='-',
               linewidth=2, label=f'Mean={mean_tracking_error * 100:.1f}%')
    ax.set_xlabel('Tracking Error (% of frame)')
    ax.set_ylabel('Count')
    ax.set_title('Tracking Error Distribution', fontsize=10)
    ax.legend(fontsize=8)

    # ─── Panel 4: Consistency per video ───
    ax = axes[1, 0]
    x_vids = np.arange(n_videos)
    ax.bar(x_vids, [c * 100 for c in slot_consistency_per_video],
           color='#4CAF50', edgecolor='black', linewidth=0.3)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% target')
    ax.axhline(y=mean_consistency, color='orange', linestyle='-',
               label=f'Mean={mean_consistency:.0f}%')
    ax.set_xlabel('Video Index')
    ax.set_ylabel('Slot Consistency (%)')
    ax.set_title('Consistency per Video', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # ─── Panel 5: Temporal degradation ───
    ax = axes[1, 1]
    frame_bins = list(range(0, n_frames, 8))
    bin_errors = []
    for b in frame_bins:
        bin_errs = []
        for fi in range(b, min(b + 8, n_frames)):
            bin_errs.extend(per_frame_errors[fi])
        bin_errors.append(np.mean(bin_errs) * 100 if bin_errs else 0)
    ax.plot(frame_bins, bin_errors, 'o-', color='#FF5722', linewidth=2, markersize=4)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Tracking Error (%)')
    ax.set_title('Temporal Degradation (8-frame bins)', fontsize=10)
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% target')
    ax.axhline(y=early_err * 100, color='green', linestyle=':', alpha=0.7,
               label=f'Early (0-15): {early_err * 100:.1f}%')
    ax.axhline(y=late_err * 100, color='purple', linestyle=':', alpha=0.7,
               label=f'Late (64-127): {late_err * 100:.1f}%')
    ax.legend(fontsize=7)

    # ─── Panel 6: Summary ───
    ax = axes[1, 2]
    ax.axis('off')

    err_ok = mean_tracking_error < 0.20
    cons_ok = mean_consistency > 30
    bind_ok = mean_binding > 85
    if err_ok and cons_ok and bind_ok:
        verdict = "SUCCESS"
    elif err_ok or cons_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = (
        f"Phase 46: SlotContrast → CLEVRER\n"
        f"{'=' * 40}\n\n"
        f"Model: Pretrained MOVi-C\n"
        f"  11 slots, 64-dim, ViT-S/14 DINOv2\n"
        f"  Zero-shot transfer (no training)\n\n"
        f"Data: {n_videos} videos × {n_frames} frames\n\n"
        f"Tracking error:   {mean_tracking_error * 100:.1f}% "
        f"({'OK' if err_ok else 'MISS'}, target <20%)\n"
        f"  median:         {median_tracking_error * 100:.1f}%\n"
        f"  early (0-15):   {early_err * 100:.1f}%\n"
        f"  late (64-127):  {late_err * 100:.1f}%\n"
        f"Slot consistency: {mean_consistency:.1f}% "
        f"({'OK' if cons_ok else 'MISS'}, target >30%)\n"
        f"Binding accuracy: {mean_binding:.1f}% "
        f"({'OK' if bind_ok else 'MISS'}, target >85%)\n\n"
        f"Attn entropy:     {mean_entropy:.3f}\n"
        f"Active slots:     {n_active_avg:.1f}/{n_slots}\n"
        f"Time:             {elapsed:.0f}s\n\n"
        f"Phase 45 baseline: err=28.8% cons=3.4%\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(
        f'Phase 46: SlotContrast Pretrained → CLEVRER\n'
        f'err={mean_tracking_error * 100:.1f}% '
        f'consistency={mean_consistency:.0f}% '
        f'binding={mean_binding:.0f}% '
        f'entropy={mean_entropy:.3f} | {verdict}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "phase46_slotcontrast_clevrer.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'=' * 60}", flush=True)
    print(f"RESULTS: Phase 46", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"│  Tracking error: {mean_tracking_error * 100:.1f}% "
          f"(median {median_tracking_error * 100:.1f}%)", flush=True)
    print(f"│  Early (0-15):   {early_err * 100:.1f}%", flush=True)
    print(f"│  Late (64-127):  {late_err * 100:.1f}%", flush=True)
    print(f"│  Consistency:    {mean_consistency:.1f}%", flush=True)
    print(f"│  Binding:        {mean_binding:.1f}%", flush=True)
    print(f"│  Entropy:        {mean_entropy:.3f}", flush=True)
    print(f"│  Active slots:   {n_active_avg:.1f}/{n_slots}", flush=True)
    print(f"│  Time:           {elapsed:.0f}s", flush=True)
    print(f"│  Saved: results/phase46_slotcontrast_clevrer.png", flush=True)
    print(f"│  VERDICT: {verdict}", flush=True)
    print(f"│  (Phase 45 baseline: err=28.8% cons=3.4%)", flush=True)
    print(f"{'=' * 60}", flush=True)

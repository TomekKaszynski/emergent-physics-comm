"""
Phase 53: Augmented Pixel-Based Elasticity Communication
=========================================================
Same as Phase 51, but with heavy data augmentation during training:
  - ColorJitter (brightness, contrast, saturation, hue)
  - RandomHorizontalFlip
  - RandomAffine (small rotation, translation)
  - RandomErasing (small patches)

Hypothesis: Phase 52 showed NO TRANSFER because the CNN learned
appearance-specific features. If augmented training produces a protocol
that transfers, the communication framework is sound — it just needs
perceptual diversity to develop invariant representations.

After training, automatically runs frozen transfer evaluation on the
Phase 52 transfer dataset (200 scenes: 100 near, 100 far).

Run from ~/AI/:
  python _phase53_augmented_training.py
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T


# ══════════════════════════════════════════════════════════════════
# Architecture (identical to Phase 51)
# ══════════════════════════════════════════════════════════════════

class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.conv(x).squeeze(-1).squeeze(-1)


class VideoEncoder(nn.Module):
    def __init__(self, hidden_dim, n_frames):
        super().__init__()
        self.frame_enc = FrameEncoder()
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, video):
        B, T = video.shape[:2]
        frames_flat = video.reshape(B * T, *video.shape[2:])
        frame_feats = self.frame_enc(frames_flat)
        frame_feats = frame_feats.reshape(B, T, 128)
        x = frame_feats.permute(0, 2, 1)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


class PixelSender(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_frames):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = VideoEncoder(hidden_dim, n_frames)
        self.to_message = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video, tau=1.0, hard=True):
        h = self.encoder(video)
        logits = self.to_message(h)
        if self.training:
            message = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            idx = logits.argmax(dim=-1)
            message = F.one_hot(idx, self.vocab_size).float()
        return message, logits


class Receiver(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


class PixelOracle(nn.Module):
    def __init__(self, hidden_dim, n_frames):
        super().__init__()
        self.enc_a = VideoEncoder(hidden_dim, n_frames)
        self.enc_b = VideoEncoder(hidden_dim, n_frames)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, vid_a, vid_b):
        ha = self.enc_a(vid_a)
        hb = self.enc_b(vid_b)
        return self.head(torch.cat([ha, hb], dim=-1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data augmentation
# ══════════════════════════════════════════════════════════════════

class VideoAugmentor:
    """Apply consistent augmentation across all frames of a video.

    VECTORIZED: all ops are batched tensor operations, no per-frame loops.
    Augmentation is consistent within a video (same color shift for all
    8 frames) but different across videos in the batch.
    """
    def __init__(self):
        self.brightness_range = (0.6, 1.4)   # ±0.4
        self.contrast_range = (0.6, 1.4)     # ±0.4
        self.saturation_range = (0.6, 1.4)   # ±0.4
        self.flip_p = 0.5
        self.erase_p = 0.2
        self.erase_scale = (0.02, 0.08)

    def __call__(self, video_batch, device):
        """Augment a batch of videos using vectorized ops.

        video_batch: (B, T, 3, H, W) — already ImageNet-normalized
        Returns: augmented batch on device, same shape
        """
        B, n_t, C, H, W = video_batch.shape

        # Denormalize to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        x = (video_batch * std + mean).clamp(0, 1)

        # Brightness: per-video random factor (B, 1, 1, 1, 1)
        bright = torch.empty(B, 1, 1, 1, 1).uniform_(*self.brightness_range)
        x = (x * bright).clamp(0, 1)

        # Contrast: lerp toward per-video spatial mean
        contrast = torch.empty(B, 1, 1, 1, 1).uniform_(*self.contrast_range)
        chan_mean = x.mean(dim=(3, 4), keepdim=True).mean(dim=1, keepdim=True)
        x = (contrast * x + (1 - contrast) * chan_mean).clamp(0, 1)

        # Saturation: lerp toward grayscale
        sat = torch.empty(B, 1, 1, 1, 1).uniform_(*self.saturation_range)
        gray_w = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 3, 1, 1)
        gray = (x * gray_w).sum(dim=2, keepdim=True)  # (B, T, 1, H, W)
        x = (sat * x + (1 - sat) * gray).clamp(0, 1)

        # Horizontal flip (per-video, consistent across frames)
        flip_mask = torch.rand(B) < self.flip_p
        if flip_mask.any():
            x[flip_mask] = x[flip_mask].flip(dims=[-1])

        # Random erasing: small rectangle per frame (independent)
        if self.erase_p > 0:
            erase_mask = torch.rand(B, n_t) < self.erase_p
            if erase_mask.any():
                for b_idx, t_idx in erase_mask.nonzero(as_tuple=False):
                    b_i, t_i = b_idx.item(), t_idx.item()
                    area = H * W
                    s = torch.empty(1).uniform_(*self.erase_scale).item()
                    eh = int(round((area * s) ** 0.5))
                    ew = eh
                    eh = min(eh, H)
                    ew = min(ew, W)
                    y0 = torch.randint(0, H - eh + 1, (1,)).item()
                    x0 = torch.randint(0, W - ew + 1, (1,)).item()
                    x[b_i, t_i, :, y0:y0+eh, x0:x0+ew] = torch.rand(3, eh, ew)

        # Re-normalize
        x = (x - mean) / std

        return x.to(device)


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_dataset(dataset_dir):
    """Load Kubric dataset, return videos tensor, restitutions, train/val split."""
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "index.json"

    with open(index_path) as f:
        index = json.load(f)

    n_sample_frames = 8
    frame_indices = np.linspace(0, 47, n_sample_frames, dtype=int)

    all_videos = []
    restitutions = []

    for meta in index:
        sid = meta["scene_id"]
        scene_dir = dataset_dir / f"scene_{sid:04d}"
        if not (scene_dir / "rgba_00000.png").exists():
            continue

        frames = []
        skip = False
        for fi in frame_indices:
            fpath = scene_dir / f"rgba_{fi:05d}.png"
            if not fpath.exists():
                skip = True
                break
            img = Image.open(fpath).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            frames.append(img_np)

        if skip:
            continue

        video = np.stack(frames).transpose(0, 3, 1, 2)
        all_videos.append(video)
        restitutions.append(meta["restitution"])

    n_scenes = len(all_videos)
    all_videos = np.stack(all_videos)
    restitutions = np.array(restitutions)

    all_videos_t = torch.tensor(all_videos, dtype=torch.float32)
    all_rest_t = torch.tensor(restitutions, dtype=torch.float32)

    # ImageNet normalization
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    all_videos_t = (all_videos_t - img_mean) / img_std

    return all_videos_t, all_rest_t, restitutions, n_scenes


def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


# ══════════════════════════════════════════════════════════════════
# Transfer evaluation (reused from Phase 52)
# ══════════════════════════════════════════════════════════════════

def evaluate_frozen(sender, receiver, videos_t, restitutions, scene_ids,
                    device, n_rounds=50, batch_size=64):
    rng = np.random.RandomState(42)
    rest_dev = torch.tensor(restitutions, dtype=torch.float32).to(device)
    correct = 0
    total = 0
    for _ in range(n_rounds):
        vi_a, vi_b = sample_pairs(scene_ids, min(batch_size, len(scene_ids)), rng)
        vv_a = videos_t[vi_a].to(device)
        vv_b = videos_t[vi_b].to(device)
        labels = (rest_dev[vi_a] > rest_dev[vi_b]).float()
        msg_a, _ = sender(vv_a)
        msg_b, _ = sender(vv_b)
        pred = receiver(msg_a, msg_b)
        correct += ((pred > 0) == labels.bool()).sum().item()
        total += len(labels)
    return correct / max(total, 1), total


def evaluate_by_difficulty(sender, receiver, videos_t, restitutions, scene_ids,
                           device, n_rounds=30, batch_size=64):
    rng = np.random.RandomState(123)
    rest_dev = torch.tensor(restitutions, dtype=torch.float32).to(device)
    gap_bins = [(0.0, 0.1, "tiny"), (0.1, 0.3, "small"),
                (0.3, 0.5, "medium"), (0.5, 1.0, "large")]
    results = {}
    for gap_lo, gap_hi, name in gap_bins:
        correct = 0
        total = 0
        for _ in range(n_rounds):
            vi_a, vi_b = sample_pairs(scene_ids, min(batch_size, len(scene_ids)), rng)
            gaps = np.abs(restitutions[vi_a] - restitutions[vi_b])
            mask = (gaps >= gap_lo) & (gaps < gap_hi)
            if mask.sum() == 0:
                continue
            vv_a = videos_t[vi_a[mask]].to(device)
            vv_b = videos_t[vi_b[mask]].to(device)
            labels = (rest_dev[vi_a[mask]] > rest_dev[vi_b[mask]]).float()
            msg_a, _ = sender(vv_a)
            msg_b, _ = sender(vv_b)
            pred = receiver(msg_a, msg_b)
            correct += ((pred > 0) == labels.bool()).sum().item()
            total += len(labels)
        if total > 0:
            results[name] = {'acc': correct / total, 'n': total}
    return results


def get_symbol_stats(sender, videos_t, restitutions, device, vocab_size=16):
    all_msg_ids = []
    for i in range(0, len(videos_t), 100):
        vids = videos_t[i:i+100].to(device)
        msgs, _ = sender(vids)
        all_msg_ids.append(msgs.argmax(dim=-1).cpu().numpy())
    all_msg_ids = np.concatenate(all_msg_ids)
    symbol_stats = {}
    for s in range(vocab_size):
        mask = all_msg_ids == s
        if mask.sum() > 0:
            symbol_stats[s] = {
                'mean_e': float(np.mean(restitutions[mask])),
                'std_e': float(np.std(restitutions[mask])),
                'count': int(mask.sum()),
            }
    counts = np.bincount(all_msg_ids, minlength=vocab_size).astype(float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
    n_used = int((probs > 0.01).sum())
    return all_msg_ids, symbol_stats, float(entropy), n_used


# ══════════════════════════════════════════════════════════════════
# Main training + transfer evaluation
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("PHASE 53: Augmented Pixel-Based Elasticity Communication", flush=True)
    print("  Same as Phase 51 + heavy data augmentation", flush=True)
    print("  Goal: learn appearance-invariant features → protocol transfer", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 0: Load data
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load training data", flush=True)
    print(f"{'=' * 60}", flush=True)

    all_videos_t, all_rest_t, restitutions, n_scenes = load_dataset(
        "kubric/output/elasticity_dataset")

    # Same split as Phase 51
    n_train = int(0.8 * n_scenes)
    perm = np.random.RandomState(42).permutation(n_scenes)
    train_ids = perm[:n_train]
    val_ids = perm[n_train:]

    print(f"│  Loaded {n_scenes} scenes, train={len(train_ids)}, val={len(val_ids)}", flush=True)
    print(f"│  Restitution range: [{restitutions.min():.3f}, {restitutions.max():.3f}]", flush=True)

    all_rest_dev = all_rest_t.to(device)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: Setup
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Setup augmented training", flush=True)
    print(f"{'=' * 60}", flush=True)

    vocab_size = 16
    hidden_dim = 128
    n_epochs = 250           # more epochs — augmentation makes learning harder
    oracle_pretrain_epochs = 50  # no augmentation in oracle — same as Phase 51
    lr = 1e-4
    batch_size = 64
    gumbel_tau_start = 3.0
    gumbel_tau_end = 1.5
    soft_warmup_epochs = 40  # longer warmup — augmented inputs are noisier
    n_frames = 8

    augmentor = VideoAugmentor()

    torch.manual_seed(42)
    sender = PixelSender(hidden_dim, vocab_size, n_frames).to(device)
    receiver = Receiver(vocab_size, hidden_dim).to(device)
    oracle = PixelOracle(hidden_dim, n_frames).to(device)

    oracle_optimizer = torch.optim.Adam(oracle.parameters(), lr=lr)

    n_sender_params = sum(p.numel() for p in sender.parameters())
    n_receiver_params = sum(p.numel() for p in receiver.parameters())

    print(f"│  Sender:   {n_sender_params:,} params", flush=True)
    print(f"│  Receiver: {n_receiver_params:,} params", flush=True)
    print(f"│  Vocab: {vocab_size}, Hidden: {hidden_dim}", flush=True)
    print(f"│  Epochs: {n_epochs} (oracle pretrain: {oracle_pretrain_epochs})", flush=True)
    print(f"│  Augmentation: ColorJitter(0.4/0.4/0.4/0.15) + HFlip + Affine + Erasing", flush=True)
    print(f"│  Soft warmup: {soft_warmup_epochs} epochs", flush=True)

    rng = np.random.RandomState(123)
    batches_per_epoch = max(1, len(train_ids) * 4 // batch_size)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1b: Oracle pretrain (WITH augmentation)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1b: Pre-train oracle WITHOUT augmentation ({oracle_pretrain_epochs} epochs)", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts1b = time.time()

    oracle.train()
    for ep in range(1, oracle_pretrain_epochs + 1):
        ep_correct = 0
        ep_total = 0
        for bi in range(batches_per_epoch):
            idx_a, idx_b = sample_pairs(train_ids, batch_size, rng)

            # NO augmentation for oracle pretrain — random weights + augmented
            # inputs means oracle can't learn (stuck at chance)
            vid_a = all_videos_t[idx_a].to(device)
            vid_b = all_videos_t[idx_b].to(device)

            rest_a = all_rest_dev[idx_a]
            rest_b = all_rest_dev[idx_b]
            labels = (rest_a > rest_b).float()

            pred = oracle(vid_a, vid_b)
            loss = F.binary_cross_entropy_with_logits(pred, labels)

            oracle_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            oracle_optimizer.step()

            with torch.no_grad():
                ep_correct += ((pred > 0) == labels.bool()).sum().item()
                ep_total += len(labels)

        if ep % 15 == 0 or ep == 1:
            # Validate WITHOUT augmentation
            oracle.eval()
            with torch.no_grad():
                vc, vt = 0, 0
                for _ in range(20):
                    vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
                    vv_a = all_videos_t[vi_a].to(device)
                    vv_b = all_videos_t[vi_b].to(device)
                    vlabels = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()
                    vp = oracle(vv_a, vv_b)
                    vc += ((vp > 0) == vlabels.bool()).sum().item()
                    vt += len(vlabels)
                val_acc = vc / max(vt, 1)
            train_acc = ep_correct / max(ep_total, 1)
            print(f"│  Oracle ep {ep:3d}/{oracle_pretrain_epochs}: "
                  f"train={train_acc:.3f} val={val_acc:.3f} [{time.time()-ts1b:.0f}s]", flush=True)
            oracle.train()

        if hasattr(torch.mps, 'empty_cache') and ep % 20 == 0:
            torch.mps.empty_cache()

    # Final oracle eval
    oracle.eval()
    with torch.no_grad():
        vc, vt = 0, 0
        for _ in range(20):
            vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
            vv_a = all_videos_t[vi_a].to(device)
            vv_b = all_videos_t[vi_b].to(device)
            vlabels = (all_rest_dev[vi_a] > all_rest_dev[vi_b]).float()
            vp = oracle(vv_a, vv_b)
            vc += ((vp > 0) == vlabels.bool()).sum().item()
            vt += len(vlabels)
        oracle_val_acc = vc / max(vt, 1)
    print(f"│  Oracle final val: {oracle_val_acc:.3f}", flush=True)

    # Bootstrap sender from oracle
    sender_enc_state = sender.encoder.state_dict()
    oracle_enc_state = oracle.enc_a.state_dict()
    for key in sender_enc_state:
        if key in oracle_enc_state:
            sender_enc_state[key] = oracle_enc_state[key].clone()
    sender.encoder.load_state_dict(sender_enc_state)
    print(f"│  Copied oracle.enc_a → sender.encoder", flush=True)
    print(f"└─ Stage 1b done [{time.time()-ts1b:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Communication training (WITH augmentation)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Train communication agents WITH augmentation", flush=True)
    print(f"{'=' * 60}", flush=True)
    ts2 = time.time()

    comm_params = list(sender.parameters()) + list(receiver.parameters())
    comm_optimizer = torch.optim.Adam(comm_params, lr=lr)
    rng = np.random.RandomState(456)

    history = {
        'epoch': [], 'train_comm': [], 'val_comm': [],
        'train_oracle': [], 'val_oracle': [],
        'msg_entropy': [], 'gumbel_tau': [],
    }

    best_val_comm = 0.0
    best_sender_state = None
    best_receiver_state = None
    best_epoch = 0
    collapse_patience = 0

    epoch_times = []
    for epoch in range(1, n_epochs + 1):
        ep_start = time.time()
        progress = min(epoch / (n_epochs * 0.7), 1.0)
        g_tau = gumbel_tau_start + (gumbel_tau_end - gumbel_tau_start) * progress

        epoch_comm_correct = 0
        epoch_oracle_correct = 0
        epoch_total = 0

        sender.train(); receiver.train(); oracle.train()

        for bi in range(batches_per_epoch):
            idx_a, idx_b = sample_pairs(train_ids, batch_size, rng)

            # AUGMENT training videos
            vid_a_raw = all_videos_t[idx_a]
            vid_b_raw = all_videos_t[idx_b]
            vid_a = augmentor(vid_a_raw, device)
            vid_b = augmentor(vid_b_raw, device)

            rest_a = all_rest_dev[idx_a]
            rest_b = all_rest_dev[idx_b]
            labels = (rest_a > rest_b).float()

            # Communication path
            use_hard = epoch > soft_warmup_epochs
            msg_a, _ = sender(vid_a, tau=g_tau, hard=use_hard)
            msg_b, _ = sender(vid_b, tau=g_tau, hard=use_hard)
            pred_comm = receiver(msg_a, msg_b)
            comm_loss = F.binary_cross_entropy_with_logits(pred_comm, labels)

            comm_optimizer.zero_grad()
            comm_loss.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            comm_optimizer.step()

            # Oracle path (also augmented)
            pred_oracle = oracle(vid_a, vid_b)
            oracle_loss = F.binary_cross_entropy_with_logits(pred_oracle, labels)

            oracle_optimizer.zero_grad()
            oracle_loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            oracle_optimizer.step()

            with torch.no_grad():
                epoch_comm_correct += ((pred_comm > 0) == labels.bool()).sum().item()
                epoch_oracle_correct += ((pred_oracle > 0) == labels.bool()).sum().item()
                epoch_total += len(labels)

        train_comm_acc = epoch_comm_correct / max(epoch_total, 1)
        train_oracle_acc = epoch_oracle_correct / max(epoch_total, 1)

        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)

        if hasattr(torch.mps, 'empty_cache') and epoch % 20 == 0:
            torch.mps.empty_cache()

        # Validation every 10 epochs (NO augmentation at eval)
        if epoch % 10 == 0 or epoch == 1:
            sender.eval(); receiver.eval(); oracle.eval()

            with torch.no_grad():
                val_correct_comm = 0
                val_correct_oracle = 0
                val_total = 0

                for _ in range(10):
                    vi_a, vi_b = sample_pairs(val_ids, batch_size, rng)
                    vv_a = all_videos_t[vi_a].to(device)
                    vv_b = all_videos_t[vi_b].to(device)
                    vr_a = all_rest_dev[vi_a]
                    vr_b = all_rest_dev[vi_b]
                    vlabels = (vr_a > vr_b).float()

                    vm_a, _ = sender(vv_a)
                    vm_b, _ = sender(vv_b)
                    vp_comm = receiver(vm_a, vm_b)
                    vp_oracle = oracle(vv_a, vv_b)

                    val_correct_comm += ((vp_comm > 0) == vlabels.bool()).sum().item()
                    val_correct_oracle += ((vp_oracle > 0) == vlabels.bool()).sum().item()
                    val_total += len(vlabels)

                val_comm_acc = val_correct_comm / max(val_total, 1)
                val_oracle_acc = val_correct_oracle / max(val_total, 1)

                # Message entropy
                n_ent = min(200, n_scenes)
                ent_vids = all_videos_t[:n_ent].to(device)
                all_msg, _ = sender(ent_vids)
                msg_ids = all_msg.argmax(dim=-1).cpu().numpy()
                counts = np.bincount(msg_ids, minlength=vocab_size).astype(float)
                probs = counts / counts.sum()
                msg_entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
                n_used = (probs > 0.01).sum()

            history['epoch'].append(epoch)
            history['train_comm'].append(train_comm_acc)
            history['val_comm'].append(val_comm_acc)
            history['train_oracle'].append(train_oracle_acc)
            history['val_oracle'].append(val_oracle_acc)
            history['msg_entropy'].append(float(msg_entropy))
            history['gumbel_tau'].append(g_tau)

            avg_time = np.mean(epoch_times[-20:])
            remaining = (n_epochs - epoch) * avg_time
            eta_str = f"{remaining/60:.0f}m" if remaining > 60 else f"{remaining:.0f}s"

            if val_comm_acc > best_val_comm:
                best_val_comm = val_comm_acc
                best_sender_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}
                best_receiver_state = {k: v.cpu().clone() for k, v in receiver.state_dict().items()}
                best_epoch = epoch

            # Collapse detection
            if msg_entropy < 0.05 and best_val_comm > 0.6:
                collapse_patience += 1
            else:
                collapse_patience = 0

            if epoch % 20 == 0 or epoch == 1 or epoch == soft_warmup_epochs + 1:
                mode_str = "soft" if epoch <= soft_warmup_epochs else "hard"
                best_str = f" *best={best_val_comm:.3f}@{best_epoch}" if best_val_comm > 0.55 else ""
                print(f"│  Epoch {epoch:4d}/{n_epochs} [{mode_str}]: "
                      f"comm={train_comm_acc:.3f}/{val_comm_acc:.3f} "
                      f"oracle={train_oracle_acc:.3f}/{val_oracle_acc:.3f} "
                      f"ent={msg_entropy:.3f}({n_used}/{vocab_size}) "
                      f"τ={g_tau:.2f} [{ep_time:.1f}s] eta={eta_str}{best_str}", flush=True)

            if collapse_patience >= 3:
                print(f"│  *** COLLAPSE DETECTED at epoch {epoch} ***", flush=True)
                print(f"│  Restoring best model from epoch {best_epoch}", flush=True)
                break

    # Restore best
    if best_sender_state is not None and best_val_comm > 0.55:
        sender.load_state_dict({k: v.to(device) for k, v in best_sender_state.items()})
        receiver.load_state_dict({k: v.to(device) for k, v in best_receiver_state.items()})
        print(f"│  Restored best model from epoch {best_epoch} "
              f"(val comm={best_val_comm:.3f})", flush=True)

    print(f"└─ Stage 2 done [{time.time()-ts2:.0f}s]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: Evaluate on original val set
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Final evaluation on original val set", flush=True)
    print(f"{'=' * 60}", flush=True)

    sender.eval(); receiver.eval(); oracle.eval()

    with torch.no_grad():
        orig_acc, orig_n = evaluate_frozen(
            sender, receiver, all_videos_t, restitutions,
            val_ids, device)
        orig_diff = evaluate_by_difficulty(
            sender, receiver, all_videos_t, restitutions,
            val_ids, device)

        _, orig_sym_stats, orig_entropy, orig_n_used = get_symbol_stats(
            sender, all_videos_t, restitutions, device, vocab_size)

        orig_ordering = sorted(
            [(s, v['mean_e']) for s, v in orig_sym_stats.items() if v['count'] >= 5],
            key=lambda x: x[1])
        orig_order_syms = [s for s, _ in orig_ordering]

    print(f"│  Original val accuracy: {orig_acc*100:.1f}% ({orig_n} pairs)", flush=True)
    print(f"│  Phase 51 (no aug):     84.5%", flush=True)
    print(f"│  Entropy: {orig_entropy:.3f}, Symbols: {orig_n_used}/{vocab_size}", flush=True)
    print(f"│  Symbol ordering: {orig_order_syms}", flush=True)
    for name in ["tiny", "small", "medium", "large"]:
        if name in orig_diff:
            d = orig_diff[name]
            print(f"│    {name:6s}: {d['acc']*100:.1f}% (n={d['n']})", flush=True)

    # Save model
    torch.save({
        'sender': sender.state_dict(),
        'receiver': receiver.state_dict(),
        'oracle': oracle.state_dict(),
        'history': history,
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'augmented': True,
    }, str(OUTPUT_DIR / "phase53_model.pt"))
    print(f"│  Saved results/phase53_model.pt", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STAGE 4: Transfer evaluation (FROZEN)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 4: Transfer evaluation (frozen augmented protocol)", flush=True)
    print(f"{'=' * 60}", flush=True)

    transfer_dir = Path("kubric/output/transfer_dataset")
    if not transfer_dir.exists():
        print(f"│  Transfer dataset not found — skipping", flush=True)
        transfer_results = {}
    else:
        xfer_videos, xfer_rest_t, xfer_rest, xfer_n = load_dataset(transfer_dir)

        if xfer_videos is None or xfer_n == 0:
            print(f"│  No rendered transfer scenes — skipping", flush=True)
            transfer_results = {}
        else:
            # Load metadata for near/far split
            with open(transfer_dir / "index.json") as f:
                xfer_index = json.load(f)

            # Match loaded scenes to metadata
            xfer_meta = []
            for meta in xfer_index:
                sid = meta["scene_id"]
                scene_dir = transfer_dir / f"scene_{sid:04d}"
                if (scene_dir / "rgba_00000.png").exists():
                    xfer_meta.append(meta)

            near_ids = np.array([i for i, m in enumerate(xfer_meta)
                                 if m.get("transfer_type") == "near"])
            far_ids = np.array([i for i, m in enumerate(xfer_meta)
                                if m.get("transfer_type") == "far"])
            all_xfer_ids = np.arange(xfer_n)

            print(f"│  Transfer dataset: {xfer_n} scenes "
                  f"(near={len(near_ids)}, far={len(far_ids)})", flush=True)

            with torch.no_grad():
                # All transfer
                all_xfer_acc, all_xfer_n = evaluate_frozen(
                    sender, receiver, xfer_videos, xfer_rest,
                    all_xfer_ids, device)
                print(f"│  All transfer:    {all_xfer_acc*100:.1f}% ({all_xfer_n} pairs)", flush=True)

                # Near transfer
                near_acc, near_n = 0, 0
                if len(near_ids) >= 10:
                    near_acc, near_n = evaluate_frozen(
                        sender, receiver, xfer_videos, xfer_rest,
                        near_ids, device)
                    near_diff = evaluate_by_difficulty(
                        sender, receiver, xfer_videos, xfer_rest,
                        near_ids, device)
                    print(f"│  Near-transfer:   {near_acc*100:.1f}% ({near_n} pairs)", flush=True)
                    for name in ["tiny", "small", "medium", "large"]:
                        if name in near_diff:
                            d = near_diff[name]
                            print(f"│    {name:6s}: {d['acc']*100:.1f}% (n={d['n']})", flush=True)

                # Far transfer
                far_acc, far_n = 0, 0
                if len(far_ids) >= 10:
                    far_acc, far_n = evaluate_frozen(
                        sender, receiver, xfer_videos, xfer_rest,
                        far_ids, device)
                    far_diff = evaluate_by_difficulty(
                        sender, receiver, xfer_videos, xfer_rest,
                        far_ids, device)
                    print(f"│  Far-transfer:    {far_acc*100:.1f}% ({far_n} pairs)", flush=True)
                    for name in ["tiny", "small", "medium", "large"]:
                        if name in far_diff:
                            d = far_diff[name]
                            print(f"│    {name:6s}: {d['acc']*100:.1f}% (n={d['n']})", flush=True)

                # Cross-domain
                rng_cross = np.random.RandomState(555)
                cross_correct = 0
                cross_total = 0
                for _ in range(50):
                    bs = min(64, len(val_ids), xfer_n)
                    idx_orig = rng_cross.choice(val_ids, size=bs)
                    idx_xfer = rng_cross.choice(all_xfer_ids, size=bs)
                    vid_orig = all_videos_t[idx_orig].to(device)
                    vid_xfer = xfer_videos[idx_xfer].to(device)
                    rest_orig_b = torch.tensor(restitutions[idx_orig], dtype=torch.float32).to(device)
                    rest_xfer_b = torch.tensor(xfer_rest[idx_xfer], dtype=torch.float32).to(device)
                    labels = (rest_orig_b > rest_xfer_b).float()
                    msg_orig, _ = sender(vid_orig)
                    msg_xfer, _ = sender(vid_xfer)
                    pred = receiver(msg_orig, msg_xfer)
                    cross_correct += ((pred > 0) == labels.bool()).sum().item()
                    cross_total += len(labels)
                cross_acc = cross_correct / max(cross_total, 1)
                print(f"│  Cross-domain:    {cross_acc*100:.1f}% ({cross_total} pairs)", flush=True)

                # Symbol consistency
                _, xfer_sym_stats, xfer_entropy, xfer_n_used = get_symbol_stats(
                    sender, xfer_videos, xfer_rest, device, vocab_size)
                print(f"│  Transfer entropy: {xfer_entropy:.3f}, Symbols: {xfer_n_used}/{vocab_size}", flush=True)

                xfer_ordering = sorted(
                    [(s, v['mean_e']) for s, v in xfer_sym_stats.items() if v['count'] >= 3],
                    key=lambda x: x[1])
                xfer_order_syms = [s for s, _ in xfer_ordering]
                print(f"│  Original ordering:  {orig_order_syms}", flush=True)
                print(f"│  Transfer ordering:  {xfer_order_syms}", flush=True)

                # Kendall tau
                common_syms = set(orig_order_syms) & set(xfer_order_syms)
                kendall_tau = 0.0
                if len(common_syms) >= 2:
                    orig_ranks = {s: i for i, s in enumerate(orig_order_syms)}
                    xfer_ranks = {s: i for i, s in enumerate(xfer_order_syms)}
                    concordant = 0
                    discordant = 0
                    common_list = sorted(common_syms)
                    for i, s1 in enumerate(common_list):
                        for s2 in common_list[i+1:]:
                            if (orig_ranks[s1] < orig_ranks[s2]) == (xfer_ranks[s1] < xfer_ranks[s2]):
                                concordant += 1
                            else:
                                discordant += 1
                    if concordant + discordant > 0:
                        kendall_tau = (concordant - discordant) / (concordant + discordant)
                    print(f"│  Kendall τ: {kendall_tau:.3f}", flush=True)

                # Symbol distribution on transfer set
                print(f"│  Transfer symbol distribution:", flush=True)
                for s in sorted(xfer_sym_stats.keys()):
                    v = xfer_sym_stats[s]
                    if v['count'] >= 3:
                        print(f"│    Symbol {s:2d}: mean_e={v['mean_e']:.3f} ± {v['std_e']:.3f} "
                              f"(n={v['count']})", flush=True)

            transfer_results = {
                'all_transfer': {'accuracy': float(all_xfer_acc), 'n_pairs': all_xfer_n},
                'near_transfer': {'accuracy': float(near_acc), 'n_pairs': near_n},
                'far_transfer': {'accuracy': float(far_acc), 'n_pairs': far_n},
                'cross_domain': {'accuracy': float(cross_acc), 'n_pairs': cross_total},
                'symbol_consistency': {
                    'kendall_tau': float(kendall_tau),
                    'original_ordering': orig_order_syms,
                    'transfer_ordering': xfer_order_syms,
                    'transfer_entropy': xfer_entropy,
                    'transfer_symbols_used': xfer_n_used,
                },
            }

    # ══════════════════════════════════════════════════════════════
    # STAGE 5: Save everything
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 5: Save results", flush=True)
    print(f"{'=' * 60}", flush=True)

    all_results = {
        'original_val': {
            'accuracy': float(orig_acc),
            'entropy': orig_entropy,
            'symbols_used': orig_n_used,
            'symbol_ordering': orig_order_syms,
        },
        'training': {
            'n_epochs': n_epochs,
            'best_epoch': best_epoch,
            'best_val_comm': float(best_val_comm),
            'augmentation': 'ColorJitter(0.4/0.4/0.4/0.15)+HFlip+Affine+Erasing',
        },
        'transfer': transfer_results,
        'comparison': {
            'phase51_val': 84.5,
            'phase52_near_transfer': 50.1,
            'phase52_far_transfer': 54.8,
        },
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_json = json.loads(json.dumps(all_results, default=convert))
    with open("results/phase53_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"│  Saved results/phase53_results.json", flush=True)

    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Training curves
    ax = axes[0, 0]
    epochs = history['epoch']
    ax.plot(epochs, history['val_comm'], 'b-', linewidth=2, label=f'Comm val')
    ax.plot(epochs, history['train_comm'], 'b--', alpha=0.5, label='Comm train')
    ax.plot(epochs, history['val_oracle'], 'g-', linewidth=2, label=f'Oracle val')
    ax.plot(epochs, history['train_oracle'], 'g--', alpha=0.5, label='Oracle train')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.axhline(y=0.845, color='red', linestyle=':', alpha=0.5, label='Phase 51 (84.5%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Phase 53: Augmented Training', fontsize=11)
    ax.legend(fontsize=7)
    ax.set_ylim(0.4, 1.05)

    # Panel 2: Entropy over training
    ax = axes[0, 1]
    ax.plot(epochs, history['msg_entropy'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Message Diversity', fontsize=11)
    ax.set_ylim(0, 1.05)

    # Panel 3: Transfer comparison (Phase 52 vs Phase 53)
    ax = axes[0, 2]
    conditions = ['Phase 52\nnear', 'Phase 53\nnear', 'Phase 52\nfar', 'Phase 53\nfar', 'Chance']
    p52_near = 50.1
    p53_near = transfer_results.get('near_transfer', {}).get('accuracy', 0) * 100
    p52_far = 54.8
    p53_far = transfer_results.get('far_transfer', {}).get('accuracy', 0) * 100
    accs = [p52_near, p53_near, p52_far, p53_far, 50.0]
    colors = ['#FFCDD2', '#4CAF50', '#FFE0B2', '#FF9800', '#9E9E9E']
    ax.bar(range(len(conditions)), accs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Transfer: No-Aug vs Augmented', fontsize=11, fontweight='bold')
    ax.set_ylim(40, 100)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Panel 4: Original val accuracy comparison
    ax = axes[1, 0]
    conditions = ['Phase 51\n(no aug)', 'Phase 53\n(augmented)', 'Chance']
    accs = [84.5, orig_acc * 100, 50.0]
    colors = ['#2196F3', '#4CAF50', '#9E9E9E']
    ax.bar(range(len(conditions)), accs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Original Val: No-Aug vs Augmented', fontsize=11)
    ax.set_ylim(40, 100)
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Panel 5: Symbol distribution on transfer set
    ax = axes[1, 1]
    if 'symbol_consistency' in transfer_results:
        sc = transfer_results['symbol_consistency']
        xfer_counts = np.zeros(vocab_size)
        for s, v in xfer_sym_stats.items():
            xfer_counts[int(s)] = v['count']
        xfer_probs = xfer_counts / max(xfer_counts.sum(), 1)
        ax.bar(range(vocab_size), xfer_probs, color='#FF9800', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Transfer Symbol Distribution (ent={xfer_entropy:.3f})', fontsize=11)
        ax.set_xticks(range(vocab_size))
    else:
        ax.text(0.5, 0.5, 'No transfer data', transform=ax.transAxes, ha='center')

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = time.time() - t0

    near_acc_val = transfer_results.get('near_transfer', {}).get('accuracy', 0)
    far_acc_val = transfer_results.get('far_transfer', {}).get('accuracy', 0)

    if near_acc_val > 0.75:
        verdict = "STRONG TRANSFER"
    elif near_acc_val > 0.65:
        verdict = "MODERATE TRANSFER"
    elif near_acc_val > 0.55:
        verdict = "WEAK TRANSFER"
    else:
        verdict = "NO TRANSFER"

    summary = (
        f"Phase 53: Augmented Training\n\n"
        f"Original val:  {orig_acc*100:.1f}%\n"
        f"  (Phase 51: 84.5%)\n\n"
        f"TRANSFER (frozen):\n"
        f"  Near:  {near_acc_val*100:.1f}%  (P52: 50.1%)\n"
        f"  Far:   {far_acc_val*100:.1f}%  (P52: 54.8%)\n"
        f"  Cross: {transfer_results.get('cross_domain', {}).get('accuracy', 0)*100:.1f}%\n\n"
        f"Symbol ordering:\n"
        f"  Kendall τ: {transfer_results.get('symbol_consistency', {}).get('kendall_tau', 0):.3f}\n\n"
        f"Augmentation:\n"
        f"  ColorJitter + HFlip\n"
        f"  + Affine + Erasing\n\n"
        f"Time: {elapsed/60:.0f}min\n\n"
        f"VERDICT: {verdict}"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center')

    fig.suptitle(f'Phase 53: Augmented Elasticity Communication\n'
                 f'orig={orig_acc*100:.0f}% near_xfer={near_acc_val*100:.0f}% '
                 f'far_xfer={far_acc_val*100:.0f}% | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/phase53_augmented.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase53_augmented.png", flush=True)

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"PHASE 53 RESULTS: {verdict}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Original val:     {orig_acc*100:.1f}% (Phase 51: 84.5%)", flush=True)
    if transfer_results:
        print(f"  Near-transfer:    {near_acc_val*100:.1f}% (Phase 52: 50.1%)", flush=True)
        print(f"  Far-transfer:     {far_acc_val*100:.1f}% (Phase 52: 54.8%)", flush=True)
        cross_a = transfer_results.get('cross_domain', {}).get('accuracy', 0)
        print(f"  Cross-domain:     {cross_a*100:.1f}% (Phase 52: 59.6%)", flush=True)
        kt = transfer_results.get('symbol_consistency', {}).get('kendall_tau', 0)
        print(f"  Kendall τ:        {kt:.3f} (Phase 52: 0.200)", flush=True)
    print(f"  Chance:           50.0%", flush=True)
    print(f"\n  Total time: {elapsed/60:.1f}min", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()

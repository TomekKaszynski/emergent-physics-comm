"""
Phase 52: Transfer Test for Emergent Communication Protocol
============================================================
Tests whether the Phase 51 learned communication protocol transfers to
visually different scenarios — proving agents learned abstract physics
concepts, not pattern matching.

Loads trained Phase 51 model (frozen), evaluates on:
  - Original val set (sanity check, should match ~84.5%)
  - Near-transfer scenes (different colors, textures, camera angles, lighting)
  - Far-transfer scenes (different ball sizes, camera elevation, drop height)

All weights completely frozen — no fine-tuning.

Run from ~/AI/:
  python _phase52_transfer_test.py
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


# ══════════════════════════════════════════════════════════════════
# Architecture (exact copy from Phase 51)
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


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_video_dataset(dataset_dir, max_scenes=None):
    """Load a video dataset. Returns videos tensor, restitutions array, metadata list."""
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "index.json"

    with open(index_path) as f:
        index = json.load(f)

    if max_scenes:
        index = index[:max_scenes]

    n_sample_frames = 8
    frame_indices = np.linspace(0, 47, n_sample_frames, dtype=int)

    all_videos = []
    restitutions = []
    metadata_list = []

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
        metadata_list.append(meta)

    if len(all_videos) == 0:
        return None, None, []

    all_videos = np.stack(all_videos)
    restitutions = np.array(restitutions)

    all_videos_t = torch.tensor(all_videos, dtype=torch.float32)

    # ImageNet normalization (same as Phase 51)
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    all_videos_t = (all_videos_t - img_mean) / img_std

    return all_videos_t, restitutions, metadata_list


def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


def evaluate_frozen(sender, receiver, videos_t, restitutions, scene_ids,
                    device, n_rounds=50, batch_size=64):
    """Evaluate frozen sender+receiver on pairwise comparison."""
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
    """Evaluate accuracy by restitution gap bins."""
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
    """Get per-symbol mean restitution and counts."""
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

    # Entropy
    counts = np.bincount(all_msg_ids, minlength=vocab_size).astype(float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(vocab_size)
    n_used = int((probs > 0.01).sum())

    return all_msg_ids, symbol_stats, float(entropy), n_used


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("PHASE 52: Transfer Test for Emergent Communication", flush=True)
    print("  Frozen Phase 51 model evaluated on visually different scenes", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    vocab_size = 16
    hidden_dim = 128

    # ══════════════════════════════════════════════════════════════
    # STAGE 0: Load trained Phase 51 model (FROZEN)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 0: Load frozen Phase 51 model", flush=True)
    print(f"{'=' * 60}", flush=True)

    ckpt = torch.load("results/phase51_model.pt", map_location=device, weights_only=False)
    sender = PixelSender(hidden_dim, vocab_size, 8).to(device)
    receiver = Receiver(vocab_size, hidden_dim).to(device)
    sender.load_state_dict(ckpt['sender'])
    receiver.load_state_dict(ckpt['receiver'])
    sender.eval()
    receiver.eval()

    # Freeze all parameters
    for p in sender.parameters():
        p.requires_grad = False
    for p in receiver.parameters():
        p.requires_grad = False

    print(f"│  Loaded sender + receiver from results/phase51_model.pt", flush=True)
    print(f"│  All parameters frozen", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: Sanity check on original dataset
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 1: Sanity check — original training distribution", flush=True)
    print(f"{'=' * 60}", flush=True)

    orig_videos, orig_rest, orig_meta = load_video_dataset(
        "kubric/output/elasticity_dataset")

    if orig_videos is not None:
        n_orig = len(orig_rest)
        # Same train/val split as Phase 51
        n_train = int(0.8 * n_orig)
        perm = np.random.RandomState(42).permutation(n_orig)
        val_ids = perm[n_train:]

        print(f"│  Original dataset: {n_orig} scenes, val={len(val_ids)}", flush=True)

        with torch.no_grad():
            orig_acc, orig_n = evaluate_frozen(
                sender, receiver, orig_videos, orig_rest, val_ids, device)
            orig_diff = evaluate_by_difficulty(
                sender, receiver, orig_videos, orig_rest, val_ids, device)

        print(f"│  Original val accuracy: {orig_acc*100:.1f}% ({orig_n} pairs)", flush=True)
        print(f"│  Phase 51 reported:     84.5%", flush=True)
        results['original'] = {
            'accuracy': float(orig_acc),
            'n_pairs': orig_n,
            'difficulty': {k: v for k, v in orig_diff.items()},
        }

        # Symbol stats on original
        _, orig_sym_stats, orig_entropy, orig_n_used = get_symbol_stats(
            sender, orig_videos, orig_rest, device, vocab_size)
        results['original']['entropy'] = orig_entropy
        results['original']['symbols_used'] = orig_n_used
        results['original']['symbol_stats'] = orig_sym_stats

        # Phase 51 symbol ordering for comparison
        orig_ordering = sorted(
            [(s, v['mean_e']) for s, v in orig_sym_stats.items() if v['count'] >= 5],
            key=lambda x: x[1])
        orig_order_syms = [s for s, _ in orig_ordering]
        print(f"│  Symbol order (low→high): {orig_order_syms}", flush=True)
        print(f"│  Entropy: {orig_entropy:.3f}, Symbols: {orig_n_used}/{vocab_size}", flush=True)
        results['original']['symbol_ordering'] = orig_order_syms
    else:
        print(f"│  WARNING: Could not load original dataset", flush=True)
        orig_order_syms = []

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Transfer dataset evaluation
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 2: Transfer dataset evaluation (FROZEN protocol)", flush=True)
    print(f"{'=' * 60}", flush=True)

    transfer_dir = Path("kubric/output/transfer_dataset")
    if not transfer_dir.exists():
        print(f"│  ERROR: Transfer dataset not found at {transfer_dir}", flush=True)
        print(f"│  Run kubric/generate_transfer_dataset.py first", flush=True)
        return results

    transfer_videos, transfer_rest, transfer_meta = load_video_dataset(transfer_dir)

    if transfer_videos is None or len(transfer_meta) == 0:
        print(f"│  ERROR: No rendered transfer scenes found", flush=True)
        return results

    n_transfer = len(transfer_rest)
    print(f"│  Transfer dataset: {n_transfer} scenes", flush=True)
    print(f"│  Restitution range: [{transfer_rest.min():.3f}, {transfer_rest.max():.3f}]", flush=True)

    # Split into near and far transfer
    near_ids = []
    far_ids = []
    for i, meta in enumerate(transfer_meta):
        ttype = meta.get("transfer_type", "unknown")
        if ttype == "near":
            near_ids.append(i)
        elif ttype == "far":
            far_ids.append(i)
    near_ids = np.array(near_ids)
    far_ids = np.array(far_ids)

    print(f"│  Near-transfer: {len(near_ids)} scenes", flush=True)
    print(f"│  Far-transfer:  {len(far_ids)} scenes", flush=True)

    all_transfer_ids = np.arange(n_transfer)

    with torch.no_grad():
        # === Overall transfer accuracy ===
        all_acc, all_n = evaluate_frozen(
            sender, receiver, transfer_videos, transfer_rest, all_transfer_ids, device)
        print(f"│", flush=True)
        print(f"│  === ALL TRANSFER ===", flush=True)
        print(f"│  Accuracy: {all_acc*100:.1f}% ({all_n} pairs)", flush=True)
        results['all_transfer'] = {'accuracy': float(all_acc), 'n_pairs': all_n}

        # === Near-transfer accuracy ===
        if len(near_ids) >= 10:
            near_acc, near_n = evaluate_frozen(
                sender, receiver, transfer_videos, transfer_rest, near_ids, device)
            near_diff = evaluate_by_difficulty(
                sender, receiver, transfer_videos, transfer_rest, near_ids, device)
            print(f"│", flush=True)
            print(f"│  === NEAR-TRANSFER ===", flush=True)
            print(f"│  Accuracy: {near_acc*100:.1f}% ({near_n} pairs)", flush=True)
            for name in ["tiny", "small", "medium", "large"]:
                if name in near_diff:
                    d = near_diff[name]
                    print(f"│    {name:6s}: {d['acc']*100:.1f}% (n={d['n']})", flush=True)
            results['near_transfer'] = {
                'accuracy': float(near_acc), 'n_pairs': near_n,
                'difficulty': {k: v for k, v in near_diff.items()},
            }

        # === Far-transfer accuracy ===
        if len(far_ids) >= 10:
            far_acc, far_n = evaluate_frozen(
                sender, receiver, transfer_videos, transfer_rest, far_ids, device)
            far_diff = evaluate_by_difficulty(
                sender, receiver, transfer_videos, transfer_rest, far_ids, device)
            print(f"│", flush=True)
            print(f"│  === FAR-TRANSFER ===", flush=True)
            print(f"│  Accuracy: {far_acc*100:.1f}% ({far_n} pairs)", flush=True)
            for name in ["tiny", "small", "medium", "large"]:
                if name in far_diff:
                    d = far_diff[name]
                    print(f"│    {name:6s}: {d['acc']*100:.1f}% (n={d['n']})", flush=True)
            results['far_transfer'] = {
                'accuracy': float(far_acc), 'n_pairs': far_n,
                'difficulty': {k: v for k, v in far_diff.items()},
            }

        # === Cross-domain: pair original scene with transfer scene ===
        if orig_videos is not None and len(near_ids) >= 10:
            print(f"│", flush=True)
            print(f"│  === CROSS-DOMAIN (original ↔ transfer) ===", flush=True)

            # Sample pairs where one scene is from original, one from transfer
            rng_cross = np.random.RandomState(555)
            cross_correct = 0
            cross_total = 0
            for _ in range(50):
                bs = min(64, len(val_ids), n_transfer)
                idx_orig = rng_cross.choice(val_ids, size=bs)
                idx_xfer = rng_cross.choice(all_transfer_ids, size=bs)

                vid_orig = orig_videos[idx_orig].to(device)
                vid_xfer = transfer_videos[idx_xfer].to(device)
                rest_orig_batch = torch.tensor(orig_rest[idx_orig], dtype=torch.float32).to(device)
                rest_xfer_batch = torch.tensor(transfer_rest[idx_xfer], dtype=torch.float32).to(device)
                labels = (rest_orig_batch > rest_xfer_batch).float()

                msg_orig, _ = sender(vid_orig)
                msg_xfer, _ = sender(vid_xfer)
                pred = receiver(msg_orig, msg_xfer)

                cross_correct += ((pred > 0) == labels.bool()).sum().item()
                cross_total += len(labels)

            cross_acc = cross_correct / max(cross_total, 1)
            print(f"│  Cross-domain accuracy: {cross_acc*100:.1f}% ({cross_total} pairs)", flush=True)
            results['cross_domain'] = {'accuracy': float(cross_acc), 'n_pairs': cross_total}

        # === Symbol consistency check ===
        print(f"│", flush=True)
        print(f"│  === SYMBOL CONSISTENCY ===", flush=True)

        xfer_msg_ids, xfer_sym_stats, xfer_entropy, xfer_n_used = get_symbol_stats(
            sender, transfer_videos, transfer_rest, device, vocab_size)

        print(f"│  Transfer entropy: {xfer_entropy:.3f}, Symbols: {xfer_n_used}/{vocab_size}", flush=True)
        print(f"│  Symbol → Mean restitution (transfer):", flush=True)
        for s in sorted(xfer_sym_stats.keys()):
            v = xfer_sym_stats[s]
            if v['count'] >= 3:
                print(f"│    Symbol {s:2d}: mean_e={v['mean_e']:.3f} ± {v['std_e']:.3f} "
                      f"(n={v['count']})", flush=True)

        # Compare ordering
        xfer_ordering = sorted(
            [(s, v['mean_e']) for s, v in xfer_sym_stats.items() if v['count'] >= 3],
            key=lambda x: x[1])
        xfer_order_syms = [s for s, _ in xfer_ordering]
        print(f"│", flush=True)
        print(f"│  Original ordering:  {orig_order_syms}", flush=True)
        print(f"│  Transfer ordering:  {xfer_order_syms}", flush=True)

        # Check if orderings match
        common_syms = set(orig_order_syms) & set(xfer_order_syms)
        if len(common_syms) >= 2:
            # Compare relative ordering of common symbols
            orig_ranks = {s: i for i, s in enumerate(orig_order_syms)}
            xfer_ranks = {s: i for i, s in enumerate(xfer_order_syms)}

            concordant = 0
            discordant = 0
            for i, s1 in enumerate(list(common_syms)):
                for s2 in list(common_syms)[i+1:]:
                    orig_order = orig_ranks[s1] < orig_ranks[s2]
                    xfer_order = xfer_ranks[s1] < xfer_ranks[s2]
                    if orig_order == xfer_order:
                        concordant += 1
                    else:
                        discordant += 1

            if concordant + discordant > 0:
                kendall_tau = (concordant - discordant) / (concordant + discordant)
            else:
                kendall_tau = 0.0

            print(f"│  Common symbols: {sorted(common_syms)}", flush=True)
            print(f"│  Kendall's τ (ordering agreement): {kendall_tau:.3f}", flush=True)
            print(f"│  (1.0 = perfect agreement, 0.0 = random, -1.0 = reversed)", flush=True)

            if kendall_tau > 0.8:
                order_verdict = "STRONG — symbol semantics preserved"
            elif kendall_tau > 0.5:
                order_verdict = "MODERATE — partial semantic transfer"
            elif kendall_tau > 0.0:
                order_verdict = "WEAK — some ordering preserved"
            else:
                order_verdict = "NONE — ordering not preserved"

            print(f"│  Ordering verdict: {order_verdict}", flush=True)
            results['symbol_consistency'] = {
                'kendall_tau': float(kendall_tau),
                'common_symbols': sorted(list(common_syms)),
                'original_ordering': orig_order_syms,
                'transfer_ordering': xfer_order_syms,
                'transfer_symbol_stats': xfer_sym_stats,
                'transfer_entropy': xfer_entropy,
                'verdict': order_verdict,
            }

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: Save and visualize
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE 3: Save results and visualize", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_json = json.loads(json.dumps(results, default=convert))
    with open("results/phase52_transfer.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"│  Saved results/phase52_transfer.json", flush=True)

    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Accuracy comparison bar chart
    ax = axes[0, 0]
    conditions = ['Original\nval']
    accuracies = [results.get('original', {}).get('accuracy', 0) * 100]
    colors = ['#2196F3']

    if 'near_transfer' in results:
        conditions.append('Near\ntransfer')
        accuracies.append(results['near_transfer']['accuracy'] * 100)
        colors.append('#4CAF50')

    if 'far_transfer' in results:
        conditions.append('Far\ntransfer')
        accuracies.append(results['far_transfer']['accuracy'] * 100)
        colors.append('#FF9800')

    if 'cross_domain' in results:
        conditions.append('Cross\ndomain')
        accuracies.append(results['cross_domain']['accuracy'] * 100)
        colors.append('#9C27B0')

    conditions.append('Chance')
    accuracies.append(50.0)
    colors.append('#9E9E9E')

    ax.bar(range(len(conditions)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Frozen Protocol Transfer', fontsize=12, fontweight='bold')
    ax.set_ylim(40, 100)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=84.5, color='blue', linestyle=':', alpha=0.3, label='Phase 51')
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: Accuracy by difficulty — near transfer
    ax = axes[0, 1]
    if 'near_transfer' in results and 'difficulty' in results['near_transfer']:
        diff = results['near_transfer']['difficulty']
        names = ['tiny', 'small', 'medium', 'large']
        accs = [diff.get(n, {}).get('acc', 0) * 100 for n in names]
        bar_colors = ['#FFCDD2', '#EF9A9A', '#E57373', '#F44336']
        ax.bar(names, accs, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Near-Transfer by Difficulty', fontsize=11)
        ax.set_ylim(40, 105)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        for i, v in enumerate(accs):
            if v > 0:
                ax.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No near-transfer data', transform=ax.transAxes, ha='center')
        ax.set_title('Near-Transfer by Difficulty')

    # Panel 3: Accuracy by difficulty — far transfer
    ax = axes[0, 2]
    if 'far_transfer' in results and 'difficulty' in results['far_transfer']:
        diff = results['far_transfer']['difficulty']
        names = ['tiny', 'small', 'medium', 'large']
        accs = [diff.get(n, {}).get('acc', 0) * 100 for n in names]
        bar_colors = ['#FFE0B2', '#FFCC80', '#FFB74D', '#FF9800']
        ax.bar(names, accs, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Far-Transfer by Difficulty', fontsize=11)
        ax.set_ylim(40, 105)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        for i, v in enumerate(accs):
            if v > 0:
                ax.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No far-transfer data', transform=ax.transAxes, ha='center')
        ax.set_title('Far-Transfer by Difficulty')

    # Panel 4: Symbol usage comparison (original vs transfer)
    ax = axes[1, 0]
    orig_counts = np.zeros(vocab_size)
    xfer_counts = np.zeros(vocab_size)
    if 'original' in results and 'symbol_stats' in results['original']:
        for s, v in results['original']['symbol_stats'].items():
            orig_counts[int(s)] = v['count']
    if 'symbol_consistency' in results and 'transfer_symbol_stats' in results['symbol_consistency']:
        for s, v in results['symbol_consistency']['transfer_symbol_stats'].items():
            xfer_counts[int(s)] = v['count']

    orig_probs = orig_counts / max(orig_counts.sum(), 1)
    xfer_probs = xfer_counts / max(xfer_counts.sum(), 1)

    x = np.arange(vocab_size)
    w = 0.35
    ax.bar(x - w/2, orig_probs, w, label='Original', color='#2196F3', alpha=0.7)
    ax.bar(x + w/2, xfer_probs, w, label='Transfer', color='#FF9800', alpha=0.7)
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Frequency')
    ax.set_title('Symbol Usage: Original vs Transfer', fontsize=11)
    ax.set_xticks(range(vocab_size))
    ax.legend(fontsize=9)

    # Panel 5: Symbol → restitution scatter (transfer)
    ax = axes[1, 1]
    if 'symbol_consistency' in results and 'transfer_symbol_stats' in results['symbol_consistency']:
        for s, v in results['symbol_consistency']['transfer_symbol_stats'].items():
            if v['count'] >= 3:
                ax.errorbar(int(s), v['mean_e'], yerr=v['std_e'],
                           fmt='o', markersize=max(3, min(12, v['count']/3)),
                           color='#FF9800', capsize=3)
                ax.annotate(f"n={v['count']}", (int(s), v['mean_e']),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Mean Restitution')
    ax.set_title('Transfer: Symbol → Elasticity', fontsize=11)
    ax.set_xticks(range(vocab_size))
    ax.set_ylim(0, 1)

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = time.time() - t0

    lines = ["Phase 52: Transfer Test\n"]
    lines.append(f"  Frozen Phase 51 protocol\n")

    if 'original' in results:
        lines.append(f"Original val:     {results['original']['accuracy']*100:.1f}%")
    if 'near_transfer' in results:
        lines.append(f"Near-transfer:    {results['near_transfer']['accuracy']*100:.1f}%")
    if 'far_transfer' in results:
        lines.append(f"Far-transfer:     {results['far_transfer']['accuracy']*100:.1f}%")
    if 'cross_domain' in results:
        lines.append(f"Cross-domain:     {results['cross_domain']['accuracy']*100:.1f}%")
    lines.append(f"Chance:           50.0%\n")

    if 'symbol_consistency' in results:
        sc = results['symbol_consistency']
        lines.append(f"Symbol ordering:")
        lines.append(f"  Kendall τ: {sc['kendall_tau']:.3f}")
        lines.append(f"  {sc['verdict']}\n")

    # Determine overall verdict
    near_acc = results.get('near_transfer', {}).get('accuracy', 0)
    far_acc = results.get('far_transfer', {}).get('accuracy', 0)

    if near_acc > 0.75:
        verdict = "STRONG TRANSFER"
    elif near_acc > 0.65:
        verdict = "MODERATE TRANSFER"
    elif near_acc > 0.55:
        verdict = "WEAK TRANSFER"
    else:
        verdict = "NO TRANSFER"

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"Time: {elapsed:.0f}s")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top')

    fig.suptitle(f'Phase 52: Communication Protocol Transfer Test\n'
                 f'near={near_acc*100:.0f}% far={far_acc*100:.0f}% | {verdict}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/phase52_transfer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  Saved results/phase52_transfer.png", flush=True)

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"PHASE 52 RESULTS: {verdict}", flush=True)
    print(f"{'=' * 70}", flush=True)
    if 'original' in results:
        print(f"  Original val:     {results['original']['accuracy']*100:.1f}%", flush=True)
    if 'near_transfer' in results:
        print(f"  Near-transfer:    {results['near_transfer']['accuracy']*100:.1f}% (target >75%=strong, >65%=moderate)", flush=True)
    if 'far_transfer' in results:
        print(f"  Far-transfer:     {results['far_transfer']['accuracy']*100:.1f}% (target >65%=remarkable)", flush=True)
    if 'cross_domain' in results:
        print(f"  Cross-domain:     {results['cross_domain']['accuracy']*100:.1f}%", flush=True)
    print(f"  Chance:           50.0%", flush=True)
    if 'symbol_consistency' in results:
        sc = results['symbol_consistency']
        print(f"  Symbol ordering:  Kendall τ = {sc['kendall_tau']:.3f} — {sc['verdict']}", flush=True)
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    print(f"{'=' * 70}", flush=True)

    return results


if __name__ == "__main__":
    main()

"""
Phase 66: One-Shot Visual Concept Learning — Referential Game
==============================================================
Tests whether compositional communication enables one-shot generalization
to NOVEL visual categories never seen during training.

Dataset: CIFAR-100 (100 classes, 20 superclasses)
- DINOv2 ViT-S/14 features (384-dim per image)
- 80 train classes (16 superclasses), 20 test classes (4 holdout superclasses)

Task: Referential game
- Sender sees reference image → discrete message
- Receiver sees K=5 candidates + message → selects match
- HARD: distractors from same superclass; EASY: random distractors

Five conditions × 10 seeds:
  1. COMPOSITIONAL_HARD: 4 pos × vocab 10, same-superclass distractors
  2. COMPOSITIONAL_EASY: 4 pos × vocab 10, random distractors
  3. HOLISTIC_HARD: 1 pos × vocab 100, same-superclass distractors
  4. HOLISTIC_EASY: 1 pos × vocab 100, random distractors
  5. NEAREST_NEIGHBOR: DINOv2 cosine similarity (no training)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase66_oneshot.py
"""

import os
import time
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from pathlib import Path

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
CIFAR_ROOT = "./cifar100_data"
CACHE_PATH = RESULTS_DIR / "phase66_dino_cifar100.pt"

HIDDEN_DIM = 128
DINO_DIM = 384
BATCH_SIZE = 64
K = 5  # candidates per episode

N_EPOCHS = 200
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 2.0
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

N_RECEIVERS = 3
RECEIVER_RESET_INTERVAL = 30
N_BATCHES_PER_EPOCH = 32  # 32 × 64 = 2048 episodes per epoch

N_SEEDS = 10
N_EVAL_EPISODES = 2000

# Holdout 4 superclasses (20 test classes)
HOLDOUT_SUPERCLASS_NAMES = {
    'large_carnivores', 'vehicles_1', 'insects', 'people'
}


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_cifar100_with_coarse(root=CIFAR_ROOT):
    """Load CIFAR-100 images with both fine and coarse labels."""
    import torchvision.datasets
    print("  Downloading CIFAR-100 (if needed)...", flush=True)
    torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=root, train=False, download=True)

    base = os.path.join(root, 'cifar-100-python')

    with open(os.path.join(base, 'meta'), 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
    fine_names = meta['fine_label_names']
    coarse_names = meta['coarse_label_names']

    with open(os.path.join(base, 'train'), 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
    with open(os.path.join(base, 'test'), 'rb') as f:
        test_data = pickle.load(f, encoding='latin1')

    # Combine train + test: 60K images
    images = np.concatenate([
        train_data['data'].reshape(-1, 3, 32, 32),
        test_data['data'].reshape(-1, 3, 32, 32),
    ], axis=0)  # (60000, 3, 32, 32)

    fine_labels = np.array(
        train_data['fine_labels'] + test_data['fine_labels'])
    coarse_labels = np.array(
        train_data['coarse_labels'] + test_data['coarse_labels'])

    print(f"  CIFAR-100: {len(images)} images, "
          f"{len(fine_names)} fine classes, {len(coarse_names)} superclasses",
          flush=True)

    return images, fine_labels, coarse_labels, fine_names, coarse_names


def extract_dino_features(images, cache_path, device):
    """Extract DINOv2 CLS tokens for all CIFAR-100 images → (60000, 384)."""
    if os.path.exists(cache_path):
        print(f"  Loading cached DINOv2 features from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=False)
        print(f"  Features shape: {data['features'].shape}", flush=True)
        return data['features']

    print("  Loading DINOv2 ViT-S/14...", flush=True)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                          pretrained=True)
    dino.eval().to(device)
    for p in dino.parameters():
        p.requires_grad = False

    dino_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    dino_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

    all_features = []
    batch_size = 64
    t_start = time.time()

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]  # (B, 3, 32, 32) uint8
        batch_t = torch.tensor(batch, dtype=torch.float32) / 255.0
        # Upscale 32×32 → 224×224
        batch_t = F.interpolate(batch_t, size=(224, 224),
                                mode='bilinear', align_corners=False)
        batch_t = batch_t.to(device)
        batch_t = (batch_t - dino_mean) / dino_std

        with torch.no_grad():
            cls_tokens = dino(batch_t)  # (B, 384)
        all_features.append(cls_tokens.cpu())

        step = i // batch_size + 1
        if step % 100 == 0:
            elapsed = time.time() - t_start
            total_steps = len(images) // batch_size
            eta = elapsed / step * (total_steps - step)
            print(f"    [{i + batch_size}/{len(images)}] "
                  f"{elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

        if device.type == 'mps' and step % 100 == 0:
            torch.mps.empty_cache()

    features = torch.cat(all_features, dim=0)  # (60000, 384)
    print(f"  DINOv2 features: {features.shape} "
          f"({features.numel() * 4 / 1e6:.1f} MB)", flush=True)

    torch.save({'features': features}, cache_path)
    print(f"  Cached to {cache_path}", flush=True)

    del dino
    if device.type == 'mps':
        torch.mps.empty_cache()

    return features


# ══════════════════════════════════════════════════════════════════
# Class splits
# ══════════════════════════════════════════════════════════════════

def get_class_splits(fine_labels, coarse_labels, coarse_names):
    """Split fine classes into train/test by superclass.

    Returns:
        train_classes: list of fine class indices for training
        test_classes: list of fine class indices for testing
        class_to_indices: dict mapping fine class → list of image indices
        fine_to_coarse: dict mapping fine class → coarse class
        coarse_to_fines: dict mapping coarse class → list of fine classes
    """
    # Map superclass names to indices
    holdout_coarse_ids = set()
    for i, name in enumerate(coarse_names):
        if name in HOLDOUT_SUPERCLASS_NAMES:
            holdout_coarse_ids.add(i)

    print(f"  Holdout superclasses: {holdout_coarse_ids}", flush=True)

    # Build fine-to-coarse mapping
    fine_to_coarse = {}
    for fl, cl in zip(fine_labels, coarse_labels):
        fine_to_coarse[int(fl)] = int(cl)

    # Split classes
    train_classes = sorted([fc for fc, cc in fine_to_coarse.items()
                           if cc not in holdout_coarse_ids])
    test_classes = sorted([fc for fc, cc in fine_to_coarse.items()
                          if cc in holdout_coarse_ids])

    # Build class→image_indices
    class_to_indices = {}
    for i, fl in enumerate(fine_labels):
        fl = int(fl)
        if fl not in class_to_indices:
            class_to_indices[fl] = []
        class_to_indices[fl].append(i)

    # Build coarse→fines
    coarse_to_fines = {}
    for fc, cc in fine_to_coarse.items():
        if cc not in coarse_to_fines:
            coarse_to_fines[cc] = []
        if fc not in coarse_to_fines[cc]:
            coarse_to_fines[cc].append(fc)

    print(f"  Train: {len(train_classes)} classes, "
          f"Test: {len(test_classes)} classes", flush=True)

    return (train_classes, test_classes, class_to_indices,
            fine_to_coarse, coarse_to_fines)


# ══════════════════════════════════════════════════════════════════
# Episode sampling
# ══════════════════════════════════════════════════════════════════

def sample_episodes(class_set, class_to_indices, fine_to_coarse,
                    coarse_to_fines, batch_size, mode='easy', rng=None):
    """Sample a batch of referential game episodes.

    Returns:
        ref_indices: (B,) image indices for sender
        candidate_indices: (B, K) image indices for receiver
        target_pos: (B,) which candidate is the match
    """
    if rng is None:
        rng = np.random.RandomState()

    ref_indices = np.zeros(batch_size, dtype=np.int64)
    candidate_indices = np.zeros((batch_size, K), dtype=np.int64)
    target_pos = np.zeros(batch_size, dtype=np.int64)

    for b in range(batch_size):
        target_class = rng.choice(class_set)
        target_images = class_to_indices[target_class]
        ref_idx, match_idx = rng.choice(target_images, size=2, replace=False)
        ref_indices[b] = ref_idx

        if mode == 'hard':
            target_super = fine_to_coarse[target_class]
            siblings = [c for c in coarse_to_fines[target_super]
                       if c != target_class and c in class_set]
            if len(siblings) >= K - 1:
                distractor_classes = rng.choice(
                    siblings, size=K - 1, replace=False)
            else:
                # Pad with random classes if not enough siblings
                extra = [c for c in class_set
                        if c != target_class and c not in siblings]
                distractor_classes = list(siblings) + list(
                    rng.choice(extra, size=K - 1 - len(siblings), replace=False))
        else:
            other = [c for c in class_set if c != target_class]
            distractor_classes = rng.choice(other, size=K - 1, replace=False)

        distractors = [rng.choice(class_to_indices[int(dc)])
                      for dc in distractor_classes]

        candidates = [match_idx] + distractors
        perm = rng.permutation(K)
        candidates = [candidates[p] for p in perm]
        target_pos[b] = int(np.where(perm == 0)[0][0])
        candidate_indices[b] = candidates

    return ref_indices, candidate_indices, target_pos


# ══════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════

class Sender(nn.Module):
    """Referential game sender. DINOv2 features → discrete message."""
    def __init__(self, input_dim=384, hidden_dim=128,
                 n_positions=4, vocab_size=10):
        super().__init__()
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_positions * vocab_size)

    def forward(self, x, tau=1.0, hard=True):
        """
        x: (B, 384) DINOv2 features
        Returns message (B, n_pos*vocab), logits (B, n_pos, vocab)
        """
        h = self.encoder(x)
        logits = self.head(h).view(-1, self.n_positions, self.vocab_size)

        if self.training:
            flat = logits.reshape(-1, self.vocab_size)
            tokens = F.gumbel_softmax(flat, tau=tau, hard=hard)
            tokens = tokens.reshape(-1, self.n_positions, self.vocab_size)
        else:
            idx = logits.argmax(dim=-1)
            tokens = F.one_hot(idx, self.vocab_size).float()

        message = tokens.reshape(-1, self.n_positions * self.vocab_size)
        return message, logits


class Receiver(nn.Module):
    """Referential game receiver. Scores candidates against message."""
    def __init__(self, msg_dim, feat_dim=384, hidden_dim=128):
        super().__init__()
        self.embed_dim = hidden_dim
        self.msg_encoder = nn.Sequential(
            nn.Linear(msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.cand_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )

    def forward(self, message, candidates):
        """
        message: (B, msg_dim) flattened one-hot
        candidates: (B, K, feat_dim)
        Returns: scores (B, K)
        """
        msg_emb = self.msg_encoder(message)  # (B, embed_dim)
        B, Kn, D = candidates.shape
        cand_emb = self.cand_encoder(
            candidates.reshape(B * Kn, D))  # (B*K, embed_dim)
        cand_emb = cand_emb.reshape(B, Kn, -1)  # (B, K, embed_dim)
        scores = torch.einsum('bd,bkd->bk', msg_emb, cand_emb)
        return scores


# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_referential(sender, receiver, features, class_set,
                         class_to_indices, fine_to_coarse, coarse_to_fines,
                         device, mode='easy', n_episodes=N_EVAL_EPISODES):
    """Evaluate referential game accuracy."""
    sender.eval()
    receiver.eval()
    rng = np.random.RandomState(12345)

    correct = 0
    total = 0

    with torch.no_grad():
        n_batches = max(1, n_episodes // BATCH_SIZE)
        for _ in range(n_batches):
            ref_idx, cand_idx, tgt = sample_episodes(
                class_set, class_to_indices, fine_to_coarse,
                coarse_to_fines, BATCH_SIZE, mode=mode, rng=rng)

            ref_feat = features[ref_idx].to(device)
            cand_feat = features[cand_idx].to(device)
            target = torch.tensor(tgt, device=device, dtype=torch.long)

            msg, _ = sender(ref_feat)
            scores = receiver(msg, cand_feat)
            preds = scores.argmax(dim=-1)

            correct += (preds == target).sum().item()
            total += len(target)

    return correct / max(total, 1)


def evaluate_population_referential(sender, receivers, features, class_set,
                                     class_to_indices, fine_to_coarse,
                                     coarse_to_fines, device, mode='easy',
                                     n_episodes=N_EVAL_EPISODES):
    """Pick best receiver from population, then evaluate."""
    best_acc = -1
    best_r = None
    for r in receivers:
        acc = evaluate_referential(
            sender, r, features, class_set,
            class_to_indices, fine_to_coarse, coarse_to_fines,
            device, mode=mode, n_episodes=min(640, n_episodes))
        if acc > best_acc:
            best_acc = acc
            best_r = r

    final_acc = evaluate_referential(
        sender, best_r, features, class_set,
        class_to_indices, fine_to_coarse, coarse_to_fines,
        device, mode=mode, n_episodes=n_episodes)
    return final_acc, best_r


# ══════════════════════════════════════════════════════════════════
# Analysis metrics
# ══════════════════════════════════════════════════════════════════

def compute_analysis(sender, features, fine_labels, device,
                     n_positions, vocab_size):
    """Compute TopSim, per-position entropy, consistency, unique messages."""
    sender.eval()
    all_tokens = []
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch = features[i:i + BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens = logits.argmax(dim=-1).cpu().numpy()
            all_tokens.append(tokens)
            all_feats.append(batch.cpu())

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, n_pos)
    all_feats = torch.cat(all_feats, dim=0)  # (N, 384)

    # Per-position entropy (normalized)
    entropies = []
    max_ent = np.log(vocab_size) if vocab_size > 1 else 1.0
    for p in range(n_positions):
        counts = np.bincount(all_tokens[:, p], minlength=vocab_size)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / max_ent))

    # Unique messages
    msg_tuples = [tuple(row) for row in all_tokens]
    n_unique = len(set(msg_tuples))

    # Message consistency per class
    classes_present = np.unique(fine_labels)
    consistencies = []
    for cls in classes_present:
        mask = fine_labels == cls
        cls_msgs = [msg_tuples[i] for i in range(len(msg_tuples)) if mask[i]]
        if len(cls_msgs) > 0:
            most_common = max(set(cls_msgs), key=cls_msgs.count)
            consistencies.append(cls_msgs.count(most_common) / len(cls_msgs))
    mean_consistency = float(np.mean(consistencies)) if consistencies else 0.0

    # Topographic similarity (Spearman: Hamming distance vs cosine distance)
    rng = np.random.RandomState(42)
    n_pairs = 5000
    idx1 = rng.randint(0, len(features), size=n_pairs)
    idx2 = rng.randint(0, len(features), size=n_pairs)
    # Avoid same-image pairs
    same = idx1 == idx2
    while same.any():
        idx2[same] = rng.randint(0, len(features), size=same.sum())
        same = idx1 == idx2

    # Hamming distance
    msg_dist = (all_tokens[idx1] != all_tokens[idx2]).sum(axis=1).astype(float)

    # Cosine distance
    f1 = all_feats[idx1]
    f2 = all_feats[idx2]
    cos_sim = F.cosine_similarity(f1, f2, dim=-1).numpy()
    feat_dist = 1.0 - cos_sim

    topsim_val, _ = stats.spearmanr(msg_dist, feat_dist)
    if np.isnan(topsim_val):
        topsim_val = 0.0

    return {
        'topsim': float(topsim_val),
        'entropies': entropies,
        'mean_entropy': float(np.mean(entropies)),
        'n_unique_messages': n_unique,
        'mean_consistency': mean_consistency,
    }


# ══════════════════════════════════════════════════════════════════
# Nearest neighbor baseline
# ══════════════════════════════════════════════════════════════════

def nearest_neighbor_baseline(features, class_set, class_to_indices,
                               fine_to_coarse, coarse_to_fines,
                               mode='easy', n_episodes=N_EVAL_EPISODES,
                               n_runs=10):
    """Cosine similarity baseline — no training."""
    accs = []
    for run in range(n_runs):
        rng = np.random.RandomState(run * 1000)
        correct = 0
        total = 0

        n_batches = max(1, n_episodes // BATCH_SIZE)
        for _ in range(n_batches):
            ref_idx, cand_idx, tgt = sample_episodes(
                class_set, class_to_indices, fine_to_coarse,
                coarse_to_fines, BATCH_SIZE, mode=mode, rng=rng)

            ref_feat = features[ref_idx]  # (B, 384)
            cand_feat = features[cand_idx]  # (B, K, 384)

            ref_norm = F.normalize(ref_feat, dim=-1).unsqueeze(1)  # (B, 1, 384)
            cand_norm = F.normalize(cand_feat, dim=-1)  # (B, K, 384)
            sims = (ref_norm * cand_norm).sum(dim=-1)  # (B, K)
            preds = sims.argmax(dim=-1)

            target = torch.tensor(tgt, dtype=torch.long)
            correct += (preds == target).sum().item()
            total += len(target)

        accs.append(correct / max(total, 1))

    return float(np.mean(accs)), float(np.std(accs))


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_seed(seed, features, fine_labels, coarse_labels,
               train_classes, test_classes, class_to_indices,
               fine_to_coarse, coarse_to_fines,
               n_positions, vocab_size, mode, device, tag=''):
    """Train one seed for one condition. Returns result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    msg_dim = n_positions * vocab_size

    sender = Sender(DINO_DIM, HIDDEN_DIM, n_positions, vocab_size).to(device)
    receivers = [Receiver(msg_dim, DINO_DIM, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    max_entropy = math.log(vocab_size)
    best_train_acc = 0.0
    best_states = None
    nan_count = 0
    t_start = time.time()

    features_dev = features.to(device)

    for epoch in range(N_EPOCHS):
        # Population IL: reset receivers
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            for i in range(N_RECEIVERS):
                receivers[i] = Receiver(msg_dim, DINO_DIM, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, N_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for _ in range(N_BATCHES_PER_EPOCH):
            ref_idx, cand_idx, tgt = sample_episodes(
                train_classes, class_to_indices, fine_to_coarse,
                coarse_to_fines, BATCH_SIZE, mode=mode, rng=rng)

            ref_feat = features_dev[ref_idx]
            cand_feat = features_dev[cand_idx]
            target = torch.tensor(tgt, device=device, dtype=torch.long)

            msg, logits = sender(ref_feat, tau=tau, hard=hard)

            # Average loss over receiver population
            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                scores = r(msg, cand_feat)
                r_loss = F.cross_entropy(scores, target)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Track accuracy (use first receiver for speed)
            with torch.no_grad():
                scores_0 = receivers[0](msg, cand_feat)
                preds = scores_0.argmax(dim=-1)
                epoch_correct += (preds == target).sum().item()
                epoch_total += len(target)

            # Entropy regularization
            for p in range(n_positions):
                pos_logits = logits[:, p, :]
                log_probs = F.log_softmax(pos_logits, dim=-1)
                probs = log_probs.exp().clamp(min=1e-8)
                ent = -(probs * log_probs).sum(dim=-1).mean()
                rel_ent = ent / max_entropy
                if rel_ent < ENTROPY_THRESHOLD:
                    loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                continue

            sender_opt.zero_grad()
            for opt in receiver_opts:
                opt.zero_grad()
            loss.backward()

            # NaN gradient check
            has_nan_grad = False
            all_params = list(sender.parameters())
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for param in all_params:
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or
                    torch.isinf(param.grad).any()
                ):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

            epoch_loss += loss.item()

        if epoch % 50 == 0 and device.type == 'mps':
            torch.mps.empty_cache()

        # Periodic evaluation
        if (epoch + 1) % 20 == 0 or epoch == 0:
            train_acc = epoch_correct / max(epoch_total, 1)
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"        Ep {epoch+1:3d}: "
                  f"train_acc={train_acc:.1%}  "
                  f"loss={epoch_loss / N_BATCHES_PER_EPOCH:.3f}{nan_str}  "
                  f"ETA {eta/60:.1f}min", flush=True)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_states = {
                    'sender': {k: v.cpu().clone()
                              for k, v in sender.state_dict().items()},
                    'receivers': [
                        {k: v.cpu().clone()
                         for k, v in r.state_dict().items()}
                        for r in receivers
                    ],
                }

    # Restore best
    if best_states is not None:
        sender.load_state_dict(best_states['sender'])
        sender.to(device)
        for i, r in enumerate(receivers):
            r.load_state_dict(best_states['receivers'][i])
            r.to(device)

    # Final evaluation: 4 settings
    results = {}

    # Pick best receiver on train-mode
    _, best_r = evaluate_population_referential(
        sender, receivers, features, train_classes,
        class_to_indices, fine_to_coarse, coarse_to_fines,
        device, mode=mode, n_episodes=N_EVAL_EPISODES)

    for eval_split, eval_classes in [('train', train_classes),
                                      ('test', test_classes)]:
        for eval_mode in ['easy', 'hard']:
            acc = evaluate_referential(
                sender, best_r, features, eval_classes,
                class_to_indices, fine_to_coarse, coarse_to_fines,
                device, mode=eval_mode, n_episodes=N_EVAL_EPISODES)
            results[f'{eval_split}_{eval_mode}'] = acc

    # Analysis on all images
    analysis = compute_analysis(
        sender, features, fine_labels, device, n_positions, vocab_size)
    results['analysis'] = analysis
    results['nan_count'] = nan_count

    return results


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("Phase 66: One-Shot Visual Concept Learning — Referential Game",
          flush=True)
    print("=" * 70, flush=True)
    t_global = time.time()

    # ── Stage 0: Load data ──
    print("\n[Stage 0] Loading CIFAR-100 + DINOv2 features...", flush=True)
    images, fine_labels, coarse_labels, fine_names, coarse_names = \
        load_cifar100_with_coarse()
    features = extract_dino_features(images, CACHE_PATH, DEVICE)

    (train_classes, test_classes, class_to_indices,
     fine_to_coarse, coarse_to_fines) = get_class_splits(
        fine_labels, coarse_labels, coarse_names)

    # Print holdout info
    for cc in sorted(coarse_to_fines.keys()):
        fines = coarse_to_fines[cc]
        if any(fc in test_classes for fc in fines):
            names = [fine_names[fc] for fc in fines]
            print(f"  Holdout superclass {coarse_names[cc]}: {names}",
                  flush=True)

    # ── Stage 1: Nearest neighbor baseline ──
    print("\n[Stage 1] Nearest neighbor baseline...", flush=True)
    nn_results = {}
    for mode_label in ['easy', 'hard']:
        for split_label, split_classes in [('train', train_classes),
                                           ('test', test_classes)]:
            acc_mean, acc_std = nearest_neighbor_baseline(
                features, split_classes, class_to_indices,
                fine_to_coarse, coarse_to_fines,
                mode=mode_label, n_episodes=N_EVAL_EPISODES, n_runs=10)
            key = f'{split_label}_{mode_label}'
            nn_results[key] = {'mean': acc_mean, 'std': acc_std}
            print(f"  NN {key}: {acc_mean:.1%} ± {acc_std:.1%}", flush=True)

    # ── Stage 2: Train all conditions ──
    conditions = [
        {'name': 'compositional_hard', 'n_pos': 4, 'vocab': 10, 'mode': 'hard'},
        {'name': 'compositional_easy', 'n_pos': 4, 'vocab': 10, 'mode': 'easy'},
        {'name': 'holistic_hard', 'n_pos': 1, 'vocab': 100, 'mode': 'hard'},
        {'name': 'holistic_easy', 'n_pos': 1, 'vocab': 100, 'mode': 'easy'},
    ]

    all_results = {}

    for cond in conditions:
        cname = cond['name']
        print(f"\n[Stage 2] Condition: {cname} "
              f"({cond['n_pos']} pos × vocab {cond['vocab']}, "
              f"{cond['mode']} distractors)", flush=True)

        seed_results = []
        for s in range(N_SEEDS):
            seed = s * 100 + 42
            print(f"    Seed {s+1}/{N_SEEDS} (seed={seed}):", flush=True)

            result = train_seed(
                seed, features, fine_labels, coarse_labels,
                train_classes, test_classes, class_to_indices,
                fine_to_coarse, coarse_to_fines,
                n_positions=cond['n_pos'], vocab_size=cond['vocab'],
                mode=cond['mode'], device=DEVICE, tag=cname)

            seed_results.append(result)

            print(f"      → train_easy={result['train_easy']:.1%}  "
                  f"train_hard={result['train_hard']:.1%}  "
                  f"test_easy={result['test_easy']:.1%}  "
                  f"test_hard={result['test_hard']:.1%}  "
                  f"topsim={result['analysis']['topsim']:.3f}",
                  flush=True)

        # Aggregate
        agg = {}
        for key in ['train_easy', 'train_hard', 'test_easy', 'test_hard']:
            vals = [r[key] for r in seed_results]
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))

        # Analysis averages
        topsim_vals = [r['analysis']['topsim'] for r in seed_results]
        unique_vals = [r['analysis']['n_unique_messages']
                      for r in seed_results]
        consist_vals = [r['analysis']['mean_consistency']
                       for r in seed_results]
        agg['topsim_mean'] = float(np.mean(topsim_vals))
        agg['topsim_std'] = float(np.std(topsim_vals))
        agg['n_unique_mean'] = float(np.mean(unique_vals))
        agg['consistency_mean'] = float(np.mean(consist_vals))
        agg['seeds'] = seed_results

        all_results[cname] = agg

        # Train-test gaps
        gap_easy = agg['train_easy_mean'] - agg['test_easy_mean']
        gap_hard = agg['train_hard_mean'] - agg['test_hard_mean']
        print(f"  {cname} summary:", flush=True)
        print(f"    Train easy: {agg['train_easy_mean']:.1%} ± "
              f"{agg['train_easy_std']:.1%}", flush=True)
        print(f"    Train hard: {agg['train_hard_mean']:.1%} ± "
              f"{agg['train_hard_std']:.1%}", flush=True)
        print(f"    Test easy:  {agg['test_easy_mean']:.1%} ± "
              f"{agg['test_easy_std']:.1%}", flush=True)
        print(f"    Test hard:  {agg['test_hard_mean']:.1%} ± "
              f"{agg['test_hard_std']:.1%}", flush=True)
        print(f"    Gap easy: {gap_easy:+.1%}  Gap hard: {gap_hard:+.1%}",
              flush=True)
        print(f"    TopSim: {agg['topsim_mean']:.3f} ± "
              f"{agg['topsim_std']:.3f}  "
              f"Unique msgs: {agg['n_unique_mean']:.0f}  "
              f"Consistency: {agg['consistency_mean']:.3f}", flush=True)

    # ── Stage 3: Summary ──
    elapsed = time.time() - t_global
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    # Header
    print(f"\n{'Condition':<25} {'Train Easy':>11} {'Train Hard':>11} "
          f"{'Test Easy':>11} {'Test Hard':>11} {'Gap Hard':>9} "
          f"{'TopSim':>8}", flush=True)
    print("-" * 95, flush=True)

    # NN baseline
    print(f"{'nearest_neighbor':<25} "
          f"{nn_results['train_easy']['mean']:>10.1%}  "
          f"{nn_results['train_hard']['mean']:>10.1%}  "
          f"{nn_results['test_easy']['mean']:>10.1%}  "
          f"{nn_results['test_hard']['mean']:>10.1%}  "
          f"{'—':>9} {'—':>8}", flush=True)

    # Trained conditions
    for cname in ['compositional_hard', 'compositional_easy',
                  'holistic_hard', 'holistic_easy']:
        if cname not in all_results:
            continue
        a = all_results[cname]
        gap = a['train_hard_mean'] - a['test_hard_mean']
        print(f"{cname:<25} "
              f"{a['train_easy_mean']:>10.1%}  "
              f"{a['train_hard_mean']:>10.1%}  "
              f"{a['test_easy_mean']:>10.1%}  "
              f"{a['test_hard_mean']:>10.1%}  "
              f"{gap:>+8.1%}  "
              f"{a['topsim_mean']:>7.3f}", flush=True)

    # Key comparisons
    if 'compositional_hard' in all_results and 'holistic_hard' in all_results:
        comp = all_results['compositional_hard']
        hol = all_results['holistic_hard']
        print(f"\nKey comparison (HARD mode):", flush=True)
        print(f"  Compositional test: {comp['test_hard_mean']:.1%} ± "
              f"{comp['test_hard_std']:.1%}", flush=True)
        print(f"  Holistic test:      {hol['test_hard_mean']:.1%} ± "
              f"{hol['test_hard_std']:.1%}", flush=True)
        diff = comp['test_hard_mean'] - hol['test_hard_mean']
        print(f"  Compositional advantage: {diff:+.1%}", flush=True)
        print(f"  TopSim: compositional={comp['topsim_mean']:.3f} "
              f"holistic={hol['topsim_mean']:.3f}", flush=True)

    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)

    # ── Stage 4: Save ──
    output = {
        'nearest_neighbor': nn_results,
        **{k: {kk: vv for kk, vv in v.items() if kk != 'seeds'}
           for k, v in all_results.items()},
        'seeds_detail': {k: v['seeds'] for k, v in all_results.items()},
    }

    out_path = RESULTS_DIR / "phase66_oneshot.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()

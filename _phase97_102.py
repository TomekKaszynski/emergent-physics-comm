"""
Phases 97-102: Sequential experiments
======================================
97: Zero-shot cross-architecture transfer
98: Protocol robustness under noise
99: Population scaling to 8 and 16 agents
100: Cross-scenario transfer
101: Full Physics 101 expansion (real video)
102: Message entropy analysis

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase97_102 import run_all; run_all()"
"""

import time, json, math, os, sys, traceback
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement, topographic_similarity, mutual_information

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
CLIP_DIM = 768
BATCH_SIZE = 32
COMM_EPOCHS = 400
EARLY_STOP_PATIENCE = 150
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3
N_HEADS = 2


# ═══════════════════════════════════════════════════════════════
# Shared architecture
# ═══════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, max(1, n_frames))
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)]
        )

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        messages, all_logits = [], []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                msg = F.one_hot(logits.argmax(dim=-1), self.vocab_size).float()
            messages.append(msg)
            all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits


class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)

    def forward(self, views, tau=1.0, hard=True):
        messages, all_logits = [], []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

_feat_cache = {}

def load_features(scenario="spring", max_clips=None):
    key = (scenario, max_clips)
    if key in _feat_cache:
        return _feat_cache[key]

    vjepa_data = torch.load(
        RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    dino_feat = dino_data["features"].float()

    # CLIP features (may not exist for non-spring)
    clip_path = RESULTS_DIR / f"phase96_phys101_{scenario}_clip.pt"
    if clip_path.exists():
        clip_data = torch.load(clip_path, weights_only=False)
        clip_feat = clip_data["features"].float()
    else:
        clip_feat = None

    if max_clips and len(obj_names) > max_clips:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(obj_names), max_clips, replace=False)
        idx.sort()
        vjepa_feat = vjepa_feat[idx]
        dino_feat = dino_feat[idx]
        if clip_feat is not None:
            clip_feat = clip_feat[idx]
        obj_names = [obj_names[i] for i in idx]
        mass_values = mass_values[idx]

    n_frames = vjepa_feat.shape[1]
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous()
    clip_temporal = clip_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous() if clip_feat is not None else None

    result = (vjepa_feat, dino_temporal, clip_temporal, obj_names, mass_values)
    _feat_cache[key] = result
    return result


def make_views(feat_tensor, n_agents):
    """Split feature tensor into per-agent views."""
    n_frames = feat_tensor.shape[1]
    fpa = max(1, n_frames // n_agents)
    views = []
    for i in range(n_agents):
        start = (i * fpa) % n_frames
        end = start + fpa
        if end <= n_frames:
            views.append(feat_tensor[:, start:end, :])
        else:
            # Wrap around for large populations
            v = torch.cat([feat_tensor[:, start:, :],
                          feat_tensor[:, :end - n_frames, :]], dim=1)
            views.append(v)
    return views, fpa


# ═══════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════

def train_model(agent_configs, mass_values, obj_names, vocab_size, seed,
                comm_epochs=COMM_EPOCHS, return_model=False):
    """Train and return results + optionally the model."""
    n_agents = len(agent_configs)
    msg_dim = n_agents * N_HEADS * vocab_size
    agent_views = [feat.float() for feat, _ in agent_configs]

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)

    rng = np.random.RandomState(seed * 1000 + 42)
    holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

    if len(holdout_ids) < 4:
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)

    senders = []
    for feat, input_dim in agent_configs:
        nf = feat.shape[1]
        enc = TemporalEncoder(HIDDEN_DIM, input_dim, n_frames=nf)
        senders.append(CompositionalSender(enc, HIDDEN_DIM, vocab_size, N_HEADS))

    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                 for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(vocab_size)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc = 0.0
    best_state = None
    best_epoch = 0
    best_recv_states = None
    t0 = time.time()

    for ep in range(comm_epochs):
        if time.time() - t0 > 600:
            break
        if ep - best_epoch > EARLY_STOP_PATIENCE and best_acc > 0.55:
            break

        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, comm_epochs - 1)
        hard = ep >= SOFT_WARMUP

        for _ in range(nb):
            ia = rng.choice(train_ids, BATCH_SIZE)
            ib = rng.choice(train_ids, BATCH_SIZE)
            same = ia == ib
            while same.any():
                ib[same] = rng.choice(train_ids, same.sum())
                same = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib])
            keep = md > 0.5
            if keep.sum() < 4:
                continue
            ia, ib = ia[keep], ib[keep]

            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard)
            mb, lb = multi(vb, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=DEVICE)
            for r in receivers:
                pred = r(ma, mb)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pred, label)
            loss = total_loss / len(receivers)

            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1)
                p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / max_ent < ENTROPY_THRESHOLD:
                    loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                continue

            s_opt.zero_grad()
            for o in r_opts:
                o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
            for rm in receivers:
                torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            s_opt.step()
            for o in r_opts:
                o.step()

        if ep % 50 == 0:
            torch.mps.empty_cache()

        if (ep + 1) % 50 == 0 or ep == 0:
            multi.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                correct = total = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs)
                    ib_h = er.choice(holdout_ids, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                        same_h = ia_h == ib_h
                    md_h = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    keep_h = md_h > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    for r in receivers:
                        pred_h = r(ma_h, mb_h) > 0
                        correct += (pred_h == la_h).sum().item()
                        total += len(la_h)
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = ep
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
                    best_recv_states = [{k: v.cpu().clone() for k, v in r.state_dict().items()}
                                        for r in receivers]

    if best_state:
        multi.load_state_dict(best_state)
    multi.eval()

    # Tokens
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    # Metrics
    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs_s = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs_s)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_c = np.digitize(obj_bins, np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_c], axis=1)

    posdis, mi_matrix, entropies = positional_disentanglement(all_tokens, attributes, vocab_size)
    topsim = topographic_similarity(all_tokens, mass_bins, obj_bins_c)

    result = {
        "accuracy": float(best_acc),
        "posdis": float(posdis),
        "topsim": float(topsim),
        "entropies": entropies,
        "mi_matrix": mi_matrix.tolist() if hasattr(mi_matrix, 'tolist') else mi_matrix,
        "tokens": all_tokens,
    }

    if return_model:
        result["multi"] = multi
        result["receivers"] = receivers
        result["best_recv_states"] = best_recv_states
        result["holdout_ids"] = holdout_ids
        result["train_ids"] = train_ids

    return result


def compute_metrics_from_tokens(tokens, mass_values, obj_names, vocab_size):
    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs_s = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs_s)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_c = np.digitize(obj_bins, np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_c], axis=1)
    posdis, mi_matrix, entropies = positional_disentanglement(tokens, attributes, vocab_size)
    topsim = topographic_similarity(tokens, mass_bins, obj_bins_c)
    return {"posdis": float(posdis), "topsim": float(topsim),
            "entropies": entropies, "mi_matrix": mi_matrix}


def summarize(results, label=""):
    """Print summary stats from a list of result dicts."""
    accs = [r["accuracy"] for r in results if r]
    pds = [r["posdis"] for r in results if r]
    tss = [r["topsim"] for r in results if r]
    if not accs:
        return {}
    s = {
        "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
        "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
        "topsim": f"{np.mean(tss):.3f}±{np.std(tss):.3f}",
        "acc_mean": float(np.mean(accs)),
        "pd_mean": float(np.mean(pds)),
        "ts_mean": float(np.mean(tss)),
    }
    if label:
        print(f"    {label}: acc={s['acc']} PD={s['posdis']} TS={s['topsim']}", flush=True)
    return s


# ═══════════════════════════════════════════════════════════════
# PHASE 97: Zero-shot cross-architecture transfer
# ═══════════════════════════════════════════════════════════════

def run_phase97():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 97: Zero-Shot Cross-Architecture Transfer         ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vjepa_feat, dino_temporal, clip_temporal, obj_names, mass_values = load_features("spring")
    n_frames = vjepa_feat.shape[1]
    fpa = n_frames // 2  # 2-agent

    archs = {
        "vjepa": (vjepa_feat[:, :fpa, :], VJEPA_DIM),
        "dino": (dino_temporal[:, :fpa, :], DINO_DIM),
    }
    if clip_temporal is not None:
        archs["clip"] = (clip_temporal[:, :fpa, :], CLIP_DIM)

    arch_names = list(archs.keys())
    vocab_size = 3
    n_seeds = 10

    # For each training architecture: train 2 agents of same type, save model
    # Then at test: swap agent 0's features for a different architecture
    results = {}

    for train_arch in arch_names:
        print(f"\n  ── Training on {train_arch} ──", flush=True)

        for seed in range(n_seeds):
            feat, dim = archs[train_arch]
            configs = [
                (feat, dim),
                (feat, dim),  # homogeneous training
            ]
            # Need agent 1 view from second half of frames
            feat2 = {
                "vjepa": vjepa_feat[:, fpa:, :],
                "dino": dino_temporal[:, fpa:, :],
            }
            if clip_temporal is not None:
                feat2["clip"] = clip_temporal[:, fpa:, :]

            configs = [
                (feat, dim),
                (feat2[train_arch], dim),
            ]

            r = train_model(configs, mass_values, obj_names, vocab_size, seed,
                           return_model=True)
            if r is None:
                continue

            baseline_key = f"{train_arch}→{train_arch}"
            if baseline_key not in results:
                results[baseline_key] = []
            results[baseline_key].append({
                "accuracy": r["accuracy"], "posdis": r["posdis"], "topsim": r["topsim"]
            })

            # Test with each other architecture swapped into agent 0
            multi = r["multi"]
            receivers_loaded = r["receivers"]
            holdout_ids = r["holdout_ids"]

            for test_arch in arch_names:
                if test_arch == train_arch:
                    continue

                swap_key = f"{train_arch}→{test_arch}"
                if swap_key not in results:
                    results[swap_key] = []

                # Swap agent 0's features
                test_feat, test_dim = archs[test_arch]

                # If dims don't match, we can't directly swap (different encoder weights)
                # Instead: build new sender with test_arch dim, copy head weights, test
                # Actually: the sender's encoder processes input_dim → hidden_dim.
                # Swapping features of different dim won't work through the same encoder.
                # The correct test: feed test_arch features through agent 0's encoder.
                # This WILL fail if dims mismatch — that's the point of zero-shot transfer.
                # We need to test: can the RECEIVER understand messages from a DIFFERENT
                # sender architecture? So: train a fresh sender on test_arch, get its messages,
                # feed to the trained receiver.

                # Simpler approach: train agent 0 on test_arch (new sender), agent 1 stays,
                # receiver stays. But that's not zero-shot — it retrains the sender.

                # TRUE zero-shot: The receiver was trained to understand messages from
                # train_arch senders. Now we run a SEPARATELY-trained test_arch sender
                # and see if the receiver understands its messages.

                # Need separately-trained sender on test_arch
                test_configs = [
                    (test_feat, test_dim),
                    (feat2[test_arch], test_dim),
                ]
                test_r = train_model(test_configs, mass_values, obj_names, vocab_size,
                                     seed, return_model=True)
                if test_r is None:
                    continue

                test_multi = test_r["multi"]

                # Get messages from test sender, evaluate with trained receiver
                agent_views_test = [test_feat.float(), feat2[test_arch].float()]
                multi.eval()
                test_multi.eval()
                for recv in receivers_loaded:
                    recv.eval()

                correct = total = 0
                mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
                er = np.random.RandomState(999)
                with torch.no_grad():
                    for _ in range(50):
                        bs = min(BATCH_SIZE, len(holdout_ids))
                        ia_h = er.choice(holdout_ids, bs)
                        ib_h = er.choice(holdout_ids, bs)
                        same_h = ia_h == ib_h
                        while same_h.any():
                            ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                            same_h = ia_h == ib_h
                        md = np.abs(mass_values[ia_h] - mass_values[ib_h])
                        keep_h = md > 0.5
                        if keep_h.sum() < 2:
                            continue
                        ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]

                        # Messages from TEST sender
                        va = [v[ia_h].to(DEVICE) for v in agent_views_test]
                        vb = [v[ib_h].to(DEVICE) for v in agent_views_test]
                        ma, _ = test_multi(va)
                        mb, _ = test_multi(vb)

                        # Decode with TRAINED receiver
                        for recv in receivers_loaded:
                            pred = recv(ma, mb) > 0
                            label = mass_dev[ia_h] > mass_dev[ib_h]
                            correct += (pred == label).sum().item()
                            total += len(label)

                swap_acc = correct / max(total, 1)

                # Get tokens from test sender for PosDis
                test_tokens = test_r["tokens"]
                test_metrics = compute_metrics_from_tokens(
                    test_tokens, mass_values, obj_names, vocab_size)

                results[swap_key].append({
                    "accuracy": float(swap_acc),
                    "posdis": test_metrics["posdis"],
                    "topsim": test_metrics["topsim"],
                })

            if (seed + 1) % 5 == 0:
                print(f"    Seed {seed} done", flush=True)
                torch.mps.empty_cache()

    # Summary
    print(f"\n  ╔═══ ZERO-SHOT TRANSFER RESULTS ═══╗", flush=True)
    all_summaries = {}
    for key, runs in sorted(results.items()):
        s = summarize(runs, key)
        all_summaries[key] = s
    print(f"  ╚══════════════════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase97_zeroshot_transfer.json"
    with open(save_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_summaries


# ═══════════════════════════════════════════════════════════════
# PHASE 98: Protocol robustness under noise
# ═══════════════════════════════════════════════════════════════

def run_phase98():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 98: Protocol Robustness Under Noise               ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vjepa_feat, dino_temporal, clip_temporal, obj_names, mass_values = load_features("spring")
    n_frames = vjepa_feat.shape[1]
    vocab_size = 3
    n_agents = 4
    fpa = n_frames // n_agents
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_seeds = 10

    # Heterogeneous 4-agent (V-JEPA + DINOv2 alternating)
    all_results = {sigma: [] for sigma in noise_levels}

    for seed in range(n_seeds):
        configs = []
        for i in range(n_agents):
            if i % 2 == 0:
                configs.append((vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs.append((dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM))

        r = train_model(configs, mass_values, obj_names, vocab_size, seed,
                        return_model=True)
        if r is None:
            continue

        multi = r["multi"]
        receivers_trained = r["receivers"]
        holdout_ids = r["holdout_ids"]
        agent_views = [feat.float() for feat, _ in configs]
        mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
        msg_dim = n_agents * N_HEADS * vocab_size

        for sigma in noise_levels:
            multi.eval()
            for recv in receivers_trained:
                recv.eval()

            correct = total = 0
            er = np.random.RandomState(999)
            with torch.no_grad():
                for _ in range(50):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs)
                    ib_h = er.choice(holdout_ids, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                        same_h = ia_h == ib_h
                    md = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    keep_h = md > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]

                    va = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb = [v[ib_h].to(DEVICE) for v in agent_views]
                    ma, _ = multi(va)
                    mb, _ = multi(vb)

                    # Add Gaussian noise to messages
                    if sigma > 0:
                        ma = ma + torch.randn_like(ma) * sigma
                        mb = mb + torch.randn_like(mb) * sigma

                    for recv in receivers_trained:
                        pred = recv(ma, mb) > 0
                        label = mass_dev[ia_h] > mass_dev[ib_h]
                        correct += (pred == label).sum().item()
                        total += len(label)

            acc = correct / max(total, 1)
            all_results[sigma].append({"accuracy": float(acc)})

        if (seed + 1) % 5 == 0:
            print(f"    Seed {seed} done", flush=True)
            torch.mps.empty_cache()

    # Summary
    print(f"\n  ╔═══ NOISE ROBUSTNESS ═══╗", flush=True)
    noise_summary = {}
    for sigma in noise_levels:
        accs = [r["accuracy"] for r in all_results[sigma]]
        noise_summary[str(sigma)] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
        }
        print(f"  ║ σ={sigma:.1f}: acc={np.mean(accs):.1%}±{np.std(accs):.1%}", flush=True)
    print(f"  ╚═════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase98_noise_robustness.json"
    with open(save_path, "w") as f:
        json.dump(noise_summary, f, indent=2)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return noise_summary


# ═══════════════════════════════════════════════════════════════
# PHASE 99: Population scaling
# ═══════════════════════════════════════════════════════════════

def run_phase99():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 99: Population Scaling (1–16 agents)              ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vjepa_feat, dino_temporal, _, obj_names, mass_values = load_features("spring")
    n_frames = vjepa_feat.shape[1]
    vocab_size = 3
    n_seeds = 10
    pop_sizes = [1, 2, 4, 8, 16]

    all_results = {}

    for n_agents in pop_sizes:
        fpa = max(1, n_frames // n_agents)

        for cond_name, make_config in [
            ("heterogeneous", lambda i: (vjepa_feat, VJEPA_DIM) if i % 2 == 0 else (dino_temporal, DINO_DIM)),
            ("homo_vjepa", lambda i: (vjepa_feat, VJEPA_DIM)),
            ("homo_dino", lambda i: (dino_temporal, DINO_DIM)),
        ]:
            label = f"{cond_name} n={n_agents}"
            print(f"\n  ── {label} ──", flush=True)
            seed_results = []

            for seed in range(n_seeds):
                configs = []
                for i in range(n_agents):
                    feat_full, dim = make_config(i)
                    start = (i * fpa) % n_frames
                    end = min(start + fpa, n_frames)
                    if end - start < fpa:
                        # Wrap around
                        view = torch.cat([feat_full[:, start:, :],
                                         feat_full[:, :fpa-(n_frames-start), :]], dim=1)
                    else:
                        view = feat_full[:, start:end, :]
                    configs.append((view, dim))

                r = train_model(configs, mass_values, obj_names, vocab_size, seed)
                if r is None:
                    continue

                seed_results.append({
                    "accuracy": r["accuracy"],
                    "posdis": r["posdis"],
                    "topsim": r["topsim"],
                })

            s = summarize(seed_results, label)
            s["seeds"] = seed_results
            all_results[label] = s

            if n_agents >= 8:
                torch.mps.empty_cache()

    # Summary
    print(f"\n  ╔═══ POPULATION SCALING ═══╗", flush=True)
    for cond in ["heterogeneous", "homo_vjepa", "homo_dino"]:
        line = f"  ║ {cond:15s}: "
        for n in pop_sizes:
            key = f"{cond} n={n}"
            if key in all_results and "pd_mean" in all_results[key]:
                line += f"n={n}:{all_results[key]['pd_mean']:.3f} "
        print(line, flush=True)
    print(f"  ╚═════════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase99_population_scaling.json"
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "seeds"} for k, v in all_results.items()}
    with open(save_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_results


# ═══════════════════════════════════════════════════════════════
# PHASE 100: Cross-scenario transfer
# ═══════════════════════════════════════════════════════════════

def run_phase100():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 100: Cross-Scenario Transfer                      ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    spring = load_features("spring")
    fall = load_features("fall")
    ramp = load_features("ramp", max_clips=500)

    scenarios = {"spring": spring, "fall": fall, "ramp": ramp}
    vocab_size = 3
    n_agents = 2
    n_seeds = 10

    all_results = {}

    for train_scenario in ["spring"]:
        vf_train, dt_train, _, obj_train, mass_train = scenarios[train_scenario]
        fpa = vf_train.shape[1] // n_agents

        for cond_name in ["heterogeneous", "homo_vjepa"]:
            print(f"\n  ── Train: {train_scenario}, {cond_name} ──", flush=True)

            for seed in range(n_seeds):
                if cond_name == "heterogeneous":
                    configs = [
                        (vf_train[:, :fpa, :], VJEPA_DIM),
                        (dt_train[:, fpa:, :], DINO_DIM),
                    ]
                else:
                    configs = [
                        (vf_train[:, :fpa, :], VJEPA_DIM),
                        (vf_train[:, fpa:, :], VJEPA_DIM),
                    ]

                r = train_model(configs, mass_train, obj_train, vocab_size, seed,
                               return_model=True)
                if r is None:
                    continue

                multi = r["multi"]
                receivers_trained = r["receivers"]

                # Test on each scenario
                for test_scenario in ["spring", "fall", "ramp"]:
                    vf_test, dt_test, _, obj_test, mass_test = scenarios[test_scenario]
                    fpa_test = vf_test.shape[1] // n_agents

                    if cond_name == "heterogeneous":
                        test_views = [vf_test[:, :fpa_test, :].float(),
                                     dt_test[:, fpa_test:, :].float()]
                    else:
                        test_views = [vf_test[:, :fpa_test, :].float(),
                                     vf_test[:, fpa_test:, :].float()]

                    # Evaluate
                    multi.eval()
                    mass_dev = torch.tensor(mass_test, dtype=torch.float32).to(DEVICE)

                    unique_objs = sorted(set(obj_test))
                    n_holdout = max(4, len(unique_objs) // 5)
                    er = np.random.RandomState(seed * 1000 + 42)
                    holdout = set(er.choice(unique_objs, n_holdout, replace=False))
                    test_ids = np.array([i for i, o in enumerate(obj_test) if o in holdout])

                    if len(test_ids) < 4:
                        continue

                    correct = total = 0
                    er2 = np.random.RandomState(999)
                    with torch.no_grad():
                        for _ in range(50):
                            bs = min(BATCH_SIZE, len(test_ids))
                            ia = er2.choice(test_ids, bs)
                            ib = er2.choice(test_ids, bs)
                            same = ia == ib
                            while same.any():
                                ib[same] = er2.choice(test_ids, same.sum())
                                same = ia == ib
                            md = np.abs(mass_test[ia] - mass_test[ib])
                            keep = md > 0.5
                            if keep.sum() < 2:
                                continue
                            ia, ib = ia[keep], ib[keep]
                            va = [v[ia].to(DEVICE) for v in test_views]
                            vb = [v[ib].to(DEVICE) for v in test_views]
                            ma, _ = multi(va)
                            mb, _ = multi(vb)
                            for recv in receivers_trained:
                                pred = recv(ma, mb) > 0
                                label = mass_dev[ia] > mass_dev[ib]
                                correct += (pred == label).sum().item()
                                total += len(label)

                    acc = correct / max(total, 1)
                    key = f"{cond_name} train={train_scenario} test={test_scenario}"
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append({"accuracy": float(acc)})

                if (seed + 1) % 5 == 0:
                    print(f"    Seed {seed} done", flush=True)
                    torch.mps.empty_cache()

    # Summary
    print(f"\n  ╔═══ CROSS-SCENARIO TRANSFER ═══╗", flush=True)
    summary = {}
    for key, runs in sorted(all_results.items()):
        accs = [r["accuracy"] for r in runs]
        summary[key] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}
        print(f"  ║ {key}: {np.mean(accs):.1%}±{np.std(accs):.1%}", flush=True)
    print(f"  ╚═══════════════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase100_cross_scenario.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return summary


# ═══════════════════════════════════════════════════════════════
# PHASE 101: Full Physics 101 expansion
# ═══════════════════════════════════════════════════════════════

def run_phase101():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 101: Full Physics 101 Expansion (Real Video)      ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vocab_size = 3
    n_seeds = 10
    all_results = {}

    for scenario, max_clips in [("spring", None), ("fall", None), ("ramp", 500)]:
        vf, dt, _, obj_names, mass_values = load_features(scenario, max_clips)
        n_frames = vf.shape[1]

        print(f"\n  ═══ {scenario} ({len(obj_names)} clips) ═══", flush=True)

        for cond_name, n_agents in [("het", 2), ("het", 4), ("homo_vv", 2), ("homo_dd", 2)]:
            fpa = n_frames // n_agents
            label = f"{scenario}_{cond_name}_n={n_agents}"
            print(f"  ── {label} ──", flush=True)
            seed_results = []

            for seed in range(n_seeds):
                configs = []
                for i in range(n_agents):
                    if cond_name == "homo_vv":
                        configs.append((vf[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
                    elif cond_name == "homo_dd":
                        configs.append((dt[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
                    else:  # het
                        if i % 2 == 0:
                            configs.append((vf[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
                        else:
                            configs.append((dt[:, i*fpa:(i+1)*fpa, :], DINO_DIM))

                r = train_model(configs, mass_values, obj_names, vocab_size, seed)
                if r is None:
                    continue
                seed_results.append({
                    "accuracy": r["accuracy"], "posdis": r["posdis"], "topsim": r["topsim"]
                })

            s = summarize(seed_results, label)
            all_results[label] = s
            torch.mps.empty_cache()

    save_path = RESULTS_DIR / "phase101_phys101_expansion.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_results


# ═══════════════════════════════════════════════════════════════
# PHASE 102: Message entropy analysis
# ═══════════════════════════════════════════════════════════════

def run_phase102():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 102: Message Entropy Analysis                     ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vf, dt, _, obj_names, mass_values = load_features("spring")
    n_frames = vf.shape[1]
    n_seeds = 10

    all_mi = {}

    for cond_name, n_agents in [("het", 2), ("het", 4), ("homo_vv", 2), ("homo_dd", 2)]:
        for K in [3, 5, 8, 16, 32]:
            fpa = n_frames // n_agents
            label = f"{cond_name}_n={n_agents}_K={K}"
            print(f"  ── {label} ──", flush=True)

            seed_mis = []
            seed_ents = []
            for seed in range(n_seeds):
                configs = []
                for i in range(n_agents):
                    if cond_name == "homo_vv":
                        configs.append((vf[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
                    elif cond_name == "homo_dd":
                        configs.append((dt[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
                    else:
                        if i % 2 == 0:
                            configs.append((vf[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
                        else:
                            configs.append((dt[:, i*fpa:(i+1)*fpa, :], DINO_DIM))

                r = train_model(configs, mass_values, obj_names, K, seed)
                if r is None:
                    continue
                seed_mis.append(r["mi_matrix"])
                seed_ents.append(r["entropies"])

            if seed_mis:
                avg_mi = np.mean(seed_mis, axis=0).tolist()
                avg_ent = np.mean(seed_ents, axis=0).tolist()
                all_mi[label] = {
                    "avg_mi_matrix": avg_mi,
                    "avg_entropies": avg_ent,
                    "n_seeds": len(seed_mis),
                }
                print(f"    MI: {np.array(avg_mi).round(3).tolist()}, "
                      f"Ent: {np.array(avg_ent).round(3).tolist()}", flush=True)

            torch.mps.empty_cache()

    # Generate MI heatmaps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Select key conditions for visualization
    fig_conditions = [
        ("het_n=2_K=3", "Hetero n=2 K=3"),
        ("het_n=4_K=3", "Hetero n=4 K=3"),
        ("homo_vv_n=2_K=3", "HomoVV n=2 K=3"),
        ("homo_dd_n=2_K=3", "HomoDD n=2 K=3"),
        ("het_n=2_K=8", "Hetero n=2 K=8"),
        ("het_n=2_K=32", "Hetero n=2 K=32"),
    ]

    n_plots = len([c for c, _ in fig_conditions if c in all_mi])
    if n_plots > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Phase 102: MI(position, attribute) Heatmaps", fontsize=14, fontweight='bold')
        axes = axes.flatten()

        for idx, (key, title) in enumerate(fig_conditions):
            if key not in all_mi or idx >= len(axes):
                continue
            ax = axes[idx]
            mi = np.array(all_mi[key]["avg_mi_matrix"])
            im = ax.imshow(mi, cmap='YlOrRd', aspect='auto')
            ax.set_xlabel("Attribute (0=mass, 1=object)")
            ax.set_ylabel("Message position")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["mass", "object"])
            ax.set_yticks(range(mi.shape[0]))
            for i in range(mi.shape[0]):
                for j in range(mi.shape[1]):
                    ax.text(j, i, f'{mi[i,j]:.2f}', ha='center', va='center', fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        save_path = RESULTS_DIR / "phase102_mi_heatmaps.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved {save_path}", flush=True)

    # Entropy profile comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 102: Entropy Profiles", fontsize=13, fontweight='bold')

    # By condition at K=3
    ax = axes[0]
    for cond in ["het_n=2_K=3", "het_n=4_K=3", "homo_vv_n=2_K=3", "homo_dd_n=2_K=3"]:
        if cond in all_mi:
            ents = all_mi[cond]["avg_entropies"]
            ax.plot(range(len(ents)), ents, 'o-', label=cond.replace("_K=3", ""))
    ax.set_xlabel("Message position")
    ax.set_ylabel("Normalized entropy")
    ax.set_title("Entropy by condition (K=3)")
    ax.legend(fontsize=8)

    # By K for het n=2
    ax = axes[1]
    for K in [3, 5, 8, 16, 32]:
        key = f"het_n=2_K={K}"
        if key in all_mi:
            ents = all_mi[key]["avg_entropies"]
            ax.plot(range(len(ents)), ents, 'o-', label=f"K={K}")
    ax.set_xlabel("Message position")
    ax.set_ylabel("Normalized entropy")
    ax.set_title("Entropy by codebook size (het n=2)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = RESULTS_DIR / "phase102_entropy_profiles.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}", flush=True)

    save_path = RESULTS_DIR / "phase102_entropy_analysis.json"
    with open(save_path, "w") as f:
        json.dump(all_mi, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_mi


# ═══════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════

def run_all():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phases 97–102: Sequential Experiments                   ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    phases = [
        (97, run_phase97),
        (98, run_phase98),
        (99, run_phase99),
        (100, run_phase100),
        (101, run_phase101),
        (102, run_phase102),
    ]

    results = {}
    for num, func in phases:
        try:
            print(f"\n{'#'*70}", flush=True)
            print(f"#  STARTING PHASE {num}", flush=True)
            print(f"{'#'*70}", flush=True)
            results[num] = func()
            _feat_cache.clear()
            torch.mps.empty_cache()
        except Exception as e:
            print(f"\n  PHASE {num} FAILED: {e}", flush=True)
            traceback.print_exc()
            results[num] = {"error": str(e)}

    total_h = (time.time() - t_total) / 3600
    print(f"\n{'='*70}", flush=True)
    print(f"  ALL PHASES COMPLETE. Total: {total_h:.1f} hours", flush=True)
    for num, r in results.items():
        status = "FAILED" if isinstance(r, dict) and "error" in r else "OK"
        print(f"  Phase {num}: {status}", flush=True)
    print(f"{'='*70}", flush=True)

    return results


if __name__ == "__main__":
    run_all()

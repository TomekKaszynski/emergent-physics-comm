"""
Phases 103-106: Systems & Scaling Experiments
==============================================
103: Latency benchmarking
104: Fine-tuning speed for new model onboarding
105: Multi-property scaling
106: Async pub-sub integration proof-of-concept

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase103_106 import run_all; run_all()"
"""

import time, json, math, os, sys, traceback, asyncio
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


# ═══ Architecture ═══

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


# ═══ Data ═══

_feat_cache = {}

def load_features(scenario="spring", max_clips=None):
    key = (scenario, max_clips)
    if key in _feat_cache:
        return _feat_cache[key]

    vjepa_data = torch.load(RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt", weights_only=False)
    dino_data = torch.load(RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt", weights_only=False)

    vf = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    df = dino_data["features"].float()

    clip_path = RESULTS_DIR / f"phase96_phys101_{scenario}_clip.pt"
    cf = torch.load(clip_path, weights_only=False)["features"].float() if clip_path.exists() else None

    if max_clips and len(obj_names) > max_clips:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(obj_names), max_clips, replace=False)
        idx.sort()
        vf, df = vf[idx], df[idx]
        if cf is not None: cf = cf[idx]
        obj_names = [obj_names[i] for i in idx]
        mass_values = mass_values[idx]

    nf = vf.shape[1]
    dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()
    ct = cf.unsqueeze(1).expand(-1, nf, -1).contiguous() if cf is not None else None

    result = (vf, dt, ct, obj_names, mass_values)
    _feat_cache[key] = result
    return result


def train_model(agent_configs, mass_values, obj_names, vocab_size, seed,
                comm_epochs=COMM_EPOCHS, return_model=False):
    """Full training loop. Returns metrics + optionally model."""
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

    senders = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, dim, n_frames=feat.shape[1]),
        HIDDEN_DIM, vocab_size, N_HEADS) for feat, dim in agent_configs]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(vocab_size)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc, best_state, best_epoch = 0.0, None, 0
    best_recv = None
    t0 = time.time()

    for ep in range(comm_epochs):
        if time.time() - t0 > 600: break
        if ep - best_epoch > EARLY_STOP_PATIENCE and best_acc > 0.55: break
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi.train()
        for r in receivers: r.train()
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
            if keep.sum() < 4: continue
            ia, ib = ia[keep], ib[keep]

            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard)
            mb, lb = multi(vb, tau=tau, hard=hard)

            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), label)
                       for r in receivers) / len(receivers)
            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1)
                p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / max_ent < ENTROPY_THRESHOLD:
                    loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                s_opt.zero_grad()
                for o in r_opts: o.zero_grad()
                continue
            s_opt.zero_grad()
            for o in r_opts: o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
            for rm in receivers: torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            s_opt.step()
            for o in r_opts: o.step()

        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep + 1) % 50 == 0 or ep == 0:
            multi.eval()
            for r in receivers: r.eval()
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
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    for r in receivers:
                        correct += ((r(ma_h, mb_h) > 0) == la_h).sum().item()
                        total += len(la_h)
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc; best_epoch = ep
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
                    best_recv = [{k: v.cpu().clone() for k, v in r.state_dict().items()} for r in receivers]

    if best_state: multi.load_state_dict(best_state)
    multi.eval()

    # Tokens
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            all_tokens.append(np.stack([l.argmax(dim=-1).cpu().numpy() for l in logits], axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    uobjs = sorted(set(obj_names))
    oi = {o: i for i, o in enumerate(uobjs)}
    obj_bins = np.digitize(np.array([oi[o] for o in obj_names]),
                            np.quantile(np.arange(len(uobjs)), [0.2, 0.4, 0.6, 0.8]))
    attrs = np.stack([mass_bins, obj_bins], axis=1)
    posdis, mi, ent = positional_disentanglement(all_tokens, attrs, vocab_size)
    topsim = topographic_similarity(all_tokens, mass_bins, obj_bins)

    result = {"accuracy": float(best_acc), "posdis": float(posdis), "topsim": float(topsim),
              "entropies": ent, "mi_matrix": mi, "tokens": all_tokens,
              "holdout_ids": holdout_ids, "train_ids": train_ids}
    if return_model:
        result["multi"] = multi
        result["receivers"] = receivers
        if best_recv:
            for i, r in enumerate(receivers):
                r.load_state_dict(best_recv[i])
    return result


# ═══════════════════════════════════════════════════════════════
# PHASE 103: Latency benchmarking
# ═══════════════════════════════════════════════════════════════

def run_phase103():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 103: Latency Benchmarking                         ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vf, dt, _, obj_names, mass_values = load_features("spring")
    n_agents, vocab_size, n_frames = 4, 3, vf.shape[1]
    fpa = n_frames // n_agents

    configs = []
    for i in range(n_agents):
        if i % 2 == 0:
            configs.append((vf[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
        else:
            configs.append((dt[:, i*fpa:(i+1)*fpa, :], DINO_DIM))

    r = train_model(configs, mass_values, obj_names, vocab_size, seed=0, return_model=True)
    multi = r["multi"]
    recv = r["receivers"][0]
    multi.eval(); recv.eval()

    agent_views = [feat.float() for feat, _ in configs]
    results = {}

    for device_name, dev in [("cpu", torch.device("cpu")), ("mps", DEVICE)]:
        multi_d = multi.to(dev)
        recv_d = recv.to(dev)

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                va = [v[:1].to(dev) for v in agent_views]
                vb = [v[1:2].to(dev) for v in agent_views]
                ma, _ = multi_d(va)
                mb, _ = multi_d(vb)
                recv_d(ma, mb)

        # Benchmark: single sample, full round-trip
        latencies = []
        for _ in range(1000):
            idx_a = np.random.randint(0, len(obj_names))
            idx_b = np.random.randint(0, len(obj_names))

            if dev.type == "mps":
                torch.mps.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                va = [v[idx_a:idx_a+1].to(dev) for v in agent_views]
                vb = [v[idx_b:idx_b+1].to(dev) for v in agent_views]
                ma, _ = multi_d(va)
                mb, _ = multi_d(vb)
                pred = recv_d(ma, mb)

            if dev.type == "mps":
                torch.mps.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)  # ms

        latencies = np.array(latencies)
        results[device_name] = {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_per_s": float(1000 / np.mean(latencies)),
        }
        print(f"  {device_name}: mean={np.mean(latencies):.2f}ms "
              f"median={np.median(latencies):.2f}ms "
              f"p95={np.percentile(latencies, 95):.2f}ms "
              f"p99={np.percentile(latencies, 99):.2f}ms "
              f"throughput={1000/np.mean(latencies):.0f}/s", flush=True)

        multi_d = multi_d.to(DEVICE)
        recv_d = recv_d.to(DEVICE)

    # Batch throughput
    print(f"\n  Batch throughput (batch=32, MPS):", flush=True)
    multi.to(DEVICE)
    recv.to(DEVICE)
    batch_latencies = []
    for _ in range(100):
        idx = np.random.randint(0, len(obj_names), 64)
        torch.mps.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            va = [v[idx[:32]].to(DEVICE) for v in agent_views]
            vb = [v[idx[32:]].to(DEVICE) for v in agent_views]
            ma, _ = multi(va)
            mb, _ = multi(vb)
            pred = recv(ma, mb)
        torch.mps.synchronize()
        batch_latencies.append((time.perf_counter() - t_start) * 1000)

    results["batch_32_mps"] = {
        "mean_ms": float(np.mean(batch_latencies)),
        "throughput_per_s": float(32 * 1000 / np.mean(batch_latencies)),
    }
    print(f"  batch=32 MPS: {np.mean(batch_latencies):.2f}ms "
          f"({32*1000/np.mean(batch_latencies):.0f} comms/s)", flush=True)

    realtime = results["mps"]["mean_ms"] < 10
    print(f"\n  Real-time viable (<10ms): {'YES' if realtime else 'NO'} "
          f"({results['mps']['mean_ms']:.2f}ms on MPS)", flush=True)

    save_path = RESULTS_DIR / "phase103_latency.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 104: Fine-tuning for new model onboarding
# ═══════════════════════════════════════════════════════════════

def run_phase104():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 104: Fine-Tuning for New Model Onboarding         ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vf, dt, ct, obj_names, mass_values = load_features("spring")
    if ct is None:
        print("  ERROR: CLIP features not found", flush=True)
        return {"error": "no clip features"}

    n_frames = vf.shape[1]
    n_agents, vocab_size = 4, 3
    fpa = n_frames // n_agents
    msg_dim = n_agents * N_HEADS * vocab_size
    n_seeds = 10
    checkpoint_interval = 50
    max_finetune_steps = 2000

    all_curves = []

    for seed in range(n_seeds):
        print(f"  Seed {seed}:", flush=True)

        # Train base system: 4-agent hetero V-JEPA + DINOv2
        configs_base = []
        for i in range(n_agents):
            if i % 2 == 0:
                configs_base.append((vf[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs_base.append((dt[:, i*fpa:(i+1)*fpa, :], DINO_DIM))

        base = train_model(configs_base, mass_values, obj_names, vocab_size, seed,
                          return_model=True)
        if base is None:
            continue

        base_acc = base["accuracy"]
        print(f"    Base accuracy: {base_acc:.1%}", flush=True)

        # Now replace agent 1 (DINOv2) with CLIP
        # Freeze agents 0, 2, 3 (senders) and receivers
        multi_base = base["multi"]
        recv_base = base["receivers"][0]

        # Create new CLIP sender for agent 1's slot
        torch.manual_seed(seed + 5000)
        clip_sender = CompositionalSender(
            TemporalEncoder(HIDDEN_DIM, CLIP_DIM, n_frames=fpa),
            HIDDEN_DIM, vocab_size, N_HEADS
        ).to(DEVICE)

        # Replace sender 1 in multi
        multi_base.senders[1] = clip_sender

        # Freeze everything except the new CLIP sender
        for name, param in multi_base.named_parameters():
            if "senders.1" not in name:
                param.requires_grad = False

        # Freeze receiver
        for param in recv_base.parameters():
            param.requires_grad = False

        # New agent views: agent 1 now uses CLIP features
        agent_views_new = [
            vf[:, 0*fpa:1*fpa, :].float(),   # agent 0: V-JEPA
            ct[:, 1*fpa:2*fpa, :].float(),    # agent 1: CLIP (NEW)
            vf[:, 2*fpa:3*fpa, :].float(),    # agent 2: V-JEPA
            dt[:, 3*fpa:4*fpa, :].float(),    # agent 3: DINOv2
        ]

        # Fine-tune only CLIP sender
        ft_opt = torch.optim.Adam(clip_sender.parameters(), lr=SENDER_LR)
        mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)

        rng = np.random.RandomState(seed * 1000 + 42)
        unique_objs = sorted(set(obj_names))
        n_holdout = max(4, len(unique_objs) // 5)
        holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
        train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
        holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

        curve = []
        target_acc = base_acc * 0.9

        for step in range(max_finetune_steps):
            multi_base.train()
            clip_sender.train()

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

            va = [v[ia].to(DEVICE) for v in agent_views_new]
            vb = [v[ib].to(DEVICE) for v in agent_views_new]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, _ = multi_base(va)
            mb, _ = multi_base(vb)
            loss = F.binary_cross_entropy_with_logits(recv_base(ma, mb), label)

            ft_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_sender.parameters(), 1.0)
            ft_opt.step()

            if (step + 1) % checkpoint_interval == 0:
                multi_base.eval()
                recv_base.eval()
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
                        mdh = np.abs(mass_values[ia_h] - mass_values[ib_h])
                        kh = mdh > 0.5
                        if kh.sum() < 2: continue
                        ia_h, ib_h = ia_h[kh], ib_h[kh]
                        va_h = [v[ia_h].to(DEVICE) for v in agent_views_new]
                        vb_h = [v[ib_h].to(DEVICE) for v in agent_views_new]
                        la_h = mass_dev[ia_h] > mass_dev[ib_h]
                        ma_h, _ = multi_base(va_h)
                        mb_h, _ = multi_base(vb_h)
                        pred = recv_base(ma_h, mb_h) > 0
                        correct += (pred == la_h).sum().item()
                        total += len(la_h)
                    acc = correct / max(total, 1)
                curve.append({"step": step + 1, "accuracy": float(acc)})

                if acc >= target_acc:
                    print(f"    Reached 90% of base ({acc:.1%}) at step {step+1}", flush=True)
                    break

        all_curves.append({
            "seed": seed,
            "base_acc": float(base_acc),
            "target_acc": float(target_acc),
            "curve": curve,
            "steps_to_90pct": curve[-1]["step"] if curve and curve[-1]["accuracy"] >= target_acc else None,
            "final_acc": curve[-1]["accuracy"] if curve else 0,
        })
        torch.mps.empty_cache()

    # Summary
    steps_list = [c["steps_to_90pct"] for c in all_curves if c["steps_to_90pct"] is not None]
    print(f"\n  ╔═══ ONBOARDING SPEED ═══╗", flush=True)
    if steps_list:
        print(f"  ║ Steps to 90%: {np.mean(steps_list):.0f}±{np.std(steps_list):.0f} "
              f"({len(steps_list)}/{len(all_curves)} reached target)", flush=True)
    else:
        print(f"  ║ No seeds reached 90% target", flush=True)
    final_accs = [c["final_acc"] for c in all_curves]
    print(f"  ║ Final acc: {np.mean(final_accs):.1%}±{np.std(final_accs):.1%}", flush=True)
    print(f"  ╚═════════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase104_onboarding.json"
    with open(save_path, "w") as f:
        json.dump(all_curves, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_curves


# ═══════════════════════════════════════════════════════════════
# PHASE 105: Multi-property scaling
# ═══════════════════════════════════════════════════════════════

def run_phase105():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 105: Multi-Property Scaling                       ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vf, dt, _, obj_names, mass_values = load_features("spring")
    n_frames = vf.shape[1]
    n_agents, vocab_size = 2, 3
    fpa = n_frames // n_agents
    n_seeds = 10

    # Create synthetic additional properties from mass
    # Property 1: mass (real)
    # Property 2: log-mass bin (coarse mass category)
    # Property 3: mass rank (ordinal)
    # Property 4: mass residual from mean (high/low within category)
    # Property 5: object material category (from name prefix)

    log_mass = np.log1p(mass_values)
    mass_rank = stats.rankdata(mass_values)
    material = np.array([0 if o.startswith(('cardboard', 'paper')) else
                         1 if o.startswith(('wood', 'w_')) else
                         2 if o.startswith(('metal', 'm_')) else
                         3 if o.startswith(('plastic', 'p_')) else 4
                         for o in obj_names])

    all_props = [mass_values, log_mass, mass_rank,
                 mass_values - np.mean(mass_values), material.astype(float)]
    prop_names = ["mass", "log_mass", "rank", "residual", "material"]

    all_results = {}

    for n_props in [2, 3, 4, 5]:
        n_heads_scaled = n_props  # One message position per property
        label = f"n_props={n_props}"
        print(f"\n  ── {label} (n_heads={n_heads_scaled}) ──", flush=True)

        # Bin each property for MI computation
        prop_bins = []
        for p in range(n_props):
            vals = all_props[p]
            bins = np.digitize(vals, np.quantile(vals, [0.2, 0.4, 0.6, 0.8]))
            prop_bins.append(bins)
        attributes = np.stack(prop_bins, axis=1)  # (N, n_props)

        seed_results = []
        for seed in range(n_seeds):
            configs = [
                (vf[:, :fpa, :], VJEPA_DIM),
                (dt[:, fpa:, :], DINO_DIM),
            ]

            # Build model with scaled n_heads
            msg_dim = n_agents * n_heads_scaled * vocab_size

            torch.manual_seed(seed)
            np.random.seed(seed)
            senders_list = [
                CompositionalSender(
                    TemporalEncoder(HIDDEN_DIM, dim, n_frames=feat.shape[1]),
                    HIDDEN_DIM, vocab_size, n_heads_scaled)
                for feat, dim in configs]
            multi = MultiAgentSender(senders_list).to(DEVICE)
            receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
            s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
            r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

            unique_objs = sorted(set(obj_names))
            n_holdout = max(4, len(unique_objs) // 5)
            rng = np.random.RandomState(seed * 1000 + 42)
            holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
            train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
            holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])
            if len(holdout_ids) < 4: continue

            mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
            agent_views = [feat.float() for feat, _ in configs]
            max_ent = math.log(vocab_size)
            nb = max(1, len(train_ids) // BATCH_SIZE)
            best_acc, best_state, best_epoch = 0.0, None, 0

            for ep in range(COMM_EPOCHS):
                if ep - best_epoch > EARLY_STOP_PATIENCE and best_acc > 0.55: break
                if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
                    for i in range(len(receivers)):
                        receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                        r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)
                multi.train()
                for r in receivers: r.train()
                tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
                hard = ep >= SOFT_WARMUP

                for _ in range(nb):
                    ia = rng.choice(train_ids, BATCH_SIZE)
                    ib = rng.choice(train_ids, BATCH_SIZE)
                    same = ia == ib
                    while same.any():
                        ib[same] = rng.choice(train_ids, same.sum()); same = ia == ib
                    md = np.abs(mass_values[ia] - mass_values[ib])
                    keep = md > 0.5
                    if keep.sum() < 4: continue
                    ia, ib = ia[keep], ib[keep]
                    va = [v[ia].to(DEVICE) for v in agent_views]
                    vb = [v[ib].to(DEVICE) for v in agent_views]
                    label = (mass_dev[ia] > mass_dev[ib]).float()
                    ma, la = multi(va, tau=tau, hard=hard)
                    mb, lb = multi(vb, tau=tau, hard=hard)
                    loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), label) for r in receivers) / len(receivers)
                    for logits in la + lb:
                        lp = F.log_softmax(logits, dim=-1); p = lp.exp().clamp(min=1e-8)
                        ent = -(p * lp).sum(dim=-1).mean()
                        if ent / max_ent < ENTROPY_THRESHOLD: loss = loss - ENTROPY_COEF * ent
                    if torch.isnan(loss) or torch.isinf(loss):
                        s_opt.zero_grad()
                        for o in r_opts: o.zero_grad()
                        continue
                    s_opt.zero_grad()
                    for o in r_opts: o.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
                    s_opt.step()
                    for o in r_opts: o.step()

                if ep % 50 == 0: torch.mps.empty_cache()
                if (ep + 1) % 50 == 0 or ep == 0:
                    multi.eval()
                    for r in receivers: r.eval()
                    with torch.no_grad():
                        correct = total = 0
                        er = np.random.RandomState(999)
                        for _ in range(30):
                            bs = min(BATCH_SIZE, len(holdout_ids))
                            ia_h = er.choice(holdout_ids, bs); ib_h = er.choice(holdout_ids, bs)
                            same_h = ia_h == ib_h
                            while same_h.any(): ib_h[same_h] = er.choice(holdout_ids, same_h.sum()); same_h = ia_h == ib_h
                            mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                            if kh.sum() < 2: continue
                            ia_h, ib_h = ia_h[kh], ib_h[kh]
                            va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                            vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                            la_h = mass_dev[ia_h] > mass_dev[ib_h]
                            ma_h, _ = multi(va_h); mb_h, _ = multi(vb_h)
                            for r in receivers:
                                correct += ((r(ma_h, mb_h) > 0) == la_h).sum().item(); total += len(la_h)
                        acc = correct / max(total, 1)
                        if acc > best_acc:
                            best_acc = acc; best_epoch = ep
                            best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}

            if best_state: multi.load_state_dict(best_state)
            multi.eval()

            all_tokens = []
            with torch.no_grad():
                for i in range(0, len(agent_views[0]), BATCH_SIZE):
                    views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
                    _, logits = multi(views)
                    all_tokens.append(np.stack([l.argmax(dim=-1).cpu().numpy() for l in logits], axis=1))
            all_tokens = np.concatenate(all_tokens, axis=0)

            posdis, mi, ent = positional_disentanglement(all_tokens, attributes, vocab_size)
            topsim = topographic_similarity(all_tokens, prop_bins[0], prop_bins[1] if n_props > 1 else prop_bins[0])

            seed_results.append({"accuracy": float(best_acc), "posdis": float(posdis), "topsim": float(topsim)})

        if seed_results:
            pds = [r["posdis"] for r in seed_results]
            accs = [r["accuracy"] for r in seed_results]
            all_results[label] = {
                "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "pd_mean": float(np.mean(pds)),
            }
            print(f"    acc={np.mean(accs):.1%} PD={np.mean(pds):.3f}", flush=True)
        torch.mps.empty_cache()

    print(f"\n  Scaling: " + " → ".join(
        f"{k}:PD={v['pd_mean']:.3f}" for k, v in all_results.items()), flush=True)

    # Ensure all values are JSON-serializable
    clean_results = {}
    for k, v in all_results.items():
        clean_results[str(k)] = {
            kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else str(vv) if not isinstance(vv, (str, int, float, bool, type(None), list, dict)) else vv
            for kk, vv in v.items()
        }

    save_path = RESULTS_DIR / "phase105_multi_property.json"
    with open(save_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_results


# ═══════════════════════════════════════════════════════════════
# PHASE 106: Async pub-sub integration PoC
# ═══════════════════════════════════════════════════════════════

def run_phase106():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 106: Async Pub-Sub Integration PoC                ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0_total = time.time()

    vf, dt, _, obj_names, mass_values = load_features("spring")
    n_frames, n_agents, vocab_size = vf.shape[1], 2, 3
    fpa = n_frames // n_agents

    configs = [
        (vf[:, :fpa, :], VJEPA_DIM),
        (dt[:, fpa:, :], DINO_DIM),
    ]

    r = train_model(configs, mass_values, obj_names, vocab_size, seed=0, return_model=True)
    multi = r["multi"]
    recv = r["receivers"][0]
    multi.eval(); recv.eval()
    multi_cpu = multi.cpu()
    recv_cpu = recv.cpu()

    agent_views = [feat.float().cpu() for feat, _ in configs]
    mass_dev = mass_values

    # ═══ Threaded pub-sub simulation ═══
    import queue
    import threading

    message_bus = queue.Queue()
    latencies = []
    exchange_count = [0]
    correct_count = [0]
    total_count = [0]

    n_rounds = 1000

    def node_a_publisher():
        """Node A (V-JEPA encoder): publishes messages."""
        rng_a = np.random.RandomState(42)
        for round_idx in range(n_rounds):
            idx_a = rng_a.randint(0, len(obj_names))
            idx_b = rng_a.randint(0, len(obj_names))
            if abs(mass_values[idx_a] - mass_values[idx_b]) < 0.5:
                continue

            t_start = time.perf_counter()

            with torch.no_grad():
                va = [v[idx_a:idx_a+1] for v in agent_views]
                vb = [v[idx_b:idx_b+1] for v in agent_views]
                ma, _ = multi_cpu(va)
                mb, _ = multi_cpu(vb)

            message_bus.put({
                "msg_a": ma,
                "msg_b": mb,
                "label": float(mass_values[idx_a] > mass_values[idx_b]),
                "t_start": t_start,
            })
            exchange_count[0] += 1

        message_bus.put(None)  # Sentinel

    def node_b_subscriber():
        """Node B (DINOv2 encoder): subscribes, decodes, predicts."""
        while True:
            msg = message_bus.get()
            if msg is None:
                break

            with torch.no_grad():
                pred = recv_cpu(msg["msg_a"], msg["msg_b"]).item() > 0

            t_end = time.perf_counter()
            latencies.append((t_end - msg["t_start"]) * 1000)
            total_count[0] += 1
            if pred == msg["label"]:
                correct_count[0] += 1

    print(f"  Running {n_rounds} threaded pub-sub rounds...", flush=True)

    thread_a = threading.Thread(target=node_a_publisher)
    thread_b = threading.Thread(target=node_b_subscriber)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    acc = correct_count[0] / max(total_count[0], 1)
    lat = np.array(latencies) if latencies else np.array([0])

    results = {
        "n_rounds": n_rounds,
        "exchanges_completed": exchange_count[0],
        "success_rate": float(exchange_count[0] / n_rounds),
        "accuracy": float(acc),
        "latency_mean_ms": float(np.mean(lat)),
        "latency_median_ms": float(np.median(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "throughput_per_s": float(1000 / np.mean(lat)) if np.mean(lat) > 0 else 0,
    }

    print(f"\n  ╔═══ PUB-SUB RESULTS ═══╗", flush=True)
    print(f"  ║ Exchanges: {exchange_count[0]}/{n_rounds} ({results['success_rate']:.1%})", flush=True)
    print(f"  ║ Accuracy: {acc:.1%}", flush=True)
    print(f"  ║ Latency: mean={np.mean(lat):.2f}ms median={np.median(lat):.2f}ms "
          f"p95={np.percentile(lat, 95):.2f}ms", flush=True)
    print(f"  ║ Throughput: {results['throughput_per_s']:.0f} comms/s", flush=True)
    print(f"  ╚═══════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase106_pubsub.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path} ({(time.time()-t0_total)/60:.1f}min)", flush=True)
    return results


# ═══ Main ═══

def run_all():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phases 103–106: Systems & Scaling                       ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    phases = [(103, run_phase103), (104, run_phase104),
              (105, run_phase105), (106, run_phase106)]
    results = {}

    for num, func in phases:
        try:
            print(f"\n{'#'*70}\n#  STARTING PHASE {num}\n{'#'*70}", flush=True)
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

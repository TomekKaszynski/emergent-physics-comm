"""
Phase 93: Heterogeneous-Agent Emergent Communication
=====================================================
Can a V-JEPA 2 agent and a DINOv2 agent develop compositional communication
from scratch when paired together during training?

Not transfer — emergence. If compositionality emerges natively in a
mixed-architecture population, the discrete bottleneck forces shared structure
during training, no alignment map needed.

Conditions:
  - Agent 1: frozen V-JEPA 2 features (temporal, 1024-dim)
  - Agent 2: frozen DINOv2 features (static, 384-dim, replicated across time)
  - Same physics task (mass pairwise comparison), same bottleneck
  - Sweep K ∈ {3, 8, 32}, 5 seeds each
  - Control: homogeneous V-JEPA 2 pair, homogeneous DINOv2 pair

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase93_heterogeneous import run_phase93; run_phase93()"
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

# Constants (match Phase 87/91/92)
HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
BATCH_SIZE = 32
COMM_EPOCHS = 400
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
# Architecture
# ═══════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, n_frames)
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
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_spring_features():
    """Load V-JEPA 2 and DINOv2 features for spring scenario."""
    vjepa_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()   # (206, 8, 1024)
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]        # (206,)
    dino_feat = dino_data["features"].float()      # (206, 384)

    assert vjepa_data["obj_names"] == dino_data["obj_names"]

    n_frames = vjepa_feat.shape[1]  # 8
    # Replicate DINOv2 static features across time dimension
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames, -1)  # (206, 8, 384)

    return vjepa_feat, dino_temporal, obj_names, mass_values


# ═══════════════════════════════════════════════════════════════
# Training loop for heterogeneous or homogeneous agents
# ═══════════════════════════════════════════════════════════════

def train_heterogeneous_comm(agent_configs, mass_values, obj_names,
                              vocab_size, n_heads, seed, comm_epochs=COMM_EPOCHS):
    """Train communication with heterogeneous agent configurations.

    Args:
        agent_configs: list of (features_tensor, input_dim) per agent.
            features_tensor: (N, T, D) where T = n_frames for that agent's view.
        mass_values: (N,) array
        obj_names: list of N object names
        vocab_size: codebook size K
        n_heads: message positions per agent
        seed: random seed
        comm_epochs: training epochs

    Returns:
        dict with accuracy, sender state, per-agent tokens, etc.
    """
    n_agents = len(agent_configs)
    msg_dim = n_agents * n_heads * vocab_size

    # Build agent views — each agent gets its own features
    agent_views = []
    for feat, _ in agent_configs:
        agent_views.append(feat.float())

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

    # Build senders — each with its own input_dim
    senders = []
    for feat, input_dim in agent_configs:
        n_frames_agent = feat.shape[1]
        enc = TemporalEncoder(HIDDEN_DIM, input_dim, n_frames=n_frames_agent)
        senders.append(CompositionalSender(enc, HIDDEN_DIM, vocab_size, n_heads))

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
    t0 = time.time()

    for ep in range(comm_epochs):
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
            mass_diff = np.abs(mass_values[ia] - mass_values[ib])
            keep = mass_diff > 0.5
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
            for r_m in receivers:
                torch.nn.utils.clip_grad_norm_(r_m.parameters(), 1.0)
            s_opt.step()
            for o in r_opts:
                o.step()

        if ep % 50 == 0:
            torch.mps.empty_cache()

        if (ep + 1) % 100 == 0 or ep == 0:
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
                    md = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    keep_h = md > 0.5
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
                    best_state = {k: v.cpu().clone()
                                  for k, v in multi.state_dict().items()}

    # Restore best and extract tokens
    if best_state:
        multi.load_state_dict(best_state)
    multi.eval()

    # Get tokens for all data
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    elapsed = time.time() - t0
    return {
        "best_acc": float(best_acc),
        "tokens": all_tokens,
        "sender_state": best_state,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# Compositionality analysis
# ═══════════════════════════════════════════════════════════════

def analyze_compositionality(tokens, mass_values, obj_names, vocab_size,
                              n_agents=2, n_heads=2):
    """Full compositionality analysis with per-agent breakdown."""
    # Bin mass into 5 bins
    mass_bins = np.digitize(mass_values,
                            np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)

    # Global PosDis
    pd_global, mi_global, ent_global = positional_disentanglement(
        tokens, attributes, vocab_size)

    # Per-agent PosDis
    mi_np = np.array(mi_global) if not isinstance(mi_global, np.ndarray) else mi_global
    per_agent = []
    for ag in range(n_agents):
        start = ag * n_heads
        agent_tokens = tokens[:, start:start + n_heads]
        pd_ag, mi_ag, ent_ag = positional_disentanglement(
            agent_tokens, attributes, vocab_size)

        # Per-position MI with mass specifically
        mass_mi = [float(mi_np[start + h, 0]) for h in range(n_heads)]

        per_agent.append({
            "posdis": float(pd_ag),
            "entropies": ent_ag,
            "mass_mi": mass_mi,
        })

    # Per-position symbol distribution (check if both agents converge to same encoding)
    position_symbols = {}
    for pos in range(tokens.shape[1]):
        unique, counts = np.unique(tokens[:, pos], return_counts=True)
        position_symbols[f"pos_{pos}"] = {
            "symbols_used": int(len(unique)),
            "distribution": {int(u): int(c) for u, c in zip(unique, counts)},
        }

    return {
        "posdis_global": float(pd_global),
        "per_agent": per_agent,
        "entropies": ent_global,
        "mi_matrix": mi_np.tolist(),
        "position_symbols": position_symbols,
    }


def check_symbol_property_mapping(tokens, mass_values, n_agents=2, n_heads=2):
    """Check whether each agent's symbols map to the same physical property ranges.

    For each position, group scenes by symbol and compute mean mass per symbol.
    If both agents encode mass monotonically, they've independently discovered
    the same encoding.
    """
    results = {}
    for pos in range(n_agents * n_heads):
        agent = pos // n_heads
        head = pos % n_heads
        symbol_masses = defaultdict(list)
        for i, sym in enumerate(tokens[:, pos]):
            symbol_masses[int(sym)].append(mass_values[i])

        # Sort symbols by mean mass
        sorted_symbols = sorted(symbol_masses.keys(),
                                 key=lambda s: np.mean(symbol_masses[s]))
        mean_masses = [np.mean(symbol_masses[s]) for s in sorted_symbols]

        # Check monotonicity (Spearman correlation with rank)
        if len(mean_masses) >= 2:
            from scipy.stats import spearmanr
            rho, p_val = spearmanr(range(len(mean_masses)), mean_masses)
        else:
            rho, p_val = 0.0, 1.0

        results[f"agent{agent}_head{head}"] = {
            "sorted_symbols": sorted_symbols,
            "mean_masses": [float(m) for m in mean_masses],
            "mass_range_per_symbol": {
                int(s): f"{np.min(symbol_masses[s]):.1f}-{np.max(symbol_masses[s]):.1f}"
                for s in sorted_symbols
            },
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "monotonic": abs(rho) > 0.8,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════

def run_phase93():
    """Run heterogeneous agent communication experiment."""
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 93: Heterogeneous-Agent Emergent Communication    ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    # Load features
    vjepa_feat, dino_temporal, obj_names, mass_values = load_spring_features()
    n_frames = vjepa_feat.shape[1]
    print(f"  V-JEPA 2: {vjepa_feat.shape}", flush=True)
    print(f"  DINOv2 (replicated): {dino_temporal.shape}", flush=True)
    print(f"  {len(obj_names)} clips, {len(set(obj_names))} unique objects", flush=True)

    vocab_sizes = [3, 8, 32]
    n_seeds = 5

    # Conditions:
    # 1. Heterogeneous: Agent 1 = V-JEPA 2, Agent 2 = DINOv2
    # 2. Homogeneous V-JEPA: Agent 1 = V-JEPA first half, Agent 2 = V-JEPA second half
    # 3. Homogeneous DINOv2: Agent 1 = DINOv2, Agent 2 = DINOv2
    conditions = {
        "hetero_VD": lambda: [
            (vjepa_feat[:, :n_frames//2, :], VJEPA_DIM),   # Agent 1: V-JEPA first 4 frames
            (dino_temporal[:, :n_frames//2, :], DINO_DIM),  # Agent 2: DINOv2 first 4 frames
        ],
        "homo_VV": lambda: [
            (vjepa_feat[:, :n_frames//2, :], VJEPA_DIM),   # Agent 1: V-JEPA first 4 frames
            (vjepa_feat[:, n_frames//2:, :], VJEPA_DIM),   # Agent 2: V-JEPA last 4 frames
        ],
        "homo_DD": lambda: [
            (dino_temporal[:, :n_frames//2, :], DINO_DIM),  # Agent 1: DINOv2 first 4 frames
            (dino_temporal[:, n_frames//2:, :], DINO_DIM),  # Agent 2: DINOv2 last 4 frames
        ],
    }

    all_results = {}

    for K in vocab_sizes:
        bits = 2 * N_HEADS * math.log2(K)
        print(f"\n{'='*60}", flush=True)
        print(f"  K={K} (capacity={bits:.1f} bits)", flush=True)
        print(f"{'='*60}", flush=True)

        k_results = {}

        for cond_name, make_configs in conditions.items():
            print(f"\n  ── {cond_name} ──", flush=True)
            cond_results = []

            for seed in range(n_seeds):
                configs = make_configs()
                result = train_heterogeneous_comm(
                    configs, mass_values, obj_names,
                    vocab_size=K, n_heads=N_HEADS, seed=seed,
                    comm_epochs=COMM_EPOCHS)

                if result is None:
                    print(f"    Seed {seed}: skipped (insufficient holdout)", flush=True)
                    continue

                # Analyze compositionality
                comp = analyze_compositionality(
                    result["tokens"], mass_values, obj_names,
                    vocab_size=K, n_agents=2, n_heads=N_HEADS)

                # Check symbol-property mapping
                sym_map = check_symbol_property_mapping(
                    result["tokens"], mass_values,
                    n_agents=2, n_heads=N_HEADS)

                # Count how many positions have monotonic mass encoding
                n_monotonic = sum(1 for v in sym_map.values() if v["monotonic"])

                cond_results.append({
                    "seed": seed,
                    "acc": result["best_acc"],
                    "posdis_global": comp["posdis_global"],
                    "agent0_posdis": comp["per_agent"][0]["posdis"],
                    "agent1_posdis": comp["per_agent"][1]["posdis"],
                    "agent0_mass_mi": comp["per_agent"][0]["mass_mi"],
                    "agent1_mass_mi": comp["per_agent"][1]["mass_mi"],
                    "entropies": comp["entropies"],
                    "n_monotonic": n_monotonic,
                    "symbol_mapping": sym_map,
                    "elapsed_s": result["elapsed_s"],
                })

                print(f"    Seed {seed}: acc={result['best_acc']:.1%} "
                      f"PD_g={comp['posdis_global']:.3f} "
                      f"PD_ag0={comp['per_agent'][0]['posdis']:.3f} "
                      f"PD_ag1={comp['per_agent'][1]['posdis']:.3f} "
                      f"mono={n_monotonic}/4 "
                      f"({result['elapsed_s']/60:.0f}min)", flush=True)

            if cond_results:
                accs = [r["acc"] for r in cond_results]
                pds = [r["posdis_global"] for r in cond_results]
                pd0s = [r["agent0_posdis"] for r in cond_results]
                pd1s = [r["agent1_posdis"] for r in cond_results]
                monos = [r["n_monotonic"] for r in cond_results]

                summary = {
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "posdis_mean": float(np.mean(pds)),
                    "posdis_std": float(np.std(pds)),
                    "agent0_posdis_mean": float(np.mean(pd0s)),
                    "agent1_posdis_mean": float(np.mean(pd1s)),
                    "monotonic_mean": float(np.mean(monos)),
                    "seeds": cond_results,
                }
                k_results[cond_name] = summary

                print(f"    Summary: acc={np.mean(accs):.1%}±{np.std(accs):.1%} "
                      f"PD={np.mean(pds):.3f}±{np.std(pds):.3f} "
                      f"mono={np.mean(monos):.1f}/4", flush=True)

        all_results[f"K={K}"] = k_results

    # ═══ Grand summary ═══
    print(f"\n{'='*70}", flush=True)
    print(f"  GRAND SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'K':>3s} │ {'Condition':>10s} │ {'Acc':>8s} │ {'PosDis':>10s} │ "
          f"{'PD_Ag0':>8s} │ {'PD_Ag1':>8s} │ {'Mono':>5s}", flush=True)
    print(f"  {'─'*3}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}─┼─"
          f"{'─'*8}─┼─{'─'*8}─┼─{'─'*5}", flush=True)

    for K_label, k_results in all_results.items():
        K = int(K_label.split("=")[1])
        for cond_name, summary in k_results.items():
            print(f"  {K:3d} │ {cond_name:>10s} │ "
                  f"{summary['acc_mean']:.1%}±{summary['acc_std']:.1%} │ "
                  f"{summary['posdis_mean']:.3f}±{summary['posdis_std']:.3f} │ "
                  f"{summary['agent0_posdis_mean']:.3f}    │ "
                  f"{summary['agent1_posdis_mean']:.3f}    │ "
                  f"{summary['monotonic_mean']:.1f}/4", flush=True)

    # ═══ The key comparison ═══
    print(f"\n  ╔═══ HETEROGENEOUS vs HOMOGENEOUS ═══╗", flush=True)
    for K_label, k_results in all_results.items():
        K = int(K_label.split("=")[1])
        if "hetero_VD" in k_results and "homo_VV" in k_results:
            h = k_results["hetero_VD"]
            vv = k_results["homo_VV"]
            dd = k_results.get("homo_DD", {"posdis_mean": 0, "acc_mean": 0})
            print(f"  ║ K={K:2d}: Hetero PD={h['posdis_mean']:.3f}  "
                  f"HomoVV PD={vv['posdis_mean']:.3f}  "
                  f"HomoDD PD={dd['posdis_mean']:.3f}", flush=True)
            emerges = h["posdis_mean"] > 0.3
            print(f"  ║       Compositionality {'EMERGES' if emerges else 'FAILS'} "
                  f"in hetero pairing", flush=True)
    print(f"  ╚════════════════════════════════════╝", flush=True)

    # ═══ Visualization ═══
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 93: Heterogeneous-Agent Emergent Communication",
                 fontsize=13, fontweight='bold')

    ks = [int(k.split("=")[1]) for k in all_results.keys()]
    cond_colors = {"hetero_VD": "red", "homo_VV": "blue", "homo_DD": "green"}
    cond_labels = {"hetero_VD": "V-JEPA+DINOv2", "homo_VV": "V-JEPA+V-JEPA",
                   "homo_DD": "DINOv2+DINOv2"}

    # Plot 1: PosDis
    ax = axes[0]
    for cond_name in ["hetero_VD", "homo_VV", "homo_DD"]:
        vals = []
        errs = []
        for K_label in all_results:
            if cond_name in all_results[K_label]:
                vals.append(all_results[K_label][cond_name]["posdis_mean"])
                errs.append(all_results[K_label][cond_name]["posdis_std"])
            else:
                vals.append(0)
                errs.append(0)
        ax.errorbar(ks, vals, yerr=errs, fmt='o-', color=cond_colors[cond_name],
                    label=cond_labels[cond_name], capsize=3)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label="Threshold")
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("PosDis (global)")
    ax.set_title("Compositionality")
    ax.legend(fontsize=8)
    ax.set_xscale('log', base=2)

    # Plot 2: Accuracy
    ax = axes[1]
    for cond_name in ["hetero_VD", "homo_VV", "homo_DD"]:
        vals = []
        errs = []
        for K_label in all_results:
            if cond_name in all_results[K_label]:
                vals.append(all_results[K_label][cond_name]["acc_mean"])
                errs.append(all_results[K_label][cond_name]["acc_std"])
            else:
                vals.append(0.5)
                errs.append(0)
        ax.errorbar(ks, vals, yerr=errs, fmt='o-', color=cond_colors[cond_name],
                    label=cond_labels[cond_name], capsize=3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("Holdout accuracy")
    ax.set_title("Task Performance")
    ax.legend(fontsize=8)
    ax.set_xscale('log', base=2)

    # Plot 3: Per-agent PosDis comparison (hetero only)
    ax = axes[2]
    ag0_vals, ag1_vals = [], []
    for K_label in all_results:
        if "hetero_VD" in all_results[K_label]:
            ag0_vals.append(all_results[K_label]["hetero_VD"]["agent0_posdis_mean"])
            ag1_vals.append(all_results[K_label]["hetero_VD"]["agent1_posdis_mean"])
        else:
            ag0_vals.append(0)
            ag1_vals.append(0)
    ax.plot(ks, ag0_vals, 'bo-', label="Agent 0 (V-JEPA 2)")
    ax.plot(ks, ag1_vals, 'rs-', label="Agent 1 (DINOv2)")
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("Per-agent PosDis")
    ax.set_title("Per-Agent Compositionality (Hetero)")
    ax.legend(fontsize=8)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    save_path = RESULTS_DIR / "phase93_heterogeneous.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {save_path}", flush=True)

    # Save results (strip sender_state and large arrays from JSON)
    json_results = {}
    for K_label, k_results in all_results.items():
        json_results[K_label] = {}
        for cond_name, summary in k_results.items():
            json_summary = {k: v for k, v in summary.items() if k != "seeds"}
            # Simplified seeds without symbol_mapping detail
            json_summary["seeds"] = []
            for s in summary["seeds"]:
                json_summary["seeds"].append({
                    "seed": s["seed"],
                    "acc": s["acc"],
                    "posdis_global": s["posdis_global"],
                    "agent0_posdis": s["agent0_posdis"],
                    "agent1_posdis": s["agent1_posdis"],
                    "n_monotonic": s["n_monotonic"],
                    "entropies": s["entropies"],
                    "agent0_mass_mi": s["agent0_mass_mi"],
                    "agent1_mass_mi": s["agent1_mass_mi"],
                })
            json_results[K_label][cond_name] = json_summary

    save_path = RESULTS_DIR / "phase93_heterogeneous.json"
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved {save_path}", flush=True)

    total_min = (time.time() - t_total) / 60
    print(f"\n  Total elapsed: {total_min:.1f} minutes", flush=True)

    return all_results


if __name__ == "__main__":
    run_phase93()

"""
Phase 94: Full Empirical Sweep
===============================
Systematic grid: architecture pairing × codebook size × population size × seeds.
450 runs on Physics 101 spring scenario. Priority-ordered for key results first.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase94_sweep import run_phase94; run_phase94()"
"""

import time, json, math, os, sys, signal, traceback
from datetime import datetime
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

# Constants
HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
BATCH_SIZE = 32
COMM_EPOCHS = 400
EARLY_STOP_PATIENCE = 150  # Stop if no improvement for this many epochs
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
RUN_TIMEOUT_S = 600  # 10 minutes per run


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

_feature_cache = {}

def load_features(scenario="spring", max_clips=None):
    """Load and cache V-JEPA 2 and DINOv2 features.

    Args:
        scenario: Physics 101 scenario name.
        max_clips: If set, randomly subsample to this many clips (deterministic seed).
    """
    cache_key = (scenario, max_clips)
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]

    vjepa_data = torch.load(
        RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    dino_feat = dino_data["features"].float()

    # Subsample if requested
    if max_clips is not None and len(obj_names) > max_clips:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(obj_names), max_clips, replace=False)
        idx.sort()
        vjepa_feat = vjepa_feat[idx]
        dino_feat = dino_feat[idx]
        obj_names = [obj_names[i] for i in idx]
        mass_values = mass_values[idx]
        print(f"  Subsampled {scenario}: {len(idx)} clips "
              f"({len(set(obj_names))} unique objects)", flush=True)

    n_frames = vjepa_feat.shape[1]
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous()

    result = (vjepa_feat, dino_temporal, obj_names, mass_values)
    _feature_cache[cache_key] = result
    return result


# ═══════════════════════════════════════════════════════════════
# Agent configuration
# ═══════════════════════════════════════════════════════════════

def make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal):
    """Create agent configurations for a given pairing type.

    Returns list of (features_view, input_dim) per agent.
    """
    n_frames = vjepa_feat.shape[1]  # 8
    fpa = n_frames // n_agents

    if pairing == "vjepa_homo":
        return [
            (vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM)
            for i in range(n_agents)
        ]
    elif pairing == "dinov2_homo":
        return [
            (dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM)
            for i in range(n_agents)
        ]
    elif pairing == "heterogeneous":
        configs = []
        for i in range(n_agents):
            if n_agents == 2:
                # Agent 0 = V-JEPA, Agent 1 = DINOv2
                is_vjepa = (i == 0)
            elif n_agents == 3:
                # Agents 0,1 = V-JEPA, Agent 2 = DINOv2
                is_vjepa = (i < 2)
            elif n_agents == 4:
                # Alternating: 0,2 = V-JEPA, 1,3 = DINOv2
                is_vjepa = (i % 2 == 0)
            else:
                is_vjepa = (i % 2 == 0)

            if is_vjepa:
                configs.append((vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs.append((dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
        return configs
    else:
        raise ValueError(f"Unknown pairing: {pairing}")


# ═══════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════

def single_run(agent_configs, mass_values, obj_names, vocab_size, seed):
    """Execute a single training run. Returns metrics dict."""
    n_agents = len(agent_configs)
    msg_dim = n_agents * N_HEADS * vocab_size
    n_total_heads = n_agents * N_HEADS

    agent_views = [feat.float() for feat, _ in agent_configs]

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)

    rng = np.random.RandomState(seed * 1000 + 42)
    holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

    if len(holdout_ids) < 4:
        return {"status": "failed", "error": "insufficient holdout"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    senders = []
    for feat, input_dim in agent_configs:
        n_frames_agent = feat.shape[1]
        enc = TemporalEncoder(HIDDEN_DIM, input_dim, n_frames=n_frames_agent)
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
    last_loss = 0.0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        # Timeout check
        if time.time() - t0 > RUN_TIMEOUT_S:
            break

        # Early stopping
        if ep - best_epoch > EARLY_STOP_PATIENCE and best_acc > 0.55:
            break

        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= SOFT_WARMUP
        epoch_loss = 0.0
        epoch_steps = 0

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
            epoch_loss += loss.item()
            epoch_steps += 1

        if epoch_steps > 0:
            last_loss = epoch_loss / epoch_steps

        if ep % 50 == 0:
            torch.mps.empty_cache()

        # Evaluate every 50 epochs
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
                    best_epoch = ep
                    best_state = {k: v.cpu().clone()
                                  for k, v in multi.state_dict().items()}

    converge_epoch = best_epoch + 1

    # Restore best and extract tokens + compute AUC
    if best_state:
        multi.load_state_dict(best_state)
    multi.eval()

    # Extract tokens
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    # Compute AUC on holdout using messages
    from sklearn.metrics import roc_auc_score
    auc = 0.5
    try:
        multi.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for i in range(len(holdout_ids)):
                for j in range(i + 1, len(holdout_ids)):
                    gi, gj = holdout_ids[i], holdout_ids[j]
                    if abs(mass_values[gi] - mass_values[gj]) < 0.5:
                        continue
                    va_i = [v[gi:gi+1].to(DEVICE) for v in agent_views]
                    va_j = [v[gj:gj+1].to(DEVICE) for v in agent_views]
                    mi_msg, _ = multi(va_i)
                    mj_msg, _ = multi(va_j)
                    # Use best receiver (first one after last reset)
                    pred_val = torch.sigmoid(receivers[0](mi_msg, mj_msg)).item()
                    preds_all.append(pred_val)
                    labels_all.append(float(mass_values[gi] > mass_values[gj]))
        if len(preds_all) > 5:
            auc = roc_auc_score(labels_all, preds_all)
    except Exception:
        pass

    # Compositionality
    mass_bins = np.digitize(mass_values,
                            np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)

    pd_global, mi_global, ent_global = positional_disentanglement(
        all_tokens, attributes, vocab_size)

    # Per-agent PosDis
    per_agent_pd = []
    for ag in range(n_agents):
        start = ag * N_HEADS
        agent_tokens = all_tokens[:, start:start + N_HEADS]
        pd_ag, _, _ = positional_disentanglement(agent_tokens, attributes, vocab_size)
        per_agent_pd.append(float(pd_ag))

    # Monotonicity: how many positions show monotonic symbol→mass mapping
    n_monotonic = 0
    for pos in range(n_total_heads):
        symbol_masses = defaultdict(list)
        for i, sym in enumerate(all_tokens[:, pos]):
            symbol_masses[int(sym)].append(mass_values[i])
        if len(symbol_masses) >= 2:
            sorted_syms = sorted(symbol_masses.keys(),
                                  key=lambda s: np.mean(symbol_masses[s]))
            means = [np.mean(symbol_masses[s]) for s in sorted_syms]
            from scipy.stats import spearmanr
            rho, _ = spearmanr(range(len(means)), means)
            if abs(rho) > 0.8:
                n_monotonic += 1

    elapsed = time.time() - t0

    return {
        "status": "success",
        "accuracy": float(best_acc),
        "auc": float(auc),
        "posdis": float(pd_global),
        "posdis_per_agent": per_agent_pd,
        "monotonicity": f"{n_monotonic}/{n_total_heads}",
        "epochs": converge_epoch,
        "final_loss": float(last_loss),
        "entropies": ent_global,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# Run queue with priority ordering
# ═══════════════════════════════════════════════════════════════

def build_run_queue(scenario="spring"):
    """Build priority-ordered run queue for a given scenario."""
    queue = []

    pop_sizes = [2, 3, 4]
    seeds = list(range(10))

    # Priority 1: Heterogeneous, K=3, all pop sizes
    for n_agents in pop_sizes:
        for seed in seeds:
            queue.append(("heterogeneous", 3, n_agents, scenario, seed, 1))

    # Priority 2: Homogeneous controls at K=3
    for pairing in ["vjepa_homo", "dinov2_homo"]:
        for n_agents in pop_sizes:
            for seed in seeds:
                queue.append((pairing, 3, n_agents, scenario, seed, 2))

    # Priority 3: Heterogeneous, other K values
    for K in [5, 8, 16, 32]:
        for n_agents in pop_sizes:
            for seed in seeds:
                queue.append(("heterogeneous", K, n_agents, scenario, seed, 3))

    # Priority 4: Homogeneous, other K values
    for pairing in ["vjepa_homo", "dinov2_homo"]:
        for K in [5, 8, 16, 32]:
            for n_agents in pop_sizes:
                for seed in seeds:
                    queue.append((pairing, K, n_agents, scenario, seed, 4))

    return queue


# ═══════════════════════════════════════════════════════════════
# Output generation
# ═══════════════════════════════════════════════════════════════

def generate_summary_tables(results_list):
    """Generate markdown summary tables."""
    lines = ["# Phase 94: Full Empirical Sweep — Summary Tables\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    # Group results
    grouped = defaultdict(list)
    for r in results_list:
        if r["status"] != "success":
            continue
        key = (r["pairing"], r["K"], r["n_agents"])
        grouped[key].append(r)

    # Main PosDis table
    lines.append("## PosDis (mean ± std across seeds)\n")
    lines.append("| Pairing | K | 2 agents | 3 agents | 4 agents |")
    lines.append("|---------|---|----------|----------|----------|")

    for pairing in ["heterogeneous", "vjepa_homo", "dinov2_homo"]:
        for K in [3, 5, 8, 16, 32]:
            cells = [f"| {pairing} | {K} |"]
            for n_ag in [2, 3, 4]:
                runs = grouped.get((pairing, K, n_ag), [])
                if runs:
                    pds = [r["posdis"] for r in runs]
                    cells.append(f" {np.mean(pds):.3f}±{np.std(pds):.3f} |")
                else:
                    cells.append(" — |")
            lines.append("".join(cells))

    # Accuracy table
    lines.append("\n## Accuracy (mean ± std)\n")
    lines.append("| Pairing | K | 2 agents | 3 agents | 4 agents |")
    lines.append("|---------|---|----------|----------|----------|")

    for pairing in ["heterogeneous", "vjepa_homo", "dinov2_homo"]:
        for K in [3, 5, 8, 16, 32]:
            cells = [f"| {pairing} | {K} |"]
            for n_ag in [2, 3, 4]:
                runs = grouped.get((pairing, K, n_ag), [])
                if runs:
                    accs = [r["accuracy"] for r in runs]
                    cells.append(f" {np.mean(accs):.1%}±{np.std(accs):.1%} |")
                else:
                    cells.append(" — |")
            lines.append("".join(cells))

    # Heterogeneity advantage table
    lines.append("\n## Heterogeneity Advantage (hetero PosDis − best homo PosDis)\n")
    lines.append("| K | 2 agents | 3 agents | 4 agents |")
    lines.append("|---|----------|----------|----------|")

    for K in [3, 5, 8, 16, 32]:
        cells = [f"| {K} |"]
        for n_ag in [2, 3, 4]:
            het_runs = grouped.get(("heterogeneous", K, n_ag), [])
            vv_runs = grouped.get(("vjepa_homo", K, n_ag), [])
            dd_runs = grouped.get(("dinov2_homo", K, n_ag), [])
            if het_runs and (vv_runs or dd_runs):
                het_pd = np.mean([r["posdis"] for r in het_runs])
                best_homo = max(
                    np.mean([r["posdis"] for r in vv_runs]) if vv_runs else 0,
                    np.mean([r["posdis"] for r in dd_runs]) if dd_runs else 0,
                )
                adv = het_pd - best_homo
                sign = "+" if adv >= 0 else ""
                cells.append(f" {sign}{adv:.3f} |")
            else:
                cells.append(" — |")
        lines.append("".join(cells))

    return "\n".join(lines)


def generate_heatmaps(results_list, scenario="spring"):
    """Generate heatmap visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for r in results_list:
        if r["status"] != "success":
            continue
        key = (r["pairing"], r["K"], r["n_agents"])
        grouped[key].append(r)

    pairings = ["heterogeneous", "vjepa_homo", "dinov2_homo"]
    pairing_labels = ["V-JEPA + DINOv2\n(heterogeneous)", "V-JEPA + V-JEPA\n(homogeneous)",
                      "DINOv2 + DINOv2\n(homogeneous)"]
    metrics = ["posdis", "accuracy", "auc"]
    metric_labels = ["PosDis", "Accuracy", "AUC"]
    ks = [3, 5, 8, 16, 32]
    n_agents_list = [2, 3, 4]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f"Phase 94: Full Empirical Sweep — {scenario.capitalize()} Heatmaps",
                 fontsize=14, fontweight='bold')

    for row, (pairing, p_label) in enumerate(zip(pairings, pairing_labels)):
        for col, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            data = np.full((len(n_agents_list), len(ks)), np.nan)
            for i, n_ag in enumerate(n_agents_list):
                for j, K in enumerate(ks):
                    runs = grouped.get((pairing, K, n_ag), [])
                    if runs:
                        vals = [r[metric] for r in runs]
                        data[i, j] = np.mean(vals)

            im = ax.imshow(data, cmap='YlOrRd', aspect='auto',
                          vmin=0, vmax=1)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels(ks)
            ax.set_yticks(range(len(n_agents_list)))
            ax.set_yticklabels(n_agents_list)

            # Annotate cells
            for i in range(len(n_agents_list)):
                for j in range(len(ks)):
                    val = data[i, j]
                    if not np.isnan(val):
                        color = 'white' if val > 0.6 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                               fontsize=8, color=color)

            if row == 2:
                ax.set_xlabel("Codebook K")
            if col == 0:
                ax.set_ylabel(f"{p_label}\nn_agents")
            if row == 0:
                ax.set_title(m_label)

    plt.tight_layout()
    suffix = f"_{scenario}" if scenario != "spring" else ""
    save_path = RESULTS_DIR / f"phase94_heatmaps{suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def generate_advantage_plot(results_list, scenario="spring"):
    """Generate heterogeneity advantage plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for r in results_list:
        if r["status"] != "success":
            continue
        key = (r["pairing"], r["K"], r["n_agents"])
        grouped[key].append(r)

    ks = [3, 5, 8, 16, 32]
    n_agents_list = [2, 3, 4]
    colors = {2: 'blue', 3: 'red', 4: 'green'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 94: Heterogeneity Advantage", fontsize=13, fontweight='bold')

    for ax, metric, label in [(axes[0], "posdis", "PosDis"),
                               (axes[1], "accuracy", "Accuracy")]:
        for n_ag in n_agents_list:
            advantages = []
            valid_ks = []
            for K in ks:
                het = grouped.get(("heterogeneous", K, n_ag), [])
                vv = grouped.get(("vjepa_homo", K, n_ag), [])
                dd = grouped.get(("dinov2_homo", K, n_ag), [])
                if het and (vv or dd):
                    het_val = np.mean([r[metric] for r in het])
                    best_homo = max(
                        np.mean([r[metric] for r in vv]) if vv else 0,
                        np.mean([r[metric] for r in dd]) if dd else 0,
                    )
                    advantages.append(het_val - best_homo)
                    valid_ks.append(K)

            if advantages:
                ax.plot(valid_ks, advantages, 'o-', color=colors[n_ag],
                       label=f'{n_ag} agents')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel(f"Δ{label} (hetero − best homo)")
        ax.set_title(f"Heterogeneity Advantage: {label}")
        ax.legend()
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    suffix = f"_{scenario}" if scenario != "spring" else ""
    save_path = RESULTS_DIR / f"phase94_heterogeneity_advantage{suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ═══════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════

def run_phase94(scenario="spring", max_clips=None):
    """Run the full empirical sweep for a given scenario."""
    print(f"╔══════════════════════════════════════════════════════════╗", flush=True)
    print(f"║  Phase 94: Full Empirical Sweep — {scenario:>10s}              ║", flush=True)
    print(f"╚══════════════════════════════════════════════════════════╝", flush=True)

    start_time = datetime.now()
    t_total = time.time()

    queue = build_run_queue(scenario)
    total_runs = len(queue)
    print(f"  Total runs queued: {total_runs}", flush=True)
    print(f"  Scenario: {scenario}", flush=True)
    print(f"  Priority 1 (hetero K=3): 30 runs", flush=True)
    print(f"  Priority 2 (homo K=3): 60 runs", flush=True)
    print(f"  Priority 3 (hetero K=5,8,16,32): 120 runs", flush=True)
    print(f"  Priority 4 (homo K=5,8,16,32): 240 runs", flush=True)
    print(f"  Start time: {start_time.isoformat()}", flush=True)

    # Load features once
    vjepa_feat, dino_temporal, obj_names, mass_values = load_features(scenario, max_clips)
    print(f"  Data: {len(obj_names)} clips, {len(set(obj_names))} objects", flush=True)

    results_list = []
    completed = 0
    failed = 0
    current_priority = 0

    suffix = f"_{scenario}" if scenario != "spring" else "_full"
    save_path_json = RESULTS_DIR / f"phase94{suffix}_sweep.json"

    for run_idx, (pairing, K, n_agents, scenario, seed, priority) in enumerate(queue):
        if priority != current_priority:
            current_priority = priority
            print(f"\n  ═══ Priority {priority} ═══", flush=True)

        try:
            configs = make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal)
            result = single_run(configs, mass_values, obj_names, K, seed)

            result.update({
                "pairing": pairing,
                "K": K,
                "n_agents": n_agents,
                "scenario": scenario,
                "seed": seed,
                "priority": priority,
            })

            results_list.append(result)

            if result["status"] == "success":
                completed += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1
            results_list.append({
                "pairing": pairing,
                "K": K,
                "n_agents": n_agents,
                "scenario": scenario,
                "seed": seed,
                "priority": priority,
                "status": "failed",
                "error": str(e),
            })

        # Progress report every 10 runs
        if (run_idx + 1) % 10 == 0:
            elapsed = time.time() - t_total
            rate = (run_idx + 1) / elapsed * 60  # runs per minute
            eta = (total_runs - run_idx - 1) / max(rate, 0.1)
            print(f"  Completed {completed}/{total_runs} runs "
                  f"({failed} failed). "
                  f"Elapsed: {elapsed/60:.1f}min. "
                  f"ETA: {eta:.0f}min. "
                  f"[{pairing[:5]} K={K} n={n_agents}]", flush=True)

        # Checkpoint every 50 runs
        if (run_idx + 1) % 50 == 0:
            _save_checkpoint(results_list, completed, failed,
                             start_time, save_path_json)
            torch.mps.empty_cache()

        # Fail-safe: stop if >20% failure rate after 50 runs
        if run_idx >= 50 and failed / (run_idx + 1) > 0.2:
            print(f"\n  WARNING: {failed}/{run_idx+1} runs failed (>{20}%). "
                  f"Stopping early.", flush=True)
            break

    # Final save
    end_time = datetime.now()
    total_elapsed = time.time() - t_total

    _save_checkpoint(results_list, completed, failed,
                     start_time, save_path_json, end_time=end_time)

    # Generate outputs
    print(f"\n  Generating summary tables...", flush=True)
    tables_md = generate_summary_tables(results_list)
    tables_path = RESULTS_DIR / f"phase94{suffix}_tables.md"
    with open(tables_path, "w") as f:
        f.write(tables_md)
    print(f"  Saved {tables_path}", flush=True)

    print(f"  Generating heatmaps...", flush=True)
    heatmap_path = generate_heatmaps(results_list, scenario)
    print(f"  Saved {heatmap_path}", flush=True)

    print(f"  Generating advantage plot...", flush=True)
    adv_path = generate_advantage_plot(results_list, scenario)
    print(f"  Saved {adv_path}", flush=True)

    # Compute summary stats
    successful = [r for r in results_list if r["status"] == "success"]
    het_pds = [r["posdis"] for r in successful if r["pairing"] == "heterogeneous"]
    homo_pds = [r["posdis"] for r in successful if r["pairing"] != "heterogeneous"]
    het_adv = np.mean(het_pds) - np.mean(homo_pds) if het_pds and homo_pds else 0

    best_run = max(successful, key=lambda r: r["posdis"]) if successful else None

    print(f"\n=== PHASE 94 COMPLETE ===", flush=True)
    print(f"Total runs: {completed}/{total_runs} completed ({failed} failed)", flush=True)
    print(f"Total time: {total_elapsed/3600:.1f} hours {(total_elapsed%3600)/60:.0f} minutes", flush=True)
    print(f"Key finding: heterogeneity advantage = {het_adv:+.3f} PosDis "
          f"(averaged across conditions)", flush=True)
    if best_run:
        print(f"Best condition: {best_run['pairing']} K={best_run['K']} "
              f"n_agents={best_run['n_agents']} — "
              f"PosDis = {best_run['posdis']:.3f}", flush=True)
    print(f"Files saved: phase94{suffix}_sweep.json, phase94{suffix}_tables.md, "
          f"phase94_heatmaps{'_'+scenario if scenario != 'spring' else ''}.png, "
          f"phase94_heterogeneity_advantage{'_'+scenario if scenario != 'spring' else ''}.png",
          flush=True)

    return results_list


def _save_checkpoint(results_list, completed, failed, start_time,
                     save_path, end_time=None):
    """Save checkpoint of results."""
    metadata = {
        "total_runs": len(results_list),
        "completed_runs": completed,
        "failed_runs": failed,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
        "total_time_seconds": (
            (end_time - start_time).total_seconds() if end_time
            else (datetime.now() - start_time).total_seconds()
        ),
    }

    # Strip non-serializable fields
    clean_results = []
    for r in results_list:
        clean = {k: v for k, v in r.items()
                 if k != "sender_state"}
        clean_results.append(clean)

    with open(save_path, "w") as f:
        json.dump({"metadata": metadata, "results": clean_results},
                  f, indent=2, default=str)


def run_phase94_all_scenarios():
    """Run the full sweep on spring, then fall, then ramp."""
    scenarios = ["spring", "fall", "ramp"]
    all_results = {}

    for scenario in scenarios:
        print(f"\n{'#'*70}", flush=True)
        print(f"#  SCENARIO: {scenario.upper()}", flush=True)
        print(f"{'#'*70}\n", flush=True)

        try:
            # Check data exists
            vjepa_path = RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt"
            dino_path = RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt"
            if not vjepa_path.exists() or not dino_path.exists():
                print(f"  SKIPPING {scenario}: feature files not found", flush=True)
                continue

            results = run_phase94(scenario=scenario)
            all_results[scenario] = len([r for r in results if r.get("status") == "success"])
            # Clear feature cache to free memory
            _feature_cache.clear()
            torch.mps.empty_cache()

        except Exception as e:
            print(f"  FAILED {scenario}: {e}", flush=True)
            traceback.print_exc()
            all_results[scenario] = f"FAILED: {e}"
            _feature_cache.clear()
            continue

    print(f"\n{'='*70}", flush=True)
    print(f"  ALL SCENARIOS COMPLETE", flush=True)
    for sc, count in all_results.items():
        print(f"  {sc}: {count} successful runs" if isinstance(count, int)
              else f"  {sc}: {count}", flush=True)
    print(f"{'='*70}", flush=True)

    return all_results


if __name__ == "__main__":
    run_phase94()

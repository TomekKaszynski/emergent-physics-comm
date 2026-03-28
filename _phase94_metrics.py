"""
Phase 94 metrics augmentation: Add TopSim and BosDis to all 1,350 runs.
Re-trains each condition with the same seed to get deterministic tokens,
then computes all three compositionality metrics.

Optimizations vs full sweep:
  - Eval only at end (no periodic holdout eval)
  - No AUC computation
  - Early stopping still active
  - Tokens extracted once, all metrics computed together

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase94_metrics import run_metrics_augmentation; run_metrics_augmentation()"
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
from metrics import positional_disentanglement, topographic_similarity

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
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
RUN_TIMEOUT_S = 600


# ═══ Architecture (identical to Phase 94) ═══

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
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


# ═══ Data ═══

_feature_cache = {}

def load_features(scenario, max_clips=None):
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

    if max_clips is not None and len(obj_names) > max_clips:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(obj_names), max_clips, replace=False)
        idx.sort()
        vjepa_feat = vjepa_feat[idx]
        dino_feat = dino_feat[idx]
        obj_names = [obj_names[i] for i in idx]
        mass_values = mass_values[idx]

    n_frames = vjepa_feat.shape[1]
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous()

    result = (vjepa_feat, dino_temporal, obj_names, mass_values)
    _feature_cache[cache_key] = result
    return result


def make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal):
    n_frames = vjepa_feat.shape[1]
    fpa = n_frames // n_agents

    if pairing == "vjepa_homo":
        return [(vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM) for i in range(n_agents)]
    elif pairing == "dinov2_homo":
        return [(dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM) for i in range(n_agents)]
    elif pairing == "heterogeneous":
        configs = []
        for i in range(n_agents):
            if n_agents == 2:
                is_vjepa = (i == 0)
            elif n_agents == 3:
                is_vjepa = (i < 2)
            else:
                is_vjepa = (i % 2 == 0)
            if is_vjepa:
                configs.append((vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs.append((dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
        return configs


# ═══ Metrics ═══

def compute_bosdis(tokens, attributes, vocab_size):
    """Bag-of-Symbols Disentanglement.
    For each unique symbol across all positions, compute MI with each attribute.
    Measure how much each symbol specializes for one attribute.
    Unlike PosDis (which asks 'does position p encode attribute a?'),
    BosDis asks 'does symbol s encode attribute a, regardless of position?'
    """
    n_samples, n_pos = tokens.shape
    n_attr = attributes.shape[1]

    # Build bag of (position, symbol) indicators for each unique symbol value
    # For each symbol value s (0..vocab_size-1), create binary indicator:
    #   contains_s[i] = 1 if any position in sample i has symbol s
    from metrics import mutual_information

    symbol_attr_mi = np.zeros((vocab_size, n_attr))
    symbol_active = np.zeros(vocab_size)

    for s in range(vocab_size):
        # Binary: does this sample contain symbol s in ANY position?
        contains_s = np.any(tokens == s, axis=1).astype(int)
        if contains_s.sum() == 0 or contains_s.sum() == n_samples:
            continue  # No variation, skip
        symbol_active[s] = 1
        for a in range(n_attr):
            symbol_attr_mi[s, a] = mutual_information(contains_s, attributes[:, a])

    # BosDis: for each active symbol, gap between best and second-best attribute MI
    bosdis = 0.0
    n_active = 0
    for s in range(vocab_size):
        if symbol_active[s] == 0:
            continue
        sorted_mi = np.sort(symbol_attr_mi[s])[::-1]
        if sorted_mi[0] > 1e-10:
            if len(sorted_mi) > 1 and sorted_mi[1] > 1e-10:
                bosdis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
            else:
                bosdis += 1.0  # Perfect specialization
            n_active += 1

    return float(bosdis / max(n_active, 1))


def compute_all_metrics(tokens, mass_values, obj_names, vocab_size):
    """Compute PosDis, TopSim, and BosDis from tokens."""
    mass_bins = np.digitize(mass_values,
                            np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)

    # PosDis
    posdis, mi_matrix, entropies = positional_disentanglement(
        tokens, attributes, vocab_size)

    # TopSim
    topsim = topographic_similarity(tokens, mass_bins, obj_bins_coarse,
                                     n_pairs=5000, seed=42)

    # BosDis
    bosdis = compute_bosdis(tokens, attributes, vocab_size)

    return {
        "posdis": float(posdis),
        "topsim": float(topsim),
        "bosdis": float(bosdis),
        "mi_matrix": mi_matrix.tolist() if hasattr(mi_matrix, 'tolist') else mi_matrix,
    }


# ═══ Lean training (tokens only, no AUC) ═══

def train_and_extract_tokens(agent_configs, mass_values, obj_names, vocab_size, seed):
    """Train model with same seed as Phase 94, extract tokens. Minimal eval overhead."""
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
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if time.time() - t0 > RUN_TIMEOUT_S:
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
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
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

        # Lightweight eval every 50 epochs (just accuracy, no AUC)
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

    # Restore best and extract tokens
    if best_state:
        multi.load_state_dict(best_state)
    multi.eval()

    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))

    return np.concatenate(all_tokens, axis=0)


# ═══ Main augmentation loop ═══

def augment_scenario(scenario, max_clips=None):
    """Augment one scenario's sweep JSON with TopSim and BosDis."""
    suffix = f"_{scenario}" if scenario != "spring" else "_full"
    json_path = RESULTS_DIR / f"phase94{suffix}_sweep.json"

    print(f"\n═══ Augmenting {scenario} ({json_path}) ═══", flush=True)

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    n_total = len(results)
    print(f"  {n_total} runs to augment", flush=True)

    vjepa_feat, dino_temporal, obj_names, mass_values = load_features(scenario, max_clips)
    print(f"  Data: {len(obj_names)} clips", flush=True)

    t0 = time.time()
    augmented = 0
    skipped = 0

    for idx, r in enumerate(results):
        # Skip failed runs
        if r.get("status") != "success":
            skipped += 1
            continue

        # Skip already augmented
        if "topsim" in r and "bosdis" in r:
            augmented += 1
            continue

        pairing = r["pairing"]
        K = r["K"]
        n_agents = r["n_agents"]
        seed = r["seed"]

        try:
            configs = make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal)
            tokens = train_and_extract_tokens(configs, mass_values, obj_names, K, seed)

            if tokens is None:
                r["topsim"] = None
                r["bosdis"] = None
                skipped += 1
                continue

            metrics = compute_all_metrics(tokens, mass_values, obj_names, K)
            r["topsim"] = metrics["topsim"]
            r["bosdis"] = metrics["bosdis"]
            r["posdis_recomputed"] = metrics["posdis"]
            augmented += 1

        except Exception as e:
            r["topsim"] = None
            r["bosdis"] = None
            r["metrics_error"] = str(e)
            skipped += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed * 60
            eta = (n_total - idx - 1) / max(rate, 0.1)
            print(f"  {idx+1}/{n_total} ({augmented} augmented, {skipped} skipped). "
                  f"Elapsed: {elapsed/60:.1f}min. ETA: {eta:.0f}min", flush=True)

        # Checkpoint every 50
        if (idx + 1) % 50 == 0:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            torch.mps.empty_cache()

    # Final save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"  Done: {augmented} augmented, {skipped} skipped. "
          f"Elapsed: {elapsed/60:.1f}min", flush=True)

    return data


def generate_triple_metric_table(scenarios_data):
    """Generate summary table with PosDis, TopSim, BosDis side by side."""
    lines = ["# Phase 94: Triple Compositionality Metrics\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("Three independent measures of compositionality confirm the finding.\n")

    for scenario, data in scenarios_data.items():
        results = [r for r in data["results"] if r.get("status") == "success"
                   and r.get("topsim") is not None]

        if not results:
            continue

        lines.append(f"\n## {scenario.capitalize()}\n")
        lines.append("| Pairing | K | n_agents | PosDis | TopSim | BosDis | Divergent? |")
        lines.append("|---------|---|----------|--------|--------|--------|------------|")

        grouped = defaultdict(list)
        for r in results:
            grouped[(r["pairing"], r["K"], r["n_agents"])].append(r)

        for pairing in ["heterogeneous", "vjepa_homo", "dinov2_homo"]:
            for K in [3, 5, 8, 16, 32]:
                for n_ag in [2, 3, 4]:
                    runs = grouped.get((pairing, K, n_ag), [])
                    if not runs:
                        continue
                    pd = np.mean([r["posdis"] for r in runs])
                    ts = np.mean([r["topsim"] for r in runs if r["topsim"] is not None])
                    bd = np.mean([r["bosdis"] for r in runs if r["bosdis"] is not None])

                    # Flag divergence: high PosDis but low TopSim or vice versa
                    divergent = ""
                    if pd > 0.5 and ts < 0.1:
                        divergent = "PosDis>>TopSim"
                    elif ts > 0.3 and pd < 0.2:
                        divergent = "TopSim>>PosDis"
                    elif pd > 0.5 and bd < 0.3:
                        divergent = "PosDis>>BosDis"

                    lines.append(
                        f"| {pairing} | {K} | {n_ag} | "
                        f"{pd:.3f} | {ts:.3f} | {bd:.3f} | {divergent} |")

    # Aggregate summary
    lines.append("\n## Aggregate Summary\n")
    lines.append("| Scenario | Pairing | PosDis | TopSim | BosDis |")
    lines.append("|----------|---------|--------|--------|--------|")

    for scenario, data in scenarios_data.items():
        results = [r for r in data["results"]
                   if r.get("status") == "success" and r.get("topsim") is not None]
        for pairing in ["heterogeneous", "vjepa_homo", "dinov2_homo"]:
            runs = [r for r in results if r["pairing"] == pairing]
            if not runs:
                continue
            pd = np.mean([r["posdis"] for r in runs])
            ts = np.mean([r["topsim"] for r in runs if r["topsim"] is not None])
            bd = np.mean([r["bosdis"] for r in runs if r["bosdis"] is not None])
            lines.append(f"| {scenario} | {pairing} | {pd:.3f} | {ts:.3f} | {bd:.3f} |")

    return "\n".join(lines)


def run_metrics_augmentation():
    """Augment all three scenario JSONs with TopSim and BosDis."""
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 94: Triple Metrics Augmentation                   ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    scenarios_data = {}

    for scenario, max_clips in [("spring", None), ("fall", None), ("ramp", 500)]:
        try:
            data = augment_scenario(scenario, max_clips)
            scenarios_data[scenario] = data
        except Exception as e:
            print(f"  FAILED {scenario}: {e}", flush=True)
            traceback.print_exc()

    # Generate triple metric summary
    print("\n  Generating triple metric table...", flush=True)
    table = generate_triple_metric_table(scenarios_data)
    table_path = RESULTS_DIR / "phase94_triple_metrics.md"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"  Saved {table_path}", flush=True)

    total_min = (time.time() - t_total) / 60
    print(f"\n  Total elapsed: {total_min:.1f} minutes", flush=True)

    # Print quick summary
    print("\n═══ QUICK SUMMARY ═══", flush=True)
    for scenario, data in scenarios_data.items():
        results = [r for r in data["results"]
                   if r.get("status") == "success" and r.get("topsim") is not None]
        for pairing in ["heterogeneous", "vjepa_homo", "dinov2_homo"]:
            runs = [r for r in results if r["pairing"] == pairing]
            if not runs:
                continue
            pd = np.mean([r["posdis"] for r in runs])
            ts = np.mean([r["topsim"] for r in runs if r.get("topsim") is not None])
            bd = np.mean([r["bosdis"] for r in runs if r.get("bosdis") is not None])
            print(f"  {scenario:8s} {pairing:15s}: PD={pd:.3f} TS={ts:.3f} BD={bd:.3f}",
                  flush=True)


if __name__ == "__main__":
    run_metrics_augmentation()

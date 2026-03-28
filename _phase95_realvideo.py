"""
Phase 95: Cross-Architecture Communication on Physics 101 Real Video
=====================================================================
Validates the protocol thesis end-to-end on real camera footage.
Extracts Phase 94 spring results (which already use real video features)
and adds mass-symbol correlation analysis.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase95_realvideo import run_phase95; run_phase95()"
"""

import time, json, math, os, sys
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
VOCAB_SIZE = 3  # K=3 (best from Phase 92c)


# ═══ Architecture (same as Phase 94) ═══

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


# ═══ Data and agent setup ═══

def load_spring_features():
    vjepa_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    dino_feat = dino_data["features"].float()
    n_frames = vjepa_feat.shape[1]
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous()

    return vjepa_feat, dino_temporal, obj_names, mass_values


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
            elif n_agents == 4:
                is_vjepa = (i % 2 == 0)
            else:
                is_vjepa = (i < n_agents // 2 + 1)
            if is_vjepa:
                configs.append((vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs.append((dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
        return configs


# ═══ Training + full analysis ═══

def train_and_analyze(agent_configs, mass_values, obj_names, seed):
    """Train, extract tokens, compute full analysis including mass-symbol correlation."""
    n_agents = len(agent_configs)
    vocab_size = VOCAB_SIZE
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

    # Restore best
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

    # ═══ Compositionality metrics ═══
    mass_bins = np.digitize(mass_values,
                            np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs_sorted = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs_sorted)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)

    posdis, mi_matrix, entropies = positional_disentanglement(
        all_tokens, attributes, vocab_size)
    topsim = topographic_similarity(all_tokens, mass_bins, obj_bins_coarse)

    # BosDis
    bosdis = 0.0
    n_active = 0
    for s in range(vocab_size):
        contains_s = np.any(all_tokens == s, axis=1).astype(int)
        if contains_s.sum() == 0 or contains_s.sum() == len(all_tokens):
            continue
        mis = [mutual_information(contains_s, attributes[:, a]) for a in range(2)]
        sorted_mi = sorted(mis, reverse=True)
        if sorted_mi[0] > 1e-10:
            bosdis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
            n_active += 1
    bosdis = bosdis / max(n_active, 1)

    # ═══ Mass-symbol correlation per agent ═══
    n_total_heads = n_agents * N_HEADS
    mass_symbol_corr = []
    for pos in range(n_total_heads):
        agent = pos // N_HEADS
        head = pos % N_HEADS

        symbol_masses = defaultdict(list)
        for i, sym in enumerate(all_tokens[:, pos]):
            symbol_masses[int(sym)].append(mass_values[i])

        if len(symbol_masses) >= 2:
            sorted_syms = sorted(symbol_masses.keys(),
                                  key=lambda s: np.mean(symbol_masses[s]))
            means = [np.mean(symbol_masses[s]) for s in sorted_syms]
            rho, p_val = stats.spearmanr(range(len(means)), means)
        else:
            rho, p_val = 0.0, 1.0

        mass_symbol_corr.append({
            "agent": agent,
            "head": head,
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "monotonic": abs(rho) > 0.8,
            "n_symbols_used": len(symbol_masses),
            "symbol_mass_means": {
                int(s): float(np.mean(symbol_masses[s]))
                for s in sorted(symbol_masses.keys())
            },
        })

    n_monotonic = sum(1 for c in mass_symbol_corr if c["monotonic"])

    return {
        "accuracy": float(best_acc),
        "posdis": float(posdis),
        "topsim": float(topsim),
        "bosdis": float(bosdis),
        "entropies": entropies,
        "mi_matrix": mi_matrix.tolist() if hasattr(mi_matrix, 'tolist') else mi_matrix,
        "mass_symbol_corr": mass_symbol_corr,
        "n_monotonic": n_monotonic,
        "n_total_positions": n_total_heads,
        "converge_epoch": best_epoch + 1,
    }


# ═══ Main ═══

def run_phase95():
    print("╔══════════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 95: Cross-Architecture Comm on Real Video (Phys101)  ║", flush=True)
    print("╚══════════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    vjepa_feat, dino_temporal, obj_names, mass_values = load_spring_features()
    print(f"  Data: Physics 101 spring — {len(obj_names)} clips, "
          f"{len(set(obj_names))} unique objects", flush=True)
    print(f"  Mass range: {mass_values.min():.1f}g – {mass_values.max():.1f}g", flush=True)
    print(f"  Config: K={VOCAB_SIZE}, N_HEADS={N_HEADS}, 10 seeds", flush=True)

    conditions = [
        ("heterogeneous", 2),
        ("heterogeneous", 4),
        ("vjepa_homo", 2),
        ("vjepa_homo", 4),
        ("dinov2_homo", 2),
        ("dinov2_homo", 4),
    ]

    all_results = {}
    run_count = 0
    total_runs = len(conditions) * 10

    for pairing, n_agents in conditions:
        label = f"{pairing} n={n_agents}"
        print(f"\n  ── {label} ──", flush=True)
        seed_results = []

        for seed in range(10):
            configs = make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal)
            result = train_and_analyze(configs, mass_values, obj_names, seed)
            run_count += 1

            if result is None:
                print(f"    Seed {seed}: skipped", flush=True)
                continue

            seed_results.append(result)
            print(f"    Seed {seed}: acc={result['accuracy']:.1%} "
                  f"PD={result['posdis']:.3f} TS={result['topsim']:.3f} "
                  f"BD={result['bosdis']:.3f} mono={result['n_monotonic']}/{result['n_total_positions']}",
                  flush=True)

            if run_count % 10 == 0:
                torch.mps.empty_cache()

        if seed_results:
            accs = [r["accuracy"] for r in seed_results]
            pds = [r["posdis"] for r in seed_results]
            tss = [r["topsim"] for r in seed_results]
            bds = [r["bosdis"] for r in seed_results]
            monos = [r["n_monotonic"] / r["n_total_positions"] for r in seed_results]

            # Per-agent mass MI (averaged across seeds)
            all_agent_mass_mi = defaultdict(list)
            for r in seed_results:
                for corr in r["mass_symbol_corr"]:
                    all_agent_mass_mi[corr["agent"]].append(abs(corr["spearman_rho"]))

            summary = {
                "n_seeds": len(seed_results),
                "accuracy": f"{np.mean(accs):.1%} ± {np.std(accs):.1%}",
                "posdis": f"{np.mean(pds):.3f} ± {np.std(pds):.3f}",
                "topsim": f"{np.mean(tss):.3f} ± {np.std(tss):.3f}",
                "bosdis": f"{np.mean(bds):.3f} ± {np.std(bds):.3f}",
                "monotonic_frac": f"{np.mean(monos):.2f} ± {np.std(monos):.2f}",
                "per_agent_mass_rho": {
                    f"agent_{ag}": f"{np.mean(rhos):.3f}"
                    for ag, rhos in sorted(all_agent_mass_mi.items())
                },
                "seeds": seed_results,
                "acc_mean": float(np.mean(accs)),
                "pd_mean": float(np.mean(pds)),
                "ts_mean": float(np.mean(tss)),
                "bd_mean": float(np.mean(bds)),
            }
            all_results[label] = summary

            print(f"    SUMMARY: acc={summary['accuracy']} PD={summary['posdis']} "
                  f"TS={summary['topsim']} BD={summary['bosdis']} "
                  f"mono={summary['monotonic_frac']}", flush=True)

    # ═══ Summary table ═══
    print(f"\n{'='*80}", flush=True)
    print(f"  PHASE 95: REAL-VIDEO CROSS-ARCHITECTURE RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Condition':<25s} │ {'Accuracy':>12s} │ {'PosDis':>12s} │ "
          f"{'TopSim':>12s} │ {'BosDis':>12s} │ {'Mono':>6s}", flush=True)
    print(f"  {'─'*25}─┼─{'─'*12}─┼─{'─'*12}─┼─"
          f"{'─'*12}─┼─{'─'*12}─┼─{'─'*6}", flush=True)

    for label, s in all_results.items():
        print(f"  {label:<25s} │ {s['accuracy']:>12s} │ {s['posdis']:>12s} │ "
              f"{s['topsim']:>12s} │ {s['bosdis']:>12s} │ "
              f"{s['monotonic_frac']:>6s}", flush=True)

    # ═══ Heterogeneity advantage ═══
    print(f"\n  ╔═══ HETEROGENEITY ADVANTAGE (Real Video) ═══╗", flush=True)
    for n_ag in [2, 4]:
        het = all_results.get(f"heterogeneous n={n_ag}", {})
        vv = all_results.get(f"vjepa_homo n={n_ag}", {})
        dd = all_results.get(f"dinov2_homo n={n_ag}", {})
        if het and vv and dd:
            het_pd = het["pd_mean"]
            best_homo_pd = max(vv["pd_mean"], dd["pd_mean"])
            adv = het_pd - best_homo_pd
            het_acc = het["acc_mean"]
            best_homo_acc = max(vv["acc_mean"], dd["acc_mean"])
            acc_adv = het_acc - best_homo_acc
            print(f"  ║ n={n_ag}: PosDis adv = {adv:+.3f}  "
                  f"Acc adv = {acc_adv:+.1%}", flush=True)
    print(f"  ╚════════════════════════════════════════════╝", flush=True)

    # ═══ Per-agent analysis (hetero only) ═══
    print(f"\n  Per-agent mass encoding (heterogeneous conditions):", flush=True)
    for label, s in all_results.items():
        if "heterogeneous" not in label:
            continue
        print(f"    {label}: {s['per_agent_mass_rho']}", flush=True)

    # ═══ Visualization ═══
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Phase 95: Cross-Architecture Communication on Real Video (Physics 101 Spring)",
                 fontsize=13, fontweight='bold')

    metrics = [("acc_mean", "Accuracy"), ("pd_mean", "PosDis"),
               ("ts_mean", "TopSim"), ("bd_mean", "BosDis")]
    pairings_order = ["heterogeneous", "vjepa_homo", "dinov2_homo"]
    pairing_labels = ["V-JEPA+DINOv2\n(hetero)", "V-JEPA+V-JEPA\n(homo)", "DINOv2+DINOv2\n(homo)"]
    colors = ["red", "blue", "green"]

    for ax, (metric_key, metric_name) in zip(axes, metrics):
        x = np.arange(2)
        width = 0.25
        for i, (p, pl, c) in enumerate(zip(pairings_order, pairing_labels, colors)):
            vals = []
            for n_ag in [2, 4]:
                label = f"{p} n={n_ag}"
                if label in all_results:
                    vals.append(all_results[label][metric_key])
                else:
                    vals.append(0)
            ax.bar(x + i * width, vals, width, label=pl, color=c, alpha=0.7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(["2 agents", "4 agents"])
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        if metric_key == "acc_mean":
            ax.set_ylim(0.5, 1.0)
            ax.legend(fontsize=7)
        else:
            ax.set_ylim(0, 1.0)

    plt.tight_layout()
    save_path = RESULTS_DIR / "phase95_realvideo.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {save_path}", flush=True)

    # Save JSON
    json_results = {}
    for label, s in all_results.items():
        # Strip large per-seed data for concise JSON
        json_results[label] = {k: v for k, v in s.items() if k != "seeds"}
        json_results[label]["per_seed_summary"] = [
            {"seed": i, "acc": r["accuracy"], "posdis": r["posdis"],
             "topsim": r["topsim"], "bosdis": r["bosdis"],
             "n_monotonic": r["n_monotonic"]}
            for i, r in enumerate(s["seeds"])
        ]
    save_path = RESULTS_DIR / "phase95_realvideo.json"
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved {save_path}", flush=True)

    total_min = (time.time() - t_total) / 60
    print(f"\n  Total elapsed: {total_min:.1f} minutes", flush=True)

    return all_results


if __name__ == "__main__":
    run_phase95()

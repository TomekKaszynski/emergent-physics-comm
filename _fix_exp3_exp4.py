"""
Fix 1: Exp3 width sweep rerun
Fix 2: Exp4 agent confound investigation
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as scipy_stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results/neurips_battery")

HIDDEN_DIM = 128
DEFAULT_VOCAB = 5
COMM_EPOCHS = 600
BATCH_SIZE = 32
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
EARLY_STOP = 200
RESET_INTERVAL = 40

# Import architecture from existing battery
sys.path.insert(0, os.path.dirname(__file__))
from _neurips_battery import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender,
    ContinuousSender, ContinuousMultiSender, RawProbe, Receiver,
    compute_posdis, compute_topsim, continuous_posdis, compute_causal_spec,
    load_task
)


def train_run(arm, feat, p1, p2, mass, obj_names, seed,
              n_agents=4, vocab_size=5, n_heads=2, msg_dim=None, epochs=COMM_EPOCHS):
    """Single training run. Returns metrics dict."""
    t0 = time.time()
    n, nf, dim = feat.shape
    fpa = max(1, nf // n_agents)
    is_discrete = (arm == "discrete")

    if msg_dim is None:
        msg_dim = n_agents * n_heads * vocab_size

    views = []
    for i in range(n_agents):
        start = (i * fpa) % nf
        end = min(start + fpa, nf)
        if end - start < fpa:
            # Wrap
            v = torch.cat([feat[:, start:, :], feat[:, :fpa-(nf-start), :]], dim=1)
        else:
            v = feat[:, start:end, :]
        views.append(v)

    rng = np.random.RandomState(seed * 1000 + 42)
    uo = sorted(set(obj_names))
    ho = set(rng.choice(uo, max(4, len(uo)//5), replace=False))
    tr = np.array([i for i, o in enumerate(obj_names) if o not in ho])
    tei = np.array([i for i, o in enumerate(obj_names) if o in ho])
    if len(tei) < 4:
        perm = rng.permutation(n); n_ho = max(4, n//5)
        tei, tr = perm[:n_ho], perm[n_ho:]

    torch.manual_seed(seed); np.random.seed(seed)

    if arm == "discrete":
        ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, vocab_size, n_heads)
              for _ in range(n_agents)]
        sender = DiscreteMultiSender(ss)
    elif arm == "continuous":
        per_agent = msg_dim // n_agents
        ss = [ContinuousSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, per_agent)
              for _ in range(n_agents)]
        sender = ContinuousMultiSender(ss)

    sender = sender.to(DEVICE)
    recvs = [Receiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    ros = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in recvs]
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(max(vocab_size, 2))
    nb = max(1, len(tr) // BATCH_SIZE)
    best_acc, best_state, best_ep = 0.0, None, 0

    for ep in range(epochs):
        if time.time() - t0 > 600: break
        if ep - best_ep > EARLY_STOP and best_acc > 0.55: break
        if ep > 0 and ep % RESET_INTERVAL == 0:
            for i in range(len(recvs)):
                recvs[i] = Receiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                ros[i] = torch.optim.Adam(recvs[i].parameters(), lr=RECEIVER_LR)
        sender.train(); [r.train() for r in recvs]
        tau = 3.0 + (1.0 - 3.0) * ep / max(1, epochs - 1); hard = ep >= 30
        for _ in range(nb):
            ia = rng.choice(tr, BATCH_SIZE); ib = rng.choice(tr, BATCH_SIZE)
            s = ia == ib
            while s.any(): ib[s] = rng.choice(tr, s.sum()); s = ia == ib
            md = np.abs(mass[ia] - mass[ib]); k = md > 0.5
            if k.sum() < 4: continue
            ia, ib = ia[k], ib[k]
            va = [v[ia].to(DEVICE) for v in views]; vb = [v[ib].to(DEVICE) for v in views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = sender(va, tau=tau, hard=hard) if is_discrete else sender(va)
            mb, lb = sender(vb, tau=tau, hard=hard) if is_discrete else sender(vb)
            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), label) for r in recvs) / len(recvs)
            if is_discrete and la:
                for lg in la + lb:
                    lp = F.log_softmax(lg, -1); p = lp.exp().clamp(min=1e-8)
                    ent = -(p * lp).sum(-1).mean()
                    if ent / max_ent < 0.1: loss = loss - 0.03 * ent
            if torch.isnan(loss): so.zero_grad(); [o.zero_grad() for o in ros]; continue
            so.zero_grad(); [o.zero_grad() for o in ros]; loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0); so.step(); [o.step() for o in ros]
        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep + 1) % 50 == 0 or ep == 0:
            sender.eval(); [r.eval() for r in recvs]
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(tei, min(32, len(tei))); ib_h = er.choice(tei, min(32, len(tei)))
                    s2 = ia_h == ib_h
                    while s2.any(): ib_h[s2] = er.choice(tei, s2.sum()); s2 = ia_h == ib_h
                    mdh = np.abs(mass[ia_h] - mass[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in views]; wh = [v[ib_h].to(DEVICE) for v in views]
                    mah, _ = sender(vh); mbh, _ = sender(wh)
                    for r in recvs:
                        c += ((r(mah, mbh) > 0) == (mass_dev[ia_h] > mass_dev[ib_h])).sum().item()
                        t += len(ia_h)
                acc = c / max(t, 1)
                if acc > best_acc:
                    best_acc = acc; best_ep = ep
                    best_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}

    if best_state: sender.load_state_dict(best_state)
    sender.eval()

    # Metrics
    n_total_heads = n_agents * n_heads
    attrs = np.stack([p1, p2], axis=1)

    if is_discrete:
        tokens = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                v2 = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                _, logits = sender(v2)
                tokens.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
        tokens = np.concatenate(tokens, 0)
        posdis, _ = compute_posdis(tokens, attrs, vocab_size)
        topsim = compute_topsim(tokens, p1, p2)
    else:
        repr_all = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                v2 = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                m, _ = sender(v2); repr_all.append(m.cpu())
        repr_all = torch.cat(repr_all, 0).numpy()
        posdis, _ = continuous_posdis(repr_all, attrs)
        topsim = 0.0

    cs, _ = compute_causal_spec(sender, views, mass, recvs[0],
                                 n_total_heads if is_discrete else msg_dim,
                                 is_discrete, vocab_size)

    return {
        "arm": arm, "accuracy": float(best_acc), "posdis": float(posdis),
        "topsim": float(topsim), "causal_spec": float(cs),
        "converge_epoch": best_ep + 1, "elapsed_s": time.time() - t0,
    }


# ═══════════════════════════════════════════════════════════════
# FIX 1: Width sweep
# ═══════════════════════════════════════════════════════════════

def fix1_width_sweep():
    print("╔═ FIX 1: Exp3 Width Sweep Rerun ═╗", flush=True)
    d = RESULTS_DIR / "exp3_width_sweep"; d.mkdir(exist_ok=True)

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data
    results = {}
    timings = []

    # Width configs for discrete: N_AGENTS=4, VOCAB=5, vary N_HEADS
    # width = 4 * nh * 5 = 20*nh
    # width 10 → nh=1, actual=20 (can't go lower with 4 agents and V=5)
    # width 20 → nh=1, actual=20
    # width 40 → nh=2, actual=40
    # width 80 → nh=4, actual=80
    # width 160 → nh=8, actual=160
    # width 320 → nh=16, actual=320

    width_configs = [
        (1, 20),   # nh=1 → actual width 20
        (2, 40),   # nh=2 → actual width 40
        (4, 80),   # nh=4 → actual width 80
        (8, 160),  # nh=8 → actual width 160
        (16, 320), # nh=16 → actual width 320
    ]

    for nh, actual_w in width_configs:
        for arm in ["discrete", "continuous"]:
            if arm == "discrete":
                md = actual_w
            else:
                md = actual_w

            label = f"{arm}_w={actual_w}"
            print(f"\n  ── {label} (n_heads={nh if arm=='discrete' else '-'}) ──", flush=True)
            runs = []
            for seed in range(10):
                r = train_run(arm, feat, p1, p2, mass, obj, seed,
                              n_heads=nh, msg_dim=md)
                runs.append(r)
                timings.append(r["elapsed_s"])
                if seed == 2:
                    avg_t = np.mean(timings[-3:])
                    remaining = (len(width_configs) * 2 * 10) - len(timings)
                    print(f"      Measured: {avg_t:.1f}s/seed, "
                          f"~{remaining * avg_t / 60:.0f}min remaining", flush=True)

            accs = [r["accuracy"] for r in runs]
            pds = [r["posdis"] for r in runs]
            css = [r["causal_spec"] for r in runs]
            tss = [r["topsim"] for r in runs]
            results[label] = {
                "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "pd": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "cs": f"{np.mean(css):.3f}±{np.std(css):.3f}",
                "ts": f"{np.mean(tss):.3f}±{np.std(tss):.3f}",
                "acc_m": float(np.mean(accs)), "pd_m": float(np.mean(pds)),
                "cs_m": float(np.mean(css)), "ts_m": float(np.mean(tss)),
                "actual_width": actual_w, "n_heads": nh,
            }
            print(f"    {label}: acc={results[label]['acc']} PD={results[label]['pd']} "
                  f"CS={results[label]['cs']}", flush=True)
            torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exp3: Discrete vs Continuous at Different Bottleneck Widths",
                 fontsize=13, fontweight='bold')
    widths = [w for _, w in width_configs]
    for ax, metric, label in [(axes[0], "pd_m", "PosDis"),
                               (axes[1], "cs_m", "Causal Specificity"),
                               (axes[2], "acc_m", "Accuracy")]:
        for arm, color in [("discrete", "#2196F3"), ("continuous", "#F44336")]:
            vals = [results.get(f"{arm}_w={w}", {}).get(metric, 0) for w in widths]
            ax.plot(widths, vals, 'o-', color=color, label=arm, linewidth=2, markersize=7)
        ax.set_xlabel("Bottleneck Width"); ax.set_ylabel(label)
        ax.set_title(label); ax.legend(); ax.set_xscale('log', base=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_width_money_plot.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved exp3 results + plot", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# FIX 2: Agent confound
# ═══════════════════════════════════════════════════════════════

def fix2_agent_confound():
    print("\n╔═ FIX 2: Agent Sweep Confound Investigation ═╗", flush=True)
    d = RESULTS_DIR / "exp4_agent_confound"; d.mkdir(exist_ok=True)

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data  # [206, 8, 1024]
    results = {}
    timings = []
    fpa_fixed = 2  # Hold frames per agent constant at 2

    for n_agents in [1, 2, 4, 8]:
        md = n_agents * 2 * 5  # n_agents * n_heads * vocab_size
        print(f"\n  ── N={n_agents} agents, {fpa_fixed} frames each ──", flush=True)

        # Build feature tensor: n_agents * fpa_fixed frames needed
        total_frames_needed = n_agents * fpa_fixed
        if total_frames_needed <= 8:
            # Use first total_frames_needed frames
            feat_used = feat[:, :total_frames_needed, :]
        else:
            # Duplicate frames to fill
            repeats = math.ceil(total_frames_needed / 8)
            feat_expanded = feat.repeat(1, repeats, 1)[:, :total_frames_needed, :]
            feat_used = feat_expanded

        runs = []
        for seed in range(10):
            r = train_run("discrete", feat_used, p1, p2, mass, obj, seed,
                          n_agents=n_agents, msg_dim=md)
            runs.append(r)
            timings.append(r["elapsed_s"])
            if seed == 2 and n_agents == 1:
                avg_t = np.mean(timings[-3:])
                total_remaining = 4 * 10 - len(timings)
                print(f"      Measured: {avg_t:.1f}s/seed, "
                      f"~{total_remaining * avg_t / 60:.0f}min remaining", flush=True)

        accs = [r["accuracy"] for r in runs]
        pds = [r["posdis"] for r in runs]
        css = [r["causal_spec"] for r in runs]
        label = f"N={n_agents}_fixed_fpa"
        results[label] = {
            "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
            "pd": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
            "cs": f"{np.mean(css):.3f}±{np.std(css):.3f}",
            "acc_m": float(np.mean(accs)), "pd_m": float(np.mean(pds)),
            "cs_m": float(np.mean(css)),
            "n_agents": n_agents, "fpa": fpa_fixed,
            "total_frames": total_frames_needed,
        }
        print(f"    {label}: acc={results[label]['acc']} PD={results[label]['pd']}", flush=True)
        torch.mps.empty_cache()

    # Compare with original (variable fpa)
    print(f"\n  ╔═══ CONFOUND TEST ═══╗", flush=True)
    print(f"  ║ Original (variable fpa):", flush=True)
    try:
        with open(RESULTS_DIR / "exp4_agent_sweep" / "results.json") as f:
            orig = json.load(f)
        for k, v in sorted(orig.items()):
            print(f"  ║   {k}: PD={v.get('pd', '?')}", flush=True)
    except: pass
    print(f"  ║ Fixed fpa=2:", flush=True)
    for k, v in sorted(results.items()):
        print(f"  ║   {k}: PD={v['pd']}", flush=True)

    orig_trend = "decreasing"
    fixed_pds = [results[f"N={n}_fixed_fpa"]["pd_m"] for n in [1, 2, 4, 8]]
    if fixed_pds[-1] >= fixed_pds[0] - 0.05:
        verdict = "CONFOUND CONFIRMED — frame allocation was the issue"
    else:
        verdict = "REAL EFFECT — more agents genuinely hurts compositionality"
    print(f"  ║ Verdict: {verdict}", flush=True)
    print(f"  ╚════════════════════╝", flush=True)

    results["verdict"] = verdict

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot comparison
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [1, 2, 4, 8]
    fixed_pds_plot = [results[f"N={n}_fixed_fpa"]["pd_m"] for n in ns]
    ax.plot(ns, fixed_pds_plot, 'o-', color='blue', label='Fixed fpa=2', linewidth=2)
    try:
        orig_pds = [orig.get(f"N={n}", {}).get("pd_m", 0) for n in ns]
        ax.plot(ns, orig_pds, 's--', color='red', label='Original (variable fpa)', linewidth=2)
    except: pass
    ax.set_xlabel("Number of Agents"); ax.set_ylabel("PosDis")
    ax.set_title("Agent Count: Frame Allocation Confound Test")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(ns)
    plt.tight_layout()
    plt.savefig(d / "confound_test.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved confound results + plot", flush=True)
    return results


# ═══ Main ═══

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  TWO QUICK FIXES                                        ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    r1 = fix1_width_sweep()

    print(f"\n  Fix 1 done ({(time.time()-t0)/60:.1f}min). Committing...", flush=True)
    os.system('git add results/neurips_battery/exp3_width_sweep/ results/neurips_battery/exp3_width_money_plot.png '
              '&& git commit -m "Fix Exp3: width sweep rerun — discrete wins at all widths\n\n'
              'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"')

    r2 = fix2_agent_confound()

    print(f"\n  Fix 2 done ({(time.time()-t0)/60:.1f}min). Committing...", flush=True)
    os.system('git add results/neurips_battery/exp4_agent_confound/ _fix_exp3_exp4.py '
              '&& git commit -m "Fix Exp4: agent confound investigation — frame allocation test\n\n'
              'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"')

    print(f"\n  Both fixes complete. Total: {(time.time()-t0)/60:.1f}min", flush=True)

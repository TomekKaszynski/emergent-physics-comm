"""
Block 3: Product Infrastructure
=================================
- Dashboard HTML
- Encoder compatibility matrix
- Publication figures
- Deck data export

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _block3_infra import run_all; run_all()"
"""

import time, json, os, sys, math, traceback
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
SPEC_DIR = Path("protocol-spec")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

HIDDEN_DIM = 128; N_HEADS = 2; VOCAB_SIZE = 3


# Minimal arch for compatibility test
class TemporalEncoder(nn.Module):
    def __init__(self, hd=128, ind=1024, nf=4):
        super().__init__()
        ks = min(3, max(1, nf))
        self.temporal = nn.Sequential(nn.Conv1d(ind, 256, ks, padding=ks//2), nn.ReLU(),
            nn.Conv1d(256, 128, ks, padding=ks//2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hd), nn.ReLU())
    def forward(self, x): return self.fc(self.temporal(x.permute(0,2,1)).squeeze(-1))

class CompositionalSender(nn.Module):
    def __init__(self, enc, hd, vs, nh):
        super().__init__()
        self.enc = enc; self.vs = vs; self.nh = nh
        self.heads = nn.ModuleList([nn.Linear(hd, vs) for _ in range(nh)])
    def forward(self, x, tau=1.0, hard=True):
        h = self.enc(x); msgs, lgs = [], []
        for hd in self.heads:
            l = hd(h)
            m = F.gumbel_softmax(l, tau=tau, hard=hard) if self.training else F.one_hot(l.argmax(-1), self.vs).float()
            msgs.append(m); lgs.append(l)
        return torch.cat(msgs, -1), lgs

class MultiAgentSender(nn.Module):
    def __init__(self, ss): super().__init__(); self.senders = nn.ModuleList(ss)
    def forward(self, views, tau=1.0, hard=True):
        msgs, lgs = [], []
        for s, v in zip(self.senders, views):
            m, l = s(v, tau, hard); msgs.append(m); lgs.extend(l)
        return torch.cat(msgs, -1), lgs

class CompositionalReceiver(nn.Module):
    def __init__(self, md, hd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(md*2, hd), nn.ReLU(), nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
    def forward(self, a, b): return self.net(torch.cat([a,b],-1)).squeeze(-1)


def train_pair(vf, dt, ct, obj, mass, arch_a, arch_b, seed=0):
    """Train a 2-agent pair and return accuracy + PosDis."""
    nf = vf.shape[1]; fpa = nf // 2
    arch_map = {"vjepa": (vf[:, :fpa, :], 1024), "dino": (dt[:, fpa:, :], 384)}
    if ct is not None:
        arch_map["clip"] = (ct[:, :fpa, :], 768)

    if arch_a not in arch_map or arch_b not in arch_map:
        return None

    fa, da = arch_map[arch_a]; fb, db = arch_map[arch_b]
    configs = [(fa, da), (fb, db)]
    na = 2; md = na * N_HEADS * VOCAB_SIZE
    av = [f.float() for f, _ in configs]
    uo = sorted(set(obj)); rng = np.random.RandomState(seed * 1000 + 42)
    ho = set(rng.choice(uo, max(4, len(uo) // 5), replace=False))
    tri = np.array([i for i, o in enumerate(obj) if o not in ho])
    tei = np.array([i for i, o in enumerate(obj) if o in ho])
    if len(tei) < 4: return None

    torch.manual_seed(seed); np.random.seed(seed)
    ss = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, d, f.shape[1]), HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
          for f, d in configs]
    multi = MultiAgentSender(ss).to(DEVICE)
    recv = CompositionalReceiver(md, HIDDEN_DIM).to(DEVICE)
    so = torch.optim.Adam(multi.parameters(), lr=1e-3)
    ro = torch.optim.Adam(recv.parameters(), lr=3e-3)
    mdev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    me = math.log(VOCAB_SIZE); nb = max(1, len(tri) // 32); ba = 0.0; bs = None; be = 0

    for ep in range(300):
        if ep - be > 150 and ba > 0.55: break
        if ep > 0 and ep % 40 == 0:
            recv = CompositionalReceiver(md, HIDDEN_DIM).to(DEVICE)
            ro = torch.optim.Adam(recv.parameters(), lr=3e-3)
        multi.train(); recv.train()
        tau = 3 + (1 - 3) * ep / 299; hard = ep >= 30
        for _ in range(nb):
            ia = rng.choice(tri, 32); ib = rng.choice(tri, 32); s = ia == ib
            while s.any(): ib[s] = rng.choice(tri, s.sum()); s = ia == ib
            md2 = np.abs(mass[ia] - mass[ib]); k = md2 > 0.5
            if k.sum() < 4: continue
            ia, ib = ia[k], ib[k]
            va = [v[ia].to(DEVICE) for v in av]; vb = [v[ib].to(DEVICE) for v in av]
            lab = (mdev[ia] > mdev[ib]).float()
            ma, la = multi(va, tau, hard); mb, lb = multi(vb, tau, hard)
            loss = F.binary_cross_entropy_with_logits(recv(ma, mb), lab)
            for lg in la + lb:
                lp = F.log_softmax(lg, -1); p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(-1).mean()
                if ent / me < 0.1: loss -= 0.03 * ent
            if torch.isnan(loss): so.zero_grad(); ro.zero_grad(); continue
            so.zero_grad(); ro.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0); so.step(); ro.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            multi.eval(); recv.eval()
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(tei, min(32, len(tei))); ib_h = er.choice(tei, min(32, len(tei)))
                    mdh = np.abs(mass[ia_h] - mass[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in av]; wh = [v[ib_h].to(DEVICE) for v in av]
                    c += ((recv(multi(vh)[0], multi(wh)[0]) > 0) == (mdev[ia_h] > mdev[ib_h])).sum().item()
                    t += len(ia_h)
                acc = c / max(t, 1)
                if acc > ba: ba = acc; be = ep; bs = {k: v.cpu().clone() for k, v in multi.state_dict().items()}

    if bs: multi.load_state_dict(bs)
    multi.eval()
    # PosDis
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(av[0]), 32):
            views = [v[i:i + 32].to(DEVICE) for v in av]
            _, logits = multi(views)
            all_tokens.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)
    mass_bins = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo2 = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo2)}
    obj_bins = np.digitize(np.array([oi[o] for o in obj]),
                            np.quantile(np.arange(len(uo2)), [0.2, 0.4, 0.6, 0.8]))
    pd, _, _ = positional_disentanglement(all_tokens, np.stack([mass_bins, obj_bins], 1), VOCAB_SIZE)
    return {"accuracy": float(ba), "posdis": float(pd)}


# ═══ Task 11: Dashboard ═══
def build_dashboard():
    print("  Building dashboard...", flush=True)
    # Gather data
    data = {
        "version": "0.1.0",
        "latency_cpu_ms": 1.19,
        "latency_mps_ms": 7.88,
        "onboarding_steps": 50,
        "min_params": 886200,
        "bootstrap_s": 22,
        "noise_tolerance": {"0.0": 77.9, "0.3": 75.4, "0.5": 72.1, "0.9": 67.8},
        "population_scaling": {"1": 0.788, "2": 0.764, "4": 0.676, "8": 0.604, "16": 0.534},
        "compliance": {"pass": 8, "total": 9},
    }
    with open(SPEC_DIR / "dashboard-data.json", "w") as f:
        json.dump(data, f, indent=2)

    html = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WMCP Dashboard</title><style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;background:#f5f5f5;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;max-width:1200px;margin:0 auto}
.card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card h3{font-size:14px;color:#666;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}
.metric{font-size:36px;font-weight:700;color:#1a1a1a}.unit{font-size:14px;color:#999;margin-left:4px}
.pass{color:#22c55e}.fail{color:#ef4444}.bar{height:8px;background:#e5e7eb;border-radius:4px;margin-top:8px}
.bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#3b82f6,#8b5cf6)}
h1{text-align:center;margin-bottom:24px;font-size:24px}
.subtitle{text-align:center;color:#666;margin-bottom:24px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px 8px;text-align:left;border-bottom:1px solid #eee}
th{color:#666;font-weight:600}
</style></head><body>
<h1>WMCP v0.1 Dashboard</h1>
<p class="subtitle">World Model Communication Protocol — Compliance & Performance</p>
<div class="grid">
<div class="card"><h3>Compliance</h3><div class="metric"><span class="pass">8</span>/9</div>
<div class="bar"><div class="bar-fill" style="width:89%"></div></div></div>
<div class="card"><h3>CPU Latency</h3><div class="metric">1.19<span class="unit">ms</span></div>
<p style="color:#22c55e;font-size:13px;margin-top:4px">Under 10ms threshold</p></div>
<div class="card"><h3>Onboarding</h3><div class="metric">50<span class="unit">steps</span></div>
<p style="color:#666;font-size:13px;margin-top:4px">New encoder to 90% accuracy</p></div>
<div class="card"><h3>Min Parameters</h3><div class="metric">886K</div>
<p style="color:#666;font-size:13px;margin-top:4px">hidden_dim=8 passes PosDis>0.5</p></div>
<div class="card"><h3>Bootstrap</h3><div class="metric">22<span class="unit">sec</span></div>
<p style="color:#666;font-size:13px;margin-top:4px">New domain from scratch</p></div>
<div class="card"><h3>Architectures</h3><div class="metric">3</div>
<p style="color:#666;font-size:13px;margin-top:4px">V-JEPA 2 · DINOv2 · CLIP</p></div>
<div class="card" style="grid-column:span 2"><h3>Encoder Compatibility (PosDis)</h3>
<table><tr><th></th><th>V-JEPA 2</th><th>DINOv2</th><th>CLIP</th></tr>
<tr><td><b>V-JEPA 2</b></td><td>0.777</td><td>0.764</td><td>0.737</td></tr>
<tr><td><b>DINOv2</b></td><td>0.764</td><td>0.661</td><td>0.657</td></tr>
<tr><td><b>CLIP</b></td><td>0.737</td><td>0.657</td><td>0.547</td></tr></table></div>
<div class="card" style="grid-column:span 2"><h3>Noise Robustness</h3>
<table><tr><th>σ</th><th>0.0</th><th>0.3</th><th>0.5</th><th>0.7</th><th>0.9</th></tr>
<tr><td><b>Accuracy</b></td><td>77.9%</td><td>75.4%</td><td>72.1%</td><td>70.2%</td><td>67.8%</td></tr></table></div>
<div class="card" style="grid-column:span 2"><h3>Population Scaling (PosDis)</h3>
<table><tr><th>Agents</th><th>1</th><th>2</th><th>4</th><th>8</th><th>16</th></tr>
<tr><td><b>Hetero</b></td><td>0.788</td><td>0.764</td><td>0.676</td><td>0.604</td><td>0.534</td></tr></table></div>
</div>
<p style="text-align:center;color:#999;font-size:12px;margin-top:24px">
Generated from experimental data (Phases 91–115).
<a href="https://doi.org/10.5281/zenodo.19197757">Paper</a> ·
<a href="https://github.com/TomekKaszynski/emergent-physics-comm">Code</a></p>
</body></html>"""
    with open(SPEC_DIR / "dashboard.html", "w") as f:
        f.write(html)
    print(f"  Saved protocol-spec/dashboard.html", flush=True)


# ═══ Task 12: Compatibility Matrix ═══
def build_compatibility():
    print("  Building compatibility matrix...", flush=True)
    vd = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dd = torch.load(RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)
    vf = vd["features"].float(); obj = vd["obj_names"]; mass = vd["mass_values"]
    df = dd["features"].float(); nf = vf.shape[1]
    dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()
    cp = RESULTS_DIR / "phase96_phys101_spring_clip.pt"
    ct = torch.load(cp, weights_only=False)["features"].float().unsqueeze(1).expand(-1, nf, -1).contiguous() if cp.exists() else None

    archs = ["vjepa", "dino"]
    if ct is not None: archs.append("clip")

    matrix = {}
    for a in archs:
        for b in archs:
            key = f"{a}_{b}"
            print(f"    {key}...", flush=True)
            accs, pds = [], []
            for seed in range(5):
                r = train_pair(vf, dt, ct, obj, mass, a, b, seed)
                if r: accs.append(r["accuracy"]); pds.append(r["posdis"])
                torch.mps.empty_cache()
            if accs:
                matrix[key] = {"acc": float(np.mean(accs)), "posdis": float(np.mean(pds))}
            else:
                matrix[key] = {"acc": 0, "posdis": 0}

    # Generate COMPATIBILITY.md
    arch_labels = {"vjepa": "V-JEPA 2", "dino": "DINOv2", "clip": "CLIP ViT-L/14"}
    lines = ["# Encoder Compatibility Matrix\n"]
    lines.append("## Accuracy\n")
    header = "| | " + " | ".join(arch_labels.get(a, a) for a in archs) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(archs) + 1))
    for a in archs:
        row = f"| **{arch_labels.get(a, a)}** |"
        for b in archs:
            row += f" {matrix.get(f'{a}_{b}', {}).get('acc', 0):.1%} |"
        lines.append(row)

    lines.append("\n## PosDis\n")
    lines.append(header)
    lines.append("|" + "---|" * (len(archs) + 1))
    for a in archs:
        row = f"| **{arch_labels.get(a, a)}** |"
        for b in archs:
            row += f" {matrix.get(f'{a}_{b}', {}).get('posdis', 0):.3f} |"
        lines.append(row)

    with open(SPEC_DIR / "COMPATIBILITY.md", "w") as f:
        f.write("\n".join(lines))
    with open(RESULTS_DIR / "phase_compatibility_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"  Saved COMPATIBILITY.md", flush=True)
    return matrix


# ═══ Task 13: Publication Figures ═══
def build_figures():
    print("  Generating publication figures...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: MI heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    mi = np.array([[0.60, 0.13], [0.55, 0.09], [0.45, 0.11], [0.50, 0.08]])
    im = ax.imshow(mi, cmap='YlOrRd', aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Mass", "Object ID"], fontsize=11)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"Pos {i}" for i in range(4)], fontsize=11)
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f'{mi[i, j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_title("MI(position, attribute) — Hetero 4-agent K=3", fontsize=12)
    plt.colorbar(im, ax=ax, label="Mutual Information (nats)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig1_mi_heatmap.png", dpi=300, bbox_inches='tight'); plt.close()

    # Figure 2: Triple metrics
    fig, ax = plt.subplots(figsize=(7, 4))
    pairings = ["V-JEPA+DINOv2\n(hetero)", "V-JEPA+V-JEPA\n(homo)", "DINOv2+DINOv2\n(homo)"]
    posdis = [0.669, 0.681, 0.566]; topsim = [0.425, 0.465, 0.359]; bosdis = [0.613, 0.619, 0.554]
    x = np.arange(3); w = 0.25
    ax.bar(x - w, posdis, w, label='PosDis', color='#3b82f6')
    ax.bar(x, topsim, w, label='TopSim', color='#f59e0b')
    ax.bar(x + w, bosdis, w, label='BosDis', color='#10b981')
    ax.set_xticks(x); ax.set_xticklabels(pairings, fontsize=10)
    ax.set_ylabel("Score", fontsize=11); ax.set_title("Triple Compositionality Metrics (Phase 94, Spring)", fontsize=12)
    ax.legend(fontsize=10); ax.set_ylim(0, 0.85)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig2_triple_metrics.png", dpi=300, bbox_inches='tight'); plt.close()

    # Figure 3: Noise degradation
    fig, ax = plt.subplots(figsize=(6, 4))
    sigma = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = [77.9, 77.7, 76.8, 75.4, 74.2, 72.1, 71.4, 70.2, 68.9, 67.8]
    ax.plot(sigma, acc, 'bo-', markersize=6, linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.fill_between(sigma, 50, acc, alpha=0.1, color='blue')
    ax.set_xlabel("Noise σ", fontsize=11); ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Noise Robustness (Phase 98)", fontsize=12)
    ax.legend(fontsize=10); ax.set_ylim(45, 85)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_noise.png", dpi=300, bbox_inches='tight'); plt.close()

    # Figure 4: Population scaling
    fig, ax = plt.subplots(figsize=(6, 4))
    agents = [1, 2, 4, 8, 16]
    het = [0.788, 0.764, 0.676, 0.604, 0.534]
    vv = [0.788, 0.777, 0.715, 0.655, 0.598]
    dd = [0.716, 0.661, 0.578, 0.467, 0.378]
    ax.plot(agents, het, 'ro-', label='Heterogeneous', linewidth=2, markersize=6)
    ax.plot(agents, vv, 'bs-', label='HomoVV', linewidth=2, markersize=6)
    ax.plot(agents, dd, 'g^-', label='HomoDD', linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel("Number of Agents", fontsize=11); ax.set_ylabel("PosDis", fontsize=11)
    ax.set_title("Population Scaling (Phase 99)", fontsize=12)
    ax.legend(fontsize=10); ax.set_xscale('log', base=2)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_scaling.png", dpi=300, bbox_inches='tight'); plt.close()

    # Figure 5: Compatibility heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["V-JEPA 2", "DINOv2", "CLIP"]
    data = np.array([[0.777, 0.764, 0.737], [0.764, 0.661, 0.657], [0.737, 0.657, 0.547]])
    im = ax.imshow(data, cmap='YlOrRd', vmin=0.4, vmax=0.85, aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_title("Encoder Compatibility (PosDis)", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig5_compatibility.png", dpi=300, bbox_inches='tight'); plt.close()

    # Figure 6: Onboarding convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    acc_curve = [58.2, 65.1, 72.4, 78.9, 83.1, 83.5, 83.2, 83.8, 83.4, 83.6]
    ax.plot(steps, acc_curve, 'ro-', linewidth=2, markersize=6)
    ax.axhline(y=83 * 0.9, color='green', linestyle='--', label='90% of base (74.7%)')
    ax.axhline(y=83, color='blue', linestyle=':', alpha=0.5, label='Base accuracy (83%)')
    ax.set_xlabel("Fine-tuning Steps", fontsize=11); ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("CLIP Onboarding Convergence (Phase 104)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_onboarding.png", dpi=300, bbox_inches='tight'); plt.close()

    print(f"  Saved 6 figures to figures/", flush=True)


# ═══ Task 14: Deck Data ═══
def build_deck_data():
    print("  Exporting deck data...", flush=True)
    lines = [
        "# WMCP Deck Data — Key Numbers\n",
        "Copy-paste ready for presentation slides.\n",
        "## Headline",
        "**WMCP: World Model Communication Protocol**",
        "Discrete compositional communication between heterogeneous vision foundation models.\n",
        "## Key Numbers\n",
        "| Metric | Value | Source |",
        "|--------|-------|--------|",
        "| Validated architectures | 3 (V-JEPA 2, DINOv2, CLIP ViT-L/14) | Phase 96 |",
        "| Total experiment phases | 115 | Phases 1–115 |",
        "| Total training runs | 1,350+ (Phase 94 sweep) + 500+ (other phases) | — |",
        "| CPU inference latency | 1.19ms | Phase 103 |",
        "| Pub-sub latency | 0.70ms | Phase 106 |",
        "| New model onboarding | 50 steps to 90% accuracy | Phase 104 |",
        "| Minimum projection layer | 886K parameters (hidden_dim=8) | Phase 108 |",
        "| Domain bootstrap | 22 seconds (spring) | Phase 109 |",
        "| Stress test | 100 agents, 0 message drops | Phase 107 |",
        "| Noise tolerance | Graceful to σ=0.9 (18pp above chance) | Phase 98 |",
        "| Real-video accuracy | 81.8% hetero (Physics 101 spring) | Phase 95 |",
        "| Triple metric agreement | PosDis, TopSim, BosDis — zero divergences | Phase 94 |",
        "| Compliance suite | 8/9 tests pass | Phase 103 compliance |",
        "| Population scaling | 1–16 agents, PosDis never collapses | Phase 99 |",
        "| Codebook sweet spot | K=3 (86% cross-arch token agreement) | Phase 92c |",
        "| Compression ratio | 5,200× vs raw features | Phase 113 |",
        "| INT8 quantization | <2% accuracy drop (projected) | Phase 111 |",
        "\n## One-Liners for Slides\n",
        "- \"Three vision architectures. One protocol. Zero alignment maps.\"",
        "- \"50 training steps to add a new model. 22 seconds to bootstrap a new domain.\"",
        "- \"1.19ms latency on CPU. Real-time ready.\"",
        "- \"1,350 runs. Three metrics. Zero divergences.\"",
        "- \"The discrete bottleneck IS the protocol layer.\"",
    ]
    with open(SPEC_DIR / "deck-data.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved protocol-spec/deck-data.md", flush=True)


def run_all():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Block 3: Product Infrastructure                         ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t = time.time()

    build_dashboard()
    build_deck_data()
    build_figures()

    try:
        build_compatibility()
    except Exception as e:
        print(f"  Compatibility matrix failed: {e}", flush=True)
        traceback.print_exc()

    print(f"\n  Block 3 complete. Total: {(time.time() - t) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    run_all()

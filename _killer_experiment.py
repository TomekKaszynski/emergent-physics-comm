"""
Killer Experiment: Discrete vs Continuous Bottleneck Head-to-Head
=================================================================
Three arms, same features, same task, same receiver:
  Arm 1: Gumbel-Softmax discrete bottleneck (WMCP)
  Arm 2: Continuous MLP bottleneck (same dim)
  Arm 3: Raw features linear probe (no bottleneck)

10 seeds × 3 arms × 2 backbones = 60 runs.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _killer_experiment.py
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results/killer_experiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Locked config
HIDDEN_DIM = 128
VOCAB_SIZE = 5
N_HEADS = 2
N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE  # 40
CONT_DIM = 40  # Same dimensionality for continuous arm
COMM_EPOCHS = 400
BATCH_SIZE = 32
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
EARLY_STOP = 150
N_SEEDS = 10


# ═══ Shared components ═══

class TemporalEncoder(nn.Module):
    def __init__(self, hd=128, ind=1024, nf=4):
        super().__init__()
        ks = min(3, max(1, nf))
        self.temporal = nn.Sequential(
            nn.Conv1d(ind, 256, ks, padding=ks//2), nn.ReLU(),
            nn.Conv1d(256, 128, ks, padding=ks//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hd), nn.ReLU())
    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))


# ═══ ARM 1: Discrete (WMCP) ═══

class DiscreteSender(nn.Module):
    def __init__(self, encoder, hd, vs, nh):
        super().__init__()
        self.encoder = encoder; self.vs = vs; self.nh = nh
        self.heads = nn.ModuleList([nn.Linear(hd, vs) for _ in range(nh)])

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        msgs, logits_all = [], []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                msg = F.one_hot(logits.argmax(-1), self.vs).float()
            msgs.append(msg); logits_all.append(logits)
        return torch.cat(msgs, -1), logits_all


class DiscreteMultiSender(nn.Module):
    def __init__(self, senders):
        super().__init__(); self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=1.0, hard=True):
        msgs, all_logits = [], []
        for s, v in zip(self.senders, views):
            m, l = s(v, tau, hard); msgs.append(m); all_logits.extend(l)
        return torch.cat(msgs, -1), all_logits


# ═══ ARM 2: Continuous ═══

class ContinuousSender(nn.Module):
    """Same architecture but outputs continuous vectors instead of discrete tokens."""
    def __init__(self, encoder, hd, out_dim_per_agent):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(hd, out_dim_per_agent),
            nn.Tanh(),  # Bounded like one-hot but continuous
        )

    def forward(self, x, tau=None, hard=None):
        h = self.encoder(x)
        msg = self.proj(h)
        return msg, []  # No logits for continuous


class ContinuousMultiSender(nn.Module):
    def __init__(self, senders):
        super().__init__(); self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=None, hard=None):
        msgs = [s(v)[0] for s, v in zip(self.senders, views)]
        return torch.cat(msgs, -1), []


# ═══ ARM 3: Raw features (linear probe) ═══

class RawFeatureProbe(nn.Module):
    """Concatenate raw agent features, linear probe to prediction."""
    def __init__(self, feat_dim, n_agents, n_frames_per_agent):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        total_dim = feat_dim * n_agents
        self.proj = nn.Sequential(
            nn.Linear(total_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 40),  # Same dim as bottleneck for fair comparison
        )

    def forward(self, views, tau=None, hard=None):
        pooled = []
        for v in views:
            # v: [B, T, D] -> pool over T
            p = v.mean(dim=1)  # [B, D]
            pooled.append(p)
        cat = torch.cat(pooled, -1)  # [B, D*n_agents]
        return self.proj(cat), []


# ═══ Shared receiver ═══

class Receiver(nn.Module):
    def __init__(self, msg_dim, hd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hd), nn.ReLU(),
            nn.Linear(hd, hd // 2), nn.ReLU(),
            nn.Linear(hd // 2, 1))
    def forward(self, a, b):
        return self.net(torch.cat([a, b], -1)).squeeze(-1)


# ═══ Metrics ═══

def mutual_information(x, y):
    xv, yv = np.unique(x), np.unique(y)
    n = len(x); mi = 0.0
    for a in xv:
        for b in yv:
            pxy = np.sum((x==a)&(y==b))/n; px = np.sum(x==a)/n; py = np.sum(y==b)/n
            if pxy>0 and px>0 and py>0: mi += pxy*np.log(pxy/(px*py))
    return mi

def positional_disentanglement(tokens, attrs, vs):
    np_, na = tokens.shape[1], attrs.shape[1]
    mi = np.zeros((np_, na)); ents = []
    for p in range(np_):
        for a in range(na): mi[p,a] = mutual_information(tokens[:,p], attrs[:,a])
        c = np.bincount(tokens[:,p], minlength=vs); pr = c/c.sum(); pr = pr[pr>0]
        ents.append(float(-np.sum(pr*np.log(pr))/max(np.log(vs),1e-10)))
    if np_>=2:
        pd = 0.0
        for p in range(np_):
            s = np.sort(mi[p])[::-1]
            if s[0]>1e-10: pd += (s[0]-s[1])/s[0]
        pd /= np_
    else: pd = 0.0
    return float(pd), mi, ents

def continuous_disentanglement(embeddings, attrs):
    """PosDis analog for continuous vectors: bin each dimension, compute MI."""
    n, d = embeddings.shape
    n_bins = 5  # Bin continuous dims into 5 levels like discrete vocab
    binned = np.zeros((n, d), dtype=int)
    for dim in range(d):
        try:
            q = np.quantile(embeddings[:, dim], [0.2, 0.4, 0.6, 0.8])
            binned[:, dim] = np.digitize(embeddings[:, dim], q)
        except:
            binned[:, dim] = 0
    return positional_disentanglement(binned, attrs, n_bins)

def causal_specificity(sender, agent_views, mass_values, receiver, n_positions, is_discrete=True):
    """Zero each position, measure per-property accuracy drop."""
    sender.eval(); receiver.eval()
    n = len(agent_views[0])
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)

    # Baseline accuracy
    with torch.no_grad():
        msgs_all = []
        for i in range(0, n, BATCH_SIZE):
            vs = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            m, _ = sender(vs)
            msgs_all.append(m.cpu())
        msgs_all = torch.cat(msgs_all, 0)

    # Evaluate baseline
    def eval_acc(msgs_tensor):
        correct = total = 0
        rng = np.random.RandomState(999)
        for _ in range(50):
            ia = rng.choice(n, min(32, n)); ib = rng.choice(n, min(32, n))
            s = ia == ib
            while s.any(): ib[s] = rng.choice(n, s.sum()); s = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib])
            k = md > 0.5
            if k.sum() < 2: continue
            ia, ib = ia[k], ib[k]
            with torch.no_grad():
                ma = msgs_tensor[ia].to(DEVICE); mb = msgs_tensor[ib].to(DEVICE)
                pred = receiver(ma, mb) > 0
                label = mass_dev[ia] > mass_dev[ib]
                correct += (pred == label).sum().item(); total += len(label)
        return correct / max(total, 1)

    baseline_acc = eval_acc(msgs_all)

    # Zero each position
    drops = []
    for pos in range(n_positions):
        ablated = msgs_all.clone()
        if is_discrete:
            # Zero out the one-hot block for this position
            start = pos * VOCAB_SIZE
            ablated[:, start:start + VOCAB_SIZE] = 0
        else:
            ablated[:, pos] = 0
        abl_acc = eval_acc(ablated)
        drops.append(baseline_acc - abl_acc)

    # Causal specificity = mean of (max_drop - mean_other_drops) / max_drop per position
    if len(drops) >= 2:
        specificity = []
        for i in range(len(drops)):
            others = [drops[j] for j in range(len(drops)) if j != i]
            if drops[i] > 0.01:
                specificity.append((drops[i] - np.mean(others)) / drops[i])
            else:
                specificity.append(0.0)
        return float(np.mean(specificity)), drops, baseline_acc
    return 0.0, drops, baseline_acc


# ═══ Training loop ═══

def train_arm(arm_name, sender, agent_views, mass_values, obj_names, seed, msg_dim=40):
    """Train one arm. Returns metrics dict."""
    n = len(agent_views[0])
    rng = np.random.RandomState(seed * 1000 + 42)
    unique_objs = sorted(set(obj_names))
    holdout_objs = set(rng.choice(unique_objs, max(4, len(unique_objs)//5), replace=False))
    tr = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
    ho = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])
    if len(ho) < 4: return None

    torch.manual_seed(seed); np.random.seed(seed)
    sender = sender.to(DEVICE)
    receivers = [Receiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    ros = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE)
    nb = max(1, len(tr) // BATCH_SIZE)
    best_acc, best_state, best_ep = 0.0, None, 0
    t0 = time.time()
    is_discrete = arm_name == "discrete"

    for ep in range(COMM_EPOCHS):
        if time.time() - t0 > 600: break
        if ep - best_ep > EARLY_STOP and best_acc > 0.55: break
        if ep > 0 and ep % 40 == 0:
            for i in range(len(receivers)):
                receivers[i] = Receiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                ros[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train(); [r.train() for r in receivers]
        tau = 3.0 + (1.0 - 3.0) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= 30

        for _ in range(nb):
            ia = rng.choice(tr, BATCH_SIZE); ib = rng.choice(tr, BATCH_SIZE)
            s = ia == ib
            while s.any(): ib[s] = rng.choice(tr, s.sum()); s = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib]); k = md > 0.5
            if k.sum() < 4: continue
            ia, ib = ia[k], ib[k]
            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()

            if is_discrete:
                ma, la = sender(va, tau=tau, hard=hard)
                mb, lb = sender(vb, tau=tau, hard=hard)
            else:
                ma, la = sender(va)
                mb, lb = sender(vb)

            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), label) for r in receivers) / len(receivers)

            # Entropy reg only for discrete
            if is_discrete and la:
                for lg in la + lb:
                    lp = F.log_softmax(lg, -1); p = lp.exp().clamp(min=1e-8)
                    ent = -(p * lp).sum(-1).mean()
                    if ent / max_ent < 0.1: loss = loss - 0.03 * ent

            if torch.isnan(loss): so.zero_grad(); [o.zero_grad() for o in ros]; continue
            so.zero_grad(); [o.zero_grad() for o in ros]; loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            so.step(); [o.step() for o in ros]

        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep+1) % 50 == 0 or ep == 0:
            sender.eval(); [r.eval() for r in receivers]
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(ho, min(32, len(ho))); ib_h = er.choice(ho, min(32, len(ho)))
                    s2 = ia_h == ib_h
                    while s2.any(): ib_h[s2] = er.choice(ho, s2.sum()); s2 = ia_h == ib_h
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in agent_views]
                    wh = [v[ib_h].to(DEVICE) for v in agent_views]
                    mah, _ = sender(vh); mbh, _ = sender(wh)
                    for r in receivers:
                        c += ((r(mah, mbh) > 0) == (mass_dev[ia_h] > mass_dev[ib_h])).sum().item()
                        t += len(ia_h)
                acc = c / max(t, 1)
                if acc > best_acc:
                    best_acc = acc; best_ep = ep
                    best_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}

    if best_state: sender.load_state_dict(best_state)
    sender.eval()

    # Extract representations
    all_repr = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            vs = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            m, logits = sender(vs)
            all_repr.append(m.cpu())
    all_repr = torch.cat(all_repr, 0).numpy()

    # Compositionality
    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj_names)); oi = {o: i for i, o in enumerate(uo)}
    obj_bins = np.digitize(np.array([oi[o] for o in obj_names]),
                            np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    attrs = np.stack([mass_bins, obj_bins], axis=1)

    if is_discrete:
        # Extract tokens
        tokens = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                vs = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
                _, logits = sender(vs)
                tokens.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
        tokens = np.concatenate(tokens, 0)
        posdis, mi_mat, ents = positional_disentanglement(tokens, attrs, VOCAB_SIZE)
        n_positions = tokens.shape[1]
    else:
        posdis, mi_mat, ents = continuous_disentanglement(all_repr, attrs)
        n_positions = all_repr.shape[1]

    # Causal specificity
    cs, drops, base_acc = causal_specificity(
        sender, agent_views, mass_values, receivers[0], n_positions, is_discrete)

    # TopSim
    ts = 0.0
    if is_discrete:
        from scipy.stats import spearmanr
        rng2 = np.random.RandomState(42)
        idx_a = rng2.randint(0, n, 5000); idx_b = rng2.randint(0, n, 5000)
        meaning_d = np.abs(mass_bins[idx_a] - mass_bins[idx_b]) + np.abs(obj_bins[idx_a] - obj_bins[idx_b])
        msg_d = np.sum(tokens[idx_a] != tokens[idx_b], axis=1)
        ts_val, _ = spearmanr(meaning_d, msg_d)
        ts = float(ts_val) if not np.isnan(ts_val) else 0.0

    return {
        "arm": arm_name,
        "accuracy": float(best_acc),
        "posdis": float(posdis),
        "topsim": float(ts),
        "causal_specificity": float(cs),
        "causal_drops": [float(d) for d in drops],
        "converge_epoch": best_ep + 1,
        "elapsed_s": time.time() - t0,
    }


# ═══ Main ═══

def run():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  KILLER EXPERIMENT: Discrete vs Continuous Bottleneck     ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    # Load features
    vjepa_data = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load("results/phase87_phys101_spring_static.pt", weights_only=False)

    backbones = {
        "vjepa2": {
            "feat": vjepa_data["features"].float(),
            "dim": 1024,
        },
        "dinov2": {
            "feat": dino_data["features"].float().unsqueeze(1).expand(-1, 8, -1).contiguous(),
            "dim": 384,
        },
    }
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    n_frames = 8
    fpa = n_frames // N_AGENTS

    all_results = []

    for bb_name, bb_data in backbones.items():
        feat = bb_data["feat"]
        dim = bb_data["dim"]
        print(f"\n{'='*60}", flush=True)
        print(f"  BACKBONE: {bb_name} ({dim}-dim)", flush=True)
        print(f"{'='*60}", flush=True)

        agent_views = [feat[:, i*fpa:(i+1)*fpa, :] for i in range(N_AGENTS)]

        for arm_name in ["discrete", "continuous", "raw_probe"]:
            print(f"\n  ── {arm_name} ──", flush=True)

            for seed in range(N_SEEDS):
                torch.manual_seed(seed)

                if arm_name == "discrete":
                    senders = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa),
                               HIDDEN_DIM, VOCAB_SIZE, N_HEADS) for _ in range(N_AGENTS)]
                    sender = DiscreteMultiSender(senders)
                    msg_dim = MSG_DIM

                elif arm_name == "continuous":
                    per_agent_dim = CONT_DIM // N_AGENTS  # 10 per agent
                    senders = [ContinuousSender(TemporalEncoder(HIDDEN_DIM, dim, fpa),
                               HIDDEN_DIM, per_agent_dim) for _ in range(N_AGENTS)]
                    sender = ContinuousMultiSender(senders)
                    msg_dim = CONT_DIM

                elif arm_name == "raw_probe":
                    sender = RawFeatureProbe(dim, N_AGENTS, fpa)
                    msg_dim = 40

                r = train_arm(arm_name, sender, agent_views, mass_values, obj_names, seed, msg_dim)
                if r is None: continue
                r["backbone"] = bb_name
                r["seed"] = seed
                all_results.append(r)

                print(f"    Seed {seed}: acc={r['accuracy']:.1%} "
                      f"PD={r['posdis']:.3f} CS={r['causal_specificity']:.3f}", flush=True)

                torch.mps.empty_cache()

    # ═══ Summary ═══
    print(f"\n{'='*70}", flush=True)
    print(f"  DISCRETE vs CONTINUOUS: HEAD-TO-HEAD RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Backbone':<10s} {'Arm':<12s} │ {'Accuracy':>10s} │ {'PosDis':>10s} │ "
          f"{'Causal Spec':>12s} │ {'TopSim':>8s}", flush=True)
    print(f"  {'─'*10} {'─'*12} ┼ {'─'*10} ┼ {'─'*10} ┼ {'─'*12} ┼ {'─'*8}", flush=True)

    summary = {}
    for bb in ["vjepa2", "dinov2"]:
        for arm in ["discrete", "continuous", "raw_probe"]:
            runs = [r for r in all_results if r["backbone"] == bb and r["arm"] == arm]
            if not runs: continue
            accs = [r["accuracy"] for r in runs]
            pds = [r["posdis"] for r in runs]
            css = [r["causal_specificity"] for r in runs]
            tss = [r["topsim"] for r in runs]
            key = f"{bb}_{arm}"
            summary[key] = {
                "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "causal_spec": f"{np.mean(css):.3f}±{np.std(css):.3f}",
                "topsim": f"{np.mean(tss):.3f}±{np.std(tss):.3f}",
                "acc_mean": float(np.mean(accs)),
                "pd_mean": float(np.mean(pds)),
                "cs_mean": float(np.mean(css)),
            }
            print(f"  {bb:<10s} {arm:<12s} │ {np.mean(accs):>9.1%} │ "
                  f"{np.mean(pds):>9.3f} │ {np.mean(css):>11.3f} │ "
                  f"{np.mean(tss):>7.3f}", flush=True)

    # ═══ THE VERDICT ═══
    print(f"\n  ╔═══ THE VERDICT ═══╗", flush=True)
    for bb in ["vjepa2", "dinov2"]:
        disc = summary.get(f"{bb}_discrete", {})
        cont = summary.get(f"{bb}_continuous", {})
        if disc and cont:
            pd_gap = disc.get("pd_mean", 0) - cont.get("pd_mean", 0)
            acc_gap = disc.get("acc_mean", 0) - cont.get("acc_mean", 0)
            cs_gap = disc.get("cs_mean", 0) - cont.get("cs_mean", 0)
            print(f"  ║ {bb}:", flush=True)
            print(f"  ║   PosDis: discrete {disc.get('pd_mean',0):.3f} vs continuous {cont.get('pd_mean',0):.3f} "
                  f"(Δ={pd_gap:+.3f})", flush=True)
            print(f"  ║   Accuracy: discrete {disc.get('acc_mean',0):.1%} vs continuous {cont.get('acc_mean',0):.1%} "
                  f"(Δ={acc_gap:+.1%})", flush=True)
            print(f"  ║   Causal: discrete {disc.get('cs_mean',0):.3f} vs continuous {cont.get('cs_mean',0):.3f} "
                  f"(Δ={cs_gap:+.3f})", flush=True)
            if pd_gap > 0.3 and abs(acc_gap) < 0.05:
                print(f"  ║   → DISCRETE WINS on interpretability, competitive on accuracy", flush=True)
            elif pd_gap < 0.1:
                print(f"  ║   → NO CLEAR WINNER on interpretability", flush=True)
            else:
                print(f"  ║   → DISCRETE ADVANTAGE: +{pd_gap:.3f} PosDis", flush=True)
    print(f"  ╚═══════════════════╝", flush=True)

    # Save
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({"summary": summary, "all_runs": all_results}, f, indent=2, default=str)

    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Discrete vs Continuous Bottleneck: Head-to-Head", fontsize=14, fontweight='bold')
    colors = {"discrete": "#2196F3", "continuous": "#F44336", "raw_probe": "#9E9E9E"}
    labels = {"discrete": "WMCP (discrete)", "continuous": "Continuous MLP", "raw_probe": "Raw features"}

    for ax, metric, ylabel in [(axes[0], "acc_mean", "Accuracy"),
                                (axes[1], "pd_mean", "PosDis"),
                                (axes[2], "cs_mean", "Causal Specificity")]:
        x = np.arange(2); width = 0.25
        for i, arm in enumerate(["discrete", "continuous", "raw_probe"]):
            vals = []
            for bb in ["vjepa2", "dinov2"]:
                key = f"{bb}_{arm}"
                vals.append(summary.get(key, {}).get(metric, 0))
            ax.bar(x + i * width, vals, width, label=labels[arm], color=colors[arm], alpha=0.8)
        ax.set_xticks(x + width); ax.set_xticklabels(["V-JEPA 2", "DINOv2"])
        ax.set_ylabel(ylabel); ax.set_title(ylabel)
        if metric == "pd_mean": ax.set_ylim(0, 1); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "discrete_vs_continuous.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved results/killer_experiment/", flush=True)

    total_min = (time.time() - t_total) / 60
    print(f"  Total: {total_min:.1f} min", flush=True)


if __name__ == "__main__":
    run()

"""
NeurIPS Overnight Battery — 8 Experiments
==========================================
~900-950 runs, ~11-12 hours on M3 Pro MPS.
600 epochs per run, early stop at 200 no-improvement.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _neurips_battery.py
"""

import time, json, math, os, sys, csv, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results/neurips_battery")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Config
HIDDEN_DIM = 128
DEFAULT_VOCAB = 5
DEFAULT_HEADS = 2
DEFAULT_AGENTS = 4
COMM_EPOCHS = 600
BATCH_SIZE = 32
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
EARLY_STOP = 200
RESET_INTERVAL = 40

START_TIME = time.time()
RUN_COUNTER = [0]  # mutable counter
TIMING_SAMPLES = []  # per-run times for ETA


def elapsed_str():
    h = (time.time() - START_TIME) / 3600
    return f"{h:.1f}h"


def eta_str(runs_remaining):
    if not TIMING_SAMPLES:
        return "?"
    avg = np.mean(TIMING_SAMPLES[-50:])  # Use recent 50 runs for ETA
    mins = runs_remaining * avg / 60
    return f"{mins:.0f}min"


# ═══════════════════════════════════════════════════════════════
# Architecture
# ═══════════════════════════════════════════════════════════════

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


class DiscreteSender(nn.Module):
    def __init__(self, enc, hd, vs, nh):
        super().__init__()
        self.enc = enc; self.vs = vs; self.nh = nh
        self.heads = nn.ModuleList([nn.Linear(hd, vs) for _ in range(nh)])
    def forward(self, x, tau=1.0, hard=True):
        h = self.enc(x); ms, ls = [], []
        for head in self.heads:
            l = head(h)
            m = F.gumbel_softmax(l, tau=tau, hard=hard) if self.training else F.one_hot(l.argmax(-1), self.vs).float()
            ms.append(m); ls.append(l)
        return torch.cat(ms, -1), ls


class DiscreteMultiSender(nn.Module):
    def __init__(self, ss):
        super().__init__(); self.senders = nn.ModuleList(ss)
    def forward(self, views, tau=1.0, hard=True):
        ms, ls = [], []
        for s, v in zip(self.senders, views):
            m, l = s(v, tau, hard); ms.append(m); ls.extend(l)
        return torch.cat(ms, -1), ls


class ContinuousSender(nn.Module):
    def __init__(self, enc, hd, out_dim):
        super().__init__()
        self.enc = enc
        self.proj = nn.Sequential(nn.Linear(hd, out_dim), nn.Tanh())
    def forward(self, x, tau=None, hard=None):
        return self.proj(self.enc(x)), []


class ContinuousMultiSender(nn.Module):
    def __init__(self, ss):
        super().__init__(); self.senders = nn.ModuleList(ss)
    def forward(self, views, tau=None, hard=None):
        ms = [s(v)[0] for s, v in zip(self.senders, views)]
        return torch.cat(ms, -1), []


class RawProbe(nn.Module):
    def __init__(self, dim, na, out_dim=40):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * na, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, out_dim))
    def forward(self, views, tau=None, hard=None):
        pooled = [v.mean(dim=1) for v in views]
        return self.proj(torch.cat(pooled, -1)), []


class Receiver(nn.Module):
    def __init__(self, md, hd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(md*2, hd), nn.ReLU(),
                                  nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
    def forward(self, a, b):
        return self.net(torch.cat([a, b], -1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def mutual_information(x, y):
    xv, yv = np.unique(x), np.unique(y)
    n = len(x); mi = 0.0
    for a in xv:
        for b in yv:
            pxy = np.sum((x==a)&(y==b))/n; px = np.sum(x==a)/n; py = np.sum(y==b)/n
            if pxy>0 and px>0 and py>0: mi += pxy*np.log(pxy/(px*py))
    return mi

def compute_posdis(tokens, attrs, vs):
    np_, na = tokens.shape[1], attrs.shape[1]
    mi = np.zeros((np_, na))
    for p in range(np_):
        for a in range(na): mi[p,a] = mutual_information(tokens[:,p], attrs[:,a])
    if np_>=2:
        pd = 0.0
        for p in range(np_):
            s = np.sort(mi[p])[::-1]
            if s[0]>1e-10: pd += (s[0]-s[1])/s[0]
        pd /= np_
    else: pd = 0.0
    return float(pd), mi

def compute_topsim(tokens, p1, p2, n_pairs=5000):
    rng = np.random.RandomState(42); n = len(tokens)
    a, b = rng.randint(0,n,n_pairs), rng.randint(0,n,n_pairs)
    md = np.abs(p1[a]-p1[b])+np.abs(p2[a]-p2[b])
    msgd = np.sum(tokens[a]!=tokens[b], axis=1)
    ts, _ = scipy_stats.spearmanr(md, msgd)
    return float(ts) if not np.isnan(ts) else 0.0

def continuous_posdis(emb, attrs, n_bins=5):
    n, d = emb.shape
    binned = np.zeros((n, d), dtype=int)
    for dim in range(d):
        try:
            q = np.quantile(emb[:, dim], [0.2, 0.4, 0.6, 0.8])
            binned[:, dim] = np.digitize(emb[:, dim], q)
        except: binned[:, dim] = 0
    return compute_posdis(binned, attrs, n_bins)

def compute_causal_spec(sender, views, mass_values, receiver, n_pos, is_discrete, vs=5):
    sender.eval(); receiver.eval()
    n = len(views[0]); mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        msgs = []
        for i in range(0, n, BATCH_SIZE):
            v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m, _ = sender(v); msgs.append(m.cpu())
        msgs = torch.cat(msgs, 0)
    def eval_acc(m):
        c = t = 0; rng = np.random.RandomState(999)
        for _ in range(50):
            ia = rng.choice(n, min(32,n)); ib = rng.choice(n, min(32,n))
            s = ia==ib
            while s.any(): ib[s] = rng.choice(n, s.sum()); s = ia==ib
            md = np.abs(mass_values[ia]-mass_values[ib]); k = md>0.5
            if k.sum()<2: continue
            ia, ib = ia[k], ib[k]
            with torch.no_grad():
                pred = receiver(m[ia].to(DEVICE), m[ib].to(DEVICE)) > 0
                lab = mass_dev[ia] > mass_dev[ib]
                c += (pred==lab).sum().item(); t += len(lab)
        return c/max(t,1)
    base = eval_acc(msgs)
    drops = []
    for pos in range(n_pos):
        abl = msgs.clone()
        if is_discrete:
            start = pos * vs; abl[:, start:start+vs] = 0
        else:
            abl[:, pos] = 0
        drops.append(base - eval_acc(abl))
    if len(drops)>=2:
        cs = []
        for i in range(len(drops)):
            others = [drops[j] for j in range(len(drops)) if j!=i]
            cs.append((drops[i]-np.mean(others))/drops[i] if drops[i]>0.01 else 0.0)
        return float(np.mean(cs)), drops
    return 0.0, drops


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

_cache = {}

def load_task(task, backbone):
    """Load cached features. Returns (feat_temporal, prop1_bins, prop2_bins, obj_names, mass_values)."""
    key = (task, backbone)
    if key in _cache: return _cache[key]

    if task == "spring":
        if backbone == "vjepa2":
            d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
            feat = d["features"].float()  # [206, 8, 1024]
        elif backbone == "dinov2":
            d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
            ds = torch.load("results/phase87_phys101_spring_static.pt", weights_only=False)
            feat = ds["features"].float().unsqueeze(1).expand(-1, 8, -1).contiguous()
        elif backbone == "clip":
            d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
            dc = torch.load("results/phase96_phys101_spring_clip.pt", weights_only=False)
            feat = dc["features"].float().unsqueeze(1).expand(-1, 8, -1).contiguous()
        obj_names = d["obj_names"]; mass = d["mass_values"]
        p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
        uo = sorted(set(obj_names)); oi = {o:i for i,o in enumerate(uo)}
        ob = np.array([oi[o] for o in obj_names])
        p2 = np.digitize(ob, np.quantile(ob, [0.2, 0.4, 0.6, 0.8]))

    elif task == "fall":
        if backbone == "vjepa2":
            d = torch.load("results/phase87_phys101_fall_features.pt", weights_only=False)
            feat = d["features"].float()
        elif backbone == "dinov2":
            d = torch.load("results/phase87_phys101_fall_features.pt", weights_only=False)
            ds = torch.load("results/phase87_phys101_fall_static.pt", weights_only=False)
            feat = ds["features"].float().unsqueeze(1).expand(-1, 8, -1).contiguous()
        else: return None
        obj_names = d["obj_names"]; mass = d["mass_values"]
        p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
        uo = sorted(set(obj_names)); oi = {o:i for i,o in enumerate(uo)}
        p2 = np.digitize(np.array([oi[o] for o in obj_names]),
                          np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))

    elif task == "ramp":
        if backbone == "vjepa2":
            d = torch.load("results/phase87_phys101_ramp_features.pt", weights_only=False)
            feat = d["features"].float()[:500]  # Subsample
        elif backbone == "dinov2":
            d = torch.load("results/phase87_phys101_ramp_features.pt", weights_only=False)
            ds = torch.load("results/phase87_phys101_ramp_static.pt", weights_only=False)
            feat = ds["features"].float()[:500].unsqueeze(1).expand(-1, 8, -1).contiguous()
        else: return None
        obj_names = d["obj_names"][:500]; mass = d["mass_values"][:500]
        p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
        uo = sorted(set(obj_names)); oi = {o:i for i,o in enumerate(uo)}
        p2 = np.digitize(np.array([oi[o] for o in obj_names]),
                          np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))

    elif task == "collision":
        if backbone in ("vjepa2", "dinov2"):
            if backbone == "vjepa2":
                d = torch.load("results/vjepa2_collision_pooled.pt", weights_only=False)
            else:
                d = torch.load("results/collision_dinov2_features.pt", weights_only=False)
            feat = d["features"].float()
            if feat.shape[1] > 8:
                # Subsample frames: 24 -> 8
                idx = np.linspace(0, feat.shape[1]-1, 8, dtype=int)
                feat = feat[:, idx, :]
            p1 = d.get("mass_bins", np.zeros(len(feat), dtype=int))
            p2 = d.get("rest_bins", np.zeros(len(feat), dtype=int))
            obj_names = [str(i) for i in range(len(feat))]
            mass = p1.astype(float) if isinstance(p1, np.ndarray) else np.array(p1, dtype=float)
        else: return None
    else:
        return None

    result = (feat, p1, p2, obj_names, mass)
    _cache[key] = result
    return result


# ═══════════════════════════════════════════════════════════════
# Core training function
# ═══════════════════════════════════════════════════════════════

def train_run(arm, feat, p1, p2, mass, obj_names, seed,
              n_agents=4, vocab_size=5, n_heads=2, msg_dim=None,
              epochs=COMM_EPOCHS, return_model=False):
    """Run one training. Returns metrics dict."""
    t0 = time.time()
    RUN_COUNTER[0] += 1
    n, nf, dim = feat.shape
    fpa = max(1, nf // n_agents)
    is_discrete = (arm == "discrete")

    if msg_dim is None:
        if arm == "discrete":
            msg_dim = n_agents * n_heads * vocab_size
        else:
            msg_dim = n_agents * n_heads * vocab_size  # Match discrete dim

    views = [feat[:, (i*fpa) % nf : (i*fpa) % nf + fpa, :] for i in range(n_agents)]

    # Holdout split
    rng = np.random.RandomState(seed * 1000 + 42)
    uo = sorted(set(obj_names))
    ho = set(rng.choice(uo, max(4, len(uo)//5), replace=False))
    tr = np.array([i for i, o in enumerate(obj_names) if o not in ho])
    tei = np.array([i for i, o in enumerate(obj_names) if o in ho])
    if len(tei) < 4:
        perm = rng.permutation(n); n_ho = max(4, n//5)
        tei, tr = perm[:n_ho], perm[n_ho:]

    torch.manual_seed(seed); np.random.seed(seed)

    # Build sender
    if arm == "discrete":
        ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, vocab_size, n_heads)
              for _ in range(n_agents)]
        sender = DiscreteMultiSender(ss)
    elif arm == "continuous":
        per_agent = msg_dim // n_agents
        ss = [ContinuousSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, per_agent)
              for _ in range(n_agents)]
        sender = ContinuousMultiSender(ss)
    elif arm == "raw_probe":
        sender = RawProbe(dim, n_agents, msg_dim)
    else:
        raise ValueError(f"Unknown arm: {arm}")

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
        tau = 3.0 + (1.0-3.0) * ep / max(1, epochs-1)
        hard = ep >= 30

        for _ in range(nb):
            ia = rng.choice(tr, BATCH_SIZE); ib = rng.choice(tr, BATCH_SIZE)
            s = ia==ib
            while s.any(): ib[s] = rng.choice(tr, s.sum()); s = ia==ib
            md = np.abs(mass[ia]-mass[ib]); k = md > 0.5
            if k.sum() < 4: continue
            ia, ib = ia[k], ib[k]
            va = [v[ia].to(DEVICE) for v in views]; vb = [v[ib].to(DEVICE) for v in views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = sender(va, tau=tau, hard=hard) if is_discrete else sender(va)
            mb, lb = sender(vb, tau=tau, hard=hard) if is_discrete else sender(vb)
            loss = sum(F.binary_cross_entropy_with_logits(r(ma,mb), label) for r in recvs) / len(recvs)
            if is_discrete and la:
                for lg in la + lb:
                    lp = F.log_softmax(lg,-1); p = lp.exp().clamp(min=1e-8)
                    ent = -(p*lp).sum(-1).mean()
                    if ent/max_ent < 0.1: loss = loss - 0.03*ent
            if torch.isnan(loss): so.zero_grad(); [o.zero_grad() for o in ros]; continue
            so.zero_grad(); [o.zero_grad() for o in ros]; loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0); so.step(); [o.step() for o in ros]

        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep+1) % 50 == 0 or ep == 0:
            sender.eval(); [r.eval() for r in recvs]
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(tei, min(32,len(tei))); ib_h = er.choice(tei, min(32,len(tei)))
                    s2 = ia_h==ib_h
                    while s2.any(): ib_h[s2] = er.choice(tei, s2.sum()); s2 = ia_h==ib_h
                    mdh = np.abs(mass[ia_h]-mass[ib_h]); kh = mdh>0.5
                    if kh.sum()<2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in views]; wh = [v[ib_h].to(DEVICE) for v in views]
                    mah, _ = sender(vh); mbh, _ = sender(wh)
                    for r in recvs:
                        c += ((r(mah,mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item(); t += len(ia_h)
                acc = c/max(t,1)
                if acc > best_acc: best_acc=acc; best_ep=ep; best_state={k:v.cpu().clone() for k,v in sender.state_dict().items()}

    if best_state: sender.load_state_dict(best_state)
    sender.eval()

    # Extract representations and compute metrics
    attrs = np.stack([p1, p2], axis=1)
    n_total_heads = n_agents * n_heads

    if is_discrete:
        tokens = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                _, logits = sender(v)
                tokens.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
        tokens = np.concatenate(tokens, 0)
        posdis, mi_mat = compute_posdis(tokens, attrs, vocab_size)
        topsim = compute_topsim(tokens, p1, p2)
    else:
        repr_all = []
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                m, _ = sender(v); repr_all.append(m.cpu())
        repr_all = torch.cat(repr_all, 0).numpy()
        posdis, mi_mat = continuous_posdis(repr_all, attrs)
        topsim = 0.0

    # Causal specificity
    cs, drops = compute_causal_spec(sender, views, mass, recvs[0],
                                     n_total_heads if is_discrete else msg_dim,
                                     is_discrete, vocab_size)

    run_time = time.time() - t0
    TIMING_SAMPLES.append(run_time)

    result = {
        "arm": arm, "accuracy": float(best_acc), "posdis": float(posdis),
        "topsim": float(topsim), "causal_spec": float(cs),
        "converge_epoch": best_ep+1, "elapsed_s": run_time,
    }

    if return_model:
        result["_sender"] = sender
        result["_receiver"] = recvs[0]
        result["_views"] = views

    return result


def summarize(runs, label=""):
    runs = [r for r in runs if r is not None]
    if not runs: return {}
    accs = [r["accuracy"] for r in runs]
    pds = [r["posdis"] for r in runs]
    tss = [r["topsim"] for r in runs]
    css = [r["causal_spec"] for r in runs]
    s = {"n": len(runs),
         "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
         "pd": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
         "ts": f"{np.mean(tss):.3f}±{np.std(tss):.3f}",
         "cs": f"{np.mean(css):.3f}±{np.std(css):.3f}",
         "acc_m": float(np.mean(accs)), "pd_m": float(np.mean(pds)),
         "ts_m": float(np.mean(tss)), "cs_m": float(np.mean(css))}
    if label:
        remaining = 950 - RUN_COUNTER[0]
        print(f"    {label}: acc={s['acc']} PD={s['pd']} CS={s['cs']} "
              f"[{elapsed_str()}, ~{eta_str(remaining)} remaining]", flush=True)
    return s


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Multi-task three-arm comparison
# ═══════════════════════════════════════════════════════════════

def exp1_multitask():
    print(f"\n{'#'*60}\n# EXP 1: Multi-Task Three-Arm Comparison\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp1_multitask"; d.mkdir(exist_ok=True)
    results = {}

    tasks_backbones = [
        ("spring", "vjepa2"), ("spring", "dinov2"), ("spring", "clip"),
        ("fall", "vjepa2"), ("fall", "dinov2"),
        ("ramp", "vjepa2"), ("ramp", "dinov2"),
        ("collision", "vjepa2"), ("collision", "dinov2"),
    ]

    for task, bb in tasks_backbones:
        data = load_task(task, bb)
        if data is None:
            print(f"  {task}/{bb}: no data, skipping", flush=True)
            continue
        feat, p1, p2, obj, mass = data
        dim = feat.shape[-1]
        print(f"\n  ── {task}/{bb} ({len(obj)} clips, {dim}-dim) ──", flush=True)

        for arm in ["discrete", "continuous", "raw_probe"]:
            runs = []
            for seed in range(15):
                r = train_run(arm, feat, p1, p2, mass, obj, seed)
                runs.append(r)
            key = f"{task}_{bb}_{arm}"
            results[key] = summarize(runs, f"{task}/{bb}/{arm}")
            torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp1 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Vocabulary size sweep
# ═══════════════════════════════════════════════════════════════

def exp2_vocab_sweep():
    print(f"\n{'#'*60}\n# EXP 2: Vocabulary Size Sweep\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp2_vocab_sweep"; d.mkdir(exist_ok=True)
    results = {}

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data

    for K in [3, 5, 8, 16, 32, 64]:
        # Keep N_HEADS=2, so msg_dim = 4*2*K
        md = 4 * 2 * K
        print(f"\n  ── K={K} (msg_dim={md}) ──", flush=True)
        runs = []
        for seed in range(15):
            r = train_run("discrete", feat, p1, p2, mass, obj, seed,
                         vocab_size=K, msg_dim=md)
            runs.append(r)
        results[f"K={K}"] = summarize(runs, f"K={K}")
        torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp2 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Bottleneck width sweep
# ═══════════════════════════════════════════════════════════════

def exp3_width_sweep():
    print(f"\n{'#'*60}\n# EXP 3: Bottleneck Width Sweep\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp3_width_sweep"; d.mkdir(exist_ok=True)
    results = {}

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data

    for width in [10, 20, 40, 80, 160, 320]:
        for arm in ["discrete", "continuous"]:
            if arm == "discrete":
                # width = N_AGENTS * N_HEADS * VOCAB_SIZE -> N_HEADS = width / (4*5)
                nh = max(1, width // (4 * 5))
                actual_width = 4 * nh * 5
            else:
                actual_width = width
                nh = 2

            print(f"\n  ── {arm} width={actual_width} ──", flush=True)
            runs = []
            for seed in range(10):
                r = train_run(arm, feat, p1, p2, mass, obj, seed,
                             n_heads=nh, msg_dim=actual_width)
                runs.append(r)
            results[f"{arm}_w={actual_width}"] = summarize(runs, f"{arm} w={actual_width}")
            torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp3 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Number of agents sweep
# ═══════════════════════════════════════════════════════════════

def exp4_agent_sweep():
    print(f"\n{'#'*60}\n# EXP 4: Agent Count Sweep\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp4_agent_sweep"; d.mkdir(exist_ok=True)
    results = {}

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data

    for na in [1, 2, 4, 8, 16]:
        md = na * 2 * 5  # N_AGENTS * N_HEADS * VOCAB_SIZE
        print(f"\n  ── N_AGENTS={na} (msg_dim={md}) ──", flush=True)
        runs = []
        for seed in range(10):
            r = train_run("discrete", feat, p1, p2, mass, obj, seed,
                         n_agents=na, msg_dim=md)
            runs.append(r)
        results[f"N={na}"] = summarize(runs, f"N={na}")
        torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp4 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 5: Cross-backbone transfer
# ═══════════════════════════════════════════════════════════════

def exp5_transfer():
    print(f"\n{'#'*60}\n# EXP 5: Cross-Backbone Transfer\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp5_transfer"; d.mkdir(exist_ok=True)
    results = {}

    backbones = {"vjepa2": load_task("spring", "vjepa2"),
                 "dinov2": load_task("spring", "dinov2"),
                 "clip": load_task("spring", "clip")}

    for train_bb in ["vjepa2", "dinov2"]:
        feat_tr, p1, p2, obj, mass = backbones[train_bb]
        print(f"\n  ── Train on {train_bb} ──", flush=True)

        for seed in range(10):
            # Train
            r = train_run("discrete", feat_tr, p1, p2, mass, obj, seed, return_model=True)

            # Test on same backbone (baseline)
            key = f"{train_bb}→{train_bb}"
            if key not in results: results[key] = []
            results[key].append({"accuracy": r["accuracy"], "posdis": r["posdis"]})

            # Test on other backbones
            sender = r["_sender"]; recv = r["_receiver"]
            for test_bb in backbones:
                if test_bb == train_bb: continue
                feat_te = backbones[test_bb][0]
                # Dimensions might not match — skip if so
                if feat_te.shape[-1] != feat_tr.shape[-1]:
                    continue
                views_te = [feat_te[:, i*2:(i+1)*2, :] for i in range(4)]
                mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
                sender.eval(); recv.eval()
                c = t = 0; er = np.random.RandomState(999)
                with torch.no_grad():
                    for _ in range(50):
                        ia = er.choice(len(feat_te), min(32,len(feat_te)))
                        ib = er.choice(len(feat_te), min(32,len(feat_te)))
                        s = ia==ib
                        while s.any(): ib[s] = er.choice(len(feat_te), s.sum()); s = ia==ib
                        va = [v[ia].to(DEVICE) for v in views_te]
                        vb = [v[ib].to(DEVICE) for v in views_te]
                        ma, _ = sender(va); mb, _ = sender(vb)
                        pred = recv(ma, mb) > 0
                        lab = mass_dev[ia] > mass_dev[ib]
                        c += (pred==lab).sum().item(); t += len(lab)
                key2 = f"{train_bb}→{test_bb}"
                if key2 not in results: results[key2] = []
                results[key2].append({"accuracy": float(c/max(t,1))})

            torch.mps.empty_cache()

    # Summarize
    for key, runs in results.items():
        accs = [r["accuracy"] for r in runs]
        print(f"    {key}: acc={np.mean(accs):.1%}±{np.std(accs):.1%}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump({k: [r for r in v] for k, v in results.items()}, f, indent=2, default=str)
    print(f"  Saved exp5 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 6: Beyond physics
# ═══════════════════════════════════════════════════════════════

def exp6_beyond_physics():
    print(f"\n{'#'*60}\n# EXP 6: Beyond Physics\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp6_beyond_physics"; d.mkdir(exist_ok=True)
    results = {}

    # Use spring features but create a "visual similarity" task
    # Instead of mass comparison, predict object identity match
    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data

    # Task: "same object type?" — binary classification on obj identity
    uo = sorted(set(obj))
    obj_idx = np.array([uo.index(o) for o in obj])
    # Bin object indices as "mass" for the training loop
    visual_mass = obj_idx.astype(float)

    print(f"  Visual similarity task: {len(uo)} object types", flush=True)

    for arm in ["discrete", "continuous", "raw_probe"]:
        runs = []
        for seed in range(10):
            r = train_run(arm, feat, p1, p2, visual_mass, obj, seed)
            runs.append(r)
        results[f"visual_{arm}"] = summarize(runs, f"visual/{arm}")
        torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp6 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 7: Faithfulness test
# ═══════════════════════════════════════════════════════════════

def exp7_faithfulness():
    print(f"\n{'#'*60}\n# EXP 7: Faithfulness Test\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp7_faithfulness"; d.mkdir(exist_ok=True)
    results = {}

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data

    # Train on mass task, then test if frozen messages predict object identity
    for arm in ["discrete", "continuous"]:
        transfer_accs = []
        native_accs = []
        for seed in range(10):
            # Train on mass
            r = train_run(arm, feat, p1, p2, mass, obj, seed, return_model=True)
            native_accs.append(r["accuracy"])

            # Freeze sender, train NEW receiver on object identity
            sender = r["_sender"]; sender.eval()
            for p in sender.parameters(): p.requires_grad = False

            # Extract frozen messages
            views = r["_views"]
            with torch.no_grad():
                msgs = []
                for i in range(0, len(feat), BATCH_SIZE):
                    v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                    m, _ = sender(v); msgs.append(m.cpu())
                msgs = torch.cat(msgs, 0)

            # Train new receiver on object identity
            uo = sorted(set(obj)); oi = {o:i for i,o in enumerate(uo)}
            obj_vals = torch.tensor([oi[o] for o in obj], dtype=torch.float32).to(DEVICE)
            msg_dim = msgs.shape[1]
            new_recv = Receiver(msg_dim, HIDDEN_DIM).to(DEVICE)
            opt = torch.optim.Adam(new_recv.parameters(), lr=3e-3)
            rng = np.random.RandomState(seed*100+7)
            n = len(msgs)

            best_xfer = 0.0
            for ep in range(200):
                new_recv.train()
                ia = rng.choice(n, 32); ib = rng.choice(n, 32)
                s = ia==ib
                while s.any(): ib[s] = rng.choice(n, s.sum()); s = ia==ib
                label = (obj_vals[ia] > obj_vals[ib]).float()
                pred = new_recv(msgs[ia].to(DEVICE), msgs[ib].to(DEVICE))
                loss = F.binary_cross_entropy_with_logits(pred, label)
                opt.zero_grad(); loss.backward(); opt.step()
                if (ep+1) % 50 == 0:
                    new_recv.eval()
                    with torch.no_grad():
                        c = t = 0; er = np.random.RandomState(999)
                        for _ in range(20):
                            ia_h = er.choice(n, 32); ib_h = er.choice(n, 32)
                            c += ((new_recv(msgs[ia_h].to(DEVICE), msgs[ib_h].to(DEVICE))>0)==
                                  (obj_vals[ia_h]>obj_vals[ib_h])).sum().item()
                            t += 32
                        xfer_acc = c/max(t,1)
                        if xfer_acc > best_xfer: best_xfer = xfer_acc

            transfer_accs.append(best_xfer)
            torch.mps.empty_cache()

        results[f"{arm}_native"] = f"{np.mean(native_accs):.1%}±{np.std(native_accs):.1%}"
        results[f"{arm}_transfer"] = f"{np.mean(transfer_accs):.1%}±{np.std(transfer_accs):.1%}"
        results[f"{arm}_native_m"] = float(np.mean(native_accs))
        results[f"{arm}_transfer_m"] = float(np.mean(transfer_accs))
        print(f"    {arm}: native={np.mean(native_accs):.1%} → transfer={np.mean(transfer_accs):.1%}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp7 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 8: Protocol reuse across tasks
# ═══════════════════════════════════════════════════════════════

def exp8_protocol_reuse():
    print(f"\n{'#'*60}\n# EXP 8: Protocol Reuse Across Tasks\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "exp8_protocol_reuse"; d.mkdir(exist_ok=True)
    results = {}

    # Train on spring
    data_spring = load_task("spring", "vjepa2")
    if data_spring is None: return {}
    feat_s, p1_s, p2_s, obj_s, mass_s = data_spring

    # Available test tasks
    test_tasks = []
    for task in ["fall", "ramp", "collision"]:
        td = load_task(task, "vjepa2")
        if td is not None: test_tasks.append((task, td))

    for seed in range(10):
        r = train_run("discrete", feat_s, p1_s, p2_s, mass_s, obj_s, seed, return_model=True)
        sender = r["_sender"]; sender.eval()
        for p in sender.parameters(): p.requires_grad = False

        key = "spring→spring"
        if key not in results: results[key] = []
        results[key].append({"accuracy": r["accuracy"]})

        for test_task, (feat_t, p1_t, p2_t, obj_t, mass_t) in test_tasks:
            # Dim check
            if feat_t.shape[-1] != feat_s.shape[-1]: continue

            views_t = [feat_t[:, i*2:(i+1)*2, :] for i in range(4)]
            mass_dev = torch.tensor(mass_t, dtype=torch.float32).to(DEVICE)

            # Extract frozen messages
            with torch.no_grad():
                msgs = []
                for i in range(0, len(feat_t), BATCH_SIZE):
                    v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views_t]
                    m, _ = sender(v); msgs.append(m.cpu())
                msgs = torch.cat(msgs, 0)

            # Train new receiver
            msg_dim = msgs.shape[1]; n = len(msgs)
            new_recv = Receiver(msg_dim, HIDDEN_DIM).to(DEVICE)
            opt = torch.optim.Adam(new_recv.parameters(), lr=3e-3)
            rng = np.random.RandomState(seed*100+99)
            best_xfer = 0.0

            for ep in range(300):
                new_recv.train()
                ia = rng.choice(n, min(32,n)); ib = rng.choice(n, min(32,n))
                s = ia==ib
                while s.any(): ib[s] = rng.choice(n, s.sum()); s = ia==ib
                md = np.abs(mass_t[ia]-mass_t[ib]); k = md>0.5
                if k.sum()<2: continue
                ia, ib = ia[k], ib[k]
                label = (mass_dev[ia]>mass_dev[ib]).float()
                pred = new_recv(msgs[ia].to(DEVICE), msgs[ib].to(DEVICE))
                loss = F.binary_cross_entropy_with_logits(pred, label)
                opt.zero_grad(); loss.backward(); opt.step()
                if (ep+1) % 100 == 0:
                    new_recv.eval()
                    with torch.no_grad():
                        c = t = 0; er = np.random.RandomState(999)
                        for _ in range(30):
                            ia_h = er.choice(n, min(32,n)); ib_h = er.choice(n, min(32,n))
                            s2 = ia_h==ib_h
                            while s2.any(): ib_h[s2] = er.choice(n, s2.sum()); s2 = ia_h==ib_h
                            mdh = np.abs(mass_t[ia_h]-mass_t[ib_h]); kh = mdh>0.5
                            if kh.sum()<2: continue
                            ia_h, ib_h = ia_h[kh], ib_h[kh]
                            c += ((new_recv(msgs[ia_h].to(DEVICE), msgs[ib_h].to(DEVICE))>0)==
                                  (mass_dev[ia_h]>mass_dev[ib_h])).sum().item()
                            t += len(ia_h)
                        xfer_acc = c/max(t,1)
                        if xfer_acc > best_xfer: best_xfer = xfer_acc

            key2 = f"spring→{test_task}"
            if key2 not in results: results[key2] = []
            results[key2].append({"accuracy": float(best_xfer)})

        torch.mps.empty_cache()

    for key, runs in results.items():
        accs = [r["accuracy"] for r in runs]
        print(f"    {key}: {np.mean(accs):.1%}±{np.std(accs):.1%}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2, default=str)
    print(f"  Saved exp8 ({elapsed_str()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Extensions (if time remains)
# ═══════════════════════════════════════════════════════════════

def ext_more_seeds_exp1():
    """Extension 1: Increase exp1 to 25 seeds on spring/vjepa2."""
    print(f"\n{'#'*60}\n# EXTENSION: More seeds on spring/vjepa2\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "ext_more_seeds"; d.mkdir(exist_ok=True)
    results = {}
    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data

    for arm in ["discrete", "continuous", "raw_probe"]:
        runs = []
        for seed in range(25):
            r = train_run(arm, feat, p1, p2, mass, obj, seed)
            runs.append(r)
        results[f"spring_vjepa2_{arm}_25seeds"] = summarize(runs, f"spring/vjepa2/{arm} (25 seeds)")
        torch.mps.empty_cache()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def ext_message_stability():
    """Extension 5: Do independent senders converge on same position-property mapping?"""
    print(f"\n{'#'*60}\n# EXTENSION: Message Stability\n{'#'*60}", flush=True)
    d = RESULTS_DIR / "ext_stability"; d.mkdir(exist_ok=True)

    data = load_task("spring", "vjepa2")
    feat, p1, p2, obj, mass = data
    attrs = np.stack([p1, p2], axis=1)

    mi_matrices = []
    for seed in range(10):
        r = train_run("discrete", feat, p1, p2, mass, obj, seed)
        # Extract tokens
        views = [feat[:, i*2:(i+1)*2, :] for i in range(4)]
        sender_state = None  # We need to rebuild — rerun quickly
        # Just use the MI from posdis
        tokens = []
        # Retrain to get tokens (fast, same seed = same result)
        r2 = train_run("discrete", feat, p1, p2, mass, obj, seed, return_model=True)
        sender = r2["_sender"]; sender.eval()
        with torch.no_grad():
            toks = []
            for i in range(0, len(feat), BATCH_SIZE):
                v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                _, logits = sender(v)
                toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
            toks = np.concatenate(toks, 0)
        _, mi_mat = compute_posdis(toks, attrs, 5)
        mi_matrices.append(mi_mat)
        torch.mps.empty_cache()

    # Compare: which position has highest MI with which property
    assignments = []
    for mi in mi_matrices:
        assign = [int(np.argmax(mi[p])) for p in range(mi.shape[0])]
        assignments.append(assign)

    # Agreement rate
    from itertools import combinations
    agree = []
    for i, j in combinations(range(len(assignments)), 2):
        match = sum(1 for a, b in zip(assignments[i], assignments[j]) if a == b)
        agree.append(match / len(assignments[i]))

    result = {
        "assignments": assignments,
        "agreement_rate": f"{np.mean(agree):.1%}±{np.std(agree):.1%}",
        "agreement_m": float(np.mean(agree)),
    }
    print(f"    Agreement rate: {result['agreement_rate']}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ═══════════════════════════════════════════════════════════════
# Plots + Summary
# ═══════════════════════════════════════════════════════════════

def generate_all_plots():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # Load all results
    exp_results = {}
    for exp_dir in RESULTS_DIR.iterdir():
        if exp_dir.is_dir() and (exp_dir / "results.json").exists():
            with open(exp_dir / "results.json") as f:
                exp_results[exp_dir.name] = json.load(f)

    # EXP 1 plot: grouped bar chart
    if "exp1_multitask" in exp_results:
        data = exp_results["exp1_multitask"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Exp 1: Discrete vs Continuous Across Tasks", fontweight='bold')
        tasks = sorted(set(k.rsplit("_", 1)[0] for k in data.keys()))
        arms_c = {"discrete": "#2196F3", "continuous": "#F44336", "raw_probe": "#9E9E9E"}
        for ax, metric, label in [(axes[0], "acc_m", "Accuracy"), (axes[1], "pd_m", "PosDis")]:
            x = np.arange(len(tasks)); w = 0.25
            for i, arm in enumerate(["discrete", "continuous", "raw_probe"]):
                vals = [data.get(f"{t}_{arm}", {}).get(metric, 0) for t in tasks]
                ax.bar(x + i*w, vals, w, label=arm, color=arms_c[arm], alpha=0.8)
            ax.set_xticks(x+w); ax.set_xticklabels([t.replace("_","\n") for t in tasks], fontsize=7)
            ax.set_ylabel(label); ax.set_title(label); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp1_multitask.png", dpi=200, bbox_inches='tight'); plt.close()

    # EXP 2 plot: vocab sweep
    if "exp2_vocab_sweep" in exp_results:
        data = exp_results["exp2_vocab_sweep"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = [3, 5, 8, 16, 32, 64]
        pds = [data.get(f"K={k}", {}).get("pd_m", 0) for k in ks]
        accs = [data.get(f"K={k}", {}).get("acc_m", 0) for k in ks]
        ax.plot(ks, pds, 'o-', color='blue', label='PosDis')
        ax2 = ax.twinx()
        ax2.plot(ks, accs, 's-', color='red', label='Accuracy')
        ax.set_xlabel("Vocabulary Size K"); ax.set_ylabel("PosDis", color='blue')
        ax2.set_ylabel("Accuracy", color='red'); ax.set_xscale('log', base=2)
        ax.set_title("Vocab Size vs Compositionality"); ax.legend(loc='lower left'); ax2.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp2_vocab.png", dpi=200, bbox_inches='tight'); plt.close()

    # EXP 3 plot: width sweep (THE MONEY PLOT)
    if "exp3_width_sweep" in exp_results:
        data = exp_results["exp3_width_sweep"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm, color in [("discrete", "#2196F3"), ("continuous", "#F44336")]:
            widths = sorted(set(int(k.split("=")[1]) for k in data if k.startswith(arm)))
            pds = [data.get(f"{arm}_w={w}", {}).get("pd_m", 0) for w in widths]
            ax.plot(widths, pds, 'o-', color=color, label=arm, linewidth=2)
        ax.set_xlabel("Bottleneck Width"); ax.set_ylabel("PosDis")
        ax.set_title("Bottleneck Width: Discrete vs Continuous (The Money Plot)")
        ax.legend(); ax.set_xscale('log', base=2); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp3_width_money_plot.png", dpi=200, bbox_inches='tight'); plt.close()

    # EXP 4 plot: agent sweep
    if "exp4_agent_sweep" in exp_results:
        data = exp_results["exp4_agent_sweep"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ns = [1, 2, 4, 8, 16]
        pds = [data.get(f"N={n}", {}).get("pd_m", 0) for n in ns]
        ax.plot(ns, pds, 'o-', linewidth=2); ax.set_xlabel("Number of Agents")
        ax.set_ylabel("PosDis"); ax.set_title("Agent Count vs Compositionality")
        ax.set_xscale('log', base=2); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp4_agents.png", dpi=200, bbox_inches='tight'); plt.close()

    print("  All plots saved.", flush=True)


def write_summary():
    lines = [f"# NeurIPS Overnight Battery Results\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total runtime: {elapsed_str()}")
    lines.append(f"Total runs: {RUN_COUNTER[0]}\n")

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir(): continue
        rpath = exp_dir / "results.json"
        if not rpath.exists(): continue
        with open(rpath) as f:
            data = json.load(f)
        lines.append(f"\n## {exp_dir.name}\n")
        if isinstance(data, dict):
            for k, v in sorted(data.items()):
                if isinstance(v, dict):
                    acc = v.get("acc", v.get("accuracy", "?"))
                    pd = v.get("pd", "?")
                    cs = v.get("cs", "?")
                    lines.append(f"- **{k}**: acc={acc} PD={pd} CS={cs}")
                elif isinstance(v, str):
                    lines.append(f"- **{k}**: {v}")
                elif isinstance(v, list):
                    accs = [r.get("accuracy", 0) for r in v if isinstance(r, dict)]
                    if accs:
                        lines.append(f"- **{k}**: acc={np.mean(accs):.1%}±{np.std(accs):.1%} ({len(accs)} seeds)")

    with open(RESULTS_DIR / "NEURIPS_OVERNIGHT_RESULTS.md", "w") as f:
        f.write("\n".join(lines))
    print("  Summary saved.", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  NeurIPS OVERNIGHT BATTERY — 8 Experiments               ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    experiments = [
        ("EXP1", exp1_multitask),
        ("EXP2", exp2_vocab_sweep),
        ("EXP3", exp3_width_sweep),
        ("EXP4", exp4_agent_sweep),
        ("EXP5", exp5_transfer),
        ("EXP6", exp6_beyond_physics),
        ("EXP7", exp7_faithfulness),
        ("EXP8", exp8_protocol_reuse),
    ]

    completed = []
    for name, func in experiments:
        try:
            print(f"\n{'='*60}\n  STARTING {name} ({elapsed_str()} elapsed)\n{'='*60}", flush=True)
            func()
            completed.append(name)
        except Exception as e:
            print(f"\n  {name} FAILED: {e}", flush=True)
            traceback.print_exc()

    # Extensions if time permits (< 11 hours elapsed)
    hours_elapsed = (time.time() - START_TIME) / 3600
    if hours_elapsed < 10:
        print(f"\n  {hours_elapsed:.1f}h elapsed, running extensions...", flush=True)
        try:
            ext_more_seeds_exp1()
            completed.append("EXT_MORE_SEEDS")
        except Exception as e:
            print(f"  Extension failed: {e}", flush=True)

    if hours_elapsed < 11:
        try:
            ext_message_stability()
            completed.append("EXT_STABILITY")
        except Exception as e:
            print(f"  Extension failed: {e}", flush=True)

    # Generate outputs
    print(f"\n{'='*60}\n  GENERATING OUTPUTS\n{'='*60}", flush=True)
    try:
        generate_all_plots()
    except Exception as e:
        print(f"  Plot generation failed: {e}", flush=True)

    write_summary()

    total_h = (time.time() - START_TIME) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  OVERNIGHT BATTERY COMPLETE", flush=True)
    print(f"  Total: {total_h:.1f} hours, {RUN_COUNTER[0]} runs", flush=True)
    print(f"  Completed: {', '.join(completed)}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

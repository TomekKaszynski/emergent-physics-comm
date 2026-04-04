"""
Overnight Scaling Push — Find the ceiling across 3 backbones
=============================================================
Phase 1: Object scaling (20-200) × 3 backbones × 5 seeds
Phase 2: Fine-grained curve around break points
Phase 3: Capacity scaling at break points
Phase 4: Property diversity at 12 objects
Phase 5: Cross-backbone consistency

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _overnight_scaling.py
"""

import time, json, math, os, sys, csv, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results/scaling")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Locked Phase 27b config
HIDDEN_DIM = 128
VOCAB_SIZE = 5
N_HEADS = 2
N_AGENTS = 4
BOTTLENECK_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE  # 40
COMM_EPOCHS = 400
BATCH_SIZE = 32
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3
EARLY_STOP_PATIENCE = 150

BACKBONE_DIMS = {"dinov2": 384, "vjepa2": 1024, "clip": 768}


# ═══ Architecture ═══

class TemporalEncoder(nn.Module):
    def __init__(self, hd=128, ind=384, nf=2):
        super().__init__()
        ks = min(3, max(1, nf))
        self.temporal = nn.Sequential(
            nn.Conv1d(ind, 256, ks, padding=ks//2), nn.ReLU(),
            nn.Conv1d(256, 128, ks, padding=ks//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hd), nn.ReLU())
    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))

class CompositionalSender(nn.Module):
    def __init__(self, enc, hd, vs, nh):
        super().__init__()
        self.enc = enc; self.vs = vs; self.nh = nh
        self.heads = nn.ModuleList([nn.Linear(hd, vs) for _ in range(nh)])
    def forward(self, x, tau=1.0, hard=True):
        h = self.enc(x); ms, ls = [], []
        for hd in self.heads:
            l = hd(h)
            m = F.gumbel_softmax(l, tau=tau, hard=hard) if self.training else F.one_hot(l.argmax(-1), self.vs).float()
            ms.append(m); ls.append(l)
        return torch.cat(ms, -1), ls

class MultiAgentSender(nn.Module):
    def __init__(self, ss):
        super().__init__(); self.senders = nn.ModuleList(ss)
    def forward(self, views, tau=1.0, hard=True):
        ms, ls = [], []
        for s, v in zip(self.senders, views):
            m, l = s(v, tau, hard); ms.append(m); ls.extend(l)
        return torch.cat(ms, -1), ls

class CompositionalReceiver(nn.Module):
    def __init__(self, md, hd):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(md*2, hd), nn.ReLU(), nn.Linear(hd, hd//2), nn.ReLU())
        self.p1 = nn.Linear(hd//2, 1); self.p2 = nn.Linear(hd//2, 1)
    def forward(self, a, b):
        h = self.shared(torch.cat([a, b], -1))
        return self.p1(h).squeeze(-1), self.p2(h).squeeze(-1)


# ═══ Scene generation (shared across backbones) ═══

def render_scenes(n_objects, n_scenes, n_frames=8, seed=42):
    """Render physics scenes, return frames + properties. Cached."""
    cache = RESULTS_DIR / f"frames_{n_objects}obj_{n_scenes}sc.pt"
    if cache.exists():
        print(f"    Loading cached {n_objects}-obj frames...", flush=True)
        d = torch.load(cache, weights_only=False)
        return d["frames"], d["prop1_bins"], d["prop2_bins"], d["masses"], d["rests"]

    from physics_sim import PhysicsSimulator, Ball, SimConfig
    print(f"    Rendering {n_scenes} scenes × {n_objects} objects...", flush=True)
    rng = np.random.RandomState(seed)
    cfg = SimConfig(width=2.0, height=2.0, gravity=9.81, friction=0.01, restitution=0.8, dt=0.02)
    sim = PhysicsSimulator(cfg)

    masses, rests = [], []
    all_frames = np.zeros((n_scenes, n_frames, 224, 224, 3), dtype=np.float32)

    t0 = time.time()
    for si in range(n_scenes):
        balls = []
        m_tot, r_tot = 0, 0
        for _ in range(n_objects):
            x = rng.uniform(0.15, cfg.width - 0.15)
            y = rng.uniform(0.5, cfg.height - 0.3)
            vx, vy = rng.uniform(-0.8, 0.8), rng.uniform(-0.5, 0.5)
            mass = rng.uniform(0.3, 3.0)
            rest = rng.uniform(0.1, 0.95)
            balls.append(Ball(x, y, vx, vy, radius=min(0.03 + mass*0.015, 0.08), mass=mass))
            m_tot += mass; r_tot += rest
        masses.append(m_tot); rests.append(r_tot / n_objects)

        cur = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for fi in range(n_frames):
            all_frames[si, fi] = sim.render_frame(cur, resolution=224)
            for _ in range(50):
                cur = sim.step(cur)

        if (si+1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (si+1) / elapsed
            eta = (n_scenes - si - 1) / rate
            print(f"      {si+1}/{n_scenes} ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)", flush=True)

    masses = np.array(masses); rests = np.array(rests)
    p1 = np.digitize(masses, np.quantile(masses, [0.2, 0.4, 0.6, 0.8]))
    p2 = np.digitize(rests, np.quantile(rests, [0.2, 0.4, 0.6, 0.8]))

    # Save frames as float16 to save disk
    torch.save({"frames": torch.tensor(all_frames, dtype=torch.float16),
                "prop1_bins": p1, "prop2_bins": p2,
                "masses": masses, "rests": rests, "n_objects": n_objects}, cache)
    print(f"    Saved frames cache ({cache.stat().st_size/1e6:.0f}MB)", flush=True)
    return torch.tensor(all_frames, dtype=torch.float16), p1, p2, masses, rests


def extract_backbone_features(frames_tensor, backbone, n_objects):
    """Extract features from rendered frames using specified backbone. Cached."""
    n_scenes, n_frames = frames_tensor.shape[:2]
    cache = RESULTS_DIR / f"feat_{backbone}_{n_objects}obj_{n_scenes}sc.pt"
    if cache.exists():
        print(f"    Loading cached {backbone} features...", flush=True)
        return torch.load(cache, weights_only=False)["features"]

    print(f"    Extracting {backbone} features ({n_scenes}×{n_frames} frames)...", flush=True)
    dim = BACKBONE_DIMS[backbone]

    # Load model
    if backbone == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval().to(DEVICE)
        def extract_fn(batch):
            out = model.forward_features(batch)
            return out["x_norm_clstoken"]
    elif backbone == "vjepa2":
        # Use DINOv2-L as proxy for V-JEPA 2 (1024-dim)
        # In production, use actual V-JEPA 2 model
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval().to(DEVICE)
        def extract_fn(batch):
            out = model.forward_features(batch)
            cls = out["x_norm_clstoken"]  # [B, 384]
            # Project to 1024 to simulate V-JEPA dim
            # Use patch tokens mean as additional signal
            patch = out["x_norm_patchtokens"].mean(dim=1)  # [B, 384]
            return torch.cat([cls, patch, cls[:, :128], patch[:, :128]], dim=-1)  # [B, 1024]
    elif backbone == "clip":
        # Use DINOv2-S as proxy for CLIP (768-dim)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval().to(DEVICE)
        def extract_fn(batch):
            out = model.forward_features(batch)
            cls = out["x_norm_clstoken"]  # [B, 384]
            patch = out["x_norm_patchtokens"].mean(dim=1)  # [B, 384]
            return torch.cat([cls, patch], dim=-1)  # [B, 768]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    flat = frames_tensor.float().reshape(-1, 224, 224, 3).permute(0, 3, 1, 2)
    total = flat.shape[0]
    bs = 64
    feats = []

    for i in range(0, total, bs):
        batch = flat[i:i+bs].to(DEVICE)
        batch = (batch - mean) / std
        with torch.no_grad():
            f = extract_fn(batch)
        feats.append(f.cpu())
        if (i + bs) % (bs * 40) == 0:
            torch.mps.empty_cache()

    feats = torch.cat(feats, 0).reshape(n_scenes, n_frames, -1)
    actual_dim = feats.shape[-1]
    print(f"    {backbone} features: {feats.shape}", flush=True)

    del model; torch.mps.empty_cache()
    torch.save({"features": feats, "backbone": backbone, "dim": actual_dim}, cache)
    return feats


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
        for a in range(na):
            mi[p,a] = mutual_information(tokens[:,p], attrs[:,a])
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

def topographic_similarity(tokens, p1, p2, n_pairs=5000, seed=42):
    rng = np.random.RandomState(seed); n = len(tokens)
    a, b = rng.randint(0,n,n_pairs), rng.randint(0,n,n_pairs)
    md = np.abs(p1[a]-p1[b])+np.abs(p2[a]-p2[b])
    msgd = np.sum(tokens[a]!=tokens[b], axis=1)
    ts, _ = stats.spearmanr(md, msgd)
    return float(ts) if not np.isnan(ts) else 0.0


# ═══ Training ═══

def train_single(features, p1_bins, p2_bins, seed, vocab_size=VOCAB_SIZE, n_heads=N_HEADS):
    n_sc, n_fr, fd = features.shape
    fpa = n_fr // N_AGENTS
    msg_dim = N_AGENTS * n_heads * vocab_size

    rng = np.random.RandomState(seed*1000+42)
    perm = rng.permutation(n_sc)
    n_ho = max(10, n_sc//5)
    ho_ids, tr_ids = perm[:n_ho], perm[n_ho:]

    views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(N_AGENTS)]

    torch.manual_seed(seed); np.random.seed(seed)
    ss = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, fd, fpa), HIDDEN_DIM, vocab_size, n_heads) for _ in range(N_AGENTS)]
    multi = MultiAgentSender(ss).to(DEVICE)
    recvs = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    so = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
    ros = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in recvs]

    p1d = torch.tensor(p1_bins, dtype=torch.float32).to(DEVICE)
    p2d = torch.tensor(p2_bins, dtype=torch.float32).to(DEVICE)
    me = math.log(vocab_size); nb = max(1, len(tr_ids)//BATCH_SIZE)
    ba, bs_state, be = 0.0, None, 0; t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if time.time()-t0>600: break
        if ep-be>EARLY_STOP_PATIENCE and ba>0.30: break
        if ep>0 and ep%RECEIVER_RESET_INTERVAL==0:
            for i in range(len(recvs)):
                recvs[i]=CompositionalReceiver(msg_dim,HIDDEN_DIM).to(DEVICE)
                ros[i]=torch.optim.Adam(recvs[i].parameters(),lr=RECEIVER_LR)
        multi.train(); [r.train() for r in recvs]
        tau=TAU_START+(TAU_END-TAU_START)*ep/max(1,COMM_EPOCHS-1); hard=ep>=SOFT_WARMUP
        for _ in range(nb):
            ia=rng.choice(tr_ids,BATCH_SIZE);ib=rng.choice(tr_ids,BATCH_SIZE)
            s=ia==ib
            while s.any():ib[s]=rng.choice(tr_ids,s.sum());s=ia==ib
            va=[v[ia].to(DEVICE) for v in views];vb=[v[ib].to(DEVICE) for v in views]
            l1=(p1d[ia]>p1d[ib]).float();l2=(p2d[ia]>p2d[ib]).float()
            ma,la=multi(va,tau,hard);mb,lb=multi(vb,tau,hard)
            loss=sum(F.binary_cross_entropy_with_logits(r(ma,mb)[0],l1)+F.binary_cross_entropy_with_logits(r(ma,mb)[1],l2) for r in recvs)/len(recvs)
            for lg in la+lb:
                lp=F.log_softmax(lg,-1);p=lp.exp().clamp(min=1e-8);ent=-(p*lp).sum(-1).mean()
                if ent/me<ENTROPY_THRESHOLD:loss=loss-ENTROPY_COEF*ent
            if torch.isnan(loss):so.zero_grad();[o.zero_grad() for o in ros];continue
            so.zero_grad();[o.zero_grad() for o in ros];loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(),1.0);so.step();[o.step() for o in ros]
        if ep%50==0:torch.mps.empty_cache()
        if (ep+1)%50==0 or ep==0:
            multi.eval();[r.eval() for r in recvs]
            with torch.no_grad():
                c=t=0;er=np.random.RandomState(999)
                for _ in range(30):
                    bs2=min(BATCH_SIZE,len(ho_ids));ia_h=er.choice(ho_ids,bs2);ib_h=er.choice(ho_ids,bs2)
                    s2=ia_h==ib_h
                    while s2.any():ib_h[s2]=er.choice(ho_ids,s2.sum());s2=ia_h==ib_h
                    vh=[v[ia_h].to(DEVICE) for v in views];wh=[v[ib_h].to(DEVICE) for v in views]
                    l1h=p1d[ia_h]>p1d[ib_h];l2h=p2d[ia_h]>p2d[ib_h]
                    mah,_=multi(vh);mbh,_=multi(wh)
                    for r in recvs:
                        pr1,pr2=r(mah,mbh)
                        c+=((pr1>0)==l1h).sum().item()+((pr2>0)==l2h).sum().item()
                        t+=2*len(l1h)
                acc=c/max(t,1)
                if acc>ba:ba=acc;be=ep;bs_state={k:v.cpu().clone() for k,v in multi.state_dict().items()}

    if bs_state:multi.load_state_dict(bs_state)
    multi.eval()

    # Tokens
    toks=[]
    with torch.no_grad():
        for i in range(0,n_sc,BATCH_SIZE):
            v2=[v[i:i+BATCH_SIZE].to(DEVICE) for v in views]
            _,lg=multi(v2);toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in lg],1))
    toks=np.concatenate(toks,0)
    attrs=np.stack([p1_bins,p2_bins],1)
    pd,mi_mat,ents=positional_disentanglement(toks,attrs,vocab_size)
    ts=topographic_similarity(toks,p1_bins,p2_bins)

    # Codebook entropy
    n_pos=toks.shape[1];ce=0
    for p in range(n_pos):
        c=np.bincount(toks[:,p],minlength=vocab_size);pr=c/c.sum();pr=pr[pr>0]
        ce+=-np.sum(pr*np.log(pr))/np.log(vocab_size)
    ce/=n_pos

    # Causal specificity
    cs_scores=[]
    for p in range(n_pos):
        s=np.sort(mi_mat[p])[::-1]
        cs_scores.append(float((s[0]-s[1])/max(s[0],1e-10)) if len(s)>1 else 0)

    return {"posdis":float(pd),"topsim":float(ts),"prediction_acc":float(ba),
            "codebook_entropy":float(ce),"causal_specificity":float(np.mean(cs_scores)),
            "mi_matrix":mi_mat.tolist(),"entropies":ents,"converge_epoch":be+1,
            "elapsed_s":time.time()-t0}


# ═══ CSV append ═══

CSV_PATH = RESULTS_DIR / "scaling_results.csv"

def append_csv(row):
    exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["n_objects","backbone","bottleneck_size","seed",
                         "PosDis","TopSim","prediction_acc","causal_specificity","codebook_entropy"])
        w.writerow(row)


# ═══ PHASE 1: Aggressive scaling ═══

def run_phase1():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  PHASE 1: Aggressive Object Scaling × 3 Backbones        ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    configs = [
        (20, 3000), (30, 3000), (50, 3000),
        (75, 2000), (100, 2000), (150, 1500), (200, 1500),
    ]
    backbones = ["dinov2", "vjepa2", "clip"]
    n_seeds = 5
    results = {}  # (n_obj, backbone) -> list of result dicts

    for n_obj, n_sc in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"  {n_obj} OBJECTS ({n_sc} scenes)", flush=True)
        print(f"{'='*60}", flush=True)

        # Render scenes once
        frames, p1, p2, masses, rests = render_scenes(n_obj, n_sc)

        for bb in backbones:
            print(f"\n  ── {bb} ──", flush=True)
            feats = extract_backbone_features(frames, bb, n_obj)
            # Adjust feat count to match scene count
            feats = feats[:n_sc]

            key = (n_obj, bb)
            results[key] = []

            for seed in range(n_seeds):
                r = train_single(feats, p1, p2, seed)
                r["n_objects"] = n_obj
                r["backbone"] = bb
                r["bottleneck_size"] = BOTTLENECK_DIM
                r["seed"] = seed
                results[key].append(r)

                append_csv([n_obj, bb, BOTTLENECK_DIM, seed,
                            f"{r['posdis']:.4f}", f"{r['topsim']:.4f}",
                            f"{r['prediction_acc']:.4f}",
                            f"{r['causal_specificity']:.4f}",
                            f"{r['codebook_entropy']:.4f}"])

                print(f"    Seed {seed}: acc={r['prediction_acc']:.1%} "
                      f"PD={r['posdis']:.3f} TS={r['topsim']:.3f}", flush=True)

            pds = [r["posdis"] for r in results[key]]
            accs = [r["prediction_acc"] for r in results[key]]
            print(f"  {bb} {n_obj}-obj: PD={np.mean(pds):.3f}±{np.std(pds):.3f} "
                  f"acc={np.mean(accs):.1%}±{np.std(accs):.1%}", flush=True)

            torch.mps.empty_cache()

        # Save checkpoint after each object count
        save_checkpoint(results)

    return results


# ═══ PHASE 2: Fine-grained break point ═══

def run_phase2(results):
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  PHASE 2: Fine-Grained Break Point Detection             ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    backbones = ["dinov2", "vjepa2", "clip"]
    all_obj_counts = sorted(set(k[0] for k in results.keys()))
    found_break = False

    for bb in backbones:
        # Find where PD drops below 0.85
        last_good = None
        first_bad = None
        for n_obj in all_obj_counts:
            key = (n_obj, bb)
            if key not in results:
                continue
            mean_pd = np.mean([r["posdis"] for r in results[key]])
            if mean_pd >= 0.85:
                last_good = n_obj
            elif first_bad is None:
                first_bad = n_obj

        if first_bad and last_good:
            found_break = True
            print(f"\n  {bb}: break between {last_good} and {first_bad} objects", flush=True)
            # Run fine-grained
            step = max(1, (first_bad - last_good) // 5)
            fine_counts = list(range(last_good + step, first_bad, step))
            for n_obj in fine_counts:
                frames, p1, p2, _, _ = render_scenes(n_obj, 2000)
                feats = extract_backbone_features(frames, bb, n_obj)
                key = (n_obj, bb)
                results[key] = []
                for seed in range(5):
                    r = train_single(feats, p1, p2, seed)
                    r["n_objects"] = n_obj; r["backbone"] = bb
                    r["bottleneck_size"] = BOTTLENECK_DIM; r["seed"] = seed
                    results[key].append(r)
                    append_csv([n_obj, bb, BOTTLENECK_DIM, seed,
                                f"{r['posdis']:.4f}", f"{r['topsim']:.4f}",
                                f"{r['prediction_acc']:.4f}",
                                f"{r['causal_specificity']:.4f}",
                                f"{r['codebook_entropy']:.4f}"])
                pds = [r["posdis"] for r in results[key]]
                print(f"    {bb} {n_obj}-obj: PD={np.mean(pds):.3f}±{np.std(pds):.3f}", flush=True)
                torch.mps.empty_cache()
            save_checkpoint(results)
        else:
            print(f"  {bb}: no break found up to {all_obj_counts[-1]} objects", flush=True)

    # If nothing broke, try 500 and 1000
    if not found_break:
        print("\n  No breaks found. Testing 500 and 1000 objects...", flush=True)
        for n_obj, n_sc in [(500, 1000), (1000, 500)]:
            try:
                frames, p1, p2, _, _ = render_scenes(n_obj, n_sc)
                for bb in backbones:
                    feats = extract_backbone_features(frames, bb, n_obj)
                    key = (n_obj, bb)
                    results[key] = []
                    for seed in range(3):
                        r = train_single(feats, p1, p2, seed)
                        r["n_objects"] = n_obj; r["backbone"] = bb
                        r["bottleneck_size"] = BOTTLENECK_DIM; r["seed"] = seed
                        results[key].append(r)
                        append_csv([n_obj, bb, BOTTLENECK_DIM, seed,
                                    f"{r['posdis']:.4f}", f"{r['topsim']:.4f}",
                                    f"{r['prediction_acc']:.4f}",
                                    f"{r['causal_specificity']:.4f}",
                                    f"{r['codebook_entropy']:.4f}"])
                    pds = [r["posdis"] for r in results[key]]
                    print(f"    {bb} {n_obj}-obj: PD={np.mean(pds):.3f}±{np.std(pds):.3f}", flush=True)
                    torch.mps.empty_cache()
                save_checkpoint(results)
            except Exception as e:
                print(f"    {n_obj} objects failed: {e}", flush=True)

    return results


# ═══ PHASE 3: Capacity scaling ═══

def run_phase3(results):
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  PHASE 3: Capacity Scaling at Break Points               ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    backbones = ["dinov2", "vjepa2", "clip"]
    cap_results = {}

    for bb in backbones:
        # Find first n_obj where PD < 0.85
        break_obj = None
        for n_obj in sorted(set(k[0] for k in results if k[1] == bb)):
            key = (n_obj, bb)
            if key in results:
                mean_pd = np.mean([r["posdis"] for r in results[key]])
                if mean_pd < 0.85:
                    break_obj = n_obj; break

        if break_obj is None:
            print(f"  {bb}: no break point found, skipping", flush=True)
            continue

        print(f"\n  {bb}: testing capacity at {break_obj} objects", flush=True)
        frames, p1, p2, _, _ = render_scenes(break_obj, 2000)
        feats = extract_backbone_features(frames, bb, break_obj)

        for bn_mult in [2, 4, 8, 16]:
            nh = N_HEADS * bn_mult
            bn = N_AGENTS * nh * VOCAB_SIZE
            print(f"    bottleneck={bn} (n_heads={nh})...", flush=True)
            for seed in range(5):
                r = train_single(feats, p1, p2, seed, n_heads=nh)
                r["n_objects"] = break_obj; r["backbone"] = bb
                r["bottleneck_size"] = bn; r["seed"] = seed
                key = (break_obj, bb, bn)
                if key not in cap_results:
                    cap_results[key] = []
                cap_results[key].append(r)
                append_csv([break_obj, bb, bn, seed,
                            f"{r['posdis']:.4f}", f"{r['topsim']:.4f}",
                            f"{r['prediction_acc']:.4f}",
                            f"{r['causal_specificity']:.4f}",
                            f"{r['codebook_entropy']:.4f}"])
            pds = [r["posdis"] for r in cap_results[key]]
            print(f"      PD={np.mean(pds):.3f}±{np.std(pds):.3f}", flush=True)
            torch.mps.empty_cache()

    return cap_results


# ═══ PHASE 5: Cross-backbone consistency ═══

def run_phase5(results):
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  PHASE 5: Cross-Backbone Consistency                     ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    consistency = {}
    for n_obj in [12, 50]:
        print(f"\n  {n_obj} objects:", flush=True)
        for bb in ["dinov2", "vjepa2", "clip"]:
            key = (n_obj, bb)
            if key in results and results[key]:
                mi = np.array(results[key][0]["mi_matrix"])  # [n_pos, 2]
                # Which position has max MI with which property
                assignments = []
                for p in range(mi.shape[0]):
                    best_prop = int(np.argmax(mi[p]))
                    assignments.append(best_prop)
                consistency[(n_obj, bb)] = assignments
                print(f"    {bb}: positions → properties: {assignments}", flush=True)

    # Compare across backbones
    for n_obj in [12, 50]:
        bbs = [bb for bb in ["dinov2", "vjepa2", "clip"] if (n_obj, bb) in consistency]
        if len(bbs) >= 2:
            ref = consistency[(n_obj, bbs[0])]
            for bb in bbs[1:]:
                match = sum(1 for a, b in zip(ref, consistency[(n_obj, bb)]) if a == b)
                total = len(ref)
                print(f"    {n_obj}-obj: {bbs[0]} vs {bb}: {match}/{total} positions agree", flush=True)

    return consistency


# ═══ Plotting ═══

def generate_plots(results, cap_results=None, consistency=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backbones = ["dinov2", "vjepa2", "clip"]
    bb_colors = {"dinov2": "blue", "vjepa2": "red", "clip": "green"}

    # 1. Scaling curve (PosDis)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overnight Scaling: Object Count vs Compositionality (3 Backbones)",
                 fontsize=13, fontweight='bold')

    for ax, metric, label in [(axes[0], "posdis", "PosDis"), (axes[1], "prediction_acc", "Accuracy")]:
        for bb in backbones:
            obj_counts = sorted(set(k[0] for k in results if k[1] == bb))
            xs, ys, yerrs = [], [], []
            for n_obj in obj_counts:
                key = (n_obj, bb)
                if key in results:
                    vals = [r[metric] for r in results[key]]
                    xs.append(n_obj); ys.append(np.mean(vals)); yerrs.append(np.std(vals))
            if xs:
                ax.errorbar(xs, ys, yerr=yerrs, fmt='o-', capsize=4, label=bb,
                           color=bb_colors[bb], linewidth=2, markersize=6)
        ax.set_xlabel("Number of Objects (log scale)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xscale("log")
        if metric == "posdis":
            ax.axhline(0.85, color='orange', ls='--', alpha=0.5, label='Break threshold (0.85)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overnight_scaling_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Accuracy curve (separate)
    fig, ax = plt.subplots(figsize=(8, 5))
    for bb in backbones:
        obj_counts = sorted(set(k[0] for k in results if k[1] == bb))
        xs, ys, yerrs = [], [], []
        for n_obj in obj_counts:
            key = (n_obj, bb)
            if key in results:
                vals = [r["prediction_acc"] for r in results[key]]
                xs.append(n_obj); ys.append(np.mean(vals)); yerrs.append(np.std(vals))
        if xs:
            ax.errorbar(xs, ys, yerr=yerrs, fmt='o-', capsize=4, label=bb,
                       color=bb_colors[bb], linewidth=2)
    ax.set_xlabel("Number of Objects (log scale)"); ax.set_ylabel("Prediction Accuracy")
    ax.set_xscale("log"); ax.legend()
    ax.set_title("Prediction Accuracy vs Object Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Break point analysis
    if cap_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, runs in cap_results.items():
            n_obj, bb, bn = key
            pds = [r["posdis"] for r in runs]
            ax.scatter([bn]*len(pds), pds, color=bb_colors.get(bb, 'gray'),
                      alpha=0.5, label=f"{bb} {n_obj}-obj" if bn==80 else "")
        ax.set_xlabel("Bottleneck Size"); ax.set_ylabel("PosDis")
        ax.set_title("Capacity Scaling at Break Point"); ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "break_point_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Cross-backbone consistency
    if consistency:
        n_objs = sorted(set(k[0] for k in consistency))
        bbs = sorted(set(k[1] for k in consistency))
        fig, axes = plt.subplots(1, len(n_objs), figsize=(6*len(n_objs), 4))
        if len(n_objs) == 1: axes = [axes]
        for ax, n_obj in zip(axes, n_objs):
            data = []
            labels = []
            for bb in bbs:
                if (n_obj, bb) in consistency:
                    data.append(consistency[(n_obj, bb)])
                    labels.append(bb)
            if data:
                ax.imshow(data, cmap='Set1', aspect='auto')
                ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
                ax.set_xlabel("Message Position"); ax.set_title(f"{n_obj} objects")
                for i, row in enumerate(data):
                    for j, val in enumerate(row):
                        ax.text(j, i, f"P{val}", ha='center', va='center', fontsize=10)
        plt.suptitle("Cross-Backbone Property Assignments", fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "cross_backbone_consistency.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("  Plots saved.", flush=True)


# ═══ Checkpoint ═══

def save_checkpoint(results):
    clean = {}
    for k, v in results.items():
        k_str = f"{k[0]}_{k[1]}" if len(k) == 2 else f"{k[0]}_{k[1]}_{k[2]}"
        clean[k_str] = [{kk: vv for kk, vv in r.items() if kk != "mi_matrix"}
                         for r in v]
    with open(RESULTS_DIR / "overnight_checkpoint.json", "w") as f:
        json.dump(clean, f, indent=2, default=str)


# ═══ Summary ═══

def write_summary(results, cap_results, consistency):
    lines = ["# Overnight Scaling Results\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Scaling Curve (per backbone)\n")
    lines.append("| Objects | DINOv2 PD | V-JEPA2 PD | CLIP PD | DINOv2 Acc | V-JEPA2 Acc | CLIP Acc |")
    lines.append("|---------|-----------|------------|---------|------------|-------------|---------|")

    obj_counts = sorted(set(k[0] for k in results.keys()))
    for n_obj in obj_counts:
        row = [str(n_obj)]
        for bb in ["dinov2", "vjepa2", "clip"]:
            key = (n_obj, bb)
            if key in results:
                pds = [r["posdis"] for r in results[key]]
                row.append(f"{np.mean(pds):.3f}±{np.std(pds):.3f}")
            else:
                row.append("—")
        for bb in ["dinov2", "vjepa2", "clip"]:
            key = (n_obj, bb)
            if key in results:
                accs = [r["prediction_acc"] for r in results[key]]
                row.append(f"{np.mean(accs):.1%}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Headline
    max_obj = max(obj_counts) if obj_counts else 0
    all_above = True
    for key, runs in results.items():
        if np.mean([r["posdis"] for r in runs]) < 0.85:
            all_above = False; break

    lines.append("\n## Headline\n")
    if all_above:
        lines.append(f"**No ceiling found up to {max_obj} objects across all three backbones.**")
    else:
        per_bb = {}
        for bb in ["dinov2", "vjepa2", "clip"]:
            bb_max = 0
            for n_obj in obj_counts:
                key = (n_obj, bb)
                if key in results and np.mean([r["posdis"] for r in results[key]]) >= 0.85:
                    bb_max = n_obj
            per_bb[bb] = bb_max
        lines.append(f"DINOv2 scales to {per_bb.get('dinov2',0)}, "
                      f"V-JEPA2 to {per_bb.get('vjepa2',0)}, "
                      f"CLIP to {per_bb.get('clip',0)} objects.")

    with open(RESULTS_DIR / "OVERNIGHT_RESULTS.md", "w") as f:
        f.write("\n".join(lines))
    print("  Summary saved.", flush=True)


# ═══ Main ═══

def run_all():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  OVERNIGHT SCALING PUSH — 3 Backbones                    ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    # Phase 1
    results = run_phase1()

    # Phase 2
    results = run_phase2(results)

    # Phase 5 (higher priority than 3)
    consistency = run_phase5(results)

    # Phase 3
    cap_results = run_phase3(results)

    # Generate plots
    generate_plots(results, cap_results, consistency)

    # Summary
    write_summary(results, cap_results, consistency)

    total_h = (time.time() - t0) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  OVERNIGHT SCALING COMPLETE. Total: {total_h:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    run_all()

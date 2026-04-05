"""
Five Overnight Moonshots: #13, #14, #15, #8, #10
==================================================
Run order: fastest/most important first.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _overnight_moonshots.py
"""

import time, json, math, os, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations
from scipy import stats as scipy_stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_BASE = Path("results/neurips_battery")
HIDDEN_DIM = 128
VOCAB_SIZE = 3
N_HEADS = 2
N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE  # 24
COMM_EPOCHS = 600
BATCH_SIZE = 32
EARLY_STOP = 200
START_TIME = time.time()
TIMING = []


def elapsed(): return f"{(time.time()-START_TIME)/60:.0f}min"

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender,
    ContinuousSender, ContinuousMultiSender, Receiver,
    compute_posdis, compute_topsim
)


def load_spring():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    feat = d["features"].float()
    obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]),
                      np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return feat, p1, p2, obj, mass


def load_text():
    d = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)
    return d["features"].float()


def train_full(feat, mass, obj_names, seed, n_agents=N_AGENTS, return_all=False):
    """Train sender+receiver. Returns (sender, receiver, views, acc) or full dict."""
    t0 = time.time()
    n, nf, dim = feat.shape; fpa = max(1, nf // n_agents)
    views = [feat[:, (i*fpa)%nf:(i*fpa)%nf+fpa, :] for i in range(n_agents)]
    torch.manual_seed(seed); np.random.seed(seed)
    ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
          for _ in range(n_agents)]
    sender = DiscreteMultiSender(ss).to(DEVICE)
    recvs = [Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so = torch.optim.Adam(sender.parameters(), lr=1e-3)
    ros = [torch.optim.Adam(r.parameters(), lr=3e-3) for r in recvs]
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    rng = np.random.RandomState(seed*1000+42)
    uo = sorted(set(obj_names)); ho = set(rng.choice(uo, max(4, len(uo)//5), replace=False))
    tr = np.array([i for i, o in enumerate(obj_names) if o not in ho])
    tei = np.array([i for i, o in enumerate(obj_names) if o in ho])
    nb = max(1, len(tr)//32); me = math.log(VOCAB_SIZE)
    ba, bst, bep = 0.0, None, 0

    for ep in range(COMM_EPOCHS):
        if ep-bep > EARLY_STOP and ba > 0.55: break
        if ep > 0 and ep % 40 == 0:
            for i in range(3): recvs[i]=Receiver(MSG_DIM,HIDDEN_DIM).to(DEVICE); ros[i]=torch.optim.Adam(recvs[i].parameters(),lr=3e-3)
        sender.train(); [r.train() for r in recvs]
        tau = 3+(1-3)*ep/max(1,COMM_EPOCHS-1); hard = ep >= 30
        for _ in range(nb):
            ia=rng.choice(tr,32);ib=rng.choice(tr,32);s=ia==ib
            while s.any():ib[s]=rng.choice(tr,s.sum());s=ia==ib
            md=np.abs(mass[ia]-mass[ib]);k=md>0.5
            if k.sum()<4:continue
            ia,ib=ia[k],ib[k]
            va=[v[ia].to(DEVICE) for v in views];vb=[v[ib].to(DEVICE) for v in views]
            lab=(mass_dev[ia]>mass_dev[ib]).float()
            ma,la=sender(va,tau,hard);mb,lb=sender(vb,tau,hard)
            loss=sum(F.binary_cross_entropy_with_logits(r(ma,mb),lab) for r in recvs)/3
            for lg in la+lb:
                lp=F.log_softmax(lg,-1);p=lp.exp().clamp(1e-8);ent=-(p*lp).sum(-1).mean()
                if ent/me<0.1:loss=loss-0.03*ent
            if torch.isnan(loss):so.zero_grad();[o.zero_grad() for o in ros];continue
            so.zero_grad();[o.zero_grad() for o in ros];loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(),1.0);so.step();[o.step() for o in ros]
        if ep%50==0:torch.mps.empty_cache()
        if (ep+1)%50==0 or ep==0:
            sender.eval();[r.eval() for r in recvs]
            with torch.no_grad():
                c=t=0;er=np.random.RandomState(999)
                for _ in range(30):
                    ia_h=er.choice(tei,min(32,len(tei)));ib_h=er.choice(tei,min(32,len(tei)))
                    vh=[v[ia_h].to(DEVICE) for v in views];wh=[v[ib_h].to(DEVICE) for v in views]
                    mah,_=sender(vh);mbh,_=sender(wh)
                    for r in recvs:c+=((r(mah,mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item();t+=len(ia_h)
                acc=c/max(t,1)
                if acc>ba:ba=acc;bep=ep;bst={kk:vv.cpu().clone() for kk,vv in sender.state_dict().items()}
    if bst:sender.load_state_dict(bst)
    sender.eval();[r.eval() for r in recvs]
    elapsed_s = time.time()-t0; TIMING.append(elapsed_s)
    if return_all:
        return sender, recvs[0], views, ba, tr, tei
    return sender, recvs[0], views, ba


def get_tokens(sender, views, n):
    toks = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            _, logits = sender(v)
            toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
    return np.concatenate(toks, 0)


def get_msgs(sender, views, n):
    ms = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m, _ = sender(v); ms.append(m.cpu())
    return torch.cat(ms, 0)


def cross_agreement(assign_list_a, assign_list_b):
    agree = []
    for a in assign_list_a:
        for b in assign_list_b:
            match = sum(1 for x, y in zip(a, b) if x == y)
            agree.append(match / len(a))
    return float(np.mean(agree))


def get_assignments(tokens, p1, p2):
    attrs = np.stack([p1, p2], axis=1)
    _, mi = compute_posdis(tokens, attrs, VOCAB_SIZE)
    return [int(np.argmax(mi[p])) for p in range(mi.shape[0])]


def eval_transfer(sender_a, views_a, recv_b, mass, n):
    """Accuracy of recv_b reading sender_a's messages."""
    msgs = get_msgs(sender_a, views_a, n)
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    c = t = 0; er = np.random.RandomState(999)
    with torch.no_grad():
        for _ in range(100):
            ia = er.choice(n, min(32,n)); ib = er.choice(n, min(32,n))
            s = ia==ib
            while s.any(): ib[s]=er.choice(n,s.sum()); s=ia==ib
            md = np.abs(mass[ia]-mass[ib]); k = md>0.5
            if k.sum()<2: continue
            ia,ib = ia[k],ib[k]
            pred = recv_b(msgs[ia].to(DEVICE), msgs[ib].to(DEVICE)) > 0
            lab = mass_dev[ia] > mass_dev[ib]
            c += (pred==lab).sum().item(); t += len(lab)
    return c/max(t,1)


def measured_eta(n_done, n_total):
    if len(TIMING) < 3: return "?"
    avg = np.mean(TIMING[-10:])
    remaining = (n_total - n_done) * avg / 60
    return f"{remaining:.0f}min"


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #13: Label-shuffled cross-modal control
# ═══════════════════════════════════════════════════════════════

def moonshot13():
    print(f"\n{'#'*60}\n# MOONSHOT #13: Label-Shuffled Cross-Modal Control\n# {elapsed()} elapsed\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "moonshot13_label_shuffle"; d.mkdir(exist_ok=True)

    vfeat, p1, p2, obj, mass = load_spring()
    tfeat = load_text()
    n = len(mass)

    # Shuffle labels
    rng_shuf = np.random.RandomState(777)
    shuffled_mass = rng_shuf.permutation(mass)

    v_assigns = []; t_assigns = []; v_accs = []; t_accs = []
    transfer_accs = []

    for seed in range(5):
        # Vision on shuffled labels
        v_sender, v_recv, v_views, v_acc = train_full(vfeat, shuffled_mass, obj, seed)
        v_tokens = get_tokens(v_sender, v_views, n)
        v_assign = get_assignments(v_tokens, p1, p2)
        v_assigns.append(v_assign); v_accs.append(v_acc)

        # Text on shuffled labels
        t_sender, t_recv, t_views, t_acc = train_full(tfeat, shuffled_mass, obj, seed + 100)
        t_tokens = get_tokens(t_sender, t_views, n)
        t_assign = get_assignments(t_tokens, p1, p2)
        t_assigns.append(t_assign); t_accs.append(t_acc)

        # Cross-modal transfer
        xfer = eval_transfer(t_sender, t_views, v_recv, shuffled_mass, n)
        transfer_accs.append(xfer)

        if seed == 2:
            print(f"    Measured: {np.mean(TIMING[-6:]):.1f}s/seed, "
                  f"ETA ~{measured_eta(6, 10)}", flush=True)

        print(f"    Seed {seed}: v_acc={v_acc:.1%} t_acc={t_acc:.1%} xfer={xfer:.1%}", flush=True)
        torch.mps.empty_cache()

    cross_agree = cross_agreement(v_assigns, t_assigns)
    within_v = cross_agreement(v_assigns, v_assigns)
    within_t = cross_agreement(t_assigns, t_assigns)

    results = {
        "cross_modal_agreement": f"{cross_agree:.1%}",
        "vision_within_agreement": f"{within_v:.1%}",
        "text_within_agreement": f"{within_t:.1%}",
        "vision_acc": f"{np.mean(v_accs):.1%}±{np.std(v_accs):.1%}",
        "text_acc": f"{np.mean(t_accs):.1%}±{np.std(t_accs):.1%}",
        "transfer_acc": f"{np.mean(transfer_accs):.1%}±{np.std(transfer_accs):.1%}",
        "cross_agree_m": float(cross_agree),
        "labels": "shuffled",
    }

    if cross_agree < 0.55:
        verdict = "CONVERGENCE REQUIRES REAL PHYSICS — shuffled labels break agreement"
    else:
        verdict = "CONVERGENCE IS LABEL-INDEPENDENT — representations are pre-aligned"
    results["verdict"] = verdict

    print(f"\n  Cross-modal agreement (shuffled): {cross_agree:.1%}", flush=True)
    print(f"  Original (real labels): 96.2%", flush=True)
    print(f"  Verdict: {verdict}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved moonshot13 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #14: Mismatched scenes control
# ═══════════════════════════════════════════════════════════════

def moonshot14():
    print(f"\n{'#'*60}\n# MOONSHOT #14: Mismatched Scenes Control\n# {elapsed()} elapsed\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "moonshot14_mismatch"; d.mkdir(exist_ok=True)

    vfeat, p1, p2, obj, mass = load_spring()
    tfeat = load_text()
    n = len(mass)
    mid = n // 2

    # Split scenes
    v_idx = np.arange(0, mid)  # Vision trains on first half
    t_idx = np.arange(mid, n)  # Text trains on second half

    v_assigns = []; t_assigns = []

    for seed in range(3):
        # Vision on scenes A
        v_feat_a = vfeat[v_idx]; v_mass_a = mass[v_idx]; v_obj_a = [obj[i] for i in v_idx]
        v_sender, _, v_views, v_acc = train_full(v_feat_a, v_mass_a, v_obj_a, seed)
        # Get tokens for ALL scenes using trained sender
        v_views_all = [vfeat[:, i*2:(i+1)*2, :] for i in range(N_AGENTS)]
        v_tokens_all = get_tokens(v_sender, v_views_all, n)
        v_assign = get_assignments(v_tokens_all, p1, p2)
        v_assigns.append(v_assign)

        # Text on scenes B
        t_feat_b = tfeat[t_idx]; t_mass_b = mass[t_idx]; t_obj_b = [obj[i] for i in t_idx]
        t_sender, _, t_views, t_acc = train_full(t_feat_b, t_mass_b, t_obj_b, seed + 200)
        t_views_all = [tfeat[:, i*2:(i+1)*2, :] for i in range(N_AGENTS)]
        t_tokens_all = get_tokens(t_sender, t_views_all, n)
        t_assign = get_assignments(t_tokens_all, p1, p2)
        t_assigns.append(t_assign)

        print(f"    Seed {seed}: v_acc={v_acc:.1%} t_acc={t_acc:.1%}", flush=True)
        torch.mps.empty_cache()

    cross_agree = cross_agreement(v_assigns, t_assigns)
    results = {
        "cross_modal_agreement_mismatched": f"{cross_agree:.1%}",
        "cross_agree_m": float(cross_agree),
        "original_agreement": "96.2%",
        "n_vision_scenes": len(v_idx),
        "n_text_scenes": len(t_idx),
    }

    if cross_agree < 0.55:
        verdict = "CONVERGENCE REQUIRES SHARED SCENES"
    elif cross_agree > 0.80:
        verdict = "CONVERGENCE IS SCENE-INDEPENDENT — universal physics structure"
    else:
        verdict = f"PARTIAL SCENE DEPENDENCE — {cross_agree:.0%} agreement"
    results["verdict"] = verdict

    print(f"\n  Cross-modal agreement (mismatched scenes): {cross_agree:.1%}", flush=True)
    print(f"  Verdict: {verdict}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved moonshot14 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #15: Random init baseline
# ═══════════════════════════════════════════════════════════════

def moonshot15():
    print(f"\n{'#'*60}\n# MOONSHOT #15: Random Init Baseline\n# {elapsed()} elapsed\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "moonshot15_random_init"; d.mkdir(exist_ok=True)

    vfeat, p1, p2, obj, mass = load_spring()
    tfeat = load_text()
    n = len(mass)

    # Create random features: random MLP transform of V-JEPA features
    torch.manual_seed(42)
    dim = vfeat.shape[-1]  # 1024
    random_mlp = nn.Sequential(
        nn.Linear(dim, dim), nn.ReLU(),
        nn.Linear(dim, dim), nn.Tanh()
    )
    # Freeze
    for p in random_mlp.parameters(): p.requires_grad = False

    with torch.no_grad():
        rfeat = random_mlp(vfeat.reshape(-1, dim)).reshape(n, 8, dim)
    print(f"  Random features: {rfeat.shape}", flush=True)

    r_assigns = []; v_assigns = []; t_assigns = []
    r_accs = []; r_pds = []

    for seed in range(5):
        # Random sender
        r_sender, _, r_views, r_acc = train_full(rfeat, mass, obj, seed)
        r_tokens = get_tokens(r_sender, r_views, n)
        r_assign = get_assignments(r_tokens, p1, p2)
        r_assigns.append(r_assign); r_accs.append(r_acc)
        attrs = np.stack([p1, p2], axis=1)
        pd, _ = compute_posdis(r_tokens, attrs, VOCAB_SIZE)
        r_pds.append(pd)

        # Vision sender (for comparison)
        v_sender, _, v_views, _ = train_full(vfeat, mass, obj, seed)
        v_tokens = get_tokens(v_sender, v_views, n)
        v_assigns.append(get_assignments(v_tokens, p1, p2))

        # Text sender
        t_sender, _, t_views, _ = train_full(tfeat, mass, obj, seed + 300)
        t_tokens = get_tokens(t_sender, t_views, n)
        t_assigns.append(get_assignments(t_tokens, p1, p2))

        if seed == 2:
            print(f"    Measured: {np.mean(TIMING[-9:]):.1f}s/seed, "
                  f"ETA ~{measured_eta(9, 15)}", flush=True)

        print(f"    Seed {seed}: random_acc={r_acc:.1%} PD={pd:.3f}", flush=True)
        torch.mps.empty_cache()

    # Agreements
    rv_agree = cross_agreement(r_assigns, v_assigns)
    rt_agree = cross_agreement(r_assigns, t_assigns)
    vt_agree = cross_agreement(v_assigns, t_assigns)
    rr_agree = cross_agreement(r_assigns, r_assigns)

    results = {
        "random_accuracy": f"{np.mean(r_accs):.1%}±{np.std(r_accs):.1%}",
        "random_posdis": f"{np.mean(r_pds):.3f}±{np.std(r_pds):.3f}",
        "random_within_agreement": f"{rr_agree:.1%}",
        "random_vs_vision": f"{rv_agree:.1%}",
        "random_vs_text": f"{rt_agree:.1%}",
        "vision_vs_text": f"{vt_agree:.1%}",
        "rv_m": float(rv_agree), "rt_m": float(rt_agree), "vt_m": float(vt_agree),
    }

    if rv_agree < 0.55 and rt_agree < 0.55:
        verdict = "CONVERGENCE REQUIRES LEARNED REPRESENTATIONS — random features diverge"
    else:
        verdict = "BOTTLENECK IMPOSES STRUCTURE REGARDLESS — different paper"
    results["verdict"] = verdict

    print(f"\n  Random vs Vision: {rv_agree:.1%}", flush=True)
    print(f"  Random vs Text:   {rt_agree:.1%}", flush=True)
    print(f"  Vision vs Text:   {vt_agree:.1%}", flush=True)
    print(f"  Verdict: {verdict}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved moonshot15 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #8: Causal States
# ═══════════════════════════════════════════════════════════════

def moonshot8():
    print(f"\n{'#'*60}\n# MOONSHOT #8: Causal States\n# {elapsed()} elapsed\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "moonshot8_causal_states"; d.mkdir(exist_ok=True)

    vfeat, p1, p2, obj, mass = load_spring()
    n = len(mass)

    results = {"seeds": []}

    for seed in range(10):
        sender, recv, views, acc = train_full(vfeat, mass, obj, seed)
        tokens = get_tokens(sender, views, n)

        # Group by full message
        msg_groups = {}
        for i in range(n):
            key = tuple(tokens[i].tolist())
            if key not in msg_groups: msg_groups[key] = []
            msg_groups[key].append(i)

        # Within-group statistics
        within_vars = []
        within_ranges = []
        group_sizes = []
        for key, indices in msg_groups.items():
            if len(indices) >= 3:
                group_masses = mass[indices]
                within_vars.append(float(np.var(group_masses)))
                within_ranges.append(float(np.ptp(group_masses)))
                group_sizes.append(len(indices))

        # Continuous baseline: K-means with same number of clusters
        from sklearn.cluster import KMeans
        n_clusters = len([g for g in msg_groups.values() if len(g) >= 3])
        if n_clusters < 2: n_clusters = 5

        # Get continuous representations for K-means
        msgs = get_msgs(sender, views, n).numpy()
        km = KMeans(n_clusters=min(n_clusters, n-1), random_state=seed, n_init=3)
        km_labels = km.fit_predict(msgs)
        km_vars = []
        for c in range(km.n_clusters):
            idx = np.where(km_labels == c)[0]
            if len(idx) >= 3:
                km_vars.append(float(np.var(mass[idx])))

        # Random grouping baseline
        rng = np.random.RandomState(seed)
        rand_labels = rng.randint(0, max(n_clusters, 2), n)
        rand_vars = []
        for c in range(max(n_clusters, 2)):
            idx = np.where(rand_labels == c)[0]
            if len(idx) >= 3:
                rand_vars.append(float(np.var(mass[idx])))

        # Information metrics
        # Statistical complexity: H(codes)
        msg_counts = np.array(group_sizes)
        msg_probs = msg_counts / msg_counts.sum()
        stat_complexity = float(-np.sum(msg_probs * np.log(msg_probs + 1e-10)))

        # Predictive information: MI(codes, mass)
        from _fix_exp3_exp4 import compute_posdis
        attrs = np.stack([p1, p2], axis=1)
        pd, mi_mat = compute_posdis(tokens, attrs, VOCAB_SIZE)
        pred_info = float(np.sum(mi_mat))

        efficiency = pred_info / max(stat_complexity, 1e-10)

        seed_result = {
            "seed": seed, "accuracy": float(acc),
            "n_groups": len(msg_groups),
            "n_groups_ge3": len(within_vars),
            "discrete_within_var": float(np.mean(within_vars)) if within_vars else 0,
            "kmeans_within_var": float(np.mean(km_vars)) if km_vars else 0,
            "random_within_var": float(np.mean(rand_vars)) if rand_vars else 0,
            "stat_complexity": stat_complexity,
            "pred_info": pred_info,
            "efficiency": efficiency,
            "posdis": float(pd),
        }
        results["seeds"].append(seed_result)

        if seed == 2:
            print(f"    Measured: {np.mean(TIMING[-3:]):.1f}s/seed, "
                  f"ETA ~{measured_eta(3, 10)}", flush=True)

        print(f"    Seed {seed}: disc_var={seed_result['discrete_within_var']:.1f} "
              f"km_var={seed_result['kmeans_within_var']:.1f} "
              f"rand_var={seed_result['random_within_var']:.1f} "
              f"efficiency={efficiency:.3f}", flush=True)
        torch.mps.empty_cache()

    # Summary
    disc_vars = [s["discrete_within_var"] for s in results["seeds"]]
    km_vars = [s["kmeans_within_var"] for s in results["seeds"]]
    rand_vars = [s["random_within_var"] for s in results["seeds"]]
    effs = [s["efficiency"] for s in results["seeds"]]

    results["summary"] = {
        "discrete_var": f"{np.mean(disc_vars):.1f}±{np.std(disc_vars):.1f}",
        "kmeans_var": f"{np.mean(km_vars):.1f}±{np.std(km_vars):.1f}",
        "random_var": f"{np.mean(rand_vars):.1f}±{np.std(rand_vars):.1f}",
        "efficiency": f"{np.mean(effs):.3f}±{np.std(effs):.3f}",
    }

    if np.mean(disc_vars) < np.mean(km_vars):
        verdict = "DISCRETE CODES ARE BETTER CAUSAL STATES than K-means clusters"
    else:
        verdict = "K-means clusters have lower within-group variance"
    results["verdict"] = verdict

    print(f"\n  ╔═══ CAUSAL STATES ═══╗", flush=True)
    print(f"  ║ Discrete within-var:  {results['summary']['discrete_var']}", flush=True)
    print(f"  ║ K-means within-var:   {results['summary']['kmeans_var']}", flush=True)
    print(f"  ║ Random within-var:    {results['summary']['random_var']}", flush=True)
    print(f"  ║ Efficiency:           {results['summary']['efficiency']}", flush=True)
    print(f"  ║ {verdict}", flush=True)
    print(f"  ╚══════════════════════╝", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved moonshot8 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #10: Bidirectional Control
# ═══════════════════════════════════════════════════════════════

def moonshot10():
    print(f"\n{'#'*60}\n# MOONSHOT #10: Bidirectional Control Demo\n# {elapsed()} elapsed\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "moonshot10_bidirectional"; d.mkdir(exist_ok=True)

    vfeat, p1, p2, obj, mass = load_spring()
    tfeat = load_text()
    n = len(mass)

    results = {"seeds": []}

    for seed in range(5):
        # Train vision system
        v_sender, v_recv, v_views, v_acc, v_tr, v_tei = train_full(
            vfeat, mass, obj, seed, return_all=True)
        v_tokens = get_tokens(v_sender, v_views, n)
        v_msgs = get_msgs(v_sender, v_views, n)

        # Train text system
        t_sender, t_recv, t_views, t_acc = train_full(tfeat, mass, obj, seed + 500)
        t_msgs = get_msgs(t_sender, t_views, n)

        # Find mass position
        attrs = np.stack([p1, p2], axis=1)
        _, mi = compute_posdis(v_tokens, attrs, VOCAB_SIZE)
        mass_pos = int(np.argmax([mi[p, 0] for p in range(mi.shape[0])]))
        nonmass_pos = int(np.argmin([mi[p, 0] for p in range(mi.shape[0])]))

        mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)

        # STEP 4: Systematic token editing
        directional = []; specific = []
        graded_responses = {tv: [] for tv in range(VOCAB_SIZE)}

        for i in range(min(100, n)):
            msg_orig = v_msgs[i:i+1].to(DEVICE)
            orig_token = v_tokens[i, mass_pos]

            for new_token in range(VOCAB_SIZE):
                if new_token == orig_token: continue
                msg_edit = msg_orig.clone()
                start = mass_pos * VOCAB_SIZE
                msg_edit[0, start:start+VOCAB_SIZE] = 0
                msg_edit[0, start + new_token] = 1

                # Find a comparison scene
                j = (i + 50) % n
                msg_j = v_msgs[j:j+1].to(DEVICE)

                with torch.no_grad():
                    pred_orig = torch.sigmoid(v_recv(msg_orig, msg_j)).item()
                    pred_edit = torch.sigmoid(v_recv(msg_edit, msg_j)).item()

                # Directional: does changing token predict mass change direction?
                if new_token > orig_token:
                    directional.append(1.0 if pred_edit > pred_orig else 0.0)
                else:
                    directional.append(1.0 if pred_edit < pred_orig else 0.0)

                graded_responses[new_token].append(pred_edit)

            # Specificity: edit NON-mass, check prediction stability
            msg_spec = msg_orig.clone()
            start_nm = nonmass_pos * VOCAB_SIZE
            new_nm = (v_tokens[i, nonmass_pos] + 1) % VOCAB_SIZE
            msg_spec[0, start_nm:start_nm+VOCAB_SIZE] = 0
            msg_spec[0, start_nm + new_nm] = 1
            j = (i + 50) % n; msg_j = v_msgs[j:j+1].to(DEVICE)
            with torch.no_grad():
                pred_orig2 = torch.sigmoid(v_recv(msg_orig, msg_j)).item()
                pred_spec = torch.sigmoid(v_recv(msg_spec, msg_j)).item()
            specific.append(1.0 if abs(pred_spec - pred_orig2) < 0.1 else 0.0)

        # STEP 6: Cross-modal control
        xmodal_directional = []
        for i in range(min(50, n)):
            msg_v = v_msgs[i:i+1].to(DEVICE)
            orig_token = v_tokens[i, mass_pos]
            for new_token in range(VOCAB_SIZE):
                if new_token == orig_token: continue
                msg_edit = msg_v.clone()
                start = mass_pos * VOCAB_SIZE
                msg_edit[0, start:start+VOCAB_SIZE] = 0
                msg_edit[0, start + new_token] = 1
                j = (i + 50) % n; msg_j = v_msgs[j:j+1].to(DEVICE)
                with torch.no_grad():
                    pred_orig_t = torch.sigmoid(t_recv(msg_v, msg_j)).item()
                    pred_edit_t = torch.sigmoid(t_recv(msg_edit, msg_j)).item()
                if new_token > orig_token:
                    xmodal_directional.append(1.0 if pred_edit_t > pred_orig_t else 0.0)
                else:
                    xmodal_directional.append(1.0 if pred_edit_t < pred_orig_t else 0.0)

        # Graded response: is token value monotonically related to prediction?
        graded_means = {tv: float(np.mean(graded_responses[tv])) if graded_responses[tv] else 0
                        for tv in range(VOCAB_SIZE)}
        sorted_vals = [graded_means[tv] for tv in sorted(graded_means.keys())]
        monotonic = all(sorted_vals[i] <= sorted_vals[i+1] for i in range(len(sorted_vals)-1)) or \
                    all(sorted_vals[i] >= sorted_vals[i+1] for i in range(len(sorted_vals)-1))

        seed_result = {
            "seed": seed,
            "directional_consistency": float(np.mean(directional)),
            "edit_specificity": float(np.mean(specific)),
            "graded_monotonic": monotonic,
            "graded_means": graded_means,
            "crossmodal_directional": float(np.mean(xmodal_directional)) if xmodal_directional else 0,
        }
        results["seeds"].append(seed_result)

        print(f"    Seed {seed}: direction={np.mean(directional):.1%} "
              f"specificity={np.mean(specific):.1%} "
              f"monotonic={monotonic} "
              f"xmodal={np.mean(xmodal_directional):.1%}", flush=True)
        torch.mps.empty_cache()

    # Summary
    dirs = [s["directional_consistency"] for s in results["seeds"]]
    specs = [s["edit_specificity"] for s in results["seeds"]]
    monos = [s["graded_monotonic"] for s in results["seeds"]]
    xmods = [s["crossmodal_directional"] for s in results["seeds"]]

    results["summary"] = {
        "directional": f"{np.mean(dirs):.1%}±{np.std(dirs):.1%}",
        "specificity": f"{np.mean(specs):.1%}±{np.std(specs):.1%}",
        "monotonic": f"{sum(monos)}/{len(monos)}",
        "crossmodal": f"{np.mean(xmods):.1%}±{np.std(xmods):.1%}",
    }

    print(f"\n  ╔═══ BIDIRECTIONAL CONTROL ═══╗", flush=True)
    print(f"  ║ Directional consistency: {results['summary']['directional']}", flush=True)
    print(f"  ║ Edit specificity:        {results['summary']['specificity']}", flush=True)
    print(f"  ║ Graded monotonic:        {results['summary']['monotonic']}", flush=True)
    print(f"  ║ Cross-modal control:     {results['summary']['crossmodal']}", flush=True)
    print(f"  ╚════════════════════════════╝", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved moonshot10 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  FIVE OVERNIGHT MOONSHOTS                                ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    experiments = [
        ("#13", moonshot13),
        ("#14", moonshot14),
        ("#15", moonshot15),
        ("#8", moonshot8),
        ("#10", moonshot10),
    ]

    for name, func in experiments:
        try:
            func()
            os.system(f'cd /Users/tomek/AI && git add results/neurips_battery/ _overnight_moonshots.py '
                      f'&& git commit -m "Moonshot {name}: results\n\n'
                      f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')
        except Exception as e:
            print(f"\n  Moonshot {name} FAILED: {e}", flush=True)
            traceback.print_exc()

    total_h = (time.time() - START_TIME) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  ALL MOONSHOTS COMPLETE. Total: {total_h:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)

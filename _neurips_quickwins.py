"""
NeurIPS Quick Wins: 4 experiments
Run: PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _neurips_quickwins.py
"""

import time, json, math, os, sys, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=scipy_stats.ConstantInputWarning)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HIDDEN_DIM = 128; VOCAB_SIZE = 3; N_HEADS = 2; N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE; N_POS = N_AGENTS * N_HEADS
COMM_EPOCHS = 600; BATCH_SIZE = 32; EARLY_STOP = 200
START_TIME = time.time(); TIMING = []
RD = Path("results/neurips_quickwins"); RD.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender, Receiver, compute_posdis
)


def elapsed(): return f"{(time.time()-START_TIME)/60:.0f}min"


def safe_mi(x, y):
    """MI with zero-variance check."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    xv, yv = np.unique(x), np.unique(y)
    if len(xv) < 2 or len(yv) < 2:
        return 0.0
    n = len(x); mi = 0.0
    for a in xv:
        for b in yv:
            pxy = np.sum((x == a) & (y == b)) / n
            px = np.sum(x == a) / n; py = np.sum(y == b) / n
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))
    return float(mi)


def safe_spearman(x, y):
    """Spearman with zero-variance check."""
    try:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        rho, _ = scipy_stats.spearmanr(x, y)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


def load_meta():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]),
                      np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return p1, p2, obj, mass


def load_feat(name):
    if name == "vjepa2":
        return torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)["features"].float()
    elif name == "dinov2":
        f = torch.load("results/phase87_phys101_spring_static.pt", weights_only=False)["features"].float()
        return f.unsqueeze(1).expand(-1, 8, -1).contiguous()
    elif name == "clip":
        f = torch.load("results/phase96_phys101_spring_clip.pt", weights_only=False)["features"].float()
        return f.unsqueeze(1).expand(-1, 8, -1).contiguous()
    elif name == "text":
        return torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()
    elif name == "audio":
        return torch.load("results/crossmodal/audio/audio_features.pt", weights_only=False)["features"].float()
    elif name == "dinov2_48":
        p = Path("results/phase88_dinov2_48frame_features.pt")
        if p.exists():
            return torch.load(p, weights_only=False)["features"].float()
    return None


def train_d(feat, mass, obj, seed):
    """Train discrete sender, return tokens + acc."""
    t0 = time.time()
    n, nf, dim = feat.shape; fpa = max(1, nf // N_AGENTS)
    views = [feat[:, (i*fpa)%nf:(i*fpa)%nf+fpa, :] for i in range(N_AGENTS)]
    torch.manual_seed(seed); np.random.seed(seed)
    ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
          for _ in range(N_AGENTS)]
    sender = DiscreteMultiSender(ss).to(DEVICE)
    recvs = [Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so = torch.optim.Adam(sender.parameters(), lr=1e-3)
    ros = [torch.optim.Adam(r.parameters(), lr=3e-3) for r in recvs]
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    rng = np.random.RandomState(seed * 1000 + 42)
    perm = rng.permutation(n); n_ho = max(10, n // 5)
    tr = perm[n_ho:]; tei = perm[:n_ho]
    nb = max(1, len(tr) // 32); me = math.log(VOCAB_SIZE)
    ba = 0; bst = None; bep = 0
    for ep in range(COMM_EPOCHS):
        if ep - bep > EARLY_STOP and ba > 0.55: break
        if ep > 0 and ep % 40 == 0:
            for i in range(3):
                recvs[i] = Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE)
                ros[i] = torch.optim.Adam(recvs[i].parameters(), lr=3e-3)
        sender.train(); [r.train() for r in recvs]
        tau = 3 + (1 - 3) * ep / max(1, COMM_EPOCHS - 1); hard = ep >= 30
        for _ in range(nb):
            ia = rng.choice(tr, 32); ib = rng.choice(tr, 32); s = ia == ib
            while s.any(): ib[s] = rng.choice(tr, s.sum()); s = ia == ib
            md = np.abs(mass[ia] - mass[ib]); k = md > 0.5
            if k.sum() < 4: continue
            ia, ib = ia[k], ib[k]
            va = [v[ia].to(DEVICE) for v in views]; vb = [v[ib].to(DEVICE) for v in views]
            lab = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = sender(va, tau, hard); mb, lb = sender(vb, tau, hard)
            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), lab) for r in recvs) / 3
            for lg in la + lb:
                lp = F.log_softmax(lg, -1); p = lp.exp().clamp(1e-8)
                ent = -(p * lp).sum(-1).mean()
                if ent / me < 0.1: loss = loss - 0.03 * ent
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
                    vh = [v[ia_h].to(DEVICE) for v in views]; wh = [v[ib_h].to(DEVICE) for v in views]
                    mah, _ = sender(vh); mbh, _ = sender(wh)
                    for r in recvs:
                        c += ((r(mah, mbh) > 0) == (mass_dev[ia_h] > mass_dev[ib_h])).sum().item()
                        t += len(ia_h)
                acc = c / max(t, 1)
                if acc > ba:
                    ba = acc; bep = ep
                    bst = {kk: vv.cpu().clone() for kk, vv in sender.state_dict().items()}
    if bst: sender.load_state_dict(bst)
    sender.eval()
    toks = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v2 = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            _, logits = sender(v2)
            toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
    TIMING.append(time.time() - t0)
    return np.concatenate(toks, 0), ba


def get_primary(tokens, p1, p2):
    """Get primary property per position via MI."""
    n_pos = tokens.shape[1]
    attrs = np.stack([p1, p2], axis=1)
    assignments = []
    for p in range(n_pos):
        mi_vals = [safe_mi(tokens[:, p], attrs[:, a]) for a in range(attrs.shape[1])]
        assignments.append(int(np.argmax(mi_vals)))
    return assignments


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Permutation null
# ═══════════════════════════════════════════════════════════════

def exp1_permutation_null():
    print(f"\n{'#'*60}\n# EXP 1: Permutation Null for Structural Agreement\n# {elapsed()}\n{'#'*60}", flush=True)

    p1, p2, obj, mass = load_meta()
    n = len(mass)
    n_perms = 1000

    pairs = [
        ("vjepa2", "text"),
        ("vjepa2", "dinov2"),
        ("dinov2", "text"),
    ]

    results = {}

    for mod_a, mod_b in pairs:
        print(f"\n  ── {mod_a} ↔ {mod_b} ──", flush=True)
        feat_a = load_feat(mod_a); feat_b = load_feat(mod_b)
        if feat_a is None or feat_b is None:
            print(f"    Skipped (missing features)", flush=True)
            continue

        # Train 5 senders per modality, get assignments
        a_assigns = []; b_assigns = []
        for seed in range(5):
            ta, _ = train_d(feat_a, mass, obj, seed)
            tb, _ = train_d(feat_b, mass, obj, seed + 100)
            a_assigns.append(get_primary(ta, p1, p2))
            b_assigns.append(get_primary(tb, p1, p2))
            torch.mps.empty_cache()

            if seed == 2 and mod_a == "vjepa2" and mod_b == "text":
                avg = np.mean(TIMING[-6:])
                total_remaining = (5 - seed - 1) * 2 + (len(pairs) - 1) * 10
                print(f"    Measured: {avg:.1f}s/seed, ~{total_remaining * avg / 60:.0f}min remaining for Exp1", flush=True)

        # Observed structural agreement
        observed_agrees = []
        for aa in a_assigns:
            for ab in b_assigns:
                match = sum(1 for x, y in zip(aa, ab) if x == y) / len(aa)
                observed_agrees.append(match)
        observed = float(np.mean(observed_agrees))

        # Permutation null: randomly shuffle property assignments
        rng = np.random.RandomState(42)
        n_props = 2  # mass and object
        null_agrees = []
        for _ in range(n_perms):
            # Random assignment: each position gets a random property
            rand_a = rng.randint(0, n_props, N_POS).tolist()
            rand_b = rng.randint(0, n_props, N_POS).tolist()
            match = sum(1 for x, y in zip(rand_a, rand_b) if x == y) / N_POS
            null_agrees.append(match)

        null_mean = float(np.mean(null_agrees))
        null_std = float(np.std(null_agrees))
        # p-value: fraction of null >= observed
        p_value = float(np.mean([na >= observed for na in null_agrees]))

        pair_key = f"{mod_a}_vs_{mod_b}"
        results[pair_key] = {
            "observed": f"{observed:.1%}",
            "null_mean": f"{null_mean:.1%}",
            "null_std": f"{null_std:.1%}",
            "p_value": f"{p_value:.4f}",
            "n_permutations": n_perms,
            "observed_m": observed,
            "null_m": null_mean,
            "significant": p_value < 0.05,
        }
        print(f"    Observed: {observed:.1%}, Null: {null_mean:.1%}±{null_std:.1%}, p={p_value:.4f}", flush=True)

    with open(RD / "permutation_null.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: RSA quantization confound
# ═══════════════════════════════════════════════════════════════

def exp2_rsa_quantization():
    print(f"\n{'#'*60}\n# EXP 2: RSA Quantization Confound Control\n# {elapsed()}\n{'#'*60}", flush=True)

    p1, p2, obj, mass = load_meta()
    n = len(mass)
    from sklearn.metrics.pairwise import cosine_distances

    mods = ["vjepa2", "dinov2", "clip", "text", "audio"]

    # Compute continuous RDMs
    cont_rdms = {}
    for mod in mods:
        feat = load_feat(mod)
        if feat is None: continue
        pooled = feat.mean(dim=1).numpy()
        try:
            rdm = cosine_distances(pooled)
            cont_rdms[mod] = rdm
        except Exception as e:
            print(f"  {mod}: RDM failed ({e})", flush=True)

    # Naive quantization: bin each dim to K=3 equal-width bins
    naive_rdms = {}
    for mod in mods:
        feat = load_feat(mod)
        if feat is None: continue
        pooled = feat.mean(dim=1).numpy()
        binned = np.zeros_like(pooled, dtype=int)
        for d in range(pooled.shape[1]):
            col = pooled[:, d]
            if np.std(col) < 1e-10:
                binned[:, d] = 0
            else:
                q = np.quantile(col, [0.33, 0.67])
                binned[:, d] = np.digitize(col, q)
        # Hamming RDM on binned
        rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d_val = np.sum(binned[i] != binned[j]) / binned.shape[1]
                rdm[i, j] = d_val; rdm[j, i] = d_val
        naive_rdms[mod] = rdm

    # WMCP trained RDMs (retrain quickly, 1 seed each)
    wmcp_rdms = {}
    for mod in mods:
        feat = load_feat(mod)
        if feat is None: continue
        tokens, _ = train_d(feat, mass, obj, 0)
        rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d_val = np.sum(tokens[i] != tokens[j])
                rdm[i, j] = d_val; rdm[j, i] = d_val
        wmcp_rdms[mod] = rdm
        torch.mps.empty_cache()

    # Compute RSA for each pair
    avail = [m for m in mods if m in cont_rdms and m in naive_rdms and m in wmcp_rdms]
    tri = np.triu_indices(n, k=1)

    results = {"pairs": {}}
    cont_rsas = []; naive_rsas = []; wmcp_rsas = []

    for i, mi in enumerate(avail):
        for j, mj in enumerate(avail):
            if i >= j: continue
            pair = f"{mi}_vs_{mj}"
            try:
                rsa_cont = safe_spearman(cont_rdms[mi][tri], cont_rdms[mj][tri])
                rsa_naive = safe_spearman(naive_rdms[mi][tri], naive_rdms[mj][tri])
                rsa_wmcp = safe_spearman(wmcp_rdms[mi][tri], wmcp_rdms[mj][tri])
            except Exception as e:
                print(f"  {pair}: failed ({e})", flush=True)
                continue

            results["pairs"][pair] = {
                "rsa_continuous": round(rsa_cont, 3),
                "rsa_naive_quantized": round(rsa_naive, 3),
                "rsa_wmcp_trained": round(rsa_wmcp, 3),
            }
            cont_rsas.append(rsa_cont)
            naive_rsas.append(rsa_naive)
            wmcp_rsas.append(rsa_wmcp)
            print(f"  {pair}: cont={rsa_cont:.3f} naive={rsa_naive:.3f} wmcp={rsa_wmcp:.3f}", flush=True)

    results["summary"] = {
        "mean_continuous": round(float(np.mean(cont_rsas)), 3) if cont_rsas else 0,
        "mean_naive_quantized": round(float(np.mean(naive_rsas)), 3) if naive_rsas else 0,
        "mean_wmcp_trained": round(float(np.mean(wmcp_rsas)), 3) if wmcp_rsas else 0,
    }

    if naive_rsas and wmcp_rsas:
        if np.mean(naive_rsas) > np.mean(wmcp_rsas) * 0.8:
            verdict = "CONFOUND: naive quantization achieves similar RSA to trained WMCP"
        else:
            verdict = "TRAINING DRIVES RSA: naive quantization is much lower than trained WMCP"
        results["verdict"] = verdict
        print(f"\n  {verdict}", flush=True)
    print(f"  Mean: cont={results['summary']['mean_continuous']:.3f} "
          f"naive={results['summary']['mean_naive_quantized']:.3f} "
          f"wmcp={results['summary']['mean_wmcp_trained']:.3f}", flush=True)

    with open(RD / "rsa_quantization_control.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Real text via diverse descriptions
# ═══════════════════════════════════════════════════════════════

def exp3_real_text():
    print(f"\n{'#'*60}\n# EXP 3: Real Text (Diverse Descriptions)\n# {elapsed()}\n{'#'*60}", flush=True)

    p1, p2, obj, mass = load_meta()
    n = len(mass)
    mass_q = np.quantile(mass, [0.2, 0.4, 0.6, 0.8])

    # Material mapping
    mat_map = {
        'cardboard': ('cardboard', 'box'), 'rubber': ('rubber', 'ball'),
        'metal': ('metal', 'weight'), 'wood': ('wooden', 'block'),
        'plastic': ('plastic', 'container'), 'foam': ('foam', 'cube'),
        'dough': ('dough', 'lump'), 'porcelain': ('porcelain', 'dish'),
        'm': ('metal', 'piece'), 'p': ('plastic', 'item'), 'w': ('wood', 'stick'),
    }

    # 15 diverse templates (not just slot-filling — varied structure)
    templates = [
        "A {wt} {mat} {shape} oscillates on a spring. It weighs about {m:.0f} grams.",
        "On a spring, there's a {shape} made of {mat}. {wt_cap} object, {m:.0f}g.",
        "Spring experiment: {m:.0f}g {mat} {shape}. The object is {wt}.",
        "{wt_cap} {mat} {shape} ({m:.0f}g) bouncing up and down on a spring mechanism.",
        "Observation: {shape} of {mat}, mass {m:.0f}g, attached to spring. Classified as {wt}.",
        "The {m:.0f}-gram {mat} {shape} compresses the spring. It's a {wt} object.",
        "A spring supports a {wt} {shape}. Material: {mat}. Recorded mass: {m:.0f}g.",
        "Physics demo: {mat} {shape} on spring, {wt}, measured at {m:.0f} grams.",
        "This {shape} is {wt} and made of {mat}. When placed on the spring it weighs {m:.0f}g.",
        "I see a {wt} {mat} {shape} going up and down. The scale reads {m:.0f}g.",
        "Spring-mass system with a {m:.0f}g {shape} constructed from {mat}. {wt_cap} category.",
        "The {mat} {shape} on this spring has a mass of {m:.0f}g. That makes it {wt}.",
        "Attached to the spring: one {wt} {mat} {shape}, approximately {m:.0f} grams.",
        "Test sample: {shape} ({mat}), {m:.0f}g. Spring oscillation shows {wt} behavior.",
        "A {shape} weighing {m:.0f}g hangs on the spring. It's {mat} and considered {wt}.",
    ]

    # Generate diverse descriptions (3 per scene, randomly selected templates)
    rng = np.random.RandomState(42)
    all_descriptions = []
    desc_file_lines = []

    for i in range(n):
        base = obj[i].split('_')[0].lower()
        mat, shape = mat_map.get(base, (base, 'object'))
        m = mass[i]
        if m < mass_q[0]: wt = "very light"
        elif m < mass_q[1]: wt = "light"
        elif m < mass_q[2]: wt = "medium-weight"
        elif m < mass_q[3]: wt = "heavy"
        else: wt = "very heavy"
        wt_cap = wt.capitalize()

        # Pick 3 random templates
        chosen = rng.choice(len(templates), 3, replace=False)
        scene_descs = []
        for tidx in chosen:
            desc = templates[tidx].format(wt=wt, wt_cap=wt_cap, mat=mat, shape=shape, m=m)
            scene_descs.append(desc)
        all_descriptions.append(scene_descs)
        desc_file_lines.append(f"Scene {i}: {scene_descs[0]}")

    # Save descriptions
    with open(RD / "natural_descriptions.txt", "w") as f:
        f.write("\n".join(desc_file_lines))

    # Extract TinyLlama features for the first description of each scene
    print("  Extracting TinyLlama features for natural descriptions...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    mdl = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32, output_hidden_states=True)
    mdl.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tl = mdl.config.num_hidden_layers // 2

    natural_feats = []
    for i in range(n):
        desc = all_descriptions[i][0]  # Use first description
        inp = tok(desc, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = mdl(**inp)
            feat = out.hidden_states[tl].mean(1).squeeze(0)
        natural_feats.append(feat)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n}", flush=True)

    natural_feats = torch.stack(natural_feats).unsqueeze(1).expand(-1, 8, -1).contiguous().float()
    del mdl, tok; torch.mps.empty_cache()
    print(f"  Natural text features: {natural_feats.shape}", flush=True)

    # Also extract features for ALL 3 descriptions per scene (for paraphrase consistency)
    # Skip for now — focus on structural agreement

    # Train natural text senders
    print("  Training natural text senders...", flush=True)
    nat_assigns = []; nat_accs = []
    for seed in range(5):
        tokens, acc = train_d(natural_feats, mass, obj, seed)
        nat_assigns.append(get_primary(tokens, p1, p2))
        nat_accs.append(acc)
        torch.mps.empty_cache()

        if seed == 2:
            avg = np.mean(TIMING[-3:])
            print(f"    Measured: {avg:.1f}s/seed", flush=True)

    # Train template text senders (using existing cached features)
    template_feats = load_feat("text")
    tmpl_assigns = []; tmpl_accs = []
    for seed in range(5):
        tokens, acc = train_d(template_feats, mass, obj, seed + 200)
        tmpl_assigns.append(get_primary(tokens, p1, p2))
        tmpl_accs.append(acc)
        torch.mps.empty_cache()

    # Train V-JEPA senders for comparison
    vfeat = load_feat("vjepa2")
    v_assigns = []
    for seed in range(5):
        tokens, _ = train_d(vfeat, mass, obj, seed)
        v_assigns.append(get_primary(tokens, p1, p2))
        torch.mps.empty_cache()

    # Structural agreements
    def xagree(a_list, b_list):
        ag = []
        for a in a_list:
            for b in b_list:
                ag.append(sum(1 for x, y in zip(a, b) if x == y) / len(a))
        return float(np.mean(ag))

    nat_v = xagree(nat_assigns, v_assigns)
    tmpl_v = xagree(tmpl_assigns, v_assigns)
    nat_tmpl = xagree(nat_assigns, tmpl_assigns)

    results = {
        "natural_vs_vjepa": f"{nat_v:.1%}",
        "template_vs_vjepa": f"{tmpl_v:.1%}",
        "natural_vs_template": f"{nat_tmpl:.1%}",
        "natural_accuracy": f"{np.mean(nat_accs):.1%}±{np.std(nat_accs):.1%}",
        "template_accuracy": f"{np.mean(tmpl_accs):.1%}±{np.std(tmpl_accs):.1%}",
        "nat_v_m": nat_v, "tmpl_v_m": tmpl_v,
        "n_templates_used": len(templates),
    }

    if nat_v > 0.8:
        verdict = "NATURAL TEXT MAINTAINS STRUCTURAL CONVERGENCE"
    elif nat_v > 0.6:
        verdict = f"PARTIAL — natural text at {nat_v:.0%} vs template at {tmpl_v:.0%}"
    else:
        verdict = "NATURAL TEXT BREAKS CONVERGENCE"
    results["verdict"] = verdict

    print(f"\n  Natural↔V-JEPA: {nat_v:.1%}", flush=True)
    print(f"  Template↔V-JEPA: {tmpl_v:.1%}", flush=True)
    print(f"  Natural↔Template: {nat_tmpl:.1%}", flush=True)
    print(f"  Verdict: {verdict}", flush=True)

    with open(RD / "real_text_descriptions.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Frame-matched cross-modal
# ═══════════════════════════════════════════════════════════════

def exp4_frame_matched():
    print(f"\n{'#'*60}\n# EXP 4: Frame-Matched Cross-Modal\n# {elapsed()}\n{'#'*60}", flush=True)

    p1, p2, obj, mass = load_meta()
    n = len(mass)

    # Load features
    vjepa_feat = load_feat("vjepa2")  # [206, 8, 1024]
    dinov2_feat = load_feat("dinov2")  # [206, 8, 384] (expanded from static)
    text_feat = load_feat("text")

    # Check for 48-frame DINOv2
    dinov2_48 = load_feat("dinov2_48")
    if dinov2_48 is not None:
        print(f"  DINOv2 48-frame features found: {dinov2_48.shape}", flush=True)
    else:
        print(f"  DINOv2 48-frame features NOT found — using standard", flush=True)
        # Check phase88 for alternative
        p88 = Path("results/phase88_dinov2_48frame_features.pt")
        if p88.exists():
            d88 = torch.load(p88, weights_only=False)
            dinov2_48 = d88["features"].float()
            print(f"  Found phase88: {dinov2_48.shape}", flush=True)
            # Subsample to 8 frames for fair comparison
            if dinov2_48.shape[1] > 8:
                idx = np.linspace(0, dinov2_48.shape[1] - 1, 8, dtype=int)
                dinov2_48 = dinov2_48[:, idx, :]
                print(f"  Subsampled to: {dinov2_48.shape}", flush=True)

    # Train senders for each condition
    conditions = {
        "vjepa2_8frame": vjepa_feat,
        "dinov2_static_expanded": dinov2_feat,
        "text": text_feat,
    }
    if dinov2_48 is not None:
        conditions["dinov2_48frame"] = dinov2_48

    all_assigns = {}
    all_accs = {}

    for cond_name, feat in conditions.items():
        if feat is None: continue
        print(f"\n  ── {cond_name}: {feat.shape} ──", flush=True)
        assigns = []; accs = []
        for seed in range(5):
            tokens, acc = train_d(feat, mass, obj, seed + (hash(cond_name) % 100))
            assigns.append(get_primary(tokens, p1, p2))
            accs.append(acc)
            torch.mps.empty_cache()
        all_assigns[cond_name] = assigns
        all_accs[cond_name] = accs
        print(f"    acc={np.mean(accs):.1%}", flush=True)

    # Compute agreements vs text
    def xagree(a, b):
        ag = []
        for x in a:
            for y in b:
                ag.append(sum(1 for i, j in zip(x, y) if i == j) / len(x))
        return float(np.mean(ag))

    results = {}
    text_assigns = all_assigns.get("text", [])
    for cond_name, assigns in all_assigns.items():
        if cond_name == "text": continue
        agree = xagree(assigns, text_assigns) if text_assigns else 0
        results[cond_name] = {
            "agreement_with_text": f"{agree:.1%}",
            "accuracy": f"{np.mean(all_accs[cond_name]):.1%}±{np.std(all_accs[cond_name]):.1%}",
            "agree_m": agree,
        }
        print(f"  {cond_name} ↔ text: {agree:.1%}", flush=True)

    with open(RD / "frame_matched_crossmodal.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  NEURIPS QUICK WINS: 4 Experiments                      ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    all_results = {}
    for name, func in [
        ("Permutation null", exp1_permutation_null),
        ("RSA quantization", exp2_rsa_quantization),
        ("Real text", exp3_real_text),
        ("Frame-matched", exp4_frame_matched),
    ]:
        try:
            all_results[name] = func()
        except Exception as e:
            print(f"\n  {name} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  | {'Experiment':<25s} | {'Key result':<35s} | {'Implication':<30s} |", flush=True)
    print(f"  |{'─'*27}|{'─'*37}|{'─'*32}|", flush=True)

    if "Permutation null" in all_results:
        r = all_results["Permutation null"]
        vt = r.get("vjepa2_vs_text", {})
        print(f"  | {'Permutation null':<25s} | obs={vt.get('observed','?')} null={vt.get('null_mean','?')} p={vt.get('p_value','?'):>4s} | {'Above chance' if vt.get('significant') else 'NOT significant':<30s} |", flush=True)

    if "RSA quantization" in all_results:
        r = all_results["RSA quantization"]
        s = r.get("summary", {})
        print(f"  | {'RSA quantization':<25s} | c={s.get('mean_continuous',0):.3f} n={s.get('mean_naive_quantized',0):.3f} w={s.get('mean_wmcp_trained',0):.3f} | {r.get('verdict','?')[:30]:<30s} |", flush=True)

    if "Real text" in all_results:
        r = all_results["Real text"]
        print(f"  | {'Real text':<25s} | nat={r.get('natural_vs_vjepa','?')} tmpl={r.get('template_vs_vjepa','?'):>5s}  | {r.get('verdict','?')[:30]:<30s} |", flush=True)

    if "Frame-matched" in all_results:
        r = all_results["Frame-matched"]
        vals = " ".join(f"{k}={v.get('agreement_with_text','?')}" for k, v in r.items() if isinstance(v, dict))
        print(f"  | {'Frame-matched':<25s} | {vals[:35]:<35s} | {'See values':<30s} |", flush=True)

    print(f"\n  Total time: {elapsed()}", flush=True)

    # Commit
    os.system('cd /Users/tomek/AI && git add results/neurips_quickwins/ _neurips_quickwins.py '
              '&& git commit -m "neurips quick wins: permutation null, RSA control, real text, frame-matched\n\n'
              'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"')

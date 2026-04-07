"""
Two Critical Fixes: Multi-property structural agreement + Causal receiver
Run: PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _critical_fixes.py
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy import stats as scipy_stats

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
HIDDEN_DIM = 128; VOCAB_SIZE = 3; N_HEADS = 2; N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE; N_POS = N_AGENTS * N_HEADS  # 8
COMM_EPOCHS = 600; BATCH_SIZE = 32; EARLY_STOP = 200
START_TIME = time.time(); TIMING = []

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender, Receiver
)

def elapsed(): return f"{(time.time()-START_TIME)/60:.0f}min"
def commit(name):
    os.system(f'cd /Users/tomek/AI && git add results/ _critical_fixes.py '
              f'&& git commit -m "{name}\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')

def load_spring():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    feat = d["features"].float(); obj = d["obj_names"]; mass = d["mass_values"]
    return feat, obj, mass

def build_multi_properties(obj_names, mass_values):
    """Build 4+ distinct property binnings."""
    n = len(mass_values)
    # 1. Mass bins
    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    # 2. Material category
    materials = [o.split('_')[0] for o in obj_names]
    mat_unique = sorted(set(materials))
    mat_idx = np.array([mat_unique.index(m) for m in materials])
    mat_bins = np.digitize(mat_idx, np.quantile(np.arange(len(mat_unique)), [0.2, 0.4, 0.6, 0.8]))
    # 3. Object instance (finer than material)
    uo = sorted(set(obj_names)); oi = {o: i for i, o in enumerate(uo)}
    obj_idx = np.array([oi[o] for o in obj_names])
    obj_bins = np.digitize(obj_idx, np.quantile(obj_idx, [0.2, 0.4, 0.6, 0.8]))
    # 4. Log-mass (nonlinear transform — tests if positions encode different mass scales)
    log_mass = np.log(mass_values + 1)
    log_bins = np.digitize(log_mass, np.quantile(log_mass, [0.2, 0.4, 0.6, 0.8]))

    props = np.stack([mass_bins, mat_bins, obj_bins, log_bins], axis=1)
    prop_names = ["mass", "material", "object_instance", "log_mass"]
    return props, prop_names


def mutual_information(x, y):
    xv, yv = np.unique(x), np.unique(y)
    n = len(x); mi = 0.0
    for a in xv:
        for b in yv:
            pxy = np.sum((x == a) & (y == b)) / n
            px = np.sum(x == a) / n; py = np.sum(y == b) / n
            if pxy > 0 and px > 0 and py > 0: mi += pxy * np.log(pxy / (px * py))
    return mi


def full_mi_matrix(tokens, properties, vocab_size):
    """Compute MI between every position and every property."""
    n_pos = tokens.shape[1]; n_props = properties.shape[1]
    mi = np.zeros((n_pos, n_props))
    for p in range(n_pos):
        for a in range(n_props):
            mi[p, a] = mutual_information(tokens[:, p], properties[:, a])
    return mi


def train_sender_recv(feat, mass, obj, seed, causal_dropout=False):
    """Train sender + receiver. If causal_dropout, mask all but one position during receiver training."""
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
            bs_actual = len(ia)
            va = [v[ia].to(DEVICE) for v in views]; vb = [v[ib].to(DEVICE) for v in views]
            lab = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = sender(va, tau, hard); mb, lb = sender(vb, tau, hard)

            if causal_dropout:
                # Mask all but one random position per sample
                mask = torch.zeros_like(ma)
                keep = torch.randint(0, N_POS, (bs_actual,))
                for i in range(bs_actual):
                    start = keep[i].item() * VOCAB_SIZE
                    mask[i, start:start + VOCAB_SIZE] = 1.0
                ma_masked = ma * mask
                mb_masked = mb * mask
                loss = sum(F.binary_cross_entropy_with_logits(r(ma_masked, mb_masked), lab) for r in recvs) / 3
            else:
                loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), lab) for r in recvs) / 3

            for lg in la + lb:
                lp = F.log_softmax(lg, -1); p = lp.exp().clamp(1e-8)
                ent = -(p * lp).sum(-1).mean()
                if ent / me < 0.1: loss = loss - 0.03 * ent
            if torch.isnan(loss): so.zero_grad(); [o.zero_grad() for o in ros]; continue
            so.zero_grad(); [o.zero_grad() for o in ros]; loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            so.step(); [o.step() for o in ros]
        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep + 1) % 50 == 0 or ep == 0:
            sender.eval(); [r.eval() for r in recvs]
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(tei, min(32, len(tei))); ib_h = er.choice(tei, min(32, len(tei)))
                    vh = [v[ia_h].to(DEVICE) for v in views]; wh = [v[ib_h].to(DEVICE) for v in views]
                    mah, _ = sender(vh); mbh, _ = sender(wh)
                    # Evaluate WITHOUT dropout
                    for r in recvs:
                        c += ((r(mah, mbh) > 0) == (mass_dev[ia_h] > mass_dev[ib_h])).sum().item()
                        t += len(ia_h)
                acc = c / max(t, 1)
                if acc > ba:
                    ba = acc; bep = ep
                    bst = {kk: vv.cpu().clone() for kk, vv in sender.state_dict().items()}
    if bst: sender.load_state_dict(bst)
    sender.eval(); [r.eval() for r in recvs]
    # Extract tokens
    toks = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v2 = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            _, logits = sender(v2)
            toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
    tokens = np.concatenate(toks, 0)
    # Messages
    msgs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v2 = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m, _ = sender(v2); msgs.append(m.cpu())
    msgs = torch.cat(msgs, 0)
    TIMING.append(time.time() - t0)
    return sender, recvs[0], tokens, msgs, ba, views


def compute_iia(sender, recv, views, mass, n):
    """Interchange Intervention Accuracy — properly computed per position."""
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    sender.eval(); recv.eval()
    # Get all messages
    all_msgs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m, _ = sender(v); all_msgs.append(m.cpu())
    all_msgs = torch.cat(all_msgs, 0)

    iia_per_pos = []
    for pos in range(N_POS):
        correct = total = 0
        rng = np.random.RandomState(pos * 100 + 42)
        for _ in range(200):
            # Pick scene A (high mass) and B (low mass)
            ia = rng.randint(0, n); ib = rng.randint(0, n)
            if abs(mass[ia] - mass[ib]) < 1.0: continue

            msg_a = all_msgs[ia:ia+1].to(DEVICE)
            msg_b = all_msgs[ib:ib+1].to(DEVICE)
            # Anchor scene
            ic = rng.randint(0, n)
            msg_c = all_msgs[ic:ic+1].to(DEVICE)

            with torch.no_grad():
                pred_orig = torch.sigmoid(recv(msg_a, msg_c)).item()
                # Patch position pos of A with B's value
                msg_patch = msg_a.clone()
                start = pos * VOCAB_SIZE
                msg_patch[0, start:start+VOCAB_SIZE] = msg_b[0, start:start+VOCAB_SIZE]
                pred_patch = torch.sigmoid(recv(msg_patch, msg_c)).item()

            # IIA: did the prediction shift toward what B would produce?
            pred_b_orig = torch.sigmoid(recv(msg_b, msg_c)).item()
            # Check if patch moves prediction closer to B
            dist_before = abs(pred_orig - pred_b_orig)
            dist_after = abs(pred_patch - pred_b_orig)
            if dist_after < dist_before:
                correct += 1
            total += 1

        iia_per_pos.append(correct / max(total, 1))
    return float(np.mean(iia_per_pos)), iia_per_pos


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Multi-property structural agreement
# ═══════════════════════════════════════════════════════════════

def exp1_multi_property():
    print(f"\n{'#'*60}\n# EXP 1: Multi-Property Structural Agreement\n# {elapsed()}\n{'#'*60}", flush=True)
    d = Path("results/crossmodal/multi_property"); d.mkdir(parents=True, exist_ok=True)

    vfeat, obj, mass = load_spring()
    tfeat = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()
    n = len(mass)

    props, prop_names = build_multi_properties(obj, mass)
    n_props = len(prop_names)
    print(f"  Properties: {prop_names} ({n_props} total)", flush=True)
    print(f"  Message positions: {N_POS}", flush=True)

    results = {"prop_names": prop_names, "n_positions": N_POS, "seeds": []}

    for seed in range(5):
        # Train vision sender
        v_sender, _, v_tokens, _, v_acc, _ = train_sender_recv(vfeat, mass, obj, seed)
        # Train text sender
        t_sender, _, t_tokens, _, t_acc, _ = train_sender_recv(tfeat, mass, obj, seed + 100)

        # Full MI matrices
        v_mi = full_mi_matrix(v_tokens, props, VOCAB_SIZE)
        t_mi = full_mi_matrix(t_tokens, props, VOCAB_SIZE)

        # Primary property per position
        v_primary = [int(np.argmax(v_mi[p])) for p in range(N_POS)]
        t_primary = [int(np.argmax(t_mi[p])) for p in range(N_POS)]

        # Simple agreement: same primary property at each position
        simple_agree = sum(1 for a, b in zip(v_primary, t_primary) if a == b) / N_POS

        # Hungarian structural agreement on MI profiles
        # Cost = negative correlation between MI profiles at matched positions
        cost = np.zeros((N_POS, N_POS))
        for i in range(N_POS):
            for j in range(N_POS):
                cost[i, j] = -scipy_stats.spearmanr(v_mi[i], t_mi[j])[0]
        ri, ci = linear_sum_assignment(cost)
        hung_agree = sum(1 for r, c in zip(ri, ci) if v_primary[r] == t_primary[c]) / N_POS
        # Mean correlation of matched pairs
        matched_corr = np.mean([-cost[r, c] for r, c in zip(ri, ci)])

        # Chance baseline for N_POS positions and n_props properties
        # If each position randomly picks one of n_props, agreement = n_props^(-1) for random
        chance = 1.0 / n_props

        seed_result = {
            "seed": seed, "v_acc": float(v_acc), "t_acc": float(t_acc),
            "v_primary": v_primary, "t_primary": t_primary,
            "simple_agreement": float(simple_agree),
            "hungarian_agreement": float(hung_agree),
            "hungarian_mi_correlation": float(matched_corr),
            "v_mi_matrix": v_mi.tolist(),
            "t_mi_matrix": t_mi.tolist(),
            "chance_baseline": float(chance),
        }
        results["seeds"].append(seed_result)

        if seed == 2:
            avg = np.mean(TIMING[-6:])
            remaining = 2 * 2  # 2 more seeds × 2 modalities
            print(f"    Measured: {avg:.1f}s/seed, ~{remaining * avg / 60:.0f}min remaining", flush=True)

        print(f"    Seed {seed}: v_acc={v_acc:.1%} t_acc={t_acc:.1%}", flush=True)
        print(f"      V primary: {[prop_names[p] for p in v_primary]}", flush=True)
        print(f"      T primary: {[prop_names[p] for p in t_primary]}", flush=True)
        print(f"      Simple agree: {simple_agree:.1%} (chance={chance:.1%})", flush=True)
        print(f"      Hungarian agree: {hung_agree:.1%}, MI corr: {matched_corr:.3f}", flush=True)
        torch.mps.empty_cache()

    # Summary
    simple_agrees = [s["simple_agreement"] for s in results["seeds"]]
    hung_agrees = [s["hungarian_agreement"] for s in results["seeds"]]
    corrs = [s["hungarian_mi_correlation"] for s in results["seeds"]]
    results["summary"] = {
        "simple_agreement": f"{np.mean(simple_agrees):.1%}±{np.std(simple_agrees):.1%}",
        "hungarian_agreement": f"{np.mean(hung_agrees):.1%}±{np.std(hung_agrees):.1%}",
        "mi_correlation": f"{np.mean(corrs):.3f}±{np.std(corrs):.3f}",
        "chance": f"{results['seeds'][0]['chance_baseline']:.1%}",
    }
    print(f"\n  SUMMARY:", flush=True)
    print(f"    Simple agreement: {results['summary']['simple_agreement']} (chance={results['summary']['chance']})", flush=True)
    print(f"    Hungarian agreement: {results['summary']['hungarian_agreement']}", flush=True)
    print(f"    MI profile correlation: {results['summary']['mi_correlation']}", flush=True)

    # Plot MI matrices for one seed
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Position×Property MI Matrices: Vision vs Text", fontweight='bold')
    for ax, mi_mat, title in [(axes[0], np.array(results["seeds"][0]["v_mi_matrix"]), "V-JEPA 2"),
                               (axes[1], np.array(results["seeds"][0]["t_mi_matrix"]), "TinyLlama")]:
        im = ax.imshow(mi_mat, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n_props)); ax.set_xticklabels(prop_names, rotation=30)
        ax.set_yticks(range(N_POS)); ax.set_yticklabels([f"pos {i}" for i in range(N_POS)])
        ax.set_title(title)
        for i in range(N_POS):
            for j in range(n_props):
                ax.text(j, i, f"{mi_mat[i, j]:.2f}", ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(d / "mi_matrices.png", dpi=200, bbox_inches='tight'); plt.close()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp1 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Causal receiver with positional dropout
# ═══════════════════════════════════════════════════════════════

def exp2_causal_receiver():
    print(f"\n{'#'*60}\n# EXP 2: Causal Receiver with Positional Dropout\n# {elapsed()}\n{'#'*60}", flush=True)
    d = Path("results/crossmodal/causal_receiver"); d.mkdir(parents=True, exist_ok=True)

    vfeat, obj, mass = load_spring()
    tfeat = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()
    n = len(mass)
    props, prop_names = build_multi_properties(obj, mass)

    results = {"conditions": {}}

    for cond_name, use_dropout in [("standard", False), ("causal_dropout", True)]:
        print(f"\n  ── {cond_name} ──", flush=True)
        v_results = []; t_results = []

        for seed in range(10):
            # Vision
            v_sender, v_recv, v_tokens, v_msgs, v_acc, v_views = train_sender_recv(
                vfeat, mass, obj, seed, causal_dropout=use_dropout)
            # Text
            t_sender, t_recv, t_tokens, t_msgs, t_acc, t_views = train_sender_recv(
                tfeat, mass, obj, seed + 100, causal_dropout=use_dropout)

            # MI and PosDis
            v_mi = full_mi_matrix(v_tokens, props, VOCAB_SIZE)
            t_mi = full_mi_matrix(t_tokens, props, VOCAB_SIZE)
            v_primary = [int(np.argmax(v_mi[p])) for p in range(N_POS)]
            t_primary = [int(np.argmax(t_mi[p])) for p in range(N_POS)]

            # Structural agreement
            struct_agree = sum(1 for a, b in zip(v_primary, t_primary) if a == b) / N_POS

            # IIA
            v_iia, v_iia_per_pos = compute_iia(v_sender, v_recv, v_views, mass, n)

            # PosDis (using mass + material as attributes)
            from _fix_exp3_exp4 import compute_posdis
            pd_attrs = props[:, :2]  # mass + material
            v_pd, _ = compute_posdis(v_tokens, pd_attrs, VOCAB_SIZE)

            v_results.append({
                "acc": float(v_acc), "iia": float(v_iia), "posdis": float(v_pd),
                "struct_agree": float(struct_agree),
                "v_primary": v_primary, "t_primary": t_primary,
            })

            if seed == 2:
                avg = np.mean(TIMING[-6:])
                remaining = (10 - seed - 1) * 2 + (1 if cond_name == "standard" else 0) * 20
                print(f"    Measured: {avg:.1f}s/seed, ~{remaining * avg / 60:.0f}min remaining", flush=True)

            print(f"    Seed {seed}: acc={v_acc:.1%} IIA={v_iia:.3f} PD={v_pd:.3f} "
                  f"struct={struct_agree:.1%}", flush=True)
            torch.mps.empty_cache()

        accs = [r["acc"] for r in v_results]
        iias = [r["iia"] for r in v_results]
        pds = [r["posdis"] for r in v_results]
        structs = [r["struct_agree"] for r in v_results]

        results["conditions"][cond_name] = {
            "accuracy": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
            "iia": f"{np.mean(iias):.3f}±{np.std(iias):.3f}",
            "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
            "structural_agreement": f"{np.mean(structs):.1%}±{np.std(structs):.1%}",
            "acc_m": float(np.mean(accs)),
            "iia_m": float(np.mean(iias)),
            "pd_m": float(np.mean(pds)),
            "struct_m": float(np.mean(structs)),
        }
        print(f"\n  {cond_name}: acc={np.mean(accs):.1%} IIA={np.mean(iias):.3f} "
              f"PD={np.mean(pds):.3f} struct={np.mean(structs):.1%}", flush=True)

    # Comparison
    std = results["conditions"]["standard"]
    drp = results["conditions"]["causal_dropout"]
    print(f"\n  ╔═══ CAUSAL RECEIVER COMPARISON ═══╗", flush=True)
    print(f"  ║ {'':>20s} │ {'Standard':>10s} │ {'Dropout':>10s}", flush=True)
    print(f"  ║ {'Accuracy':>20s} │ {std['accuracy']:>10s} │ {drp['accuracy']:>10s}", flush=True)
    print(f"  ║ {'IIA':>20s} │ {std['iia']:>10s} │ {drp['iia']:>10s}", flush=True)
    print(f"  ║ {'PosDis':>20s} │ {std['posdis']:>10s} │ {drp['posdis']:>10s}", flush=True)
    print(f"  ║ {'Structural agree':>20s} │ {std['structural_agreement']:>10s} │ {drp['structural_agreement']:>10s}", flush=True)
    print(f"  ╚═══════════════════════════════════╝", flush=True)

    if drp["iia_m"] > std["iia_m"] + 0.1 and drp["struct_m"] > 0.5:
        verdict = "CAUSAL DROPOUT WORKS — IIA increases while structural agreement holds"
    elif drp["iia_m"] > std["iia_m"] + 0.1:
        verdict = "IIA IMPROVES but structural agreement drops — tradeoff"
    else:
        verdict = "NO SIGNIFICANT IIA IMPROVEMENT from dropout"
    results["verdict"] = verdict
    print(f"  Verdict: {verdict}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp2 ({elapsed()})", flush=True)
    return results


# ═══ Main ═══

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  TWO CRITICAL FIXES                                     ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    for name, func in [("Multi-property structural agreement", exp1_multi_property),
                        ("Causal receiver with dropout", exp2_causal_receiver)]:
        try:
            func()
            commit(name)
        except Exception as e:
            print(f"\n  {name} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    total_h = (time.time() - START_TIME) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  COMPLETE. Total: {total_h:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)

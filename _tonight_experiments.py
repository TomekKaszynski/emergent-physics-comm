"""
Tonight's Experiments: LLM Reading, Faithfulness, Stability Ablation
=====================================================================
Exp5: LLM reading WMCP tokens (needs API key — prepares everything, runs if key available)
Exp4: Interchange intervention faithfulness
Exp6: Stability ablation (overnight)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _tonight_experiments.py
"""

import time, json, math, os, sys
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
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE
COMM_EPOCHS = 600
BATCH_SIZE = 32
EARLY_STOP = 200

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender, Receiver,
    compute_posdis, compute_topsim
)

TIMING = []


def load_spring_vjepa():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    feat = d["features"].float()
    obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]),
                      np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return feat, p1, p2, obj, mass


def train_sender_receiver(feat, mass, obj_names, seed):
    n, nf, dim = feat.shape; fpa = nf // N_AGENTS
    views = [feat[:, i*fpa:(i+1)*fpa, :] for i in range(N_AGENTS)]
    torch.manual_seed(seed); np.random.seed(seed)
    ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, VOCAB_SIZE, N_HEADS) for _ in range(N_AGENTS)]
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
    best_acc, best_st, best_ep = 0.0, None, 0

    for ep in range(COMM_EPOCHS):
        if ep - best_ep > EARLY_STOP and best_acc > 0.55: break
        if ep > 0 and ep % 40 == 0:
            for i in range(3): recvs[i] = Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE); ros[i] = torch.optim.Adam(recvs[i].parameters(), lr=3e-3)
        sender.train(); [r.train() for r in recvs]
        tau = 3+(1-3)*ep/max(1, COMM_EPOCHS-1); hard = ep >= 30
        for _ in range(nb):
            ia = rng.choice(tr, 32); ib = rng.choice(tr, 32); s = ia == ib
            while s.any(): ib[s] = rng.choice(tr, s.sum()); s = ia == ib
            md = np.abs(mass[ia]-mass[ib]); k = md > 0.5
            if k.sum() < 4: continue
            ia, ib = ia[k], ib[k]
            va = [v[ia].to(DEVICE) for v in views]; vb = [v[ib].to(DEVICE) for v in views]
            lab = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = sender(va, tau, hard); mb, lb = sender(vb, tau, hard)
            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), lab) for r in recvs) / 3
            for lg in la + lb:
                lp = F.log_softmax(lg, -1); p = lp.exp().clamp(1e-8); ent = -(p*lp).sum(-1).mean()
                if ent/me < 0.1: loss = loss - 0.03*ent
            if torch.isnan(loss): so.zero_grad(); [o.zero_grad() for o in ros]; continue
            so.zero_grad(); [o.zero_grad() for o in ros]; loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0); so.step(); [o.step() for o in ros]
        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep+1) % 50 == 0 or ep == 0:
            sender.eval(); [r.eval() for r in recvs]
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(tei, min(32, len(tei))); ib_h = er.choice(tei, min(32, len(tei)))
                    vh = [v[ia_h].to(DEVICE) for v in views]; wh = [v[ib_h].to(DEVICE) for v in views]
                    mah, _ = sender(vh); mbh, _ = sender(wh)
                    for r in recvs: c += ((r(mah, mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item(); t += len(ia_h)
                acc = c/max(t, 1)
                if acc > best_acc: best_acc = acc; best_ep = ep; best_st = {kk: vv.cpu().clone() for kk, vv in sender.state_dict().items()}
    if best_st: sender.load_state_dict(best_st)
    sender.eval(); [r.eval() for r in recvs]
    return sender, recvs[0], views, tr, tei, best_acc


def extract_tokens(sender, views, n):
    toks = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            _, logits = sender(v)
            toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
    return np.concatenate(toks, 0)


def extract_messages(sender, views, n):
    msgs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m, _ = sender(v); msgs.append(m.cpu())
    return torch.cat(msgs, 0)


# ═══════════════════════════════════════════════════════════════
# EXP 5: LLM Reading WMCP Tokens
# ═══════════════════════════════════════════════════════════════

def exp5_llm_reading():
    print(f"\n{'#'*60}\n# EXP 5: LLM Reading WMCP Tokens\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "exp5_llm_reading"; d.mkdir(exist_ok=True)

    feat, p1, p2, obj, mass = load_spring_vjepa()
    n = len(mass)

    # Train a sender and extract mapping
    sender, recv, views, tr, tei, acc = train_sender_receiver(feat, mass, obj, seed=0)
    tokens = extract_tokens(sender, views, n)
    print(f"  Trained sender: acc={acc:.1%}", flush=True)

    # Build mapping table
    n_pos = tokens.shape[1]
    mass_bins = np.digitize(mass, np.quantile(mass, [0.33, 0.67]))
    mass_labels = {0: "light", 1: "medium", 2: "heavy"}

    uo = sorted(set(obj))
    obj_idx = np.array([uo.index(o) for o in obj])
    obj_bins = np.digitize(obj_idx, np.quantile(obj_idx, [0.33, 0.67]))
    obj_labels = {0: "group_A", 1: "group_B", 2: "group_C"}

    # Find which position encodes mass vs object
    from _fix_exp3_exp4 import compute_posdis
    attrs = np.stack([p1, p2], axis=1)
    _, mi_mat = compute_posdis(tokens, attrs, VOCAB_SIZE)

    mapping_table = []
    for pos in range(n_pos):
        best_attr = int(np.argmax(mi_mat[pos]))
        attr_name = "mass" if best_attr == 0 else "object_type"
        # Find what each token value means for this position
        token_meanings = {}
        bins = mass_bins if best_attr == 0 else obj_bins
        labels = mass_labels if best_attr == 0 else obj_labels
        for tv in range(VOCAB_SIZE):
            mask = tokens[:, pos] == tv
            if mask.sum() > 0:
                most_common_bin = int(np.bincount(bins[mask]).argmax())
                token_meanings[tv] = labels[most_common_bin]
            else:
                token_meanings[tv] = "unused"
        mapping_table.append({
            "position": pos, "property": attr_name,
            "mi": float(mi_mat[pos, best_attr]),
            "token_meanings": token_meanings
        })

    mapping_str = "WMCP Protocol Mapping:\n"
    for m in mapping_table:
        mapping_str += f"  Position {m['position']}: {m['property']} — "
        mapping_str += ", ".join(f"token {k}={v}" for k, v in m['token_meanings'].items())
        mapping_str += "\n"
    print(f"\n{mapping_str}", flush=True)

    # Generate test cases
    test_cases = []
    rng = np.random.RandomState(42)
    for _ in range(100):
        ia, ib = rng.choice(n, 2, replace=False)
        if abs(mass[ia] - mass[ib]) < 1.0: continue
        heavier = "A" if mass[ia] > mass[ib] else "B"
        tokens_a = tokens[ia].tolist()
        tokens_b = tokens[ib].tolist()
        test_cases.append({
            "scene_a_tokens": tokens_a,
            "scene_b_tokens": tokens_b,
            "question": "Which object is heavier, A or B?",
            "answer": heavier,
            "mass_a": float(mass[ia]),
            "mass_b": float(mass[ib]),
        })

    print(f"  Generated {len(test_cases)} test cases", flush=True)

    # Save mapping for reference
    with open(d / "mapping_table.json", "w") as f:
        json.dump({"mapping": mapping_table, "mapping_str": mapping_str}, f, indent=2)

    # Rule-based baseline
    mass_pos = max(range(n_pos), key=lambda p: mi_mat[p, 0])
    correct_rule = 0
    for tc in test_cases:
        ta = tc["scene_a_tokens"][mass_pos]
        tb = tc["scene_b_tokens"][mass_pos]
        pred = "A" if ta > tb else "B" if tb > ta else "A"
        if pred == tc["answer"]: correct_rule += 1
    rule_acc = correct_rule / len(test_cases)
    print(f"  Rule-based reading (mass position {mass_pos}): {rule_acc:.1%}", flush=True)

    # Use claude CLI (echo "prompt" | claude --print)
    import subprocess

    def ask_claude(system_msg, user_msg):
        prompt = f"{system_msg}\n\n{user_msg}"
        try:
            result = subprocess.run(
                ["claude", "--print"],
                input=prompt, capture_output=True, text=True, timeout=30)
            return result.stdout.strip()
        except Exception as e:
            return f"ERROR: {e}"

    # Check claude CLI works
    test_resp = ask_claude("Reply with OK", "Test")
    claude_available = "ERROR" not in test_resp and len(test_resp) > 0
    print(f"  Claude CLI: {'available' if claude_available else 'NOT available'} (test: {test_resp[:50]})", flush=True)

    results = {"rule_based_accuracy": float(rule_acc), "conditions": {}}

    if claude_available:
        conditions = [
            ("real_mapping_real_tokens", mapping_str, lambda tc: tc),
            ("shuffled_mapping",
             mapping_str.replace("mass", "color").replace("object_type", "temperature"),
             lambda tc: tc),
            ("real_mapping_random_tokens", mapping_str,
             lambda tc: {**tc, "scene_a_tokens": rng.randint(0, 3, n_pos).tolist(),
                         "scene_b_tokens": rng.randint(0, 3, n_pos).tolist()}),
        ]

        for cond_name, cond_mapping, cond_tokens_fn in conditions:
            correct = 0; total = 0
            for tc in test_cases[:30]:  # 30 cases per condition to limit API usage
                tc_mod = cond_tokens_fn(tc)
                system_msg = (f"You read a physics sensor protocol. {cond_mapping}\n"
                              f"Answer with ONLY the letter A or B. Nothing else.")
                user_msg = (f"Scene A tokens: {tc_mod['scene_a_tokens']}\n"
                            f"Scene B tokens: {tc_mod['scene_b_tokens']}\n"
                            f"Question: {tc['question']}")
                answer = ask_claude(system_msg, user_msg)
                # Parse answer
                answer_clean = answer.strip().upper()
                if "A" in answer_clean and "B" not in answer_clean:
                    answer_clean = "A"
                elif "B" in answer_clean and "A" not in answer_clean:
                    answer_clean = "B"
                elif answer_clean.startswith("A"):
                    answer_clean = "A"
                elif answer_clean.startswith("B"):
                    answer_clean = "B"
                else:
                    answer_clean = "?"

                if answer_clean == tc["answer"]:
                    correct += 1
                total += 1

                if total % 10 == 0:
                    print(f"    {cond_name}: {total}/30, acc={correct/total:.1%}", flush=True)

            results["conditions"][cond_name] = {
                "accuracy": float(correct / max(total, 1)), "n": total}
            print(f"  {cond_name}: {correct/max(total,1):.1%} ({total} cases)", flush=True)

        # BONUS: Zero-shot protocol inference
        print(f"\n  ── Zero-shot protocol inference ──", flush=True)
        examples = []
        for tc in test_cases[:20]:
            examples.append(f"Tokens A: {tc['scene_a_tokens']}, Tokens B: {tc['scene_b_tokens']} → Answer: {tc['answer']}")
        example_str = "\n".join(examples)

        correct_zs = 0; total_zs = 0
        for tc in test_cases[20:50]:
            system_msg = (f"You are analyzing a physics sensor protocol. "
                          f"Here are examples of token sequences and which object is heavier:\n{example_str}\n\n"
                          f"Infer the pattern and answer with ONLY A or B.")
            user_msg = f"Tokens A: {tc['scene_a_tokens']}, Tokens B: {tc['scene_b_tokens']} → Answer:"
            answer = ask_claude(system_msg, user_msg)
            answer_clean = answer.strip().upper()
            if answer_clean.startswith("A"): answer_clean = "A"
            elif answer_clean.startswith("B"): answer_clean = "B"
            if answer_clean == tc["answer"]: correct_zs += 1
            total_zs += 1
            if total_zs % 10 == 0:
                print(f"    Zero-shot: {total_zs}/30, acc={correct_zs/total_zs:.1%}", flush=True)

        results["zero_shot_inference"] = {
            "accuracy": float(correct_zs / max(total_zs, 1)), "n": total_zs}
        print(f"  Zero-shot inference: {correct_zs/max(total_zs,1):.1%}", flush=True)
    else:
        print("  ⚠ Claude CLI not available — using rule-based only", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp5", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXP 4: Interchange Intervention Faithfulness
# ═══════════════════════════════════════════════════════════════

def exp4_faithfulness():
    print(f"\n{'#'*60}\n# EXP 4: Interchange Intervention Faithfulness\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "exp4_faithfulness"; d.mkdir(exist_ok=True)

    feat, p1, p2, obj, mass = load_spring_vjepa()
    n = len(mass)
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)

    results = {"seeds": []}

    for seed in range(10):
        t0 = time.time()
        sender, recv, views, tr, tei, acc = train_sender_receiver(feat, mass, obj, seed)
        tokens = extract_tokens(sender, views, n)
        msgs = extract_messages(sender, views, n)

        # Find mass position (highest MI with mass)
        attrs = np.stack([p1, p2], axis=1)
        _, mi_mat = compute_posdis(tokens, attrs, VOCAB_SIZE)
        mass_pos = int(np.argmax([mi_mat[p, 0] for p in range(tokens.shape[1])]))
        nonmass_pos = int(np.argmin([mi_mat[p, 0] for p in range(tokens.shape[1])]))

        # Construct controlled pairs: high mass A, low mass B
        high_mass = np.where(p1 >= 3)[0]  # top 40%
        low_mass = np.where(p1 <= 1)[0]   # bottom 40%

        iia_scores = []
        specificity_scores = []
        necessity_mass_drops = []
        necessity_other_drops = []

        rng = np.random.RandomState(seed * 100 + 77)
        n_pairs = min(200, len(high_mass) * len(low_mass))

        for _ in range(n_pairs):
            ia = rng.choice(high_mass)
            ib = rng.choice(low_mass)
            if abs(mass[ia] - mass[ib]) < 1.0: continue

            msg_a = msgs[ia:ia+1].to(DEVICE)
            msg_b = msgs[ib:ib+1].to(DEVICE)

            # Original predictions
            with torch.no_grad():
                pred_a = torch.sigmoid(recv(msg_a, msg_b)).item()  # P(A > B)
                pred_b = torch.sigmoid(recv(msg_b, msg_a)).item()  # P(B > A)

            # INTERCHANGE: patch mass position of A with B's token
            msg_patch = msg_a.clone()
            start = mass_pos * VOCAB_SIZE
            msg_patch[0, start:start+VOCAB_SIZE] = msg_b[0, start:start+VOCAB_SIZE]

            with torch.no_grad():
                pred_patch = torch.sigmoid(recv(msg_patch, msg_b)).item()

            # IIA: does patching mass make prediction shift toward B?
            # A has high mass, so pred_a should be high (A>B).
            # After patching A's mass with B's (low), prediction should drop.
            iia = 1.0 if pred_patch < pred_a - 0.1 else 0.0
            iia_scores.append(iia)

            # SPECIFICITY: patch a NON-mass position instead
            msg_spec = msg_a.clone()
            start_nm = nonmass_pos * VOCAB_SIZE
            msg_spec[0, start_nm:start_nm+VOCAB_SIZE] = msg_b[0, start_nm:start_nm+VOCAB_SIZE]
            with torch.no_grad():
                pred_spec = torch.sigmoid(recv(msg_spec, msg_b)).item()
            # Prediction should NOT change much
            specificity_scores.append(1.0 if abs(pred_spec - pred_a) < 0.15 else 0.0)

            # NECESSITY: ablate mass position
            msg_abl = msg_a.clone()
            msg_abl[0, start:start+VOCAB_SIZE] = 0
            with torch.no_grad():
                pred_abl = torch.sigmoid(recv(msg_abl, msg_b)).item()
            necessity_mass_drops.append(abs(pred_a - pred_abl))

            # Ablate non-mass position
            msg_abl2 = msg_a.clone()
            msg_abl2[0, start_nm:start_nm+VOCAB_SIZE] = 0
            with torch.no_grad():
                pred_abl2 = torch.sigmoid(recv(msg_abl2, msg_b)).item()
            necessity_other_drops.append(abs(pred_a - pred_abl2))

        elapsed = time.time() - t0
        TIMING.append(elapsed)

        seed_result = {
            "seed": seed, "accuracy": float(acc),
            "mass_position": mass_pos,
            "iia": float(np.mean(iia_scores)),
            "specificity": float(np.mean(specificity_scores)),
            "necessity_mass_drop": float(np.mean(necessity_mass_drops)),
            "necessity_other_drop": float(np.mean(necessity_other_drops)),
            "n_pairs": len(iia_scores),
        }
        results["seeds"].append(seed_result)

        if seed == 2:
            avg_t = np.mean(TIMING[-3:])
            remaining = 7 * avg_t
            print(f"    Measured: {avg_t:.1f}s/seed, ~{remaining/60:.0f}min remaining", flush=True)

        print(f"    Seed {seed}: IIA={seed_result['iia']:.3f} "
              f"Spec={seed_result['specificity']:.3f} "
              f"MassDrop={seed_result['necessity_mass_drop']:.3f} "
              f"OtherDrop={seed_result['necessity_other_drop']:.3f}", flush=True)
        torch.mps.empty_cache()

    # Summary
    iias = [s["iia"] for s in results["seeds"]]
    specs = [s["specificity"] for s in results["seeds"]]
    m_drops = [s["necessity_mass_drop"] for s in results["seeds"]]
    o_drops = [s["necessity_other_drop"] for s in results["seeds"]]

    results["summary"] = {
        "iia": f"{np.mean(iias):.3f}±{np.std(iias):.3f}",
        "specificity": f"{np.mean(specs):.3f}±{np.std(specs):.3f}",
        "necessity_mass_drop": f"{np.mean(m_drops):.3f}±{np.std(m_drops):.3f}",
        "necessity_other_drop": f"{np.mean(o_drops):.3f}±{np.std(o_drops):.3f}",
    }

    print(f"\n  ╔═══ FAITHFULNESS RESULTS ═══╗", flush=True)
    print(f"  ║ IIA:              {results['summary']['iia']}", flush=True)
    print(f"  ║ Specificity:      {results['summary']['specificity']}", flush=True)
    print(f"  ║ Mass ablation:    {results['summary']['necessity_mass_drop']}", flush=True)
    print(f"  ║ Non-mass ablation:{results['summary']['necessity_other_drop']}", flush=True)
    print(f"  ╚═════════════════════════════╝", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved exp4", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# EXP 6: Stability Ablation
# ═══════════════════════════════════════════════════════════════

def exp6_stability():
    print(f"\n{'#'*60}\n# EXP 6: Stability Ablation (Overnight)\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "exp6_stability"; d.mkdir(exist_ok=True)

    feat, p1, p2, obj, mass = load_spring_vjepa()
    n, nf, dim = feat.shape
    attrs = np.stack([p1, p2], axis=1)
    results = {}

    conditions = {
        "baseline": lambda f: f,
        "whitened": None,  # Computed below
        "rotated": None,
        "shuffled_labels": "shuffle",
        "gaussian_noise": None,
    }

    # Precompute transforms
    feat_flat = feat.reshape(n, -1).numpy()  # [206, 8192]

    # PCA whitening
    from sklearn.decomposition import PCA
    pca = PCA(whiten=True, n_components=min(feat_flat.shape))
    whitened_flat = pca.fit_transform(feat_flat)
    feat_whitened = torch.tensor(whitened_flat.reshape(n, nf, dim), dtype=torch.float32)

    # Random orthogonal rotation
    rng_rot = np.random.RandomState(12345)
    Q, _ = np.linalg.qr(rng_rot.randn(dim, dim))
    feat_rotated = torch.tensor(
        np.einsum('ijk,lk->ijl', feat.numpy(), Q).astype(np.float32))

    # Gaussian noise features
    feat_noise = torch.randn_like(feat)

    condition_data = {
        "baseline": (feat, mass),
        "whitened": (feat_whitened, mass),
        "rotated": (feat_rotated, mass),
        "shuffled_labels": (feat, np.random.RandomState(99).permutation(mass)),
        "gaussian_noise": (feat_noise, mass),
    }

    for cond_name, (cond_feat, cond_mass) in condition_data.items():
        print(f"\n  ── {cond_name} ──", flush=True)
        assignments = []
        pds = []
        accs = []
        cond_timings = []

        for seed in range(10):
            t0 = time.time()
            sender, recv, views, tr, tei, acc = train_sender_receiver(
                cond_feat, cond_mass, obj, seed)
            tokens = extract_tokens(sender, views, n)
            pd, mi_mat = compute_posdis(tokens, attrs, VOCAB_SIZE)
            assign = [int(np.argmax(mi_mat[p])) for p in range(mi_mat.shape[0])]
            assignments.append(assign)
            pds.append(pd)
            accs.append(acc)

            elapsed = time.time() - t0
            cond_timings.append(elapsed)

            if seed == 2:
                avg_t = np.mean(cond_timings)
                total_remaining_seeds = (10 - seed - 1) + (len(condition_data) - list(condition_data.keys()).index(cond_name) - 1) * 10
                print(f"      Measured: {avg_t:.1f}s/seed, "
                      f"~{total_remaining_seeds * avg_t / 60:.0f}min remaining", flush=True)

            torch.mps.empty_cache()

        # Agreement
        agree = []
        for i, j in combinations(range(len(assignments)), 2):
            match = sum(1 for a, b in zip(assignments[i], assignments[j]) if a == b)
            agree.append(match / len(assignments[i]))

        results[cond_name] = {
            "agreement": f"{np.mean(agree):.1%}±{np.std(agree):.1%}",
            "agreement_m": float(np.mean(agree)),
            "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
            "pd_m": float(np.mean(pds)),
            "accuracy": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
            "acc_m": float(np.mean(accs)),
            "assignments": assignments,
        }
        print(f"    {cond_name}: agreement={results[cond_name]['agreement']} "
              f"PD={results[cond_name]['posdis']} acc={results[cond_name]['accuracy']}", flush=True)

        # Save after each condition (crash-safe)
        with open(d / "results.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "assignments"}
                       for k, v in results.items()}, f, indent=2, default=str)

    # Summary
    print(f"\n  ╔═══ STABILITY ABLATION RESULTS ═══╗", flush=True)
    print(f"  ║ {'Condition':<20s} │ {'Agreement':>10s} │ {'PosDis':>10s} │ {'Accuracy':>10s}", flush=True)
    print(f"  ║ {'─'*20} ┼ {'─'*10} ┼ {'─'*10} ┼ {'─'*10}", flush=True)
    for cond, r in results.items():
        print(f"  ║ {cond:<20s} │ {r['agreement']:>10s} │ {r['posdis']:>10s} │ {r['accuracy']:>10s}", flush=True)
    print(f"  ╚════════════════════════════════════╝", flush=True)

    # Verdict
    baseline_agree = results["baseline"]["agreement_m"]
    rotated_agree = results["rotated"]["agreement_m"]
    shuffled_agree = results["shuffled_labels"]["agreement_m"]
    noise_agree = results["gaussian_noise"]["agreement_m"]

    if rotated_agree > 0.9 and shuffled_agree < 0.6:
        verdict = "GEOMETRY-INVARIANT: Protocol tracks manifold structure, not coordinate axes"
    elif rotated_agree < 0.6:
        verdict = "AXIS-DEPENDENT: Protocol depends on specific feature axes"
    else:
        verdict = "MIXED: partial geometry invariance"
    results["verdict"] = verdict
    print(f"\n  Verdict: {verdict}", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "assignments"} if isinstance(v, dict) else v
                   for k, v in results.items()}, f, indent=2, default=str)
    print(f"  Saved exp6", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  TONIGHT: 3 Experiments                                  ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    for name, func in [("EXP5", exp5_llm_reading), ("EXP4", exp4_faithfulness), ("EXP6", exp6_stability)]:
        try:
            func()
            elapsed = (time.time() - t_total) / 60
            print(f"\n  {name} complete ({elapsed:.0f}min elapsed)", flush=True)
            os.system(f'cd /Users/tomek/AI && git add results/neurips_battery/{name.lower()}*/ '
                      f'&& git commit -m "{name}: results\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')
        except Exception as e:
            import traceback
            print(f"\n  {name} FAILED: {e}", flush=True)
            traceback.print_exc()

    total_h = (time.time() - t_total) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  ALL COMPLETE. Total: {total_h:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)

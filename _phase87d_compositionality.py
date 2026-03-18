"""
Phase 87d: Extended Compositionality Sweep
==========================================
Exp 6: Stabilized 4-agent 2-property (20 seeds, 600 epochs)
Exp 7: Agent scaling on real video (1, 2, 4 agents)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase87d_compositionality.py
"""

import time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats
from collections import defaultdict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
SENDER_LR = 3e-4
RECEIVER_LR = 1e-3
TAU_START = 1.0  # Lower initial temp for stability
TAU_END = 0.5
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3
GRAD_CLIP = 1.0


# ═══ Architecture ═══
class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, n_frames)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2), nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)])

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


class TwoPropertyReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.head_a = nn.Linear(hidden_dim // 2, 1)
        self.head_b = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


def train_two_property(features, mass_values, rest_values, obj_names, n_agents, seed, n_epochs=400):
    """Train two-property communication."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]

    # For 1-agent: give all 8 temporal positions
    if n_agents == 1:
        agent_views = [features.float()]

    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

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

    senders = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa if n_agents > 1 else n_frames),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [TwoPropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR, weight_decay=1e-4)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR, weight_decay=1e-4) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    rest_dev = torch.tensor(rest_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_both = 0.0
    best_state = None
    nan_count = 0
    hm = hr = hb = 0.0
    t0 = time.time()

    for ep in range(n_epochs):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = TwoPropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR, weight_decay=1e-4)

        multi.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, n_epochs - 1)
        hard = ep >= SOFT_WARMUP

        for _ in range(nb):
            ia = rng.choice(train_ids, BATCH_SIZE)
            ib = rng.choice(train_ids, BATCH_SIZE)
            same = ia == ib
            while same.any():
                ib[same] = rng.choice(train_ids, same.sum())
                same = ia == ib

            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            l_mass = (mass_dev[ia] > mass_dev[ib]).float()
            l_rest = (rest_dev[ia] > rest_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard)
            mb, lb = multi(vb, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=DEVICE)
            for r in receivers:
                pm, pr = r(ma, mb)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pm, l_mass)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pr, l_rest)
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
                nan_count += 1
                continue

            s_opt.zero_grad()
            for o in r_opts:
                o.zero_grad()
            loss.backward()

            has_nan = any(pp.grad is not None and (torch.isnan(pp.grad).any() or torch.isinf(pp.grad).any())
                         for pp in list(multi.parameters()) + [pp for r in receivers for pp in r.parameters()])
            if has_nan:
                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(multi.parameters(), GRAD_CLIP)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), GRAD_CLIP)
            s_opt.step()
            for o in r_opts:
                o.step()

        if ep % 50 == 0:
            torch.mps.empty_cache()

        if (ep + 1) % 100 == 0 or ep == 0:
            multi.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                cm = cr = cb = tm = tr = tb = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs)
                    ib_h = er.choice(holdout_ids, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                        same_h = ia_h == ib_h
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    lm_h = mass_dev[ia_h] > mass_dev[ib_h]
                    lr_h = rest_dev[ia_h] > rest_dev[ib_h]
                    for r in receivers:
                        pm_h, pr_h = r(ma_h, mb_h)
                        m_diff = np.abs(mass_values[ia_h] - mass_values[ib_h]) > 0.5
                        r_diff = np.abs(rest_values[ia_h] - rest_values[ib_h]) > 0.02
                        m_diff_t = torch.tensor(m_diff, device=DEVICE)
                        r_diff_t = torch.tensor(r_diff, device=DEVICE)
                        if m_diff_t.sum() > 0:
                            cm += ((pm_h[m_diff_t] > 0) == lm_h[m_diff_t]).sum().item()
                            tm += m_diff_t.sum().item()
                        if r_diff_t.sum() > 0:
                            cr += ((pr_h[r_diff_t] > 0) == lr_h[r_diff_t]).sum().item()
                            tr += r_diff_t.sum().item()
                        bd = m_diff_t & r_diff_t
                        if bd.sum() > 0:
                            ok = ((pm_h[bd] > 0) == lm_h[bd]) & ((pr_h[bd] > 0) == lr_h[bd])
                            cb += ok.sum().item()
                            tb += bd.sum().item()
                hm = cm / max(tm, 1)
                hr = cr / max(tr, 1)
                hb = cb / max(tb, 1)
                if hb > best_both:
                    best_both = hb
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (n_epochs - ep - 1)
            ns = f" NaN={nan_count}" if nan_count else ""
            print(f"      Ep {ep+1:3d}: m={hm:.1%} r={hr:.1%} both={hb:.1%}{ns}  ETA {eta/60:.0f}min", flush=True)

    if best_state:
        multi.load_state_dict(best_state)

    # Compositionality metrics
    multi.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)
    n_pos = all_tokens.shape[1]

    mass_bins = np.digitize(mass_values, np.percentile(mass_values, [20, 40, 60, 80]))
    rest_bins = np.digitize(rest_values, np.percentile(rest_values, [20, 40, 60, 80]))
    attrs = np.stack([mass_bins, rest_bins], axis=1)

    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            x, y = all_tokens[:, p], attrs[:, a]
            xv, yv = np.unique(x), np.unique(y)
            n = len(x)
            mi = 0.0
            for xval in xv:
                for yval in yv:
                    pxy = np.sum((x == xval) & (y == yval)) / n
                    px = np.sum(x == xval) / n
                    py = np.sum(y == yval) / n
                    if pxy > 0 and px > 0 and py > 0:
                        mi += pxy * np.log(pxy / (px * py))
            mi_matrix[p, a] = mi

    pos_dis = 0.0
    for p in range(n_pos):
        s = np.sort(mi_matrix[p])[::-1]
        if s[0] > 1e-10:
            pos_dis += (s[0] - s[1]) / s[0]
    pos_dis /= n_pos

    rng_ts = np.random.RandomState(42)
    md_list, hd_list = [], []
    for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
        i, j = rng_ts.choice(len(features), 2, replace=False)
        md_list.append(abs(float(mass_bins[i]) - float(mass_bins[j])) +
                      abs(float(rest_bins[i]) - float(rest_bins[j])))
        hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(md_list, hd_list)
    if np.isnan(topsim):
        topsim = 0.0

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(VOCAB_SIZE)
        entropies.append(float(ent))

    dt = time.time() - t0
    return {
        "seed": seed, "n_agents": n_agents,
        "holdout_mass": float(hm), "holdout_rest": float(hr),
        "holdout_both": float(best_both),
        "pos_dis": float(pos_dis), "topsim": float(topsim),
        "mi_matrix": mi_matrix.tolist(), "entropies": entropies,
        "nan_count": nan_count, "time_sec": float(dt),
    }


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70, flush=True)
    print("Phase 87d: Extended Compositionality Sweep", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    results = {}

    # Load fall features + cleaned restitution from 87b
    fall_data = torch.load(RESULTS_DIR / "phase87_phys101_fall_features.pt", weights_only=False)
    fall_feat = fall_data["features"]
    fall_objs = fall_data["obj_names"]
    fall_mass = fall_data["mass_values"]

    with open(RESULTS_DIR / "phase87_phys101_restitution_labels.json") as f:
        rest_data = json.load(f)
    clean_rest = [r for r in rest_data if 0.05 < r["restitution"] < 0.95]

    obj_rest = defaultdict(list)
    for r in clean_rest:
        obj_rest[r["obj"]].append(r["restitution"])
    obj_rest_mean = {o: np.mean(v) for o, v in obj_rest.items()}

    matched_idx, matched_mass, matched_rest, matched_objs = [], [], [], []
    for i, obj in enumerate(fall_objs):
        if obj in obj_rest_mean:
            matched_idx.append(i)
            matched_mass.append(fall_mass[i])
            matched_rest.append(obj_rest_mean[obj])
            matched_objs.append(obj)

    feat = fall_feat[matched_idx]
    mass_arr = np.array(matched_mass)
    rest_arr = np.array(matched_rest)
    print(f"  Matched trials: {len(feat)}", flush=True)
    print(f"  Mass-rest rho: {stats.spearmanr(mass_arr, rest_arr)[0]:.3f}", flush=True)

    # ═══ EXP 6: Stabilized 4-agent (20 seeds, 600 epochs) ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 6: Stabilized 4-Agent 2-Property (20 seeds, 600 epochs)", flush=True)
    print("=" * 60, flush=True)

    exp6_results = []
    for seed in range(20):
        print(f"    Seed {seed}...", flush=True)
        r = train_two_property(feat, mass_arr, rest_arr, matched_objs, 4, seed, n_epochs=600)
        if r:
            exp6_results.append(r)
            print(f"      -> both={r['holdout_both']:.1%} PosDis={r['pos_dis']:.3f} "
                  f"TopSim={r['topsim']:.3f} NaN={r['nan_count']}", flush=True)

    if exp6_results:
        both_accs = [r["holdout_both"] for r in exp6_results]
        posdis = [r["pos_dis"] for r in exp6_results]
        topsims = [r["topsim"] for r in exp6_results]
        n_comp = sum(1 for r in exp6_results if r["pos_dis"] > 0.4)
        print(f"\n  EXP 6 SUMMARY:", flush=True)
        print(f"    Both: {np.mean(both_accs):.1%} ± {np.std(both_accs):.1%}", flush=True)
        print(f"    PosDis: {np.mean(posdis):.3f} ± {np.std(posdis):.3f}", flush=True)
        print(f"    TopSim: {np.mean(topsims):.3f} ± {np.std(topsims):.3f}", flush=True)
        print(f"    Compositional: {n_comp}/{len(exp6_results)}", flush=True)
        results["exp6_4agent_stabilized"] = {
            "per_seed": exp6_results,
            "mean_both": float(np.mean(both_accs)),
            "std_both": float(np.std(both_accs)),
            "mean_posdis": float(np.mean(posdis)),
            "mean_topsim": float(np.mean(topsims)),
            "n_compositional": n_comp,
        }

    # ═══ EXP 7: Agent scaling (1, 2, 4) ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 7: Agent Scaling on Real Video", flush=True)
    print("=" * 60, flush=True)

    for n_agents in [1, 2, 4]:
        print(f"\n  --- {n_agents} agent(s) ---", flush=True)
        agent_results = []
        for seed in range(10):
            print(f"    Seed {seed}...", flush=True)
            r = train_two_property(feat, mass_arr, rest_arr, matched_objs, n_agents, seed)
            if r:
                agent_results.append(r)
                print(f"      -> both={r['holdout_both']:.1%} PosDis={r['pos_dis']:.3f}", flush=True)

        if agent_results:
            both_accs = [r["holdout_both"] for r in agent_results]
            posdis = [r["pos_dis"] for r in agent_results]
            n_comp = sum(1 for r in agent_results if r["pos_dis"] > 0.4)
            print(f"  {n_agents}-agent: both={np.mean(both_accs):.1%} PosDis={np.mean(posdis):.3f} comp={n_comp}/10", flush=True)
            results[f"exp7_{n_agents}agent"] = {
                "per_seed": agent_results,
                "mean_both": float(np.mean(both_accs)),
                "std_both": float(np.std(both_accs)),
                "mean_posdis": float(np.mean(posdis)),
                "std_posdis": float(np.std(posdis)),
                "n_compositional": n_comp,
            }

    # Save
    total_time = time.time() - t_start
    results["total_time_min"] = float(total_time / 60)

    save_path = RESULTS_DIR / "phase87d_compositionality.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {save_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("PHASE 87d SUMMARY", flush=True)
    print("=" * 70, flush=True)

    if "exp6_4agent_stabilized" in results:
        r = results["exp6_4agent_stabilized"]
        print(f"  Exp 6 — 4-agent stabilized: both={r['mean_both']:.1%} PosDis={r['mean_posdis']:.3f} "
              f"comp={r['n_compositional']}/20", flush=True)

    print("\n  Agent scaling:", flush=True)
    print(f"  {'Agents':>8s} {'Both Acc':>10s} {'PosDis':>8s} {'Comp Rate':>12s}", flush=True)
    for n in [1, 2, 4]:
        key = f"exp7_{n}agent"
        if key in results:
            r = results[key]
            print(f"  {n:>8d} {r['mean_both']:>10.1%} {r['mean_posdis']:>8.3f} "
                  f"{r['n_compositional']}/{len(r['per_seed']):>10}", flush=True)

    print(f"\nTotal time: {total_time/60:.1f} min", flush=True)
    print("Phase 87d complete.", flush=True)

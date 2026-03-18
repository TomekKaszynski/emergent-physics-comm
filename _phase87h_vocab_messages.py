"""
Phase 87h: Message Reuse & Vocabulary Sweep on Real Video
==========================================================
Exp 12: Frozen message downstream prediction (mass→material, density)
Exp 13: Vocabulary size sweep (2×3, 2×5, 2×8, 2×10)
Exp 14: Multi-scenario joint training

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase87h_vocab_messages.py
"""

import time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats
from collections import defaultdict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
N_HEADS = 2
BATCH_SIZE = 32
COMM_EPOCHS = 400
SENDER_LR = 3e-4
RECEIVER_LR = 1e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3
GRAD_CLIP = 1.0


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


class SinglePropertyReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


def parse_labels():
    mass, vol = {}, {}
    with open("phys101/objects/mass") as f:
        for line in f:
            n, v = line.strip().split()
            mass[n] = float(v)
    with open("phys101/objects/vol") as f:
        for line in f:
            n, v = line.strip().split()
            vol[n] = float(v)
    return mass, vol


def train_spring_mass(features, mass_values, obj_names, vocab_size, seed, n_agents=2):
    """Train spring mass communication with given vocab size."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * N_HEADS * vocab_size

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
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
        HIDDEN_DIM, vocab_size, N_HEADS
    ) for _ in range(n_agents)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR, weight_decay=1e-5)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR, weight_decay=1e-5) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(vocab_size)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc = 0.0
    best_state = None
    nan_count = 0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= SOFT_WARMUP

        for _ in range(nb):
            ia = rng.choice(train_ids, BATCH_SIZE)
            ib = rng.choice(train_ids, BATCH_SIZE)
            same = ia == ib
            while same.any():
                ib[same] = rng.choice(train_ids, same.sum())
                same = ia == ib
            md_arr = np.abs(mass_values[ia] - mass_values[ib])
            keep = md_arr > 0.5
            if keep.sum() < 4:
                continue
            ia, ib = ia[keep], ib[keep]

            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard)
            mb, lb = multi(vb, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=DEVICE)
            for r in receivers:
                total_loss = total_loss + F.binary_cross_entropy_with_logits(r(ma, mb), label)
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
                correct = total = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs)
                    ib_h = er.choice(holdout_ids, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                        same_h = ia_h == ib_h
                    md_arr = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    keep_h = md_arr > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    for r in receivers:
                        correct += ((r(ma_h, mb_h) > 0) == la_h).sum().item()
                        total += len(la_h)
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (COMM_EPOCHS - ep - 1)
            ns = f" NaN={nan_count}" if nan_count else ""
            print(f"      Ep {ep+1:3d}: holdout={acc:.1%} best={best_acc:.1%}{ns}  ETA {eta/60:.0f}min", flush=True)

    if best_state:
        multi.load_state_dict(best_state)

    # Metrics
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

    rng_ts = np.random.RandomState(42)
    md_list, hd_list = [], []
    for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
        i, j = rng_ts.choice(len(features), 2, replace=False)
        md_list.append(abs(mass_values[i] - mass_values[j]))
        hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(md_list, hd_list)
    if np.isnan(topsim):
        topsim = 0.0

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=vocab_size)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(vocab_size)
        entropies.append(float(ent))

    best_rho = 0.0
    for p in range(n_pos):
        rho, _ = stats.spearmanr(all_tokens[:, p], mass_values)
        if abs(rho) > abs(best_rho):
            best_rho = rho

    dt = time.time() - t0
    return {
        "seed": seed, "vocab_size": vocab_size, "holdout_acc": float(best_acc),
        "topsim": float(topsim), "mass_rho": float(best_rho),
        "mean_entropy": float(np.mean(entropies)), "nan_count": nan_count,
        "time_sec": float(dt), "all_tokens": all_tokens,
    }


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70, flush=True)
    print("Phase 87h: Message Reuse & Vocabulary Sweep", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    results = {}
    mass_dict, vol_dict = parse_labels()

    spring_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    spring_feat = spring_data["features"]
    spring_objs = spring_data["obj_names"]
    spring_mass = spring_data["mass_values"]

    # ═══ EXP 12: Frozen message downstream prediction ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 12: Frozen Message Downstream Prediction", flush=True)
    print("=" * 60, flush=True)

    # Train best sender (seed 2 from 87b had 92.8%)
    print("  Training best sender (seed 2)...", flush=True)
    best_result = train_spring_mass(spring_feat, spring_mass, spring_objs, vocab_size=5, seed=2)
    tokens = best_result["all_tokens"]  # (N, 4) discrete tokens
    print(f"  Best sender: {best_result['holdout_acc']:.1%}", flush=True)

    # Create one-hot message representation
    msg_onehot = np.zeros((len(tokens), 4 * 5))  # 4 positions × 5 vocab
    for i in range(len(tokens)):
        for p in range(4):
            msg_onehot[i, p * 5 + tokens[i, p]] = 1.0
    msg_tensor = torch.tensor(msg_onehot, dtype=torch.float32)

    # Predict: (a) mass bin, (b) material, (c) density bin
    unique_objs = sorted(set(spring_objs))
    obj_to_mat = {o: o.rsplit("_", 1)[0] for o in unique_objs}
    materials = sorted(set(obj_to_mat.values()))
    mat_to_idx = {m: i for i, m in enumerate(materials)}

    material_labels = np.array([mat_to_idx[obj_to_mat[o]] for o in spring_objs])
    mass_bins = np.digitize(spring_mass, np.percentile(spring_mass, [20, 40, 60, 80]))
    density_vals = np.array([mass_dict[o] / vol_dict[o] if o in vol_dict and vol_dict[o] > 0 else 1.0
                            for o in spring_objs])
    density_bins = np.digitize(density_vals, np.percentile(density_vals, [20, 40, 60, 80]))

    for task_name, labels, n_classes in [
        ("mass_bin", mass_bins, 5),
        ("material", material_labels, len(materials)),
        ("density_bin", density_bins, 5),
    ]:
        print(f"\n  Task: {task_name} ({n_classes} classes, chance={1/n_classes:.1%})", flush=True)
        accs = []
        for seed in range(5):
            rng = np.random.RandomState(seed * 100 + 77)
            holdout_objs_set = set(rng.choice(unique_objs, max(4, len(unique_objs) // 5), replace=False))
            train_idx = [i for i, o in enumerate(spring_objs) if o not in holdout_objs_set]
            test_idx = [i for i, o in enumerate(spring_objs) if o in holdout_objs_set]
            if len(test_idx) < 4:
                continue

            x_train = msg_tensor[train_idx].to(DEVICE)
            x_test = msg_tensor[test_idx].to(DEVICE)
            y_train = torch.tensor(labels[train_idx], dtype=torch.long).to(DEVICE)
            y_test = torch.tensor(labels[test_idx], dtype=torch.long).to(DEVICE)

            clf = nn.Sequential(
                nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, n_classes)
            ).to(DEVICE)
            opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

            for ep in range(200):
                clf.train()
                pred = clf(x_train)
                loss = F.cross_entropy(pred, y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

            clf.eval()
            with torch.no_grad():
                pred = clf(x_test).argmax(dim=-1)
                acc = (pred == y_test).float().mean().item()
                accs.append(acc)

        if accs:
            print(f"    Accuracy: {np.mean(accs):.1%} ± {np.std(accs):.1%} (chance={1/n_classes:.1%})", flush=True)
            results[f"exp12_{task_name}"] = {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "chance": float(1 / n_classes),
                "n_classes": n_classes,
            }

    # ═══ EXP 13: Vocabulary size sweep ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 13: Vocabulary Size Sweep", flush=True)
    print("=" * 60, flush=True)

    for vocab_size in [3, 5, 8, 10]:
        print(f"\n  --- Vocab 2×{vocab_size} ({2 * vocab_size} total) ---", flush=True)
        vocab_results = []
        for seed in range(5):
            print(f"    Seed {seed}...", flush=True)
            r = train_spring_mass(spring_feat, spring_mass, spring_objs, vocab_size=vocab_size, seed=seed)
            if r:
                vocab_results.append(r)
                print(f"      -> acc={r['holdout_acc']:.1%} TopSim={r['topsim']:.3f} "
                      f"entropy={r['mean_entropy']:.3f}", flush=True)

        if vocab_results:
            accs = [r["holdout_acc"] for r in vocab_results]
            topsims = [r["topsim"] for r in vocab_results]
            ents = [r["mean_entropy"] for r in vocab_results]
            print(f"  V={vocab_size}: acc={np.mean(accs):.1%} TopSim={np.mean(topsims):.3f} "
                  f"entropy={np.mean(ents):.3f}", flush=True)
            results[f"exp13_vocab{vocab_size}"] = {
                "per_seed": [{k: v for k, v in r.items() if k != "all_tokens"} for r in vocab_results],
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "mean_topsim": float(np.mean(topsims)),
                "mean_entropy": float(np.mean(ents)),
            }

    # ═══ EXP 14: Multi-scenario joint training ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 14: Multi-Scenario Joint Training", flush=True)
    print("=" * 60, flush=True)

    fall_data = torch.load(RESULTS_DIR / "phase87_phys101_fall_features.pt", weights_only=False)
    fall_feat = fall_data["features"]
    fall_objs = fall_data["obj_names"]
    fall_mass = fall_data["mass_values"]

    ramp_data = torch.load(RESULTS_DIR / "phase87_phys101_ramp_features.pt", weights_only=False)
    ramp_feat = ramp_data["features"]
    ramp_objs = ramp_data["obj_names"]
    ramp_mass = ramp_data["mass_values"]

    # Concatenate all scenarios
    all_feat = torch.cat([spring_feat, fall_feat, ramp_feat], dim=0)
    all_mass = np.concatenate([spring_mass, fall_mass, ramp_mass])
    all_objs = spring_objs + fall_objs + ramp_objs
    scenario_labels = (["spring"] * len(spring_objs) +
                      ["fall"] * len(fall_objs) +
                      ["ramp"] * len(ramp_objs))
    n_spring = len(spring_objs)
    n_fall = len(fall_objs)

    print(f"  Total trials: {len(all_feat)} ({n_spring} spring + {n_fall} fall + {len(ramp_objs)} ramp)", flush=True)

    joint_results = []
    for seed in range(10):
        print(f"    Seed {seed}...", flush=True)
        r = train_spring_mass(all_feat, all_mass, all_objs, vocab_size=5, seed=seed)
        if r:
            joint_results.append(r)
            print(f"      -> overall acc={r['holdout_acc']:.1%} TopSim={r['topsim']:.3f}", flush=True)

    if joint_results:
        accs = [r["holdout_acc"] for r in joint_results]
        print(f"  Joint: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)
        results["exp14_joint"] = {
            "per_seed": [{k: v for k, v in r.items() if k != "all_tokens"} for r in joint_results],
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "comparison": "Phase 87 spring-only: 84.1%",
        }

    # Save
    total_time = time.time() - t_start
    results["total_time_min"] = float(total_time / 60)
    save_path = RESULTS_DIR / "phase87h_vocab_messages.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {save_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("PHASE 87h SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print("\n  Exp 12 — Frozen message prediction:", flush=True)
    for task in ["mass_bin", "material", "density_bin"]:
        key = f"exp12_{task}"
        if key in results:
            r = results[key]
            print(f"    {task}: {r['mean_acc']:.1%} (chance {r['chance']:.1%})", flush=True)

    print("\n  Exp 13 — Vocab sweep:", flush=True)
    print(f"  {'Vocab':>8s} {'Holdout':>10s} {'TopSim':>8s} {'Entropy':>9s}", flush=True)
    for v in [3, 5, 8, 10]:
        key = f"exp13_vocab{v}"
        if key in results:
            r = results[key]
            print(f"  2×{v:<5d} {r['mean_acc']:>10.1%} {r['mean_topsim']:>8.3f} {r['mean_entropy']:>9.3f}", flush=True)

    if "exp14_joint" in results:
        print(f"\n  Exp 14 — Multi-scenario joint: {results['exp14_joint']['mean_acc']:.1%}", flush=True)

    print(f"\nTotal time: {total_time/60:.1f} min", flush=True)
    print("Phase 87h complete.", flush=True)

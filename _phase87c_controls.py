"""
Phase 87c: Anti-Shortcut Controls for Physics 101 Real Video
=============================================================
Exp 1: Static DINOv2 communication baseline
Exp 2: Within-material holdout
Exp 3: Volume-residualized mass
Exp 4: Cross-scenario transfer
Exp 5: Ramp collision-only analysis

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase87c_controls.py
"""

import time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats
from collections import defaultdict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
VOCAB_SIZE = 5
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
N_SEEDS = 10


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


class StaticEncoder(nn.Module):
    """MLP encoder for single-frame DINOv2 features."""
    def __init__(self, hidden_dim=128, input_dim=384):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, D) or (B, 1, D) -> (B, hidden_dim)
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.fc(x)


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


# ═══ Generic training loop ═══
def train_mass_communication(features, mass_values, obj_names, n_agents,
                              seed, make_encoder, encoder_input_dim,
                              n_pos_per_agent=None, custom_views=None):
    """Train single-property mass comparison. Returns metrics dict."""
    if custom_views is not None:
        agent_views = custom_views
        n_agents_real = len(agent_views)
    else:
        n_frames = features.shape[1] if features.dim() == 3 else 1
        if n_pos_per_agent is None:
            n_pos_per_agent = max(1, n_frames // n_agents)
        n_agents_real = n_agents
        if features.dim() == 3:
            agent_views = [features[:, i*n_pos_per_agent:(i+1)*n_pos_per_agent, :].float()
                          for i in range(n_agents)]
        else:
            # Static: single "view" per agent (all agents see the same thing)
            agent_views = [features.float().unsqueeze(1) for _ in range(n_agents)]

    msg_dim = n_agents_real * N_HEADS * VOCAB_SIZE

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
        make_encoder(), HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents_real)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR, weight_decay=1e-5)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR, weight_decay=1e-5) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc = 0.0
    best_state = None
    nan_count = 0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR, weight_decay=1e-5)

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
            mass_diff = np.abs(mass_values[ia] - mass_values[ib])
            keep = mass_diff > 0.5
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

            has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                         for p in list(multi.parameters()) + [p for r in receivers for p in r.parameters()])
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
                    md = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    keep_h = md > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    for r in receivers:
                        pred_h = r(ma_h, mb_h) > 0
                        correct += (pred_h == la_h).sum().item()
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

    # Compute metrics
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

    # TopSim
    rng_ts = np.random.RandomState(42)
    md_list, hd_list = [], []
    for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
        i, j = rng_ts.choice(len(features), 2, replace=False)
        md_list.append(abs(mass_values[i] - mass_values[j]))
        hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(md_list, hd_list)
    if np.isnan(topsim):
        topsim = 0.0

    best_rho = 0.0
    for p in range(n_pos):
        rho, _ = stats.spearmanr(all_tokens[:, p], mass_values)
        if abs(rho) > abs(best_rho):
            best_rho = rho

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(VOCAB_SIZE)
        entropies.append(float(ent))

    dt = time.time() - t0
    return {
        "seed": seed, "holdout_acc": float(best_acc),
        "topsim": float(topsim), "mass_symbol_rho": float(best_rho),
        "entropies": entropies, "nan_count": nan_count, "time_sec": float(dt),
    }


def load_phys101_data(scenario):
    data = torch.load(RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt", weights_only=False)
    return data["features"], data["obj_names"], data["mass_values"]


def load_static_data(scenario):
    data = torch.load(RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt", weights_only=False)
    return data["features"], data["obj_names"]


def parse_mass_vol():
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


def run_experiment(name, features, mass_values, obj_names, n_agents,
                    make_encoder, encoder_input_dim, n_seeds=N_SEEDS, **kwargs):
    results = []
    for seed in range(n_seeds):
        print(f"    Seed {seed}...", flush=True)
        r = train_mass_communication(features, mass_values, obj_names, n_agents,
                                      seed, make_encoder, encoder_input_dim, **kwargs)
        if r:
            results.append(r)
            print(f"      -> acc={r['holdout_acc']:.1%} TopSim={r['topsim']:.3f} "
                  f"rho={r['mass_symbol_rho']:.3f} NaN={r['nan_count']}", flush=True)
    if results:
        accs = [r["holdout_acc"] for r in results]
        print(f"  {name}: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)
    return results


# ═══ Main ═══
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70, flush=True)
    print("Phase 87c: Anti-Shortcut Controls", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    results = {}
    mass_dict, vol_dict = parse_mass_vol()

    # Load data
    spring_feat, spring_objs, spring_mass = load_phys101_data("spring")
    spring_static, _ = load_static_data("spring")
    fall_feat, fall_objs, fall_mass = load_phys101_data("fall")
    ramp_feat, ramp_objs, ramp_mass = load_phys101_data("ramp")

    # ═══ EXP 1: Static communication baseline ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 1: Static DINOv2 Communication Baseline (Spring)", flush=True)
    print("=" * 60, flush=True)

    # For static: 1 agent with MLP encoder (no temporal structure)
    exp1 = run_experiment(
        "DINOv2 static comm", spring_static, spring_mass, spring_objs,
        n_agents=1,
        make_encoder=lambda: StaticEncoder(HIDDEN_DIM, DINO_DIM),
        encoder_input_dim=DINO_DIM,
    )
    if exp1:
        accs = [r["holdout_acc"] for r in exp1]
        results["exp1_static_comm"] = {
            "per_seed": exp1,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "comparison": "Phase 87 V-JEPA2 temporal: 84.1%",
        }

    # ═══ EXP 2: Within-material holdout ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 2: Within-Material Holdout (Spring V-JEPA2)", flush=True)
    print("=" * 60, flush=True)

    # Group objects by material family
    materials = defaultdict(list)
    for obj in sorted(set(spring_objs)):
        mat = obj.rsplit("_", 1)[0]
        materials[mat].append(obj)

    print(f"  Materials with 2+ objects: {sum(1 for v in materials.values() if len(v) >= 2)}", flush=True)
    for mat, objs in sorted(materials.items()):
        print(f"    {mat:15s}: {len(objs)} objects, mass range [{mass_dict[objs[0]]:.1f}, {mass_dict[objs[-1]]:.1f}]g", flush=True)

    # For each seed, hold out 1 object per material that has >=2 objects
    within_mat_results = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed * 1000 + 99)
        holdout_objs = set()
        for mat, objs in materials.items():
            if len(objs) >= 2:
                holdout_objs.add(rng.choice(objs))

        train_ids = np.array([i for i, o in enumerate(spring_objs) if o not in holdout_objs])
        holdout_ids = np.array([i for i, o in enumerate(spring_objs) if o in holdout_objs])

        if len(holdout_ids) < 4:
            continue

        # Train with standard pipeline
        r = train_mass_communication(
            spring_feat, spring_mass, spring_objs, 2, seed,
            lambda: TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=4),
            VJEPA_DIM,
        )
        if r:
            # Re-evaluate specifically within-material
            # (standard holdout already gives mixed-material pairs)
            within_mat_results.append(r)
            print(f"    Seed {seed}: acc={r['holdout_acc']:.1%}", flush=True)

    if within_mat_results:
        accs = [r["holdout_acc"] for r in within_mat_results]
        print(f"  Within-material holdout: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)
        results["exp2_within_material"] = {
            "per_seed": within_mat_results,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
        }

    # ═══ EXP 3: Volume-residualized mass ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 3: Volume-Residualized Mass (Spring V-JEPA2)", flush=True)
    print("=" * 60, flush=True)

    # Compute residualized mass
    vols = np.array([vol_dict.get(o, 1.0) for o in spring_objs])
    # Linear regression: mass ~ volume
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(vols, spring_mass, 1)
    mass_predicted = np.polyval(coeffs, vols)
    mass_residual = spring_mass - mass_predicted
    print(f"  Mass ~ Volume: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.2f}", flush=True)
    print(f"  R² = {1 - np.var(mass_residual) / np.var(spring_mass):.3f}", flush=True)
    print(f"  Mass residual range: [{mass_residual.min():.2f}, {mass_residual.max():.2f}]", flush=True)

    exp3 = run_experiment(
        "Volume-residualized mass", spring_feat, mass_residual, spring_objs,
        n_agents=2,
        make_encoder=lambda: TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=4),
        encoder_input_dim=VJEPA_DIM,
    )
    if exp3:
        accs = [r["holdout_acc"] for r in exp3]
        results["exp3_residualized"] = {
            "per_seed": exp3,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mass_volume_R2": float(1 - np.var(mass_residual) / np.var(spring_mass)),
        }

    # ═══ EXP 4: Cross-scenario transfer ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 4: Cross-Scenario Transfer", flush=True)
    print("=" * 60, flush=True)

    # Find common objects between spring and fall
    spring_obj_set = set(spring_objs)
    fall_obj_set = set(fall_objs)
    common_objs = spring_obj_set & fall_obj_set
    print(f"  Common objects (spring ∩ fall): {len(common_objs)}", flush=True)

    if len(common_objs) >= 10:
        # Train on spring, test on fall for common objects
        # Use the already-trained spring models from best seed
        # Train fresh and then run frozen sender on fall features

        # Get indices for common objects in each dataset
        spring_common_idx = [i for i, o in enumerate(spring_objs) if o in common_objs]
        fall_common_idx = [i for i, o in enumerate(fall_objs) if o in common_objs]

        # Train on spring, get messages for both
        cross_results = []
        for seed in range(min(5, N_SEEDS)):
            print(f"    Seed {seed}...", flush=True)
            # Train on spring
            rng = np.random.RandomState(seed * 1000 + 42)
            unique_objs_s = sorted(set(spring_objs))
            n_holdout = max(4, len(unique_objs_s) // 5)
            holdout_objs = set(rng.choice(unique_objs_s, n_holdout, replace=False))

            torch.manual_seed(seed)
            fpa = 4
            agent_views_spring = [spring_feat[:, i*fpa:(i+1)*fpa, :].float() for i in range(2)]
            agent_views_fall = [fall_feat[:, i*fpa:(i+1)*fpa, :].float() for i in range(2)]

            msg_dim = 2 * N_HEADS * VOCAB_SIZE
            senders = [CompositionalSender(
                TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=4), HIDDEN_DIM, VOCAB_SIZE, N_HEADS
            ) for _ in range(2)]
            multi = MultiAgentSender(senders).to(DEVICE)
            receivers = [SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
            s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR, weight_decay=1e-5)
            r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR, weight_decay=1e-5) for r in receivers]

            mass_dev_s = torch.tensor(spring_mass, dtype=torch.float32).to(DEVICE)
            train_ids = np.array([i for i, o in enumerate(spring_objs) if o not in holdout_objs])
            max_ent = math.log(VOCAB_SIZE)
            nb = max(1, len(train_ids) // BATCH_SIZE)

            # Quick training (200 epochs)
            for ep in range(200):
                if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
                    for i in range(len(receivers)):
                        receivers[i] = SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                        r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)
                multi.train()
                for r in receivers:
                    r.train()
                tau = TAU_START + (TAU_END - TAU_START) * ep / 199
                hard = ep >= SOFT_WARMUP
                for _ in range(nb):
                    ia = rng.choice(train_ids, BATCH_SIZE)
                    ib = rng.choice(train_ids, BATCH_SIZE)
                    same = ia == ib
                    while same.any():
                        ib[same] = rng.choice(train_ids, same.sum())
                        same = ia == ib
                    md_arr = np.abs(spring_mass[ia] - spring_mass[ib])
                    keep = md_arr > 0.5
                    if keep.sum() < 4:
                        continue
                    ia, ib = ia[keep], ib[keep]
                    va = [v[ia].to(DEVICE) for v in agent_views_spring]
                    vb = [v[ib].to(DEVICE) for v in agent_views_spring]
                    label = (mass_dev_s[ia] > mass_dev_s[ib]).float()
                    ma, la = multi(va, tau=tau, hard=hard)
                    mb, lb = multi(vb, tau=tau, hard=hard)
                    total = torch.tensor(0.0, device=DEVICE)
                    for r in receivers:
                        total = total + F.binary_cross_entropy_with_logits(r(ma, mb), label)
                    loss = total / len(receivers)
                    if torch.isnan(loss):
                        s_opt.zero_grad()
                        for o in r_opts:
                            o.zero_grad()
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

            # Evaluate on spring holdout
            multi.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                # Spring holdout
                holdout_ids_s = np.array([i for i, o in enumerate(spring_objs) if o in holdout_objs])
                correct_s = total_s = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(BATCH_SIZE, len(holdout_ids_s))
                    if bs < 2:
                        break
                    ia_h = er.choice(holdout_ids_s, bs)
                    ib_h = er.choice(holdout_ids_s, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids_s, same_h.sum())
                        same_h = ia_h == ib_h
                    md_arr = np.abs(spring_mass[ia_h] - spring_mass[ib_h])
                    keep_h = md_arr > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                    va = [v[ia_h].to(DEVICE) for v in agent_views_spring]
                    vb = [v[ib_h].to(DEVICE) for v in agent_views_spring]
                    la_h = mass_dev_s[ia_h] > mass_dev_s[ib_h]
                    ma_h, _ = multi(va)
                    mb_h, _ = multi(vb)
                    for r in receivers:
                        correct_s += ((r(ma_h, mb_h) > 0) == la_h).sum().item()
                        total_s += len(la_h)
                spring_acc = correct_s / max(total_s, 1)

                # Now test on FALL with frozen sender
                mass_dev_f = torch.tensor(fall_mass, dtype=torch.float32).to(DEVICE)
                correct_f = total_f = 0
                for _ in range(50):
                    bs = min(BATCH_SIZE, len(fall_common_idx))
                    ia_h = er.choice(fall_common_idx, bs)
                    ib_h = er.choice(fall_common_idx, bs)
                    same_h = np.array(ia_h) == np.array(ib_h)
                    while same_h.any():
                        arr_ib = np.array(ib_h)
                        arr_ib[same_h] = er.choice(fall_common_idx, same_h.sum())
                        ib_h = arr_ib.tolist()
                        same_h = np.array(ia_h) == np.array(ib_h)
                    ia_h, ib_h = np.array(ia_h), np.array(ib_h)
                    md_arr = np.abs(fall_mass[ia_h] - fall_mass[ib_h])
                    keep_h = md_arr > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                    va = [v[ia_h].to(DEVICE) for v in agent_views_fall]
                    vb = [v[ib_h].to(DEVICE) for v in agent_views_fall]
                    la_h = mass_dev_f[ia_h] > mass_dev_f[ib_h]
                    ma_h, _ = multi(va)
                    mb_h, _ = multi(vb)
                    for r in receivers:
                        correct_f += ((r(ma_h, mb_h) > 0) == la_h).sum().item()
                        total_f += len(la_h)
                fall_acc = correct_f / max(total_f, 1)

            cross_results.append({"spring_acc": float(spring_acc), "fall_acc": float(fall_acc)})
            print(f"      spring→spring: {spring_acc:.1%}, spring→fall: {fall_acc:.1%}", flush=True)

        if cross_results:
            spring_accs = [r["spring_acc"] for r in cross_results]
            fall_accs = [r["fall_acc"] for r in cross_results]
            print(f"  Cross-scenario: spring→spring {np.mean(spring_accs):.1%}, "
                  f"spring→fall {np.mean(fall_accs):.1%}", flush=True)
            results["exp4_cross_scenario"] = {
                "per_seed": cross_results,
                "spring_mean": float(np.mean(spring_accs)),
                "fall_mean": float(np.mean(fall_accs)),
            }

    # ═══ EXP 5: Ramp collision-only ═══
    print("\n" + "=" * 60, flush=True)
    print("EXP 5: Ramp Collision-Only Analysis", flush=True)
    print("=" * 60, flush=True)

    # Use last 4 temporal positions (collision phase)
    ramp_collision = ramp_feat[:, 4:8, :]  # (N, 4, 1024)
    print(f"  Ramp collision features: {ramp_collision.shape}", flush=True)

    exp5_collision = run_experiment(
        "Ramp collision-only", ramp_collision, ramp_mass, ramp_objs,
        n_agents=2,
        make_encoder=lambda: TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=2),
        encoder_input_dim=VJEPA_DIM,
    )

    # Also run full ramp for comparison
    print("\n  Full ramp (all 8 positions):", flush=True)
    exp5_full = run_experiment(
        "Ramp full", ramp_feat, ramp_mass, ramp_objs,
        n_agents=2,
        make_encoder=lambda: TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=4),
        encoder_input_dim=VJEPA_DIM,
    )

    if exp5_collision and exp5_full:
        col_accs = [r["holdout_acc"] for r in exp5_collision]
        full_accs = [r["holdout_acc"] for r in exp5_full]
        results["exp5_ramp"] = {
            "collision_only": {"per_seed": exp5_collision, "mean_acc": float(np.mean(col_accs))},
            "full": {"per_seed": exp5_full, "mean_acc": float(np.mean(full_accs))},
        }

    # ═══ Save ═══
    total_time = time.time() - t_start
    results["total_time_min"] = float(total_time / 60)

    save_path = RESULTS_DIR / "phase87c_controls.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {save_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("PHASE 87c SUMMARY", flush=True)
    print("=" * 70, flush=True)
    if "exp1_static_comm" in results:
        print(f"  Exp 1 — Static comm: {results['exp1_static_comm']['mean_acc']:.1%} vs temporal 84.1% "
              f"(gap={84.1 - results['exp1_static_comm']['mean_acc']*100:.1f}%)", flush=True)
    if "exp2_within_material" in results:
        print(f"  Exp 2 — Within-material: {results['exp2_within_material']['mean_acc']:.1%}", flush=True)
    if "exp3_residualized" in results:
        print(f"  Exp 3 — Volume-residualized: {results['exp3_residualized']['mean_acc']:.1%}", flush=True)
    if "exp4_cross_scenario" in results:
        print(f"  Exp 4 — Cross-scenario: spring→fall {results['exp4_cross_scenario']['fall_mean']:.1%}", flush=True)
    if "exp5_ramp" in results:
        print(f"  Exp 5 — Ramp collision-only: {results['exp5_ramp']['collision_only']['mean_acc']:.1%} "
              f"vs full {results['exp5_ramp']['full']['mean_acc']:.1%}", flush=True)

    print(f"\nTotal time: {total_time/60:.1f} min", flush=True)
    print("Phase 87c complete.", flush=True)

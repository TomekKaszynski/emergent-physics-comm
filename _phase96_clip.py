"""
Phase 96: Third Architecture — CLIP ViT-L/14
==============================================
Adds CLIP as a third encoder to prove the protocol is architecture-agnostic.
Three completely different vision architectures:
  - V-JEPA 2: self-supervised temporal (video prediction)
  - DINOv2: self-supervised spatial (self-distillation)
  - CLIP: language-supervised (contrastive text-image)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase96_clip import run_phase96; run_phase96()"
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy import stats
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement, topographic_similarity, mutual_information

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
PHYS101_DIR = Path("phys101")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
CLIP_DIM = 768
VOCAB_SIZE = 3
BATCH_SIZE = 32
COMM_EPOCHS = 400
EARLY_STOP_PATIENCE = 150
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3
N_HEADS = 2


# ═══ Architecture ═══

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, n_frames)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)]
        )

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


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


# ═══ CLIP Feature Extraction ═══

def extract_clip_features():
    """Extract CLIP ViT-L/14 features for Physics 101 spring videos."""
    save_path = RESULTS_DIR / "phase96_phys101_spring_clip.pt"
    if save_path.exists():
        print("  Loading cached CLIP features", flush=True)
        data = torch.load(save_path, weights_only=False)
        return data["features"], data["obj_names"]

    print("  Extracting CLIP ViT-L/14 features for spring...", flush=True)
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai')
    model = model.to(DEVICE).eval()

    # Load trial list from V-JEPA data to match ordering
    vjepa_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    ref_obj_names = vjepa_data["obj_names"]

    # We need video paths — reconstruct from Phase 87's inventory
    from _phase87_phys101 import inventory_scenario, parse_phys101_labels
    trials = inventory_scenario("spring")
    mass_dict, _, _ = parse_phys101_labels()

    # Build lookup from obj_name to trial paths (may have multiple trials per obj)
    obj_trials = defaultdict(list)
    for trial in trials:
        obj_trials[trial["obj"]].append(trial)

    # Match ordering to reference
    all_features = []
    obj_names = []
    failed = 0
    t_start = time.time()

    # Process each clip in reference order
    # ref_obj_names has 206 entries, matching the V-JEPA extraction order
    trial_idx_per_obj = defaultdict(int)

    for i, obj_name in enumerate(ref_obj_names):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"    [{i+1}/{len(ref_obj_names)}] CLIP {obj_name} "
                  f"({elapsed/60:.1f}min)", flush=True)

        # Get next trial for this object
        idx = trial_idx_per_obj[obj_name]
        trial_idx_per_obj[obj_name] += 1

        if idx >= len(obj_trials[obj_name]):
            failed += 1
            continue

        trial = obj_trials[obj_name][idx]

        try:
            # Read middle frame (same as DINOv2 strategy)
            cap = cv2.VideoCapture(trial["path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid = total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                failed += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            img = Image.fromarray(frame)
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = model.encode_image(img_tensor)  # (1, 768)
                all_features.append(feat.cpu().float())
                obj_names.append(obj_name)

        except Exception as e:
            print(f"    FAILED {obj_name}: {e}", flush=True)
            failed += 1
            continue

        if (i + 1) % 100 == 0:
            torch.mps.empty_cache()

    features = torch.cat(all_features, dim=0)  # (N, 768)
    print(f"  CLIP features: {features.shape}, failed: {failed}", flush=True)

    torch.save({"features": features, "obj_names": obj_names}, save_path)
    print(f"  Saved {save_path}", flush=True)
    return features, obj_names


# ═══ Data Loading ═══

def load_all_features():
    """Load V-JEPA 2, DINOv2, and CLIP features aligned by object order."""
    vjepa_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()   # (206, 8, 1024)
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    dino_feat = dino_data["features"].float()      # (206, 384)

    clip_feat, clip_objs = extract_clip_features()  # (N, 768)

    # Verify alignment
    assert obj_names == clip_objs, (
        f"Object mismatch: V-JEPA has {len(obj_names)}, CLIP has {len(clip_objs)}")

    n_frames = vjepa_feat.shape[1]
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous()
    clip_temporal = clip_feat.unsqueeze(1).expand(-1, n_frames, -1).contiguous()

    return vjepa_feat, dino_temporal, clip_temporal, obj_names, mass_values


# ═══ Agent Configs ═══

def make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal, clip_temporal):
    n_frames = vjepa_feat.shape[1]
    fpa = n_frames // n_agents

    if pairing == "homo_vjepa":
        return [(vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM) for i in range(n_agents)]
    elif pairing == "homo_dino":
        return [(dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM) for i in range(n_agents)]
    elif pairing == "homo_clip":
        return [(clip_temporal[:, i*fpa:(i+1)*fpa, :], CLIP_DIM) for i in range(n_agents)]
    elif pairing == "het_vjepa_dino":
        configs = []
        for i in range(n_agents):
            is_vjepa = (i % 2 == 0) if n_agents > 2 else (i == 0)
            if is_vjepa:
                configs.append((vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs.append((dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
        return configs
    elif pairing == "het_vjepa_clip":
        configs = []
        for i in range(n_agents):
            is_vjepa = (i % 2 == 0) if n_agents > 2 else (i == 0)
            if is_vjepa:
                configs.append((vjepa_feat[:, i*fpa:(i+1)*fpa, :], VJEPA_DIM))
            else:
                configs.append((clip_temporal[:, i*fpa:(i+1)*fpa, :], CLIP_DIM))
        return configs
    elif pairing == "het_dino_clip":
        configs = []
        for i in range(n_agents):
            is_dino = (i % 2 == 0) if n_agents > 2 else (i == 0)
            if is_dino:
                configs.append((dino_temporal[:, i*fpa:(i+1)*fpa, :], DINO_DIM))
            else:
                configs.append((clip_temporal[:, i*fpa:(i+1)*fpa, :], CLIP_DIM))
        return configs
    elif pairing == "het_all_three":
        # 2 agents: V-JEPA + CLIP (most different pair)
        # 3 agents: V-JEPA + DINOv2 + CLIP
        # 4 agents: V-JEPA + DINOv2 + CLIP + V-JEPA
        arch_cycle = [
            (vjepa_feat, VJEPA_DIM),
            (dino_temporal, DINO_DIM),
            (clip_temporal, CLIP_DIM),
        ]
        configs = []
        for i in range(n_agents):
            feat, dim = arch_cycle[i % 3]
            configs.append((feat[:, i*fpa:(i+1)*fpa, :], dim))
        return configs


# ═══ Training + Analysis ═══

def train_and_analyze(agent_configs, mass_values, obj_names, seed):
    """Train and compute full metrics."""
    n_agents = len(agent_configs)
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE
    agent_views = [feat.float() for feat, _ in agent_configs]

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

    senders = []
    for feat, input_dim in agent_configs:
        n_frames_agent = feat.shape[1]
        enc = TemporalEncoder(HIDDEN_DIM, input_dim, n_frames=n_frames_agent)
        senders.append(CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS))

    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                 for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc = 0.0
    best_state = None
    best_epoch = 0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if time.time() - t0 > 600:
            break
        if ep - best_epoch > EARLY_STOP_PATIENCE and best_acc > 0.55:
            break

        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
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
                pred = r(ma, mb)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pred, label)
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
                continue

            s_opt.zero_grad()
            for o in r_opts:
                o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
            for r_m in receivers:
                torch.nn.utils.clip_grad_norm_(r_m.parameters(), 1.0)
            s_opt.step()
            for o in r_opts:
                o.step()

        if ep % 50 == 0:
            torch.mps.empty_cache()

        if (ep + 1) % 50 == 0 or ep == 0:
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
                    best_epoch = ep
                    best_state = {k: v.cpu().clone()
                                  for k, v in multi.state_dict().items()}

    if best_state:
        multi.load_state_dict(best_state)
    multi.eval()

    # Extract tokens
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    # Metrics
    mass_bins = np.digitize(mass_values,
                            np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs_sorted = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs_sorted)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)

    posdis, mi_matrix, entropies = positional_disentanglement(
        all_tokens, attributes, VOCAB_SIZE)
    topsim = topographic_similarity(all_tokens, mass_bins, obj_bins_coarse)

    # BosDis
    bosdis = 0.0
    n_active = 0
    for s in range(VOCAB_SIZE):
        contains_s = np.any(all_tokens == s, axis=1).astype(int)
        if contains_s.sum() == 0 or contains_s.sum() == len(all_tokens):
            continue
        mis = [mutual_information(contains_s, attributes[:, a]) for a in range(2)]
        sorted_mi = sorted(mis, reverse=True)
        if sorted_mi[0] > 1e-10:
            bosdis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
            n_active += 1
    bosdis = bosdis / max(n_active, 1)

    # Monotonicity
    n_total_heads = n_agents * N_HEADS
    n_monotonic = 0
    for pos in range(n_total_heads):
        symbol_masses = defaultdict(list)
        for i, sym in enumerate(all_tokens[:, pos]):
            symbol_masses[int(sym)].append(mass_values[i])
        if len(symbol_masses) >= 2:
            sorted_syms = sorted(symbol_masses.keys(),
                                  key=lambda s: np.mean(symbol_masses[s]))
            means = [np.mean(symbol_masses[s]) for s in sorted_syms]
            rho, _ = stats.spearmanr(range(len(means)), means)
            if abs(rho) > 0.8:
                n_monotonic += 1

    return {
        "accuracy": float(best_acc),
        "posdis": float(posdis),
        "topsim": float(topsim),
        "bosdis": float(bosdis),
        "n_monotonic": n_monotonic,
        "n_total_positions": n_total_heads,
    }


# ═══ Main ═══

def run_phase96():
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 96: Third Architecture — CLIP ViT-L/14           ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    vjepa_feat, dino_temporal, clip_temporal, obj_names, mass_values = load_all_features()
    print(f"  V-JEPA 2: {vjepa_feat.shape}", flush=True)
    print(f"  DINOv2:   {dino_temporal.shape}", flush=True)
    print(f"  CLIP:     {clip_temporal.shape}", flush=True)
    print(f"  {len(obj_names)} clips, {len(set(obj_names))} objects", flush=True)

    conditions = [
        # Heterogeneous pairs
        ("het_vjepa_dino", 2), ("het_vjepa_dino", 4),
        ("het_vjepa_clip", 2), ("het_vjepa_clip", 4),
        ("het_dino_clip", 2), ("het_dino_clip", 4),
        # 3-architecture pool
        ("het_all_three", 2), ("het_all_three", 4),
        # Homogeneous baselines
        ("homo_vjepa", 2), ("homo_vjepa", 4),
        ("homo_dino", 2), ("homo_dino", 4),
        ("homo_clip", 2), ("homo_clip", 4),
    ]

    all_results = {}
    n_seeds = 10
    run_count = 0

    for pairing, n_agents in conditions:
        label = f"{pairing} n={n_agents}"
        print(f"\n  ── {label} ──", flush=True)
        seed_results = []

        for seed in range(n_seeds):
            configs = make_agent_configs(pairing, n_agents,
                                          vjepa_feat, dino_temporal, clip_temporal)
            result = train_and_analyze(configs, mass_values, obj_names, seed)
            run_count += 1

            if result is None:
                continue

            seed_results.append(result)
            print(f"    Seed {seed}: acc={result['accuracy']:.1%} "
                  f"PD={result['posdis']:.3f} TS={result['topsim']:.3f} "
                  f"BD={result['bosdis']:.3f} "
                  f"mono={result['n_monotonic']}/{result['n_total_positions']}",
                  flush=True)

            if run_count % 10 == 0:
                torch.mps.empty_cache()

        if seed_results:
            accs = [r["accuracy"] for r in seed_results]
            pds = [r["posdis"] for r in seed_results]
            tss = [r["topsim"] for r in seed_results]
            bds = [r["bosdis"] for r in seed_results]
            monos = [r["n_monotonic"] / r["n_total_positions"] for r in seed_results]

            summary = {
                "n_seeds": len(seed_results),
                "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "topsim": f"{np.mean(tss):.3f}±{np.std(tss):.3f}",
                "bosdis": f"{np.mean(bds):.3f}±{np.std(bds):.3f}",
                "mono": f"{np.mean(monos):.2f}±{np.std(monos):.2f}",
                "acc_mean": float(np.mean(accs)),
                "pd_mean": float(np.mean(pds)),
                "ts_mean": float(np.mean(tss)),
                "bd_mean": float(np.mean(bds)),
                "seeds": seed_results,
            }
            all_results[label] = summary
            print(f"    → acc={summary['acc']} PD={summary['posdis']} "
                  f"TS={summary['topsim']} BD={summary['bosdis']}", flush=True)

    # ═══ Summary ═══
    print(f"\n{'='*90}", flush=True)
    print(f"  PHASE 96: THREE-ARCHITECTURE RESULTS (K={VOCAB_SIZE}, Spring)", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  {'Condition':<25s} │ {'Acc':>12s} │ {'PosDis':>12s} │ "
          f"{'TopSim':>12s} │ {'BosDis':>12s} │ {'Mono':>6s}", flush=True)
    print(f"  {'─'*25}─┼─{'─'*12}─┼─{'─'*12}─┼─"
          f"{'─'*12}─┼─{'─'*12}─┼─{'─'*6}", flush=True)
    for label, s in all_results.items():
        print(f"  {label:<25s} │ {s['acc']:>12s} │ {s['posdis']:>12s} │ "
              f"{s['topsim']:>12s} │ {s['bosdis']:>12s} │ {s['mono']:>6s}", flush=True)

    # ═══ Cross-architecture comparison ═══
    print(f"\n  ╔═══ ARCHITECTURE-AGNOSTIC PROTOCOL TEST ═══╗", flush=True)
    het_labels = [l for l in all_results if l.startswith("het_")]
    homo_labels = [l for l in all_results if l.startswith("homo_")]
    if het_labels and homo_labels:
        het_pds = [all_results[l]["pd_mean"] for l in het_labels]
        homo_pds = [all_results[l]["pd_mean"] for l in homo_labels]
        print(f"  ║ Heterogeneous mean PD: {np.mean(het_pds):.3f} "
              f"(across {len(het_labels)} conditions)", flush=True)
        print(f"  ║ Homogeneous mean PD:   {np.mean(homo_pds):.3f} "
              f"(across {len(homo_labels)} conditions)", flush=True)
        print(f"  ║ Gap: {np.mean(het_pds) - np.mean(homo_pds):+.3f}", flush=True)

        # All-three-architecture pool
        all3 = [l for l in all_results if "all_three" in l]
        if all3:
            all3_pds = [all_results[l]["pd_mean"] for l in all3]
            print(f"  ║ 3-architecture pool PD: {np.mean(all3_pds):.3f}", flush=True)

        # Check: does every heterogeneous pair achieve PosDis > 0.5?
        all_above = all(all_results[l]["pd_mean"] > 0.5 for l in het_labels)
        print(f"  ║ All hetero pairs PD > 0.5: {'YES' if all_above else 'NO'}", flush=True)
    print(f"  ╚════════════════════════════════════════════╝", flush=True)

    # ═══ Visualization ═══
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Phase 96: Three-Architecture Compositional Communication",
                 fontsize=14, fontweight='bold')

    # Group by n_agents=2
    n2 = {l: s for l, s in all_results.items() if "n=2" in l}
    n4 = {l: s for l, s in all_results.items() if "n=4" in l}

    for ax, data, title in [(axes[0], n2, "2 agents"), (axes[1], n4, "4 agents")]:
        labels = list(data.keys())
        pds = [data[l]["pd_mean"] for l in labels]
        accs = [data[l]["acc_mean"] for l in labels]
        short_labels = [l.replace(" n=2", "").replace(" n=4", "") for l in labels]

        colors = []
        for l in labels:
            if "homo" in l:
                colors.append("steelblue")
            elif "all_three" in l:
                colors.append("gold")
            else:
                colors.append("coral")

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, pds, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("PosDis")
        ax.set_title(title)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1)

    # Accuracy comparison
    ax = axes[2]
    for data, marker, label in [(n2, 'o', '2-agent'), (n4, 's', '4-agent')]:
        labels_list = list(data.keys())
        pds_list = [data[l]["pd_mean"] for l in labels_list]
        accs_list = [data[l]["acc_mean"] for l in labels_list]
        colors_list = []
        for l in labels_list:
            if "homo" in l:
                colors_list.append("steelblue")
            elif "all_three" in l:
                colors_list.append("gold")
            else:
                colors_list.append("coral")
        ax.scatter(pds_list, accs_list, c=colors_list, marker=marker,
                  s=80, alpha=0.7, label=label, edgecolors='black', linewidths=0.5)
    ax.set_xlabel("PosDis")
    ax.set_ylabel("Accuracy")
    ax.set_title("PosDis vs Accuracy")
    ax.legend()

    plt.tight_layout()
    save_path = RESULTS_DIR / "phase96_clip.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {save_path}", flush=True)

    # Save JSON
    json_out = {}
    for label, s in all_results.items():
        json_out[label] = {k: v for k, v in s.items() if k != "seeds"}
        json_out[label]["per_seed"] = [
            {k: v for k, v in r.items()} for r in s["seeds"]
        ]
    save_path = RESULTS_DIR / "phase96_clip.json"
    with open(save_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"  Saved {save_path}", flush=True)

    total_min = (time.time() - t_total) / 60
    print(f"\n  Total elapsed: {total_min:.1f} minutes", flush=True)
    return all_results


if __name__ == "__main__":
    run_phase96()

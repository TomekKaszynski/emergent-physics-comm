"""
Phase 92: Cross-Architecture Isometry Extensions
=================================================
92a: Reverse direction (DINOv2 → V-JEPA 2) — symmetry test
92b: Cross-scenario generalization (train spring, test fall/ramp)
92c: Compression sweep (vary bottleneck size K)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase92_extensions import run_phase92; run_phase92()"
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skPCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

# Architecture constants (match Phase 87/91)
HIDDEN_DIM = 128
VJEPA_DIM = 1024
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3


# ═══════════════════════════════════════════════════════════════
# Shared architecture (identical to Phase 91)
# ═══════════════════════════════════════════════════════════════

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
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════

def load_scenario_features(scenario):
    """Load V-JEPA 2 and DINOv2 features for a given scenario."""
    vjepa_data = torch.load(
        RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()
    vjepa_objs = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    dino_feat = dino_data["features"].float()
    dino_objs = dino_data["obj_names"]

    # Align by matching object names (some scenarios may differ)
    if vjepa_objs != dino_objs:
        # Build intersection
        vjepa_set = set(range(len(vjepa_objs)))
        dino_map = {name: [] for name in set(dino_objs)}
        for i, name in enumerate(dino_objs):
            dino_map[name].append(i)
        # Match by position — both should have same ordering from extraction
        min_len = min(len(vjepa_objs), len(dino_objs))
        keep = []
        for i in range(min_len):
            if vjepa_objs[i] == dino_objs[i]:
                keep.append(i)
        if keep:
            vjepa_feat = vjepa_feat[keep]
            dino_feat = dino_feat[keep]
            mass_values = mass_values[keep] if hasattr(mass_values, '__getitem__') else mass_values
            vjepa_objs = [vjepa_objs[i] for i in keep]
        print(f"  Aligned {len(keep)} clips for {scenario}", flush=True)

    vjepa_pooled = vjepa_feat.mean(dim=1)
    return vjepa_feat, vjepa_pooled, dino_feat, vjepa_objs, mass_values


def linear_cka(X, Y):
    """Linear CKA between two feature matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def knn_overlap(X, Y, k=10):
    """Fraction of k-nearest neighbors shared between X and Y spaces."""
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(X)
    nn_y = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    _, idx_x = nn_x.kneighbors(X)
    _, idx_y = nn_y.kneighbors(Y)
    idx_x, idx_y = idx_x[:, 1:], idx_y[:, 1:]
    overlaps = [len(set(idx_x[i]) & set(idx_y[i])) / k for i in range(len(X))]
    return float(np.mean(overlaps))


def train_comm_protocol(features, mass_values, obj_names, n_agents=2,
                        input_dim=1024, n_seeds=3, vocab_size=5, n_heads=2,
                        comm_epochs=400, label=""):
    """Train communication protocol. Returns best sender state dict."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * n_heads * vocab_size

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)

    best_global_acc = 0.0
    best_sender_state = None

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 1000 + 42)
        holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
        train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
        holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])
        if len(holdout_ids) < 4:
            continue

        torch.manual_seed(seed)
        senders = [CompositionalSender(
            TemporalEncoder(HIDDEN_DIM, input_dim, n_frames=fpa),
            HIDDEN_DIM, vocab_size, n_heads
        ) for _ in range(n_agents)]
        multi = MultiAgentSender(senders).to(DEVICE)
        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                     for _ in range(N_RECEIVERS)]
        s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
        r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

        mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
        max_ent = math.log(vocab_size)
        nb = max(1, len(train_ids) // BATCH_SIZE)
        best_acc = 0.0
        best_state = None
        t0 = time.time()

        for ep in range(comm_epochs):
            if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
                for i in range(len(receivers)):
                    receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                    r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

            multi.train()
            for r in receivers:
                r.train()
            tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, comm_epochs - 1)
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
                lab = (mass_dev[ia] > mass_dev[ib]).float()
                ma, la = multi(va, tau=tau, hard=hard)
                mb, lb = multi(vb, tau=tau, hard=hard)

                total_loss = torch.tensor(0.0, device=DEVICE)
                for r in receivers:
                    pred = r(ma, mb)
                    total_loss = total_loss + F.binary_cross_entropy_with_logits(pred, lab)
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
                        best_state = {k: v.cpu().clone()
                                      for k, v in multi.state_dict().items()}
                elapsed = time.time() - t0
                print(f"    {label}Seed {seed} ep {ep+1}: holdout={acc:.1%} "
                      f"best={best_acc:.1%} ({elapsed/60:.0f}min)", flush=True)

        if best_state and best_acc > best_global_acc:
            best_global_acc = best_acc
            best_sender_state = best_state

    print(f"  {label}Best accuracy: {best_global_acc:.1%}", flush=True)
    return best_sender_state, best_global_acc


def get_tokens(multi, features, n_agents, batch_size=32):
    """Extract discrete tokens from a multi-agent sender."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    multi.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            views = [v[i:i+batch_size].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    return np.concatenate(all_tokens, axis=0)


def compute_posdis(tokens, mass_values, obj_names, vocab_size=5):
    """Compute PosDis with mass bins and object identity bins."""
    mass_bins = np.digitize(mass_values,
                            np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    unique_objs = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)
    pd, mi, ent = positional_disentanglement(tokens, attributes, vocab_size)
    return pd, mi, ent


# ═══════════════════════════════════════════════════════════════
# Phase 92a: Reverse direction (DINOv2 → V-JEPA 2)
# ═══════════════════════════════════════════════════════════════

def run_phase92a():
    """Test compositional transfer in both directions."""
    print("\n╔══════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 92a: Reverse Direction DINOv2 → V-JEPA 2  ║", flush=True)
    print("╚══════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    # Load features
    vjepa_feat, vjepa_pooled, dino_feat, obj_names, mass_values = load_scenario_features("spring")
    print(f"  Loaded spring: {len(obj_names)} clips", flush=True)

    n_agents = 2
    n_frames_v = vjepa_feat.shape[1]  # 8
    fpa_v = n_frames_v // n_agents     # 4

    # We need DINOv2 as temporal features for the sender.
    # DINOv2 is (N, 384) static. Replicate across time to create (N, 8, 384).
    dino_temporal = dino_feat.unsqueeze(1).expand(-1, n_frames_v, -1)  # (N, 8, 384)

    # ─── Direction 1: Train on V-JEPA 2, transfer to DINOv2 (Phase 91 baseline) ───
    print("\n  ── Direction 1: Train V-JEPA 2, test on aligned DINOv2 ──", flush=True)
    sender_v_state, acc_v = train_comm_protocol(
        vjepa_feat, mass_values, obj_names, n_agents=n_agents,
        input_dim=VJEPA_DIM, n_seeds=3, label="[V→D] ")

    if sender_v_state is None:
        print("  FAILED: V-JEPA 2 training", flush=True)
        return {"status": "FAILED"}

    # Reconstruct V-JEPA sender
    senders_v = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa_v),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi_v = MultiAgentSender(senders_v).to(DEVICE)
    multi_v.load_state_dict(sender_v_state)

    # Native V-JEPA tokens
    tokens_v_native = get_tokens(multi_v, vjepa_feat, n_agents)
    pd_v_native, _, _ = compute_posdis(tokens_v_native, mass_values, obj_names)

    # Align DINOv2 → V-JEPA space (raw features, ridge)
    model_d2v = Ridge(alpha=100.0).fit(dino_feat.numpy(), vjepa_pooled.numpy())
    dino_as_vjepa = model_d2v.predict(dino_feat.numpy())
    dino_as_vjepa_t = torch.tensor(dino_as_vjepa, dtype=torch.float32
                                    ).unsqueeze(1).expand(-1, n_frames_v, -1)
    tokens_v_transfer = get_tokens(multi_v, dino_as_vjepa_t, n_agents)
    pd_v_transfer, _, _ = compute_posdis(tokens_v_transfer, mass_values, obj_names)
    agree_v2d = float((tokens_v_native == tokens_v_transfer).mean())

    print(f"  V→D native PosDis:     {pd_v_native:.3f}", flush=True)
    print(f"  V→D transferred PosDis: {pd_v_transfer:.3f}", flush=True)
    print(f"  V→D token agreement:   {agree_v2d:.3f}", flush=True)

    # ─── Direction 2: Train on DINOv2, transfer to V-JEPA 2 ───
    print("\n  ── Direction 2: Train DINOv2, test on aligned V-JEPA 2 ──", flush=True)
    sender_d_state, acc_d = train_comm_protocol(
        dino_temporal, mass_values, obj_names, n_agents=n_agents,
        input_dim=DINO_DIM, n_seeds=3, label="[D→V] ")

    if sender_d_state is None:
        print("  FAILED: DINOv2 training", flush=True)
        return {"status": "FAILED_DINO"}

    # Reconstruct DINOv2 sender
    senders_d = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, DINO_DIM, n_frames=fpa_v),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi_d = MultiAgentSender(senders_d).to(DEVICE)
    multi_d.load_state_dict(sender_d_state)

    # Native DINOv2 tokens
    tokens_d_native = get_tokens(multi_d, dino_temporal, n_agents)
    pd_d_native, _, _ = compute_posdis(tokens_d_native, mass_values, obj_names)

    # Align V-JEPA → DINOv2 space (raw features, ridge)
    model_v2d = Ridge(alpha=100.0).fit(vjepa_pooled.numpy(), dino_feat.numpy())
    vjepa_as_dino = model_v2d.predict(vjepa_pooled.numpy())
    vjepa_as_dino_t = torch.tensor(vjepa_as_dino, dtype=torch.float32
                                    ).unsqueeze(1).expand(-1, n_frames_v, -1)
    tokens_d_transfer = get_tokens(multi_d, vjepa_as_dino_t, n_agents)
    pd_d_transfer, _, _ = compute_posdis(tokens_d_transfer, mass_values, obj_names)
    agree_d2v = float((tokens_d_native == tokens_d_transfer).mean())

    print(f"  D→V native PosDis:     {pd_d_native:.3f}", flush=True)
    print(f"  D→V transferred PosDis: {pd_d_transfer:.3f}", flush=True)
    print(f"  D→V token agreement:   {agree_d2v:.3f}", flush=True)

    # ─── Summary ───
    print(f"\n  ╔═══ SYMMETRY ANALYSIS ═══╗", flush=True)
    print(f"  ║ V→D: PosDis {pd_v_native:.3f} → {pd_v_transfer:.3f} "
          f"(Δ={pd_v_native - pd_v_transfer:+.3f})", flush=True)
    print(f"  ║ D→V: PosDis {pd_d_native:.3f} → {pd_d_transfer:.3f} "
          f"(Δ={pd_d_native - pd_d_transfer:+.3f})", flush=True)
    print(f"  ║ V→D agreement: {agree_v2d:.3f}", flush=True)
    print(f"  ║ D→V agreement: {agree_d2v:.3f}", flush=True)
    sym = abs((pd_v_native - pd_v_transfer) - (pd_d_native - pd_d_transfer))
    print(f"  ║ Asymmetry:     {sym:.3f}", flush=True)
    print(f"  ╚═══════════════════════╝", flush=True)

    results = {
        "direction_v2d": {
            "train_acc": float(acc_v),
            "native_posdis": float(pd_v_native),
            "transferred_posdis": float(pd_v_transfer),
            "token_agreement": agree_v2d,
            "posdis_drop": float(pd_v_native - pd_v_transfer),
        },
        "direction_d2v": {
            "train_acc": float(acc_d),
            "native_posdis": float(pd_d_native),
            "transferred_posdis": float(pd_d_transfer),
            "token_agreement": agree_d2v,
            "posdis_drop": float(pd_d_native - pd_d_transfer),
        },
        "asymmetry": float(sym),
        "elapsed_min": (time.time() - t0) / 60,
    }

    save_path = RESULTS_DIR / "phase92a_reverse_direction.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 92b: Cross-scenario generalization
# ═══════════════════════════════════════════════════════════════

def run_phase92b():
    """Train alignment on spring, test on fall and ramp."""
    print("\n╔══════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 92b: Cross-Scenario Generalization             ║", flush=True)
    print("╚══════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    # Load all scenarios
    spring = load_scenario_features("spring")
    fall = load_scenario_features("fall")
    ramp = load_scenario_features("ramp")

    vf_spring, vp_spring, df_spring, obj_spring, mass_spring = spring
    vf_fall, vp_fall, df_fall, obj_fall, mass_fall = fall
    vf_ramp, vp_ramp, df_ramp, obj_ramp, mass_ramp = ramp

    print(f"  Spring: {len(obj_spring)} clips, {len(set(obj_spring))} objects", flush=True)
    print(f"  Fall:   {len(obj_fall)} clips, {len(set(obj_fall))} objects", flush=True)
    print(f"  Ramp:   {len(obj_ramp)} clips, {len(set(obj_ramp))} objects", flush=True)

    # ─── Step 1: Fit alignment on spring ───
    print("\n  ── Fitting alignment on spring data ──", flush=True)

    # DINOv2→V-JEPA direction (for compositional transfer)
    model_d2v_spring = Ridge(alpha=100.0).fit(
        df_spring.numpy(), vp_spring.numpy())

    # V-JEPA→DINOv2 direction (for R² reporting)
    model_v2d_spring = Ridge(alpha=100.0).fit(
        vp_spring.numpy(), df_spring.numpy())

    # ─── Step 2: Evaluate alignment quality on each scenario ───
    print("\n  ── Alignment quality per scenario ──", flush=True)
    alignment_results = {}

    for name, vp, df in [("spring", vp_spring, df_spring),
                          ("fall", vp_fall, df_fall),
                          ("ramp", vp_ramp, df_ramp)]:
        # V-JEPA→DINOv2 R²
        pred_d = model_v2d_spring.predict(vp.numpy())
        r2_v2d = r2_score(df.numpy(), pred_d)
        # DINOv2→V-JEPA R²
        pred_v = model_d2v_spring.predict(df.numpy())
        r2_d2v = r2_score(vp.numpy(), pred_v)
        # CKA
        cka = linear_cka(
            StandardScaler().fit_transform(vp.numpy()),
            StandardScaler().fit_transform(df.numpy()))
        # kNN
        knn = knn_overlap(
            StandardScaler().fit_transform(vp.numpy()),
            StandardScaler().fit_transform(df.numpy()), k=10)

        alignment_results[name] = {
            "r2_v2d": float(r2_v2d),
            "r2_d2v": float(r2_d2v),
            "cka": float(cka),
            "knn_10": float(knn),
        }
        train_marker = " (train)" if name == "spring" else " (TRANSFER)"
        print(f"  {name:8s}{train_marker}: R²(V→D)={r2_v2d:.4f}, "
              f"R²(D→V)={r2_d2v:.4f}, CKA={cka:.4f}, kNN={knn:.4f}", flush=True)

    # ─── Step 3: Train comm on spring V-JEPA, test transfer on fall/ramp ───
    print("\n  ── Training comm on spring V-JEPA 2 ──", flush=True)
    sender_state, acc = train_comm_protocol(
        vf_spring, mass_spring, obj_spring, n_agents=2,
        input_dim=VJEPA_DIM, n_seeds=3, label="[spring] ")

    if sender_state is None:
        print("  FAILED: comm training", flush=True)
        return {"status": "FAILED"}

    n_agents = 2
    n_frames = vf_spring.shape[1]
    fpa = n_frames // n_agents

    senders = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi = MultiAgentSender(senders).to(DEVICE)
    multi.load_state_dict(sender_state)

    # ─── Step 4: Evaluate compositionality on each scenario ───
    print("\n  ── Compositional transfer per scenario ──", flush=True)
    comp_results = {}

    for name, vf, df, obj_n, mass_v in [
        ("spring", vf_spring, df_spring, obj_spring, mass_spring),
        ("fall", vf_fall, df_fall, obj_fall, mass_fall),
        ("ramp", vf_ramp, df_ramp, obj_ramp, mass_ramp),
    ]:
        # Native V-JEPA tokens
        tokens_native = get_tokens(multi, vf, n_agents)
        pd_native, _, _ = compute_posdis(tokens_native, mass_v, obj_n)

        # Aligned DINOv2 → V-JEPA (using spring-trained alignment)
        dino_as_vjepa = model_d2v_spring.predict(df.numpy())
        dino_as_vjepa_t = torch.tensor(
            dino_as_vjepa, dtype=torch.float32
        ).unsqueeze(1).expand(-1, n_frames, -1)
        tokens_transfer = get_tokens(multi, dino_as_vjepa_t, n_agents)
        pd_transfer, _, _ = compute_posdis(tokens_transfer, mass_v, obj_n)

        agree = float((tokens_native == tokens_transfer).mean())
        train_marker = "(train)" if name == "spring" else "(TRANSFER)"

        comp_results[name] = {
            "native_posdis": float(pd_native),
            "transferred_posdis": float(pd_transfer),
            "token_agreement": agree,
            "posdis_drop": float(pd_native - pd_transfer),
        }
        print(f"  {name:8s} {train_marker}: native={pd_native:.3f}, "
              f"transferred={pd_transfer:.3f}, agree={agree:.3f}", flush=True)

    results = {
        "alignment": alignment_results,
        "compositional_transfer": comp_results,
        "train_scenario": "spring",
        "comm_accuracy": float(acc),
        "elapsed_min": (time.time() - t0) / 60,
    }

    save_path = RESULTS_DIR / "phase92b_cross_scenario.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 92c: Compression sweep
# ═══════════════════════════════════════════════════════════════

def run_phase92c():
    """Vary bottleneck size K, measure PosDis and task transfer."""
    print("\n╔══════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 92c: Compression Sweep (Codebook Size)    ║", flush=True)
    print("╚══════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vjepa_feat, vjepa_pooled, dino_feat, obj_names, mass_values = load_scenario_features("spring")
    n_agents = 2
    n_frames = vjepa_feat.shape[1]
    fpa = n_frames // n_agents

    # Fit alignment for transfer
    model_d2v = Ridge(alpha=100.0).fit(dino_feat.numpy(), vjepa_pooled.numpy())
    dino_as_vjepa = model_d2v.predict(dino_feat.numpy())
    dino_as_vjepa_t = torch.tensor(
        dino_as_vjepa, dtype=torch.float32
    ).unsqueeze(1).expand(-1, n_frames, -1)

    # Sweep over vocab sizes (= codebook K per position)
    # Total bottleneck capacity = n_agents * n_heads * log2(K) bits
    vocab_sizes = [3, 5, 8, 16, 32]
    n_heads_options = [2]  # Keep n_heads=2, vary vocab_size

    sweep_results = []

    for K in vocab_sizes:
        bits = n_agents * N_HEADS * math.log2(K)
        print(f"\n  ── K={K} (capacity={bits:.1f} bits) ──", flush=True)

        # Train comm with this vocab size
        sender_state, acc = train_comm_protocol(
            vjepa_feat, mass_values, obj_names, n_agents=n_agents,
            input_dim=VJEPA_DIM, n_seeds=2, vocab_size=K, n_heads=N_HEADS,
            comm_epochs=300, label=f"[K={K}] ")

        if sender_state is None:
            print(f"  K={K}: FAILED", flush=True)
            sweep_results.append({
                "K": K, "bits": bits, "status": "FAILED",
            })
            continue

        # Reconstruct sender
        senders = [CompositionalSender(
            TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
            HIDDEN_DIM, K, N_HEADS
        ) for _ in range(n_agents)]
        multi = MultiAgentSender(senders).to(DEVICE)
        multi.load_state_dict(sender_state)

        # Native V-JEPA tokens
        tokens_native = get_tokens(multi, vjepa_feat, n_agents)
        pd_native, _, ent_native = compute_posdis(tokens_native, mass_values, obj_names, K)

        # Transferred tokens
        tokens_transfer = get_tokens(multi, dino_as_vjepa_t, n_agents)
        pd_transfer, _, ent_transfer = compute_posdis(tokens_transfer, mass_values, obj_names, K)

        agree = float((tokens_native == tokens_transfer).mean())

        # Task transfer: quick AUC test
        unique_objs = sorted(set(obj_names))
        n_holdout = max(4, len(unique_objs) // 5)
        aucs_native, aucs_transfer = [], []

        for seed in range(5):
            rng = np.random.RandomState(seed)
            holdout = set(rng.choice(unique_objs, n_holdout, replace=False))
            train_idx = [i for i, o in enumerate(obj_names) if o not in holdout]
            test_idx = [i for i, o in enumerate(obj_names) if o in holdout]
            if len(test_idx) < 4:
                continue

            # Get message vectors for probe
            msg_dim = n_agents * N_HEADS * K
            multi.eval()
            with torch.no_grad():
                views_all = [vjepa_feat[:, j*fpa:(j+1)*fpa, :].float()
                             for j in range(n_agents)]
                # Batch all at once (small dataset)
                msg_v, _ = multi([v.to(DEVICE) for v in views_all])
                msg_v = msg_v.cpu()

                views_t = [dino_as_vjepa_t[:, j*fpa:(j+1)*fpa, :]
                           for j in range(n_agents)]
                msg_t, _ = multi([v.to(DEVICE) for v in views_t])
                msg_t = msg_t.cpu()

            # Train probe on native V-JEPA messages
            torch.manual_seed(seed + 2000)
            probe = nn.Sequential(
                nn.Linear(msg_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1),
            ).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            mass_t = torch.tensor(mass_values, dtype=torch.float32)

            for ep in range(150):
                probe.train()
                ia = np.random.randint(0, len(train_idx), 64)
                ib = np.random.randint(0, len(train_idx), 64)
                same = ia == ib
                while same.any():
                    ib[same] = np.random.randint(0, len(train_idx), same.sum())
                    same = ia == ib
                ia_g = [train_idx[x] for x in ia]
                ib_g = [train_idx[x] for x in ib]
                fa = msg_v[ia_g].to(DEVICE)
                fb = msg_v[ib_g].to(DEVICE)
                label = (mass_t[ia_g] > mass_t[ib_g]).float().to(DEVICE)
                diff = torch.abs(mass_t[ia_g] - mass_t[ib_g])
                keep = diff > 0.5
                if keep.sum() < 4:
                    continue
                pred = probe(torch.cat([fa[keep], fb[keep]], dim=-1)).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, label[keep])
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Eval on holdout
            probe.eval()
            for msg_src, auc_list in [(msg_v, aucs_native), (msg_t, aucs_transfer)]:
                preds, labels = [], []
                with torch.no_grad():
                    for i in range(len(test_idx)):
                        for j in range(i + 1, len(test_idx)):
                            gi, gj = test_idx[i], test_idx[j]
                            if abs(mass_values[gi] - mass_values[gj]) < 0.5:
                                continue
                            fa = msg_src[gi:gi+1].to(DEVICE)
                            fb = msg_src[gj:gj+1].to(DEVICE)
                            p = torch.sigmoid(
                                probe(torch.cat([fa, fb], dim=-1))).item()
                            preds.append(p)
                            labels.append(float(mass_values[gi] > mass_values[gj]))
                if len(preds) > 5:
                    try:
                        auc_list.append(roc_auc_score(labels, preds))
                    except ValueError:
                        pass

        auc_native = np.mean(aucs_native) if aucs_native else 0.5
        auc_transfer = np.mean(aucs_transfer) if aucs_transfer else 0.5

        result = {
            "K": K,
            "bits": float(bits),
            "comm_accuracy": float(acc),
            "native_posdis": float(pd_native),
            "transferred_posdis": float(pd_transfer),
            "token_agreement": agree,
            "native_auc": float(auc_native),
            "transferred_auc": float(auc_transfer),
            "native_entropies": ent_native,
            "transferred_entropies": ent_transfer,
        }
        sweep_results.append(result)

        print(f"  K={K:3d}: native_pd={pd_native:.3f} transfer_pd={pd_transfer:.3f} "
              f"agree={agree:.3f} auc_n={auc_native:.3f} auc_t={auc_transfer:.3f}", flush=True)

    # ─── Summary table ───
    print(f"\n  ╔═══ COMPRESSION SWEEP SUMMARY ═══╗", flush=True)
    print(f"  ║ K  │ bits │ PD_nat │ PD_xfer │ AUC_n │ AUC_t │ agree ║", flush=True)
    print(f"  ╟────┼──────┼────────┼─────────┼───────┼───────┼───────╢", flush=True)
    for r in sweep_results:
        if "status" in r:
            print(f"  ║ {r['K']:2d} │ {r['bits']:4.1f} │ FAILED                              ║", flush=True)
        else:
            print(f"  ║ {r['K']:2d} │ {r['bits']:4.1f} │ {r['native_posdis']:.3f}  │ "
                  f"{r['transferred_posdis']:.3f}   │ {r['native_auc']:.3f} │ "
                  f"{r['transferred_auc']:.3f} │ {r['token_agreement']:.3f} ║", flush=True)
    print(f"  ╚═══════════════════════════════════╝", flush=True)

    # ─── Visualization ───
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in sweep_results if "status" not in r]
    if valid:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Phase 92c: Compression Sweep — Codebook Size vs Performance",
                     fontsize=13, fontweight='bold')

        ks = [r["K"] for r in valid]
        bits = [r["bits"] for r in valid]

        ax = axes[0]
        ax.plot(ks, [r["native_posdis"] for r in valid], 'bo-', label="Native V-JEPA 2")
        ax.plot(ks, [r["transferred_posdis"] for r in valid], 'rs-', label="Transferred DINOv2")
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label="Threshold (0.3)")
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label="Target (0.5)")
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("PosDis")
        ax.set_title("Compositionality vs Compression")
        ax.legend(fontsize=8)
        ax.set_xscale('log', base=2)

        ax = axes[1]
        ax.plot(ks, [r["native_auc"] for r in valid], 'bo-', label="Native")
        ax.plot(ks, [r["transferred_auc"] for r in valid], 'rs-', label="Transferred")
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("Mass probe AUC")
        ax.set_title("Task Performance vs Compression")
        ax.legend(fontsize=8)
        ax.set_xscale('log', base=2)

        ax = axes[2]
        ax.plot(ks, [r["token_agreement"] for r in valid], 'go-')
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("Token agreement")
        ax.set_title("Cross-Architecture Agreement")
        ax.set_xscale('log', base=2)

        plt.tight_layout()
        save_path = RESULTS_DIR / "phase92c_compression_sweep.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}", flush=True)

    results = {
        "sweep": sweep_results,
        "elapsed_min": (time.time() - t0) / 60,
    }
    save_path = RESULTS_DIR / "phase92c_compression_sweep.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def run_phase92():
    """Run all Phase 92 experiments sequentially."""
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 92: Cross-Architecture Isometry Extensions        ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t_total = time.time()

    results_a = run_phase92a()
    results_b = run_phase92b()
    results_c = run_phase92c()

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 92 complete. Total elapsed: {total_min:.1f} minutes", flush=True)
    print(f"{'='*60}", flush=True)

    return {"92a": results_a, "92b": results_b, "92c": results_c}


if __name__ == "__main__":
    run_phase92()

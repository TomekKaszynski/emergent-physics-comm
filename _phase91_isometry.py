"""
Phase 91: Cross-Architecture Isometry Test — V-JEPA 2 ↔ DINOv2
================================================================
Tests whether V-JEPA 2 and DINOv2 latent representations are approximately
linearly isometric. The critical gate for the JEPA protocol layer thesis.

Methodology (inspired by Social-JEPA):
  1. Extract/load frozen features from both encoders on shared Physics 101 spring clips
  2. Fit linear alignment W via ridge regression: z_dinov2 ≈ W @ z_vjepa2
  3. Evaluate: R², CKA, Procrustes distance, kNN neighborhood overlap
  4. Task transfer: train mass probe on V-JEPA 2, test on W-aligned DINOv2
  5. Compositional transfer: train comm protocol on V-JEPA 2, test on aligned DINOv2

Success criteria:
  - R² > 0.65 on linear alignment → PROCEED with protocol thesis
  - Task transfer accuracy within 15% of native → strong signal
  - Compositional transfer PosDis > 0.3 → killer result

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase91_isometry import run_phase91; run_phase91()"
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

# Add emergent-physics-comm to path for metrics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

# Architecture constants (match Phase 87)
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
# Architecture (identical to Phase 87)
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
# Step 1: Load and prepare features
# ═══════════════════════════════════════════════════════════════

def load_aligned_features():
    """Load V-JEPA 2 and DINOv2 features for the same spring clips."""
    print("═══ Step 1: Loading features ═══", flush=True)

    vjepa_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(
        RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)

    vjepa_feat = vjepa_data["features"].float()   # (206, 8, 1024)
    vjepa_objs = vjepa_data["obj_names"]           # list of 206
    mass_values = vjepa_data["mass_values"]         # (206,)
    dino_feat = dino_data["features"].float()      # (206, 384)
    dino_objs = dino_data["obj_names"]             # list of 206

    # Verify alignment
    assert vjepa_objs == dino_objs, "Object lists don't match!"
    print(f"  V-JEPA 2: {vjepa_feat.shape}, DINOv2: {dino_feat.shape}", flush=True)
    print(f"  {len(vjepa_objs)} clips, {len(set(vjepa_objs))} unique objects", flush=True)
    print(f"  Mass range: {mass_values.min():.1f} - {mass_values.max():.1f}g", flush=True)

    # Mean-pool V-JEPA 2 temporal features -> (206, 1024)
    vjepa_pooled = vjepa_feat.mean(dim=1)
    print(f"  V-JEPA 2 pooled: {vjepa_pooled.shape}", flush=True)

    return vjepa_feat, vjepa_pooled, dino_feat, vjepa_objs, mass_values


# ═══════════════════════════════════════════════════════════════
# Step 2: Linear alignment via ridge regression
# ═══════════════════════════════════════════════════════════════

def linear_cka(X, Y):
    """Compute linear CKA (Centered Kernel Alignment) between two feature matrices.
    X: (n, d1), Y: (n, d2). Returns scalar in [0, 1]."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def procrustes_distance(X, Y):
    """Orthogonal Procrustes distance after centering and scaling.
    Both X and Y must have the same dimensionality (use after projection)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    X = X / (np.linalg.norm(X, 'fro') + 1e-10)
    Y = Y / (np.linalg.norm(Y, 'fro') + 1e-10)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return float(1.0 - np.trace(U @ Vt) / min(X.shape[1], Y.shape[1]))


def knn_overlap(X, Y, k=10):
    """Fraction of k-nearest neighbors shared between X and Y spaces.
    X, Y: (n, d) feature matrices (may have different d)."""
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(X)
    nn_y = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    _, idx_x = nn_x.kneighbors(X)
    _, idx_y = nn_y.kneighbors(Y)
    # Exclude self (first neighbor)
    idx_x = idx_x[:, 1:]
    idx_y = idx_y[:, 1:]
    overlaps = []
    for i in range(len(X)):
        overlap = len(set(idx_x[i]) & set(idx_y[i])) / k
        overlaps.append(overlap)
    return float(np.mean(overlaps))


def fit_alignment(vjepa_pooled, dino_feat, n_splits=5, pca_dim=100):
    """Fit linear alignment W via ridge regression with cross-validation.
    Uses PCA pre-reduction on V-JEPA 2 to handle n<<d (206 samples, 1024 dims)."""
    print("\n═══ Step 2: Linear alignment (ridge regression) ═══", flush=True)

    X_raw = vjepa_pooled.numpy()  # (N, 1024) — source
    Y_raw = dino_feat.numpy()     # (N, 384) — target

    # Standardize
    scaler_x = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(Y_raw)
    X_s_full = scaler_x.transform(X_raw)
    Y_s = scaler_y.transform(Y_raw)

    # PCA pre-reduction on V-JEPA 2 to avoid n<<d overfitting
    from sklearn.decomposition import PCA as skPCA
    pca_reduce = skPCA(n_components=pca_dim).fit(X_s_full)
    X_s = pca_reduce.transform(X_s_full)
    var_explained = pca_reduce.explained_variance_ratio_.sum()
    print(f"  PCA: {X_raw.shape[1]}→{pca_dim} dims, {var_explained:.1%} variance retained", flush=True)

    # CKA and kNN on full-dimensional standardized features (robust to n<<d)
    cka = linear_cka(X_s_full, Y_s)
    print(f"  Linear CKA (full dim): {cka:.4f}", flush=True)
    knn = knn_overlap(X_s_full, Y_s, k=10)
    print(f"  kNN@10 overlap (full dim): {knn:.4f}", flush=True)

    # Cross-validated R² over multiple alphas (on PCA-reduced features)
    best_alpha = 1.0
    best_r2 = -1
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        r2s = []
        n = len(X_s)
        fold_size = n // n_splits
        for fold in range(n_splits):
            test_idx = list(range(fold * fold_size, min((fold + 1) * fold_size, n)))
            train_idx = [i for i in range(n) if i not in test_idx]
            model = Ridge(alpha=alpha).fit(X_s[train_idx], Y_s[train_idx])
            pred = model.predict(X_s[test_idx])
            r2 = r2_score(Y_s[test_idx], pred)
            r2s.append(r2)
        mean_r2 = np.mean(r2s)
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_alpha = alpha
        print(f"  alpha={alpha:>7.1f}: R² = {mean_r2:.4f} ± {np.std(r2s):.4f}", flush=True)

    print(f"  Best alpha: {best_alpha}, R²: {best_r2:.4f}", flush=True)

    # Fit final model on 80/20 split for reporting
    n = len(X_s)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    model_fwd = Ridge(alpha=best_alpha).fit(X_s[train_idx], Y_s[train_idx])
    pred_fwd = model_fwd.predict(X_s[test_idx])
    r2_fwd = r2_score(Y_s[test_idx], pred_fwd)

    # Reverse direction: DINOv2 -> V-JEPA 2
    model_rev = Ridge(alpha=best_alpha).fit(Y_s[train_idx], X_s[train_idx])
    pred_rev = model_rev.predict(Y_s[test_idx])
    r2_rev = r2_score(X_s[test_idx], pred_rev)

    print(f"  R² (V-JEPA→DINOv2): {r2_fwd:.4f}", flush=True)
    print(f"  R² (DINOv2→V-JEPA): {r2_rev:.4f}", flush=True)

    # Procrustes on aligned features (project V-JEPA 2 into DINOv2 space)
    X_projected = model_fwd.predict(X_s)
    proc_dist = procrustes_distance(X_projected, Y_s)
    print(f"  Procrustes distance (aligned): {proc_dist:.4f}", flush=True)

    # Per-dimension R² breakdown
    per_dim_r2 = []
    for d in range(Y_s.shape[1]):
        r2_d = r2_score(Y_s[test_idx, d], pred_fwd[:, d])
        per_dim_r2.append(r2_d)
    per_dim_r2 = np.array(per_dim_r2)
    print(f"  Per-dim R²: mean={per_dim_r2.mean():.4f}, "
          f"median={np.median(per_dim_r2):.4f}, "
          f"min={per_dim_r2.min():.4f}, max={per_dim_r2.max():.4f}", flush=True)
    print(f"  Dims with R²>0.5: {(per_dim_r2 > 0.5).sum()}/{len(per_dim_r2)}", flush=True)
    print(f"  Dims with R²>0.3: {(per_dim_r2 > 0.3).sum()}/{len(per_dim_r2)}", flush=True)

    results = {
        "best_alpha": best_alpha,
        "pca_dim": pca_dim,
        "pca_variance_retained": float(var_explained),
        "r2_cv": float(best_r2),
        "r2_fwd": float(r2_fwd),
        "r2_rev": float(r2_rev),
        "cka": float(cka),
        "knn_overlap_10": float(knn),
        "procrustes_distance": float(proc_dist),
        "per_dim_r2_mean": float(per_dim_r2.mean()),
        "per_dim_r2_median": float(np.median(per_dim_r2)),
        "dims_r2_above_0.5": int((per_dim_r2 > 0.5).sum()),
        "dims_r2_above_0.3": int((per_dim_r2 > 0.3).sum()),
        "total_dims": int(len(per_dim_r2)),
    }

    return model_fwd, model_rev, scaler_x, scaler_y, pca_reduce, results, train_idx, test_idx


# ═══════════════════════════════════════════════════════════════
# Step 3: Task transfer test
# ═══════════════════════════════════════════════════════════════

def task_transfer_test(vjepa_pooled, dino_feat, mass_values, obj_names,
                       model_fwd, scaler_x, scaler_y, pca_reduce):
    """Train mass probe on V-JEPA 2, test on aligned DINOv2 and native DINOv2."""
    print("\n═══ Step 3: Task transfer test ═══", flush=True)

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)
    n_seeds = 10

    results_native_vjepa = []
    results_native_dino = []
    results_transferred = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
        train_idx = [i for i, o in enumerate(obj_names) if o not in holdout_objs]
        test_idx = [i for i, o in enumerate(obj_names) if o in holdout_objs]
        if len(test_idx) < 4:
            continue

        # Prepare features
        vjepa_np = vjepa_pooled.numpy()
        dino_np = dino_feat.numpy()

        # Standardize and PCA-reduce V-JEPA 2
        vjepa_s = pca_reduce.transform(scaler_x.transform(vjepa_np))  # (N, pca_dim)
        dino_s = scaler_y.transform(dino_np)                          # (N, 384)

        # Train probe on PCA-reduced V-JEPA 2 features
        # Transfer: map DINOv2 into PCA-reduced V-JEPA space via reverse ridge
        model_rev_local = Ridge(alpha=10.0).fit(
            dino_s[train_idx], vjepa_s[train_idx])
        dino_as_vjepa = model_rev_local.predict(dino_s)

        train_feat_v = torch.tensor(vjepa_s[train_idx], dtype=torch.float32).to(DEVICE)
        test_feat_v = torch.tensor(vjepa_s[test_idx], dtype=torch.float32).to(DEVICE)
        test_feat_d = torch.tensor(dino_s[test_idx], dtype=torch.float32).to(DEVICE)
        test_feat_transfer = torch.tensor(
            dino_as_vjepa[test_idx], dtype=torch.float32).to(DEVICE)

        train_mass = torch.tensor(mass_values[train_idx], dtype=torch.float32).to(DEVICE)
        test_mass = mass_values[test_idx]

        feat_dim_v = train_feat_v.shape[1]
        feat_dim_d = test_feat_d.shape[1]

        # Train pairwise probe on V-JEPA 2
        torch.manual_seed(seed + 1000)
        probe_v = nn.Sequential(
            nn.Linear(feat_dim_v * 2, 64), nn.ReLU(), nn.Linear(64, 1),
        ).to(DEVICE)
        opt_v = torch.optim.Adam(probe_v.parameters(), lr=1e-3, weight_decay=1e-4)

        # Also train native DINOv2 probe
        probe_d = nn.Sequential(
            nn.Linear(feat_dim_d * 2, 64), nn.ReLU(), nn.Linear(64, 1),
        ).to(DEVICE)
        opt_d = torch.optim.Adam(probe_d.parameters(), lr=1e-3, weight_decay=1e-4)

        train_feat_d_native = torch.tensor(
            dino_s[train_idx], dtype=torch.float32).to(DEVICE)

        for ep in range(200):
            for probe, opt, feat in [
                (probe_v, opt_v, train_feat_v),
                (probe_d, opt_d, train_feat_d_native),
            ]:
                probe.train()
                ia = np.random.randint(0, len(train_idx), 64)
                ib = np.random.randint(0, len(train_idx), 64)
                same = ia == ib
                while same.any():
                    ib[same] = np.random.randint(0, len(train_idx), same.sum())
                    same = ia == ib
                fa, fb = feat[ia], feat[ib]
                label = (train_mass[ia] > train_mass[ib]).float()
                diff = torch.abs(train_mass[ia] - train_mass[ib])
                keep = diff > 0.5
                if keep.sum() < 4:
                    continue
                pred = probe(torch.cat([fa[keep], fb[keep]], dim=-1)).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, label[keep])
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Evaluate
        def eval_probe(probe, test_f, test_m):
            probe.eval()
            preds, labels = [], []
            with torch.no_grad():
                for i in range(len(test_m)):
                    for j in range(i + 1, len(test_m)):
                        if abs(test_m[i] - test_m[j]) < 0.5:
                            continue
                        fa = test_f[i:i+1]
                        fb = test_f[j:j+1]
                        pred = torch.sigmoid(
                            probe(torch.cat([fa, fb], dim=-1))).item()
                        preds.append(pred)
                        labels.append(float(test_m[i] > test_m[j]))
            if len(preds) > 5:
                try:
                    return roc_auc_score(labels, preds)
                except ValueError:
                    return 0.5
            return 0.5

        auc_v = eval_probe(probe_v, test_feat_v, test_mass)
        auc_d = eval_probe(probe_d, test_feat_d, test_mass)
        auc_transfer = eval_probe(probe_v, test_feat_transfer, test_mass)

        results_native_vjepa.append(auc_v)
        results_native_dino.append(auc_d)
        results_transferred.append(auc_transfer)

    mean_v = np.mean(results_native_vjepa)
    mean_d = np.mean(results_native_dino)
    mean_t = np.mean(results_transferred)
    drop = (mean_v - mean_t) / mean_v * 100

    print(f"  Native V-JEPA 2 probe AUC:     {mean_v:.3f} ± {np.std(results_native_vjepa):.3f}", flush=True)
    print(f"  Native DINOv2 probe AUC:        {mean_d:.3f} ± {np.std(results_native_dino):.3f}", flush=True)
    print(f"  Transferred (DINOv2→V-JEPA) AUC: {mean_t:.3f} ± {np.std(results_transferred):.3f}", flush=True)
    print(f"  Transfer accuracy drop: {drop:.1f}%", flush=True)
    within_15 = drop < 15
    print(f"  Within 15% threshold: {'YES ✓' if within_15 else 'NO ✗'}", flush=True)

    return {
        "native_vjepa_auc": f"{mean_v:.3f} ± {np.std(results_native_vjepa):.3f}",
        "native_dino_auc": f"{mean_d:.3f} ± {np.std(results_native_dino):.3f}",
        "transferred_auc": f"{mean_t:.3f} ± {np.std(results_transferred):.3f}",
        "transfer_drop_pct": float(drop),
        "within_15pct": within_15,
        "n_seeds": len(results_native_vjepa),
    }


# ═══════════════════════════════════════════════════════════════
# Step 4: Compositional transfer test
# ═══════════════════════════════════════════════════════════════

def train_comm_on_vjepa(vjepa_feat, mass_values, obj_names, n_agents=2, n_seeds=3):
    """Train communication protocol on V-JEPA 2 features. Return best sender."""
    print("\n  Training communication protocol on V-JEPA 2...", flush=True)

    n_frames = vjepa_feat.shape[1]
    fpa = n_frames // n_agents
    agent_views = [vjepa_feat[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

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
            TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
            HIDDEN_DIM, VOCAB_SIZE, N_HEADS
        ) for _ in range(n_agents)]
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
        t0 = time.time()

        for ep in range(COMM_EPOCHS):
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
                print(f"    Seed {seed} ep {ep+1}: holdout={acc:.1%} best={best_acc:.1%} "
                      f"({elapsed/60:.0f}min)", flush=True)

        if best_state and best_acc > best_global_acc:
            best_global_acc = best_acc
            best_sender_state = best_state

    print(f"  Best communication accuracy: {best_global_acc:.1%}", flush=True)
    return best_sender_state, n_agents


def compositional_transfer_test(vjepa_feat, dino_feat, mass_values, obj_names,
                                model_fwd, scaler_x, scaler_y, pca_reduce):
    """Train comm on V-JEPA 2, test compositionality on aligned DINOv2 features."""
    print("\n═══ Step 4: Compositional transfer test ═══", flush=True)

    # Train communication protocol on V-JEPA 2
    sender_state, n_agents = train_comm_on_vjepa(
        vjepa_feat, mass_values, obj_names, n_agents=2, n_seeds=3)

    if sender_state is None:
        print("  FAILED: No successful communication training", flush=True)
        return {"status": "FAILED"}

    n_frames = vjepa_feat.shape[1]
    fpa = n_frames // n_agents

    # Reconstruct sender
    senders = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi = MultiAgentSender(senders).to(DEVICE)
    multi.load_state_dict(sender_state)
    multi.eval()

    # Get tokens on native V-JEPA 2
    agent_views_v = [vjepa_feat[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    all_tokens_v = []
    with torch.no_grad():
        for i in range(0, len(vjepa_feat), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views_v]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens_v.append(np.stack(tokens, axis=1))
    all_tokens_v = np.concatenate(all_tokens_v, axis=0)

    # Create "fake" V-JEPA 2 temporal features from DINOv2 via alignment
    # The comm protocol sees raw (unstandardized) features, so we need to map
    # DINOv2 into raw V-JEPA 2 feature space
    # Pipeline: DINOv2 raw → standardize → ridge → unstandardize V-JEPA PCA → inverse PCA → inverse standardize
    # Simpler: fit a direct mapping from raw DINOv2 to raw V-JEPA pooled features
    dino_np = dino_feat.numpy()          # (N, 384)
    vjepa_np = vjepa_feat.mean(dim=1).numpy()  # (N, 1024) raw

    model_rev_raw = Ridge(alpha=100.0).fit(dino_np, vjepa_np)
    dino_as_vjepa = model_rev_raw.predict(dino_np)  # (N, 1024) in raw V-JEPA space

    # Replicate across temporal dimension to match expected input shape
    dino_as_vjepa_temporal = torch.tensor(
        dino_as_vjepa, dtype=torch.float32
    ).unsqueeze(1).expand(-1, n_frames, -1)  # (N, 8, 1024)

    agent_views_d = [dino_as_vjepa_temporal[:, i*fpa:(i+1)*fpa, :]
                     for i in range(n_agents)]

    all_tokens_d = []
    with torch.no_grad():
        for i in range(0, len(dino_feat), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views_d]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens_d.append(np.stack(tokens, axis=1))
    all_tokens_d = np.concatenate(all_tokens_d, axis=0)

    # Compute compositionality metrics
    # Bin mass into 5 bins for MI computation
    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    # Use mass_bins as both properties (single-property setup, duplicate for PosDis)
    # Actually for single-property, PosDis measures specialization of positions
    # We can use mass_bins and a volume/density proxy. Since we only have mass,
    # create a second "property" from object identity clusters
    unique_objs = sorted(set(obj_names))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    obj_bins = np.array([obj_to_idx[o] for o in obj_names])
    obj_bins_coarse = np.digitize(obj_bins,
                                   np.quantile(obj_bins, [0.2, 0.4, 0.6, 0.8]))

    from metrics import positional_disentanglement

    # PosDis on native V-JEPA 2 tokens
    attributes = np.stack([mass_bins, obj_bins_coarse], axis=1)
    pd_v, mi_v, ent_v = positional_disentanglement(all_tokens_v, attributes, VOCAB_SIZE)

    # PosDis on transferred DINOv2 tokens
    pd_d, mi_d, ent_d = positional_disentanglement(all_tokens_d, attributes, VOCAB_SIZE)

    # Token agreement: how often do native and transferred produce same tokens?
    agreement = (all_tokens_v == all_tokens_d).mean()

    print(f"  Native V-JEPA 2 PosDis:      {pd_v:.3f}", flush=True)
    print(f"  Transferred DINOv2 PosDis:    {pd_d:.3f}", flush=True)
    print(f"  Token agreement:              {agreement:.3f}", flush=True)
    print(f"  Native entropies:             {ent_v}", flush=True)
    print(f"  Transferred entropies:        {ent_d}", flush=True)
    print(f"  PosDis > 0.3 threshold:       {'YES ✓' if pd_d > 0.3 else 'NO ✗'}", flush=True)

    return {
        "native_posdis": float(pd_v),
        "transferred_posdis": float(pd_d),
        "token_agreement": float(agreement),
        "native_entropies": ent_v,
        "transferred_entropies": ent_d,
        "native_mi": mi_v.tolist() if hasattr(mi_v, 'tolist') else mi_v,
        "transferred_mi": mi_d.tolist() if hasattr(mi_d, 'tolist') else mi_d,
        "posdis_above_0.3": pd_d > 0.3,
    }


# ═══════════════════════════════════════════════════════════════
# Step 5: Visualizations
# ═══════════════════════════════════════════════════════════════

def create_visualizations(vjepa_pooled, dino_feat, mass_values, obj_names,
                          model_fwd, scaler_x, scaler_y, pca_reduce, alignment_results):
    """Create PCA/t-SNE visualizations, scatter plots."""
    print("\n═══ Step 5: Visualizations ═══", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = pca_reduce.transform(scaler_x.transform(vjepa_pooled.numpy()))  # (N, pca_dim)
    Y = scaler_y.transform(dino_feat.numpy())  # (N, 384)
    X_aligned = model_fwd.predict(X)  # (N, 384)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Phase 91: Cross-Architecture Isometry — V-JEPA 2 ↔ DINOv2",
                 fontsize=14, fontweight='bold')

    # Color by mass
    mass_norm = (mass_values - mass_values.min()) / (mass_values.max() - mass_values.min() + 1e-10)

    # 1. PCA of V-JEPA 2 features
    pca_v = PCA(n_components=2).fit_transform(X)
    ax = axes[0, 0]
    sc = ax.scatter(pca_v[:, 0], pca_v[:, 1], c=mass_norm, cmap='viridis', s=20, alpha=0.7)
    ax.set_title("V-JEPA 2 (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Mass (normalized)")

    # 2. PCA of DINOv2 features
    pca_d = PCA(n_components=2).fit_transform(Y)
    ax = axes[0, 1]
    sc = ax.scatter(pca_d[:, 0], pca_d[:, 1], c=mass_norm, cmap='viridis', s=20, alpha=0.7)
    ax.set_title("DINOv2 (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Mass (normalized)")

    # 3. PCA of aligned V-JEPA 2 features (projected into DINOv2 space)
    pca_a = PCA(n_components=2).fit(Y)
    aligned_pca = pca_a.transform(X_aligned)
    native_pca = pca_a.transform(Y)
    ax = axes[0, 2]
    ax.scatter(native_pca[:, 0], native_pca[:, 1], c='blue', s=20, alpha=0.3, label='DINOv2 native')
    ax.scatter(aligned_pca[:, 0], aligned_pca[:, 1], c='red', s=20, alpha=0.3, label='V-JEPA→DINOv2')
    ax.set_title("Aligned overlay (DINOv2 PCA space)")
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 4. Scatter: aligned vs native (first 3 PCs)
    ax = axes[1, 0]
    for pc in range(min(3, Y.shape[1])):
        ax.scatter(Y[:, pc], X_aligned[:, pc], s=5, alpha=0.3, label=f"Dim {pc}")
    lims = [min(Y.min(), X_aligned.min()), max(Y.max(), X_aligned.max())]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel("DINOv2 native")
    ax.set_ylabel("Aligned V-JEPA 2")
    ax.set_title(f"Alignment scatter (R²={alignment_results['r2_fwd']:.3f})")
    ax.legend(fontsize=8)

    # 5. Per-dimension R² histogram
    ax = axes[1, 1]
    per_dim_r2 = []
    # Recompute from alignment
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    pred = model_fwd.predict(X[test_idx])
    for d in range(Y.shape[1]):
        r2_d = r2_score(Y[test_idx, d], pred[:, d])
        per_dim_r2.append(r2_d)
    ax.hist(per_dim_r2, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', label='R²=0.5')
    ax.axvline(x=0.3, color='orange', linestyle='--', label='R²=0.3')
    ax.axvline(x=np.mean(per_dim_r2), color='green', linestyle='-',
               label=f'mean={np.mean(per_dim_r2):.3f}')
    ax.set_xlabel("R² per dimension")
    ax.set_ylabel("Count")
    ax.set_title("Per-dimension alignment quality")
    ax.legend(fontsize=8)

    # 6. Summary text box
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"Cross-Architecture Isometry Results\n"
        f"{'='*40}\n\n"
        f"R² (V-JEPA→DINOv2):  {alignment_results['r2_fwd']:.4f}\n"
        f"R² (DINOv2→V-JEPA):  {alignment_results['r2_rev']:.4f}\n"
        f"R² (cross-val):      {alignment_results['r2_cv']:.4f}\n"
        f"Linear CKA:          {alignment_results['cka']:.4f}\n"
        f"kNN@10 overlap:      {alignment_results['knn_overlap_10']:.4f}\n"
        f"Procrustes dist:     {alignment_results['procrustes_distance']:.4f}\n\n"
        f"Dims R²>0.5:  {alignment_results['dims_r2_above_0.5']}/{alignment_results['total_dims']}\n"
        f"Dims R²>0.3:  {alignment_results['dims_r2_above_0.3']}/{alignment_results['total_dims']}\n\n"
    )
    # Decision
    r2 = alignment_results['r2_cv']
    if r2 > 0.65:
        decision = "PROCEED with protocol thesis"
    elif r2 > 0.5:
        decision = "INVESTIGATE — run ablations"
    else:
        decision = "PIVOT to adapter/certification tooling"
    summary += f"Decision: {decision}"

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    save_path = RESULTS_DIR / "phase91_isometry_overview.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}", flush=True)

    # t-SNE visualization: compare DINOv2 native vs aligned V-JEPA 2 (both 384-dim)
    print("  Computing t-SNE...", flush=True)
    combined = np.vstack([Y, X_aligned])  # Both (N, 384)
    pca50 = PCA(n_components=min(50, combined.shape[1])).fit_transform(combined)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_emb = tsne.fit_transform(pca50)
    n = len(Y)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(tsne_emb[:n, 0], tsne_emb[:n, 1],
               c=mass_norm, cmap='Blues', s=20, alpha=0.5, label='DINOv2 native')
    ax.scatter(tsne_emb[n:, 0], tsne_emb[n:, 1],
               c=mass_norm, cmap='Reds', s=20, alpha=0.5, label='V-JEPA→DINOv2 aligned')
    ax.legend()
    ax.set_title("t-SNE: DINOv2 native vs V-JEPA 2 aligned to DINOv2 space")
    save_path = RESULTS_DIR / "phase91_tsne.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def run_phase91():
    """Run the full cross-architecture isometry test."""
    print("╔══════════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 91: Cross-Architecture Isometry — V-JEPA 2 ↔ DINOv2 ║", flush=True)
    print("╚══════════════════════════════════════════════════════════════╝", flush=True)
    t_start = time.time()

    # Step 1: Load features
    vjepa_feat, vjepa_pooled, dino_feat, obj_names, mass_values = load_aligned_features()

    # Step 2: Linear alignment
    (model_fwd, model_rev, scaler_x, scaler_y, pca_reduce,
     alignment_results, train_idx, test_idx) = fit_alignment(vjepa_pooled, dino_feat)

    # Decision gate
    r2 = alignment_results['r2_cv']
    print(f"\n  ╔═══ ALIGNMENT GATE ═══╗", flush=True)
    if r2 > 0.65:
        print(f"  ║ R²={r2:.4f} > 0.65    ║", flush=True)
        print(f"  ║ → PROCEED            ║", flush=True)
    elif r2 > 0.5:
        print(f"  ║ R²={r2:.4f} ∈ [0.5,0.65] ║", flush=True)
        print(f"  ║ → INVESTIGATE         ║", flush=True)
    else:
        print(f"  ║ R²={r2:.4f} < 0.5     ║", flush=True)
        print(f"  ║ → PIVOT              ║", flush=True)
    print(f"  ╚════════════════════╝", flush=True)

    # Step 3: Task transfer
    transfer_results = task_transfer_test(
        vjepa_pooled, dino_feat, mass_values, obj_names,
        model_fwd, scaler_x, scaler_y, pca_reduce)

    # Step 4: Compositional transfer (the killer test)
    comp_results = compositional_transfer_test(
        vjepa_feat, dino_feat, mass_values, obj_names,
        model_fwd, scaler_x, scaler_y, pca_reduce)

    # Step 5: Visualizations
    create_visualizations(
        vjepa_pooled, dino_feat, mass_values, obj_names,
        model_fwd, scaler_x, scaler_y, pca_reduce, alignment_results)

    # Save all results
    all_results = {
        "alignment": alignment_results,
        "task_transfer": transfer_results,
        "compositional_transfer": comp_results,
        "metadata": {
            "n_clips": len(obj_names),
            "n_unique_objects": len(set(obj_names)),
            "vjepa_dim": VJEPA_DIM,
            "dino_dim": DINO_DIM,
            "elapsed_minutes": (time.time() - t_start) / 60,
        }
    }

    save_path = RESULTS_DIR / "phase91_alignment_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}", flush=True)

    elapsed = (time.time() - t_start) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} minutes", flush=True)

    # Final summary
    print("\n╔══════════════════════════════════════════════════╗", flush=True)
    print("║             PHASE 91 SUMMARY                    ║", flush=True)
    print("╠══════════════════════════════════════════════════╣", flush=True)
    print(f"║ Alignment R² (CV):     {alignment_results['r2_cv']:.4f}                 ║", flush=True)
    print(f"║ Linear CKA:            {alignment_results['cka']:.4f}                 ║", flush=True)
    print(f"║ kNN@10 overlap:        {alignment_results['knn_overlap_10']:.4f}                 ║", flush=True)
    print(f"║ Transfer AUC drop:     {transfer_results.get('transfer_drop_pct', 'N/A')}%            ║", flush=True)
    if isinstance(comp_results, dict) and 'transferred_posdis' in comp_results:
        print(f"║ Transferred PosDis:    {comp_results['transferred_posdis']:.3f}                  ║", flush=True)
    print("╚══════════════════════════════════════════════════╝", flush=True)

    return all_results


if __name__ == "__main__":
    run_phase91()

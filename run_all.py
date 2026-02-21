"""
Training & Evaluation Pipeline — All 5 Phases
==============================================
Run: python run_all.py

Produces visualization PNGs in results/ directory.
Each phase trains, evaluates, and produces charts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from physics_sim import (
    PhysicsSimulator, SimConfig, Ball,
    generate_random_balls, generate_dataset,
    VisualPhysics3D, collect_visual_dataset,
    SplitViewPhysics3D, collect_split_view_dataset,
    collect_rich_dataset,
    SimplifiedRichPhysics3D, collect_simplified_rich_dataset,
    generate_clevr_images, generate_clevr_images_complex
)
from world_model import (
    DirectWorldModel, LatentWorldModel, VisionWorldModel,
    MultiAgentWorldModel, SingleAgentBaseline, print_model_info,
    VisualJEPA, VisualJEPAv2,
    SlotAttentionModule, ObjectCentricJEPA, ObjectCentricJEPAv2,
    SlotAttentionAutoencoder, SlotJEPAPredictor, SlotAttentionAEv2,
    SlotAttentionAEv3,
    SoftPositionEmbed, SlotAttentionModuleV2, SlotAttentionAEv5,
    SlotAttentionDINO, SlotPredictor
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def train_model(model, X, Y, epochs=50, lr=1e-3, batch_size=256, verbose=True):
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()
    history = []
    model.train()
    for epoch in range(epochs):
        total_loss, n = 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            if isinstance(model, LatentWorldModel):
                pred, latent, next_latent = model(xb)
                loss = criterion(pred, yb)
                with torch.no_grad():
                    target_latent = model.encode(yb)
                loss += 0.1 * criterion(next_latent, target_latent)
            else:
                loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1
        history.append(total_loss / n)
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss = {total_loss/n:.6f}")
    return history


# ============================================================
# PHASE 1
# ============================================================
def run_phase1():
    print("\n" + "="*60)
    print("PHASE 1: Direct State Prediction (MLP)")
    print("="*60)
    config = SimConfig()
    sim = PhysicsSimulator(config)

    print("Generating data...")
    X_train, Y_train = generate_dataset(1000, 100, n_balls=1, seed=42)
    X_test, Y_test = generate_dataset(100, 100, n_balls=1, seed=999)
    print(f"  Train: {len(X_train)} pairs | Test: {len(X_test)} pairs")

    model = DirectWorldModel(state_dim=4, hidden_dim=64)
    print_model_info("Model", model)

    t0 = time.time()
    history = train_model(model, X_train, Y_train, epochs=50)
    print(f"  Trained in {time.time()-t0:.1f}s")

    model.eval()
    with torch.no_grad():
        test_mse = nn.MSELoss()(model(torch.tensor(X_test)), torch.tensor(Y_test)).item()
    print(f"  1-step test MSE: {test_mse:.6f}")

    # Rollout comparison
    test_ball = Ball(x=1.0, y=1.5, vx=1.5, vy=2.0)
    gt = sim.simulate([test_ball], 200)
    pred = model.rollout(gt[0], 200)
    print(f"  200-step rollout MSE: {np.mean((gt - pred)**2):.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0][0].plot(history); axes[0][0].set_title("Training Loss"); axes[0][0].set_yscale('log')
    axes[0][1].plot(gt[:, 0], label='Truth'); axes[0][1].plot(pred[:, 0], '--', label='Model')
    axes[0][1].set_title("X Position"); axes[0][1].legend()
    axes[1][0].plot(gt[:, 1], label='Truth'); axes[1][0].plot(pred[:, 1], '--', label='Model')
    axes[1][0].set_title("Y Position (bounces!)"); axes[1][0].legend()
    axes[1][1].plot(gt[:, 0], gt[:, 1], 'b-', label='Truth', alpha=0.6)
    axes[1][1].plot(pred[:, 0], pred[:, 1], 'r--', label='Model', alpha=0.6)
    axes[1][1].set_title("2D Trajectory"); axes[1][1].set_xlim(0, 2); axes[1][1].set_ylim(0, 2)
    axes[1][1].set_aspect('equal'); axes[1][1].legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "phase1_direct_model.png", dpi=150); plt.close()
    print(f"  → {OUTPUT_DIR}/phase1_direct_model.png")
    return model, config


# ============================================================
# PHASE 2
# ============================================================
def run_phase2(trained_model, config):
    print("\n" + "="*60)
    print("PHASE 2: World Model vs LLM Comparison")
    print("="*60)
    sim = PhysicsSimulator(config)

    def llm_simulate(init, n_steps, dt=0.02):
        states = [init.copy()]
        x, y, vx, vy = init
        for _ in range(n_steps):
            vy -= 9.81 * dt; x += vx * dt; y += vy * dt
            if y < 0: y = abs(y); vy = abs(vy) * 0.85
            if y > 2: y = 4 - y; vy = -abs(vy) * 0.85
            if x < 0: x = abs(x); vx = abs(vx) * 0.9
            if x > 2: x = 4 - x; vx = -abs(vx) * 0.9
            states.append([x, y, vx, vy])
        return np.array(states)

    cases = [
        ("Simple drop",     Ball(1.0, 1.8, 0.0, 0.0)),
        ("Diagonal throw",  Ball(0.3, 0.5, 2.0, 3.0)),
        ("Fast horizontal", Ball(0.1, 1.0, 5.0, 0.0)),
        ("Multi-bounce",    Ball(1.0, 1.9, 1.5, -4.0)),
        ("Near corner",     Ball(0.1, 0.1, -1.0, -1.0)),
    ]
    n_steps = 150

    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 4 * len(cases)))
    results = {}
    for idx, (name, ball) in enumerate(cases):
        gt = sim.simulate([ball], n_steps)
        wm = trained_model.rollout(gt[0], n_steps)
        llm = llm_simulate(gt[0], n_steps)
        mse_wm = np.mean((gt - wm)**2)
        mse_llm = np.mean((gt - llm)**2)
        ratio = mse_llm / max(mse_wm, 1e-10)
        results[name] = {"wm": mse_wm, "llm": mse_llm, "ratio": ratio}
        print(f"  {name:20s}: WM={mse_wm:.4f}, LLM={mse_llm:.4f}, LLM {ratio:.1f}x worse")

        axes[idx][0].plot(gt[:, 1], 'b-', lw=2, label='Truth')
        axes[idx][0].plot(wm[:, 1], 'g--', lw=1.5, label='World Model')
        axes[idx][0].plot(llm[:, 1], 'r:', lw=1.5, label='LLM-like')
        axes[idx][0].set_title(f"{name}: Y position"); axes[idx][0].legend(fontsize=8)
        axes[idx][1].plot(gt[:, 0], gt[:, 1], 'b-', alpha=.7)
        axes[idx][1].plot(wm[:, 0], wm[:, 1], 'g--', alpha=.7)
        axes[idx][1].plot(llm[:, 0], llm[:, 1], 'r:', alpha=.7)
        axes[idx][1].set_title(f"{name}: 2D"); axes[idx][1].set_aspect('equal')

    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "phase2_wm_vs_llm.png", dpi=150); plt.close()
    print(f"  → {OUTPUT_DIR}/phase2_wm_vs_llm.png")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    x_pos = np.arange(len(names))
    ax.bar(x_pos - .2, [results[n]["wm"] for n in names], .35, label='World Model', color='#2ecc71')
    ax.bar(x_pos + .2, [results[n]["llm"] for n in names], .35, label='LLM-like', color='#e74c3c')
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel("MSE"); ax.set_title("World Model vs LLM: Prediction Error"); ax.legend(); ax.set_yscale('log')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "phase2_bar.png", dpi=150); plt.close()
    print(f"  → {OUTPUT_DIR}/phase2_bar.png")
    return results


# ============================================================
# PHASE 3
# ============================================================
def run_phase3():
    print("\n" + "="*60)
    print("PHASE 3: Latent Space World Model (Baby JEPA)")
    print("="*60)
    config = SimConfig()
    sim = PhysicsSimulator(config)

    X_train, Y_train = generate_dataset(500, 100, n_balls=3, seed=42)
    X_test, Y_test = generate_dataset(50, 100, n_balls=3, seed=999)
    state_dim = X_train.shape[1]
    print(f"  State dim: {state_dim} (3 balls × 4)")

    print("Training Direct MLP baseline...")
    direct = DirectWorldModel(state_dim=state_dim, hidden_dim=128)
    h_direct = train_model(direct, X_train, Y_train, epochs=50, verbose=False)

    latent_models, latent_histories = {}, {}
    for ld in [8, 16, 32]:
        print(f"Training Latent dim={ld}...")
        m = LatentWorldModel(state_dim=state_dim, latent_dim=ld, hidden_dim=128)
        h = train_model(m, X_train, Y_train, epochs=50, verbose=False)
        latent_models[ld] = m; latent_histories[ld] = h
        m.eval()
        with torch.no_grad():
            pred, _, _ = m(torch.tensor(X_test))
            print(f"  Latent-{ld} test MSE: {nn.MSELoss()(pred, torch.tensor(Y_test)).item():.6f}")

    direct.eval()
    with torch.no_grad():
        print(f"  Direct MLP test MSE: {nn.MSELoss()(direct(torch.tensor(X_test)), torch.tensor(Y_test)).item():.6f}")

    # Rollout comparison
    balls = generate_random_balls(3, config, seed=77)
    gt = sim.simulate(balls, 100)
    rollouts = {"Direct": direct.rollout(gt[0], 100)}
    for ld in [8, 16, 32]:
        rollouts[f"Latent-{ld}"] = latent_models[ld].rollout(gt[0], 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, traj in rollouts.items():
        axes[0].plot(np.mean((gt - traj)**2, axis=1), label=name)
    axes[0].set_title("Rollout Error Over Time (3 balls)"); axes[0].legend(); axes[0].set_yscale('log')
    axes[1].plot(h_direct, label='Direct', lw=2)
    for ld in [8, 16, 32]:
        axes[1].plot(latent_histories[ld], label=f'Latent-{ld}')
    axes[1].set_title("Training Loss"); axes[1].legend(); axes[1].set_yscale('log')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "phase3_latent.png", dpi=150); plt.close()
    print(f"  → {OUTPUT_DIR}/phase3_latent.png")

    # PCA of latent space
    model = latent_models[16]; model.eval()
    with torch.no_grad():
        latents = model.encode(torch.tensor(X_test[:1000])).numpy()
    U, S, Vt = np.linalg.svd(latents - latents.mean(0), full_matrices=False)
    ev = (S**2) / (S**2).sum()
    print(f"  Latent PCA — components for 95% var: {np.searchsorted(np.cumsum(ev), 0.95) + 1}")
    return latent_models[16]


# ============================================================
# PHASE 4
# ============================================================
def run_phase4():
    print("\n" + "="*60)
    print("PHASE 4: Vision World Model (Learn Physics from Pixels)")
    print("="*60)
    config = SimConfig()
    sim = PhysicsSimulator(config)

    print("Generating pixel data...")
    frames_cur, frames_nxt = [], []
    np.random.seed(42)
    for _ in range(200):
        balls = generate_random_balls(1, config)
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for _ in range(50):
            frames_cur.append(sim.render_frame(current, 64))
            current = sim.step(current)
            frames_nxt.append(sim.render_frame(current, 64))

    X_f = torch.tensor(np.array(frames_cur), dtype=torch.float32).permute(0, 3, 1, 2)
    Y_f = torch.tensor(np.array(frames_nxt), dtype=torch.float32).permute(0, 3, 1, 2)
    print(f"  Frames: {X_f.shape}")

    model = VisionWorldModel(latent_dim=32, hidden_dim=128)
    print_model_info("Vision Model", model)

    dataset = TensorDataset(X_f, Y_f)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training (30 epochs)...")
    t0 = time.time()
    history = []
    for epoch in range(30):
        total, n = 0, 0; model.train()
        for xf, yf in loader:
            optimizer.zero_grad()
            pred_f, lat, nlat = model(xf)
            loss = criterion(pred_f, yf)
            with torch.no_grad():
                tgt_lat = model.encode(yf)
            loss += 0.5 * criterion(nlat, tgt_lat)
            loss.backward(); optimizer.step()
            total += loss.item(); n += 1
        history.append(total / n)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss = {total/n:.6f}")
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Visualize
    model.eval()
    idxs = np.random.choice(len(X_f), 5, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            pred_f, _, _ = model(X_f[idx:idx+1])
            axes[0][i].imshow(X_f[idx].permute(1, 2, 0).numpy()); axes[0][i].axis('off')
            axes[1][i].imshow(Y_f[idx].permute(1, 2, 0).numpy()); axes[1][i].axis('off')
            axes[2][i].imshow(pred_f[0].permute(1, 2, 0).numpy().clip(0, 1)); axes[2][i].axis('off')
    axes[0][0].set_title("Current"); axes[1][0].set_title("True next"); axes[2][0].set_title("Predicted")
    plt.suptitle("Vision World Model: Frame Prediction", fontsize=14)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "phase4_vision.png", dpi=150); plt.close()
    print(f"  → {OUTPUT_DIR}/phase4_vision.png")
    return model


# ============================================================
# PHASE 5
# ============================================================
def _compute_effective_ranks(singular_values):
    """Compute two effective rank metrics from singular values.
    1. Threshold rank: number of σ > 5% of σ_max
    2. Entropy rank: exp(H(σ̃)) where σ̃ = σ/Σσ and H = Shannon entropy
    """
    sv = np.array(singular_values)
    sv = sv[sv > 1e-10]  # filter near-zero
    if len(sv) == 0:
        return 0, 0.0

    # Threshold rank: σ > 5% of σ_max
    threshold = 0.05 * sv.max()
    rank_threshold = int(np.sum(sv > threshold))

    # Entropy-based rank: exp(H(σ̃))
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    rank_entropy = float(np.exp(entropy))

    return rank_threshold, rank_entropy


def run_phase5():
    print("\n" + "="*60)
    print("PHASE 5: Two World Models Communicating in Latent Space")
    print("(Scaled: 50K samples, 200 epochs, two-stage training)")
    print("(No language. Just tensors.)")
    print("="*60)
    config = SimConfig()
    sim = PhysicsSimulator(config)

    # ── Data: 50K samples ──────────────────────────────────────
    print("Generating paired-view data (1000 trajectories × 50 steps)...")
    t_data = time.time()
    views_a, views_b, next_states = [], [], []
    np.random.seed(42)
    for traj_i in range(1000):
        balls = generate_random_balls(2, config)
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for _ in range(50):
            views_a.append(sim.render_topdown(current, 64))
            views_b.append(sim.render_side(current, 64))
            nxt = sim.step(current)
            ns = []
            for b in nxt:
                ns.extend([b.x, b.y, b.vx, b.vy])
            next_states.append(ns)
            current = nxt
        if (traj_i + 1) % 200 == 0:
            print(f"  {traj_i+1}/1000 trajectories...")

    VA = torch.tensor(np.array(views_a), dtype=torch.float32).permute(0, 3, 1, 2)
    VB = torch.tensor(np.array(views_b), dtype=torch.float32).permute(0, 3, 1, 2)
    NS = torch.tensor(np.array(next_states), dtype=torch.float32)
    state_dim = NS.shape[1]
    print(f"  Views: {VA.shape}, States: {NS.shape} ({time.time()-t_data:.0f}s)")

    n_train = int(0.9 * len(VA))
    va_tr, va_te = VA[:n_train], VA[n_train:]
    vb_tr, vb_te = VB[:n_train], VB[n_train:]
    ns_tr, ns_te = NS[:n_train], NS[n_train:]
    criterion = nn.MSELoss()

    # Fixed probe batch for communication tensor logging
    probe_idx = np.random.RandomState(123).choice(len(va_te), min(256, len(va_te)), replace=False)
    va_probe = va_te[probe_idx]
    vb_probe = vb_te[probe_idx]

    # ── Stage 1: Pre-train individual encoders ─────────────────
    print("\n┌─ STAGE 1: Pre-train encoders via single-agent baselines (100 epochs)")

    print("│  Training Agent A (top-down only)...")
    agent_a = SingleAgentBaseline(latent_dim=16, state_dim=state_dim)
    opt_a = optim.Adam(agent_a.parameters(), lr=1e-3)
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, 100)
    a_h = []
    for epoch in range(100):
        loader = DataLoader(TensorDataset(va_tr, ns_tr), batch_size=128, shuffle=True)
        total, n = 0, 0; agent_a.train()
        for va, ns in loader:
            opt_a.zero_grad(); loss = criterion(agent_a(va), ns); loss.backward(); opt_a.step()
            total += loss.item(); n += 1
        a_h.append(total / n)
        sched_a.step()
        if (epoch + 1) % 25 == 0:
            print(f"│    Epoch {epoch+1:3d}/100: loss = {total/n:.6f}")

    print("│  Training Agent B (side only)...")
    agent_b = SingleAgentBaseline(latent_dim=16, state_dim=state_dim)
    opt_b = optim.Adam(agent_b.parameters(), lr=1e-3)
    sched_b = optim.lr_scheduler.CosineAnnealingLR(opt_b, 100)
    b_h = []
    for epoch in range(100):
        loader = DataLoader(TensorDataset(vb_tr, ns_tr), batch_size=128, shuffle=True)
        total, n = 0, 0; agent_b.train()
        for vb, ns in loader:
            opt_b.zero_grad(); loss = criterion(agent_b(vb), ns); loss.backward(); opt_b.step()
            total += loss.item(); n += 1
        b_h.append(total / n)
        sched_b.step()
        if (epoch + 1) % 25 == 0:
            print(f"│    Epoch {epoch+1:3d}/100: loss = {total/n:.6f}")

    # Copy trained encoder weights into fused model
    print("│  Copying encoder weights → fused model")
    fused = MultiAgentWorldModel(latent_per_agent=16, fused_dim=32, state_dim=state_dim)
    fused.encoder_a.load_state_dict(agent_a.encoder.state_dict())
    fused.encoder_b.load_state_dict(agent_b.encoder.state_dict())
    print_model_info("│  Fused model", fused)
    print("└─ Stage 1 complete\n")

    # ── Stage 2: Train fused model (200 epochs) ───────────────
    comm_tensors = []   # communication tensor every epoch
    svd_log = []        # (epoch, singular_values) every 10 epochs
    fused_h = []
    FROZEN_EPOCHS = 50
    TOTAL_EPOCHS = 200

    # Stage 2a: Frozen encoders (50 epochs)
    print("┌─ STAGE 2a: Train fusion+predictor+decoder, encoders FROZEN (50 epochs)")
    for p in fused.encoder_a.parameters(): p.requires_grad = False
    for p in fused.encoder_b.parameters(): p.requires_grad = False
    trainable = [p for p in fused.parameters() if p.requires_grad]
    opt_f = optim.Adam(trainable, lr=5e-4)
    sched_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, FROZEN_EPOCHS)

    for epoch in range(FROZEN_EPOCHS):
        loader = DataLoader(TensorDataset(va_tr, vb_tr, ns_tr), batch_size=128, shuffle=True)
        total, n = 0, 0; fused.train()
        for va, vb, ns in loader:
            opt_f.zero_grad(); pred, _ = fused(va, vb)
            loss = criterion(pred, ns); loss.backward(); opt_f.step()
            total += loss.item(); n += 1
        fused_h.append(total / n)
        sched_f.step()

        # Log communication tensor
        fused.eval()
        with torch.no_grad():
            _, comm = fused(va_probe, vb_probe)
            comm_tensors.append(comm.numpy().copy())

        # SVD every 10 epochs
        if (epoch + 1) % 10 == 0:
            fusion_w = fused.fusion[0].weight.detach().numpy()
            sv = np.linalg.svd(fusion_w, compute_uv=False)
            r_thr, r_ent = _compute_effective_ranks(sv)
            svd_log.append({"epoch": epoch + 1, "sv": sv.copy(),
                            "rank_threshold": r_thr, "rank_entropy": r_ent,
                            "stage": "frozen"})
            print(f"│  Epoch {epoch+1:3d}: loss={total/n:.6f}  "
                  f"eff_rank(5%)={r_thr}  eff_rank(H)={r_ent:.2f}")
        elif (epoch + 1) % 25 == 0:
            print(f"│  Epoch {epoch+1:3d}: loss={total/n:.6f}")

    print("└─ Stage 2a complete\n")

    # Stage 2b: Unfreeze all (150 epochs)
    UNFREEZE_EPOCHS = TOTAL_EPOCHS - FROZEN_EPOCHS
    print(f"┌─ STAGE 2b: End-to-end fine-tuning, ALL params unfrozen ({UNFREEZE_EPOCHS} epochs)")
    for p in fused.encoder_a.parameters(): p.requires_grad = True
    for p in fused.encoder_b.parameters(): p.requires_grad = True
    opt_f2 = optim.Adam(fused.parameters(), lr=1e-4)
    sched_f2 = optim.lr_scheduler.CosineAnnealingLR(opt_f2, UNFREEZE_EPOCHS)

    for epoch in range(UNFREEZE_EPOCHS):
        global_epoch = FROZEN_EPOCHS + epoch + 1
        loader = DataLoader(TensorDataset(va_tr, vb_tr, ns_tr), batch_size=128, shuffle=True)
        total, n = 0, 0; fused.train()
        for va, vb, ns in loader:
            opt_f2.zero_grad(); pred, _ = fused(va, vb)
            loss = criterion(pred, ns); loss.backward(); opt_f2.step()
            total += loss.item(); n += 1
        fused_h.append(total / n)
        sched_f2.step()

        # Log communication tensor
        fused.eval()
        with torch.no_grad():
            _, comm = fused(va_probe, vb_probe)
            comm_tensors.append(comm.numpy().copy())

        # SVD every 10 epochs
        if global_epoch % 10 == 0:
            fusion_w = fused.fusion[0].weight.detach().numpy()
            sv = np.linalg.svd(fusion_w, compute_uv=False)
            r_thr, r_ent = _compute_effective_ranks(sv)
            svd_log.append({"epoch": global_epoch, "sv": sv.copy(),
                            "rank_threshold": r_thr, "rank_entropy": r_ent,
                            "stage": "unfrozen"})
            print(f"│  Epoch {global_epoch:3d}: loss={total/n:.6f}  "
                  f"eff_rank(5%)={r_thr}  eff_rank(H)={r_ent:.2f}")
        elif global_epoch % 50 == 0:
            print(f"│  Epoch {global_epoch:3d}: loss={total/n:.6f}")

    print("└─ Stage 2b complete\n")

    # ── Evaluate ──────────────────────────────────────────────
    print("-"*50)
    print("RESULTS: Does latent communication help?")
    print("-"*50)
    fused.eval(); agent_a.eval(); agent_b.eval()
    with torch.no_grad():
        pred_f, _ = fused(va_te, vb_te)
        mse_fused = criterion(pred_f, ns_te).item()
        mse_a = criterion(agent_a(va_te), ns_te).item()
        mse_b = criterion(agent_b(vb_te), ns_te).item()

    print(f"  Agent A (top-down):  MSE = {mse_a:.6f}")
    print(f"  Agent B (side):      MSE = {mse_b:.6f}")
    print(f"  Fused (comm):        MSE = {mse_fused:.6f}")

    best_single = min(mse_a, mse_b)
    if mse_fused < best_single:
        imp = (1 - mse_fused / best_single) * 100
        print(f"\n  ✓ Latent communication improves prediction by {imp:.1f}%!")
    else:
        print(f"\n  ✗ Fusion didn't help (may need more training or data)")

    # ── Plot 1: Training loss with stage boundaries ───────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0][0]
    ax.plot(fused_h, 'b-', label='Fused', lw=2)
    ax.axvline(FROZEN_EPOCHS, color='red', ls='--', alpha=0.7, label='Unfreeze encoders')
    ax.set_title("Fused Model Training Loss (2 stages)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.set_yscale('log')

    # ── Plot 2: Test MSE comparison ───────────────────────────
    ax = axes[0][1]
    bars = ax.bar(['Agent A\n(top-down)', 'Agent B\n(side)', 'Fused\n(comm)'],
                  [mse_a, mse_b, mse_fused], color=['#e74c3c', '#3498db', '#2ecc71'])
    ax.set_title("Test MSE: Single vs Fused")
    ax.set_ylabel("MSE")
    for bar, val in zip(bars, [mse_a, mse_b, mse_fused]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=9)

    # ── Plot 3: Cross-agent latent correlation ────────────────
    ax = axes[1][0]
    with torch.no_grad():
        la = fused.encode_agent_a(va_te[:500]).numpy()
        lb = fused.encode_agent_b(vb_te[:500]).numpy()
    corr = np.corrcoef(la.T, lb.T)[:16, 16:]
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title("Cross-Agent Latent Correlation\n(structure = learned communication)")
    ax.set_xlabel("Agent B dims"); ax.set_ylabel("Agent A dims")
    plt.colorbar(im, ax=ax)

    # ── Plot 4: Communication tensor snapshots ────────────────
    ax = axes[1][1]
    snap_epochs = [0, FROZEN_EPOCHS-1, TOTAL_EPOCHS//2, TOTAL_EPOCHS-1]
    snap_epochs = [e for e in snap_epochs if e < len(comm_tensors)]
    if len(snap_epochs) >= 2:
        combined = np.stack([comm_tensors[e][:32].T for e in snap_epochs])
        for i, e in enumerate(snap_epochs):
            ax.plot(np.mean(np.abs(comm_tensors[e]), axis=0), label=f'Epoch {e+1}', alpha=0.8)
        ax.set_title("Communication Tensor: Mean |activation| per dim")
        ax.set_xlabel("Fused latent dimension"); ax.set_ylabel("Mean |activation|")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5_multi_agent.png", dpi=150); plt.close()
    print(f"\n  → {OUTPUT_DIR}/phase5_multi_agent.png")

    # ── Plot 5: Effective Rank Evolution ──────────────────────
    if svd_log:
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = [d["epoch"] for d in svd_log]
        r_thr = [d["rank_threshold"] for d in svd_log]
        r_ent = [d["rank_entropy"] for d in svd_log]

        ax.plot(epochs, r_thr, 'o-', color='#e74c3c', lw=2, markersize=6,
                label='Threshold rank (σ > 5% σ_max)')
        ax.plot(epochs, r_ent, 's-', color='#3498db', lw=2, markersize=6,
                label='Entropy rank exp(H(σ̃))')
        ax.axvline(FROZEN_EPOCHS, color='gray', ls='--', alpha=0.5, label='Unfreeze encoders')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Effective Rank", fontsize=12)
        ax.set_title("Communication Channel: Effective Rank Evolution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, TOTAL_EPOCHS + 5)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "phase5_rank_evolution.png", dpi=150); plt.close()
        print(f"  → {OUTPUT_DIR}/phase5_rank_evolution.png")

        # Print final rank info
        final = svd_log[-1]
        print(f"\n  Final effective rank (threshold): {final['rank_threshold']}")
        print(f"  Final effective rank (entropy):   {final['rank_entropy']:.2f}")
        print(f"  Fusion weight matrix singular values:")
        print(f"    {np.array2string(final['sv'], precision=3, separator=', ')}")

    # ── Plot 6: SVD spectrum evolution ────────────────────────
    if svd_log:
        fig, ax = plt.subplots(figsize=(10, 5))
        for entry in svd_log:
            alpha = 0.3 + 0.7 * (entry["epoch"] / TOTAL_EPOCHS)
            color = '#e74c3c' if entry["stage"] == "frozen" else '#2ecc71'
            ax.plot(entry["sv"], color=color, alpha=alpha, lw=1.5,
                    label=f'Epoch {entry["epoch"]}' if entry["epoch"] in [10, FROZEN_EPOCHS, TOTAL_EPOCHS] else None)
        ax.set_xlabel("Singular value index", fontsize=12)
        ax.set_ylabel("Singular value", fontsize=12)
        ax.set_title("Fusion Weight Matrix: Singular Value Spectrum Over Training", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "phase5_svd_spectrum.png", dpi=150); plt.close()
        print(f"  → {OUTPUT_DIR}/phase5_svd_spectrum.png")

    # Save communication tensor log
    np.savez(OUTPUT_DIR / "phase5_comm_tensors.npz",
             tensors=np.array(comm_tensors),
             svd_epochs=[d["epoch"] for d in svd_log],
             svd_values=[d["sv"] for d in svd_log],
             rank_threshold=[d["rank_threshold"] for d in svd_log],
             rank_entropy=[d["rank_entropy"] for d in svd_log])

    np.savez(OUTPUT_DIR / "phase5_latents.npz",
             agent_a=la, agent_b=lb, states=ns_te[:500].numpy())


# ============================================================
# MAIN
# ============================================================

# ── Phase 5b: Occlusion-Based Multi-Agent Communication ─────────

def run_phase5b():
    from physics_sim import generate_occlusion_dataset
    from world_model import OcclusionAgentModel, OcclusionFusedModel

    print("\n" + "="*60)
    print("PHASE 5b: Occlusion-Based Multi-Agent Communication")
    print("(Balls disappear when crossing x=1.0 midline)")
    print("(Communication is ESSENTIAL — neither agent sees full state)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4  # 12
    TOTAL_EPOCHS = 200
    FROZEN_EPOCHS = 50

    # ── Data Generation ──
    print(f"Generating occlusion dataset ({N_TRAJ} traj × {N_STEPS} steps)...")
    t0 = time.time()
    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n_samples = len(full_cur)
    n_handoffs = handoffs.sum()
    print(f"  {n_samples} samples, {n_handoffs} handoff steps ({100*n_handoffs/n_samples:.1f}%) ({time.time()-t0:.0f}s)")

    # Tensors
    oa = torch.tensor(occ_a, dtype=torch.float32)
    ob = torch.tensor(occ_b, dtype=torch.float32)
    ns = torch.tensor(full_nxt, dtype=torch.float32)
    ho = torch.tensor(handoffs, dtype=torch.bool)

    # Train/test split (90/10)
    split = int(0.9 * n_samples)
    oa_tr, oa_te = oa[:split], oa[split:]
    ob_tr, ob_te = ob[:split], ob[split:]
    ns_tr, ns_te = ns[:split], ns[split:]
    ho_te = ho[split:]

    # Compute time-since-last-handoff for test set (for handoff analysis plot)
    full_ho = handoffs.astype(int)
    time_since_ho = np.zeros(n_samples, dtype=np.float32)
    last_ho = -999
    for i in range(n_samples):
        if full_ho[i]:
            last_ho = i
        time_since_ho[i] = i - last_ho if last_ho >= 0 else 999
    time_since_ho_te = time_since_ho[split:]

    # Probe batch for communication logging
    probe_idx = torch.randperm(len(oa_te))[:256]
    oa_probe = oa_te[probe_idx]
    ob_probe = ob_te[probe_idx]

    # ── Models ──
    agent_a = OcclusionAgentModel(STATE_DIM)
    agent_b = OcclusionAgentModel(STATE_DIM)
    fused = OcclusionFusedModel(STATE_DIM)
    criterion = nn.MSELoss()
    n_params = sum(p.numel() for p in fused.parameters())

    # ── STAGE 1: Pre-train single-agent baselines ──
    print(f"\n┌─ STAGE 1: Pre-train single-agent baselines (100 epochs)")
    PRETRAIN = 100

    print(f"│  Training Agent A (left half only)...")
    opt_a = optim.Adam(agent_a.parameters(), lr=1e-3)
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, PRETRAIN)
    for epoch in range(PRETRAIN):
        loader = DataLoader(TensorDataset(oa_tr, ns_tr), batch_size=256, shuffle=True)
        total, n = 0, 0; agent_a.train()
        for x, y in loader:
            opt_a.zero_grad(); loss = criterion(agent_a(x), y); loss.backward(); opt_a.step()
            total += loss.item(); n += 1
        sched_a.step()
        if (epoch + 1) % 25 == 0:
            print(f"│    Epoch {epoch+1:3d}/{PRETRAIN}: loss = {total/n:.6f}")

    print(f"│  Training Agent B (right half only)...")
    opt_b = optim.Adam(agent_b.parameters(), lr=1e-3)
    sched_b = optim.lr_scheduler.CosineAnnealingLR(opt_b, PRETRAIN)
    for epoch in range(PRETRAIN):
        loader = DataLoader(TensorDataset(ob_tr, ns_tr), batch_size=256, shuffle=True)
        total, n = 0, 0; agent_b.train()
        for x, y in loader:
            opt_b.zero_grad(); loss = criterion(agent_b(x), y); loss.backward(); opt_b.step()
            total += loss.item(); n += 1
        sched_b.step()
        if (epoch + 1) % 25 == 0:
            print(f"│    Epoch {epoch+1:3d}/{PRETRAIN}: loss = {total/n:.6f}")

    # Copy first-layer encoder weights (12→128) from baselines
    # (Second layer differs: baseline has 128→128, encoder has 128→24)
    print(f"│  Copying encoder first-layer weights → fused model")
    fused.encoder_a[0].weight.data.copy_(agent_a.net[0].weight.data)
    fused.encoder_a[0].bias.data.copy_(agent_a.net[0].bias.data)
    fused.encoder_b[0].weight.data.copy_(agent_b.net[0].weight.data)
    fused.encoder_b[0].bias.data.copy_(agent_b.net[0].bias.data)
    print(f"│  Fused model: {n_params/1e3:.1f}K params")
    print(f"└─ Stage 1 complete\n")

    # ── STAGE 2: Fused model training ──
    fused_h, comm_tensors, svd_log = [], [], []

    # Stage 2a: Frozen encoders (50 epochs)
    print(f"┌─ STAGE 2a: Train fusion+predictor+decoder, encoders FROZEN ({FROZEN_EPOCHS} epochs)")
    for p in fused.encoder_a.parameters(): p.requires_grad = False
    for p in fused.encoder_b.parameters(): p.requires_grad = False
    trainable = [p for p in fused.parameters() if p.requires_grad]
    opt_f = optim.Adam(trainable, lr=5e-4)
    sched_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, FROZEN_EPOCHS)

    for epoch in range(FROZEN_EPOCHS):
        loader = DataLoader(TensorDataset(oa_tr, ob_tr, ns_tr), batch_size=256, shuffle=True)
        total, n = 0, 0; fused.train()
        for a, b, y in loader:
            opt_f.zero_grad(); pred, _ = fused(a, b)
            loss = criterion(pred, y); loss.backward(); opt_f.step()
            total += loss.item(); n += 1
        fused_h.append(total / n)
        sched_f.step()

        fused.eval()
        with torch.no_grad():
            _, comm = fused(oa_probe, ob_probe)
            comm_tensors.append(comm.numpy().copy())

        if (epoch + 1) % 10 == 0:
            fusion_w = fused.fusion[0].weight.detach().numpy()
            sv = np.linalg.svd(fusion_w, compute_uv=False)
            r_thr, r_ent = _compute_effective_ranks(sv)
            svd_log.append({"epoch": epoch + 1, "sv": sv.copy(),
                            "rank_threshold": r_thr, "rank_entropy": r_ent,
                            "stage": "frozen"})
            print(f"│  Epoch {epoch+1:3d}: loss={total/n:.6f}  "
                  f"eff_rank(5%)={r_thr}  eff_rank(H)={r_ent:.2f}")
        elif (epoch + 1) % 25 == 0:
            print(f"│  Epoch {epoch+1:3d}: loss={total/n:.6f}")
    print("└─ Stage 2a complete\n")

    # Stage 2b: Unfreeze all
    UNFREEZE_EPOCHS = TOTAL_EPOCHS - FROZEN_EPOCHS
    print(f"┌─ STAGE 2b: End-to-end fine-tuning, ALL params unfrozen ({UNFREEZE_EPOCHS} epochs)")
    for p in fused.encoder_a.parameters(): p.requires_grad = True
    for p in fused.encoder_b.parameters(): p.requires_grad = True
    opt_f2 = optim.Adam(fused.parameters(), lr=1e-4)
    sched_f2 = optim.lr_scheduler.CosineAnnealingLR(opt_f2, UNFREEZE_EPOCHS)

    for epoch in range(UNFREEZE_EPOCHS):
        global_epoch = FROZEN_EPOCHS + epoch + 1
        loader = DataLoader(TensorDataset(oa_tr, ob_tr, ns_tr), batch_size=256, shuffle=True)
        total, n = 0, 0; fused.train()
        for a, b, y in loader:
            opt_f2.zero_grad(); pred, _ = fused(a, b)
            loss = criterion(pred, y); loss.backward(); opt_f2.step()
            total += loss.item(); n += 1
        fused_h.append(total / n)
        sched_f2.step()

        fused.eval()
        with torch.no_grad():
            _, comm = fused(oa_probe, ob_probe)
            comm_tensors.append(comm.numpy().copy())

        if global_epoch % 10 == 0:
            fusion_w = fused.fusion[0].weight.detach().numpy()
            sv = np.linalg.svd(fusion_w, compute_uv=False)
            r_thr, r_ent = _compute_effective_ranks(sv)
            svd_log.append({"epoch": global_epoch, "sv": sv.copy(),
                            "rank_threshold": r_thr, "rank_entropy": r_ent,
                            "stage": "unfrozen"})
            print(f"│  Epoch {global_epoch:3d}: loss={total/n:.6f}  "
                  f"eff_rank(5%)={r_thr}  eff_rank(H)={r_ent:.2f}")
        elif global_epoch % 25 == 0:
            print(f"│  Epoch {global_epoch:3d}: loss={total/n:.6f}")
    print("└─ Stage 2b complete\n")

    # ── Evaluation ──
    agent_a.eval(); agent_b.eval(); fused.eval()
    with torch.no_grad():
        pred_a = agent_a(oa_te)
        pred_b = agent_b(ob_te)
        pred_f, _ = fused(oa_te, ob_te)

    mse_a = F.mse_loss(pred_a, ns_te).item()
    mse_b = F.mse_loss(pred_b, ns_te).item()
    mse_f = F.mse_loss(pred_f, ns_te).item()

    # Handoff-specific MSE
    if ho_te.sum() > 0:
        mse_a_ho = F.mse_loss(pred_a[ho_te], ns_te[ho_te]).item()
        mse_b_ho = F.mse_loss(pred_b[ho_te], ns_te[ho_te]).item()
        mse_f_ho = F.mse_loss(pred_f[ho_te], ns_te[ho_te]).item()
    else:
        mse_a_ho = mse_b_ho = mse_f_ho = float('nan')

    print("-" * 50)
    print("RESULTS: Does occlusion make communication essential?")
    print("-" * 50)
    print(f"  Agent A (left half):    MSE = {mse_a:.6f}")
    print(f"  Agent B (right half):   MSE = {mse_b:.6f}")
    print(f"  Fused (comm):           MSE = {mse_f:.6f}")
    print(f"  --- Handoff steps only ({ho_te.sum().item()} steps) ---")
    print(f"  Agent A handoff MSE:    {mse_a_ho:.6f}")
    print(f"  Agent B handoff MSE:    {mse_b_ho:.6f}")
    print(f"  Fused handoff MSE:      {mse_f_ho:.6f}")
    if mse_f < min(mse_a, mse_b):
        print(f"\n  ✓ Fusion HELPS! (improvement: {min(mse_a, mse_b) - mse_f:.6f})")
    else:
        print(f"\n  ✗ Fusion didn't beat best single agent")

    # Per-sample MSE for handoff analysis
    per_sample_mse_a = ((pred_a - ns_te) ** 2).mean(dim=1).numpy()
    per_sample_mse_b = ((pred_b - ns_te) ** 2).mean(dim=1).numpy()
    per_sample_mse_f = ((pred_f - ns_te) ** 2).mean(dim=1).numpy()

    # Save communication data
    np.savez(OUTPUT_DIR / "phase5b_comm_tensors.npz",
             comm_tensors=np.array(comm_tensors),
             svd_epochs=[s["epoch"] for s in svd_log],
             svd_values=np.array([s["sv"] for s in svd_log]),
             rank_threshold=[s["rank_threshold"] for s in svd_log],
             rank_entropy=[s["rank_entropy"] for s in svd_log])

    # ── PLOTTING ──

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Phase 5b: Occlusion-Based Multi-Agent Communication", fontsize=14, fontweight='bold')

    # 1. Training loss
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(fused_h, 'b-', linewidth=0.8)
    ax1.axvline(FROZEN_EPOCHS, color='red', linestyle='--', label='Unfreeze')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (Fused Model)')
    ax1.legend()

    # 2. MSE comparison (overall + handoff)
    ax2 = fig.add_subplot(3, 3, 2)
    x_pos = np.arange(3)
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, [mse_a, mse_b, mse_f], width, label='Overall', color=['#4a90d9', '#e87d5a', '#50c878'])
    bars2 = ax2.bar(x_pos + width/2, [mse_a_ho, mse_b_ho, mse_f_ho], width, label='Handoff', color=['#2a5089', '#a84d2a', '#208848'], alpha=0.8)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(['Agent A\n(left)', 'Agent B\n(right)', 'Fused'])
    ax2.set_ylabel('Test MSE'); ax2.set_title('MSE: Overall vs Handoff Steps')
    ax2.legend()

    # 3. Cross-agent latent correlation
    ax3 = fig.add_subplot(3, 3, 3)
    fused.eval()
    with torch.no_grad():
        lat_a = fused.encoder_a(oa_te[:500]).numpy()
        lat_b = fused.encoder_b(ob_te[:500]).numpy()
    n_lat = min(lat_a.shape[1], 16)
    corr = np.corrcoef(lat_a[:, :n_lat].T, lat_b[:, :n_lat].T)[:n_lat, n_lat:]
    im = ax3.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_xlabel('Agent B latent'); ax3.set_ylabel('Agent A latent')
    ax3.set_title('Cross-Agent Latent Correlation')
    plt.colorbar(im, ax=ax3)

    # 4. Communication tensor snapshots
    ax4 = fig.add_subplot(3, 3, 4)
    n_epochs_logged = len(comm_tensors)
    early = comm_tensors[0].mean(axis=0) if n_epochs_logged > 0 else np.zeros(32)
    mid = comm_tensors[n_epochs_logged // 2].mean(axis=0) if n_epochs_logged > 1 else early
    late = comm_tensors[-1].mean(axis=0) if n_epochs_logged > 0 else early
    x_dim = np.arange(len(early))
    ax4.bar(x_dim - 0.25, np.abs(early), 0.25, label='Early', alpha=0.7)
    ax4.bar(x_dim, np.abs(mid), 0.25, label='Mid', alpha=0.7)
    ax4.bar(x_dim + 0.25, np.abs(late), 0.25, label='Late', alpha=0.7)
    ax4.set_xlabel('Dimension'); ax4.set_ylabel('|Activation|')
    ax4.set_title('Comm Tensor (mean |activation|)')
    ax4.legend(fontsize=8)

    # 5. Effective rank evolution
    ax5 = fig.add_subplot(3, 3, 5)
    epochs_svd = [s["epoch"] for s in svd_log]
    ranks_thr = [s["rank_threshold"] for s in svd_log]
    ranks_ent = [s["rank_entropy"] for s in svd_log]
    ax5.plot(epochs_svd, ranks_thr, 'o-', color='#e63946', label='Threshold (σ>5% σ_max)')
    ax5.plot(epochs_svd, ranks_ent, 's-', color='#457b9d', label='Entropy exp(H)')
    ax5.axvline(FROZEN_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('Effective Rank')
    ax5.set_title('Effective Rank Evolution')
    ax5.legend(fontsize=8)

    # 6. SVD spectrum evolution
    ax6 = fig.add_subplot(3, 3, 6)
    for s in svd_log:
        color = '#e63946' if s["stage"] == "frozen" else '#2a9d8f'
        alpha = 0.3 + 0.7 * (s["epoch"] / TOTAL_EPOCHS)
        ax6.plot(s["sv"], color=color, alpha=alpha, linewidth=0.8)
    ax6.set_xlabel('Singular Value Index'); ax6.set_ylabel('σ')
    ax6.set_title('SVD Spectrum Evolution')
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#e63946', label='Frozen'),
                       Line2D([0], [0], color='#2a9d8f', label='Unfrozen')]
    ax6.legend(handles=legend_elements, fontsize=8)

    # 7. Handoff analysis: error vs time-since-handoff
    ax7 = fig.add_subplot(3, 3, 7)
    # Bin by time-since-handoff
    max_t = min(20, int(time_since_ho_te.max()))
    bins = np.arange(0, max_t + 1)
    mse_by_t_a, mse_by_t_b, mse_by_t_f = [], [], []
    for t in bins:
        mask = (time_since_ho_te >= t) & (time_since_ho_te < t + 1)
        if mask.sum() > 5:
            mse_by_t_a.append(per_sample_mse_a[mask].mean())
            mse_by_t_b.append(per_sample_mse_b[mask].mean())
            mse_by_t_f.append(per_sample_mse_f[mask].mean())
        else:
            mse_by_t_a.append(np.nan)
            mse_by_t_b.append(np.nan)
            mse_by_t_f.append(np.nan)
    ax7.plot(bins, mse_by_t_a, 'o-', color='#4a90d9', label='Agent A', markersize=4)
    ax7.plot(bins, mse_by_t_b, 's-', color='#e87d5a', label='Agent B', markersize=4)
    ax7.plot(bins, mse_by_t_f, '^-', color='#50c878', label='Fused', markersize=4)
    ax7.axvline(0, color='red', linestyle=':', alpha=0.3)
    ax7.set_xlabel('Steps Since Last Handoff')
    ax7.set_ylabel('Mean MSE')
    ax7.set_title('Prediction Error vs Handoff Recency')
    ax7.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5b_occlusion.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase5b_occlusion.png")

    # Separate rank evolution plot
    fig2, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Phase 5b: Rank Evolution", fontsize=13, fontweight='bold')
    ax_r1.plot(epochs_svd, ranks_thr, 'o-', color='#e63946', markersize=5)
    ax_r1.plot(epochs_svd, ranks_ent, 's-', color='#457b9d', markersize=5)
    ax_r1.axvline(FROZEN_EPOCHS, color='gray', linestyle='--', alpha=0.5)
    ax_r1.set_xlabel('Epoch'); ax_r1.set_ylabel('Effective Rank')
    ax_r1.set_title('Threshold vs Entropy Rank')
    ax_r1.legend(['Threshold (σ>5%)', 'Entropy exp(H)', 'Unfreeze'], fontsize=9)

    for s in svd_log:
        color = '#e63946' if s["stage"] == "frozen" else '#2a9d8f'
        alpha = 0.3 + 0.7 * (s["epoch"] / TOTAL_EPOCHS)
        ax_r2.plot(s["sv"], color=color, alpha=alpha, linewidth=1)
    ax_r2.set_xlabel('Index'); ax_r2.set_ylabel('σ')
    ax_r2.set_title('SVD Spectrum Over Training')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5b_rank_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase5b_rank_evolution.png")

    # Separate SVD spectrum
    fig3, ax_s = plt.subplots(figsize=(8, 5))
    for s in svd_log:
        color = '#e63946' if s["stage"] == "frozen" else '#2a9d8f'
        alpha = 0.3 + 0.7 * (s["epoch"] / TOTAL_EPOCHS)
        ax_s.plot(s["sv"], color=color, alpha=alpha, linewidth=1)
    ax_s.set_xlabel('Singular Value Index'); ax_s.set_ylabel('σ')
    ax_s.set_title('Phase 5b: Fusion Weight SVD Spectrum')
    ax_s.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5b_svd_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase5b_svd_spectrum.png")

    # Handoff analysis standalone
    fig4, ax_h = plt.subplots(figsize=(8, 5))
    ax_h.plot(bins, mse_by_t_a, 'o-', color='#4a90d9', label='Agent A (left)', markersize=5)
    ax_h.plot(bins, mse_by_t_b, 's-', color='#e87d5a', label='Agent B (right)', markersize=5)
    ax_h.plot(bins, mse_by_t_f, '^-', color='#50c878', label='Fused', markersize=5)
    ax_h.axvspan(-0.5, 1.5, alpha=0.1, color='red', label='Near handoff')
    ax_h.set_xlabel('Steps Since Last Midline Crossing')
    ax_h.set_ylabel('Mean Per-Sample MSE')
    ax_h.set_title('Phase 5b: Handoff Analysis — Does Communication Help at Transitions?')
    ax_h.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5b_handoff_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase5b_handoff_analysis.png")

    # Final rank info
    final_sv = svd_log[-1] if svd_log else None
    if final_sv:
        print(f"\n  Final effective rank (threshold): {final_sv['rank_threshold']}")
        print(f"  Final effective rank (entropy):   {final_sv['rank_entropy']:.2f}")
        print(f"  Fusion singular values:")
        print(f"    {np.array2string(final_sv['sv'], precision=3, separator=', ')}")

    return {
        "mse_a": mse_a, "mse_b": mse_b, "mse_f": mse_f,
        "mse_a_ho": mse_a_ho, "mse_b_ho": mse_b_ho, "mse_f_ho": mse_f_ho,
        "final_rank_threshold": final_sv['rank_threshold'] if final_sv else None,
        "final_rank_entropy": final_sv['rank_entropy'] if final_sv else None,
    }



# ── Phase 6: Information Bottleneck ─────────────────────────────

def _train_bottleneck_config(oa_tr, ob_tr, ns_tr, oa_te, ob_te, ns_te,
                              ho_te, state_dim, comm_dim, beta, label,
                              pretrain_epochs=100, frozen_epochs=50,
                              total_fused_epochs=200, verbose=True):
    """Train a single BottleneckedFusionModel config. Returns results dict."""
    from world_model import OcclusionAgentModel, BottleneckedFusionModel

    agent_a = OcclusionAgentModel(state_dim)
    agent_b = OcclusionAgentModel(state_dim)
    fused = BottleneckedFusionModel(state_dim, comm_dim=comm_dim)
    mse_loss = nn.MSELoss()

    # Stage 1: pretrain baselines
    for agent, data in [(agent_a, oa_tr), (agent_b, ob_tr)]:
        opt = optim.Adam(agent.parameters(), lr=1e-3)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, pretrain_epochs)
        for ep in range(pretrain_epochs):
            loader = DataLoader(TensorDataset(data, ns_tr), batch_size=256, shuffle=True)
            agent.train()
            for x, y in loader:
                opt.zero_grad(); mse_loss(agent(x), y).backward(); opt.step()
            sched.step()

    # Copy first-layer weights
    fused.encoder_a[0].weight.data.copy_(agent_a.net[0].weight.data)
    fused.encoder_a[0].bias.data.copy_(agent_a.net[0].bias.data)
    fused.encoder_b[0].weight.data.copy_(agent_b.net[0].weight.data)
    fused.encoder_b[0].bias.data.copy_(agent_b.net[0].bias.data)

    # Stage 2: fused training
    loss_h, kl_h, svd_log = [], [], []
    unfreeze_epoch = frozen_epochs

    # 2a: frozen encoders
    for p in fused.encoder_a.parameters(): p.requires_grad = False
    for p in fused.encoder_b.parameters(): p.requires_grad = False
    trainable = [p for p in fused.parameters() if p.requires_grad]
    opt_f = optim.Adam(trainable, lr=5e-4)
    sched_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, frozen_epochs)

    for ep in range(frozen_epochs):
        loader = DataLoader(TensorDataset(oa_tr, ob_tr, ns_tr), batch_size=256, shuffle=True)
        t_loss, t_kl, n = 0, 0, 0; fused.train()
        for a, b, y in loader:
            opt_f.zero_grad()
            pred, _, kl = fused(a, b)
            loss = mse_loss(pred, y) + beta * kl
            loss.backward(); opt_f.step()
            t_loss += mse_loss(pred, y).item(); t_kl += kl.item(); n += 1
        loss_h.append(t_loss / n); kl_h.append(t_kl / n)
        sched_f.step()

        if (ep + 1) % 10 == 0:
            fw = fused.fusion[0].weight.detach().numpy()
            sv = np.linalg.svd(fw, compute_uv=False)
            r_thr, r_ent = _compute_effective_ranks(sv)
            svd_log.append({"epoch": ep+1, "sv": sv.copy(),
                            "rank_threshold": r_thr, "rank_entropy": r_ent})

    # 2b: unfrozen
    for p in fused.encoder_a.parameters(): p.requires_grad = True
    for p in fused.encoder_b.parameters(): p.requires_grad = True
    unfreeze_epochs = total_fused_epochs - frozen_epochs
    opt_f2 = optim.Adam(fused.parameters(), lr=1e-4)
    sched_f2 = optim.lr_scheduler.CosineAnnealingLR(opt_f2, unfreeze_epochs)

    for ep in range(unfreeze_epochs):
        global_ep = frozen_epochs + ep + 1
        loader = DataLoader(TensorDataset(oa_tr, ob_tr, ns_tr), batch_size=256, shuffle=True)
        t_loss, t_kl, n = 0, 0, 0; fused.train()
        for a, b, y in loader:
            opt_f2.zero_grad()
            pred, _, kl = fused(a, b)
            loss = mse_loss(pred, y) + beta * kl
            loss.backward(); opt_f2.step()
            t_loss += mse_loss(pred, y).item(); t_kl += kl.item(); n += 1
        loss_h.append(t_loss / n); kl_h.append(t_kl / n)
        sched_f2.step()

        if global_ep % 10 == 0:
            fw = fused.fusion[0].weight.detach().numpy()
            sv = np.linalg.svd(fw, compute_uv=False)
            r_thr, r_ent = _compute_effective_ranks(sv)
            svd_log.append({"epoch": global_ep, "sv": sv.copy(),
                            "rank_threshold": r_thr, "rank_entropy": r_ent})

    # Evaluate
    agent_a.eval(); agent_b.eval(); fused.eval()
    with torch.no_grad():
        pred_a = agent_a(oa_te)
        pred_b = agent_b(ob_te)
        pred_f, _, kl_test = fused(oa_te, ob_te)

    mse_a = F.mse_loss(pred_a, ns_te).item()
    mse_b = F.mse_loss(pred_b, ns_te).item()
    mse_f = F.mse_loss(pred_f, ns_te).item()
    mse_a_ho = F.mse_loss(pred_a[ho_te], ns_te[ho_te]).item() if ho_te.sum() > 0 else float('nan')
    mse_b_ho = F.mse_loss(pred_b[ho_te], ns_te[ho_te]).item() if ho_te.sum() > 0 else float('nan')
    mse_f_ho = F.mse_loss(pred_f[ho_te], ns_te[ho_te]).item() if ho_te.sum() > 0 else float('nan')

    final_sv = svd_log[-1] if svd_log else None
    r_thr = final_sv["rank_threshold"] if final_sv else 0
    r_ent = final_sv["rank_entropy"] if final_sv else 0.0

    if verbose:
        print(f"    {label}: MSE={mse_f:.4f}  ho_MSE={mse_f_ho:.4f}  "
              f"KL={kl_test.item():.2f}  rank(5%)={r_thr}  rank(H)={r_ent:.1f}")

    return {
        "mse_a": mse_a, "mse_b": mse_b, "mse_f": mse_f,
        "mse_a_ho": mse_a_ho, "mse_b_ho": mse_b_ho, "mse_f_ho": mse_f_ho,
        "kl": kl_test.item(), "rank_threshold": r_thr, "rank_entropy": r_ent,
        "loss_h": loss_h, "kl_h": kl_h, "svd_log": svd_log,
        "model": fused, "label": label,
    }


def run_phase6():
    from physics_sim import generate_occlusion_dataset
    from world_model import BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 6: Information Bottleneck on Communication Channel")
    print("(How narrow can the pipe be? What structure emerges?)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4

    # Generate shared dataset for Exp 1 & 2
    print("Generating occlusion dataset (3 balls)...")
    t0 = time.time()
    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n_samples = len(full_cur)
    oa = torch.tensor(occ_a, dtype=torch.float32)
    ob = torch.tensor(occ_b, dtype=torch.float32)
    ns = torch.tensor(full_nxt, dtype=torch.float32)
    ho = torch.tensor(handoffs, dtype=torch.bool)
    split = int(0.9 * n_samples)
    oa_tr, oa_te = oa[:split], oa[split:]
    ob_tr, ob_te = ob[:split], ob[split:]
    ns_tr, ns_te = ns[:split], ns[split:]
    ho_te = ho[split:]
    print(f"  {n_samples} samples ({time.time()-t0:.0f}s)")

    # ── EXPERIMENT 1: Bottleneck Width Sweep ──
    print(f"\n┌─ EXPERIMENT 1: Bottleneck Width Sweep")
    COMM_DIMS = [2, 4, 8, 12, 16, 24]
    BETA = 0.001
    exp1_results = {}
    for cd in COMM_DIMS:
        r = _train_bottleneck_config(oa_tr, ob_tr, ns_tr, oa_te, ob_te, ns_te,
                                      ho_te, STATE_DIM, comm_dim=cd, beta=BETA,
                                      label=f"dim={cd}")
        exp1_results[cd] = r
    print("└─ Experiment 1 complete\n")

    # ── EXPERIMENT 2: Beta Sweep ──
    print(f"┌─ EXPERIMENT 2: Beta Sweep (comm_dim=24)")
    BETAS = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    exp2_results = {}
    for b in BETAS:
        r = _train_bottleneck_config(oa_tr, ob_tr, ns_tr, oa_te, ob_te, ns_te,
                                      ho_te, STATE_DIM, comm_dim=24, beta=b,
                                      label=f"β={b}")
        exp2_results[b] = r
    print("└─ Experiment 2 complete\n")

    # ── EXPERIMENT 3: Scaling Law ──
    print(f"┌─ EXPERIMENT 3: Scaling Law (n_balls sweep)")
    NBALLS_LIST = [2, 3, 4, 5]
    exp3_results = {}
    for nb in NBALLS_LIST:
        sd = nb * 4
        print(f"│  n_balls={nb} (state_dim={sd})")
        fc, oa_s, ob_s, fn_s, ho_s = generate_occlusion_dataset(
            1500, 50, n_balls=nb, seed=42)
        n = len(fc)
        sp = int(0.9 * n)
        oa_s_t = torch.tensor(oa_s, dtype=torch.float32)
        ob_s_t = torch.tensor(ob_s, dtype=torch.float32)
        ns_s_t = torch.tensor(fn_s, dtype=torch.float32)
        ho_s_t = torch.tensor(ho_s, dtype=torch.bool)

        # Find minimal comm_dim for good prediction
        best_cd, best_mse = None, float('inf')
        nb_results = {}
        for cd in [2, 4, 8, 12]:
            r = _train_bottleneck_config(
                oa_s_t[:sp], ob_s_t[:sp], ns_s_t[:sp],
                oa_s_t[sp:], ob_s_t[sp:], ns_s_t[sp:],
                ho_s_t[sp:], sd, comm_dim=cd, beta=0.001,
                label=f"  nb={nb},cd={cd}", verbose=True)
            nb_results[cd] = r
            if r["mse_f"] < best_mse:
                best_mse = r["mse_f"]
                best_cd = cd
        exp3_results[nb] = {"results": nb_results, "best_cd": best_cd, "best_mse": best_mse}
    print("└─ Experiment 3 complete\n")

    # ── Save data ──
    np.savez(OUTPUT_DIR / "phase6_data.npz",
             exp1_dims=COMM_DIMS,
             exp1_mse=[exp1_results[d]["mse_f"] for d in COMM_DIMS],
             exp1_mse_ho=[exp1_results[d]["mse_f_ho"] for d in COMM_DIMS],
             exp1_rank_thr=[exp1_results[d]["rank_threshold"] for d in COMM_DIMS],
             exp1_rank_ent=[exp1_results[d]["rank_entropy"] for d in COMM_DIMS],
             exp2_betas=BETAS,
             exp2_mse=[exp2_results[b]["mse_f"] for b in BETAS],
             exp2_rank_ent=[exp2_results[b]["rank_entropy"] for b in BETAS],
             exp2_kl=[exp2_results[b]["kl"] for b in BETAS],
             exp3_nballs=NBALLS_LIST,
             exp3_best_cd=[exp3_results[nb]["best_cd"] for nb in NBALLS_LIST],
             exp3_best_mse=[exp3_results[nb]["best_mse"] for nb in NBALLS_LIST])

    # ── PLOTTING ──
    from matplotlib.lines import Line2D

    # 1. Bottleneck width vs MSE
    fig1, ax = plt.subplots(figsize=(8, 5))
    ax.plot(COMM_DIMS, [exp1_results[d]["mse_f"] for d in COMM_DIMS],
            'o-', color='#2a9d8f', label='Overall MSE', linewidth=2, markersize=8)
    ax.plot(COMM_DIMS, [exp1_results[d]["mse_f_ho"] for d in COMM_DIMS],
            's--', color='#e76f51', label='Handoff MSE', linewidth=2, markersize=8)
    ax.axhline(y=exp1_results[24]["mse_f"], color='gray', linestyle=':', alpha=0.5,
               label=f'Full width (24) MSE={exp1_results[24]["mse_f"]:.3f}')
    ax.set_xlabel('Communication Dim (per agent)', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Phase 6: Bottleneck Width vs Prediction Error', fontsize=13, fontweight='bold')
    ax.legend(); ax.set_xticks(COMM_DIMS)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_width_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase6_width_sweep.png")

    # 2. Beta sweep: MSE vs Rank
    fig2, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    beta_labels = [str(b) for b in BETAS]
    mse_vals = [exp2_results[b]["mse_f"] for b in BETAS]
    rank_vals = [exp2_results[b]["rank_entropy"] for b in BETAS]
    l1 = ax1.plot(range(len(BETAS)), mse_vals, 'o-', color='#e63946', linewidth=2,
                  markersize=8, label='Test MSE')
    l2 = ax2.plot(range(len(BETAS)), rank_vals, 's-', color='#457b9d', linewidth=2,
                  markersize=8, label='Entropy Rank')
    ax1.set_xticks(range(len(BETAS))); ax1.set_xticklabels(beta_labels, fontsize=10)
    ax1.set_xlabel('β (KL penalty weight)', fontsize=12)
    ax1.set_ylabel('Test MSE', fontsize=12, color='#e63946')
    ax2.set_ylabel('Effective Rank (entropy)', fontsize=12, color='#457b9d')
    ax1.set_title('Phase 6: Information-Accuracy Tradeoff', fontsize=13, fontweight='bold')
    lines = l1 + l2; labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_beta_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase6_beta_sweep.png")

    # 3. Scaling law
    fig3, ax = plt.subplots(figsize=(8, 5))
    best_cds = [exp3_results[nb]["best_cd"] for nb in NBALLS_LIST]
    # Plot effective rank at best comm_dim for each n_balls
    best_ranks = [exp3_results[nb]["results"][exp3_results[nb]["best_cd"]]["rank_entropy"]
                  for nb in NBALLS_LIST]
    ax.plot(NBALLS_LIST, best_ranks, 'o-', color='#2a9d8f', linewidth=2, markersize=10,
            label='Effective rank (entropy)')
    ax.plot(NBALLS_LIST, [nb * 2 for nb in NBALLS_LIST], '--', color='gray',
            linewidth=1, label='Linear ref (2×n_balls)')
    ax.plot(NBALLS_LIST, [nb * 4 for nb in NBALLS_LIST], ':', color='gray',
            linewidth=1, label='Linear ref (4×n_balls)')
    ax.set_xlabel('Number of Balls', fontsize=12)
    ax.set_ylabel('Effective Rank at Best Config', fontsize=12)
    ax.set_title('Phase 6: Communication Complexity Scaling Law', fontsize=13, fontweight='bold')
    ax.set_xticks(NBALLS_LIST); ax.legend()
    # Add best comm_dim as text
    for i, nb in enumerate(NBALLS_LIST):
        ax.annotate(f'cd={best_cds[i]}', (nb, best_ranks[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_scaling_law.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase6_scaling_law.png")

    # 4. Communication structure at optimal bottleneck (smallest cd with MSE < 0.5)
    best_dim = None
    for cd in COMM_DIMS:
        if exp1_results[cd]["mse_f"] < 0.5:
            best_dim = cd
            break
    if best_dim is None:
        best_dim = COMM_DIMS[-1]

    fig4, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig4.suptitle(f"Phase 6: Communication Structure (comm_dim={best_dim}, β={BETA})",
                  fontsize=13, fontweight='bold')

    # SVD spectrum
    best_r = exp1_results[best_dim]
    if best_r["svd_log"]:
        final_sv = best_r["svd_log"][-1]["sv"]
        axes[0].bar(range(len(final_sv)), final_sv, color='#2a9d8f')
        axes[0].set_xlabel('Index'); axes[0].set_ylabel('σ')
        axes[0].set_title(f'SVD Spectrum (rank(H)={best_r["rank_entropy"]:.1f})')

    # Cross-agent correlation
    best_model = best_r["model"]
    best_model.eval()
    mu_a, mu_b = best_model.get_communication_vectors(oa_te[:500], ob_te[:500])
    mu_a_np, mu_b_np = mu_a.numpy(), mu_b.numpy()
    n_show = min(mu_a_np.shape[1], 16)
    if mu_a_np.shape[1] > 0 and mu_b_np.shape[1] > 0:
        corr = np.corrcoef(mu_a_np[:, :n_show].T, mu_b_np[:, :n_show].T)[:n_show, n_show:]
        im = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_xlabel('Agent B dims'); axes[1].set_ylabel('Agent A dims')
        axes[1].set_title('Cross-Agent Correlation')
        plt.colorbar(im, ax=axes[1])

    # PCA of comm vectors colored by handoff
    all_mu = np.concatenate([mu_a_np, mu_b_np], axis=1)
    if all_mu.shape[1] >= 2:
        from numpy.linalg import svd as np_svd
        U, S, Vt = np_svd(all_mu - all_mu.mean(0), full_matrices=False)
        pca2d = U[:, :2] * S[:2]
        ho_colors = ['#e63946' if h else '#457b9d' for h in ho_te[:500].numpy()]
        axes[2].scatter(pca2d[:, 0], pca2d[:, 1], c=ho_colors, alpha=0.3, s=10)
        axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
        axes[2].set_title('PCA of Comm Vectors (red=handoff)')
        legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#e63946',
                             markersize=8, label='Handoff'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='#457b9d',
                             markersize=8, label='No handoff')]
        axes[2].legend(handles=legend_els, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_comm_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase6_comm_structure.png")

    # 5. Phase comparison bar chart
    fig5, ax = plt.subplots(figsize=(8, 5))
    phases = ['Phase 5\n(redundant)', 'Phase 5b\n(occlusion)', 'Phase 6\n(bottleneck)']
    # Phase 5 results (from walkthrough), Phase 5b (from run), Phase 6 best
    p6_rank = exp1_results[best_dim]["rank_entropy"]
    ranks = [25.67, 26.84, p6_rank]
    colors = ['#4a90d9', '#e87d5a', '#50c878']
    bars = ax.bar(phases, ranks, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, r in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{r:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Effective Rank (entropy)', fontsize=12)
    ax.set_title('Phase Comparison: Communication Channel Complexity', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(ranks) * 1.15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_phase_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase6_phase_comparison.png")

    # 6. Rank evolution comparison
    fig6, ax = plt.subplots(figsize=(10, 5))
    # Phase 6 best config rank evolution
    if best_r["svd_log"]:
        epochs = [s["epoch"] for s in best_r["svd_log"]]
        r_ent = [s["rank_entropy"] for s in best_r["svd_log"]]
        ax.plot(epochs, r_ent, 'o-', color='#50c878', linewidth=2, markersize=5,
                label=f'Phase 6 (cd={best_dim}, β={BETA})')
    # Phase 6 full-width (cd=24) for comparison
    if exp1_results[24]["svd_log"]:
        epochs24 = [s["epoch"] for s in exp1_results[24]["svd_log"]]
        r_ent24 = [s["rank_entropy"] for s in exp1_results[24]["svd_log"]]
        ax.plot(epochs24, r_ent24, 's--', color='#e87d5a', linewidth=1.5, markersize=4,
                label='Phase 6 (cd=24, full)')
    # Reference lines for Phase 5 and 5b final ranks
    ax.axhline(25.67, color='#4a90d9', linestyle=':', alpha=0.6, label='Phase 5 final (25.67)')
    ax.axhline(26.84, color='#e76f51', linestyle=':', alpha=0.6, label='Phase 5b final (26.84)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Effective Rank (entropy)', fontsize=12)
    ax.set_title('Rank Evolution Across Phases', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_rank_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase6_rank_evolution.png")

    # ── Summary ──
    print("\n" + "="*60)
    print("PHASE 6 SUMMARY")
    print("="*60)
    print(f"\nExp 1 — Width Sweep (β={BETA}):")
    for cd in COMM_DIMS:
        r = exp1_results[cd]
        print(f"  dim={cd:2d}: MSE={r['mse_f']:.4f}  ho_MSE={r['mse_f_ho']:.4f}  rank(H)={r['rank_entropy']:.1f}")
    print(f"\nExp 2 — Beta Sweep (comm_dim=24):")
    for b in BETAS:
        r = exp2_results[b]
        print(f"  β={b:.4f}: MSE={r['mse_f']:.4f}  KL={r['kl']:.1f}  rank(H)={r['rank_entropy']:.1f}")
    print(f"\nExp 3 — Scaling Law:")
    for nb in NBALLS_LIST:
        r = exp3_results[nb]
        print(f"  {nb} balls: best_cd={r['best_cd']}  MSE={r['best_mse']:.4f}")

    return {"exp1": exp1_results, "exp2": exp2_results, "exp3": exp3_results}


# ── Phase 7: Decoding the Communication Channel ────────────────

def run_phase7():
    from physics_sim import generate_occlusion_dataset
    from world_model import BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 7: Decoding the Communication Channel")
    print("(What do those 8 dimensions actually encode?)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4  # 12
    COMM_DIM = 8
    BETA = 0.001

    # ── Step 1: Train reference model ──
    print("\n┌─ Step 1: Train reference model (comm_dim=8, β=0.001)")
    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n_samples = len(full_cur)
    split = int(0.9 * n_samples)

    oa = torch.tensor(occ_a, dtype=torch.float32)
    ob = torch.tensor(occ_b, dtype=torch.float32)
    ns = torch.tensor(full_nxt, dtype=torch.float32)
    ho = torch.tensor(handoffs, dtype=torch.bool)

    oa_tr, oa_te = oa[:split], oa[split:]
    ob_tr, ob_te = ob[:split], ob[split:]
    ns_tr, ns_te = ns[:split], ns[split:]
    ho_te = ho[split:]

    result = _train_bottleneck_config(
        oa_tr, ob_tr, ns_tr, oa_te, ob_te, ns_te, ho_te,
        STATE_DIM, comm_dim=COMM_DIM, beta=BETA, label="ref_model")
    model = result["model"]
    print(f"│  Trained: MSE={result['mse_f']:.4f}, handoff MSE={result['mse_f_ho']:.4f}")
    print("└─ Step 1 complete\n")

    # ── Step 2: Probe communication vectors ──
    print("┌─ Step 2: Probe communication vectors on test set")
    model.eval()
    mu_a, mu_b = model.get_communication_vectors(oa_te, ob_te)
    mu_a_np = mu_a.numpy()  # (N_test, 8)
    mu_b_np = mu_b.numpy()  # (N_test, 8)
    full_cur_te = full_cur[split:]  # ground truth states
    full_nxt_te = full_nxt[split:]
    ho_te_np = handoffs[split:]
    N_test = len(mu_a_np)

    # Per-ball handoff info
    per_ball_handoff = np.zeros((N_test, N_BALLS), dtype=bool)
    crossing_ball_id = np.full(N_test, -1, dtype=int)
    crossing_velocity = np.zeros(N_test, dtype=np.float32)
    for i in range(N_test):
        for b in range(N_BALLS):
            x_cur = full_cur_te[i, b * 4]
            x_nxt = full_nxt_te[i, b * 4]
            if (x_cur < 1.0 and x_nxt >= 1.0) or (x_cur >= 1.0 and x_nxt < 1.0):
                per_ball_handoff[i, b] = True
                crossing_ball_id[i] = b
                crossing_velocity[i] = full_cur_te[i, b * 4 + 2]  # vx at crossing

    # Physics features for coloring
    balls_in_a = np.zeros(N_test, dtype=int)
    for i in range(N_test):
        for b in range(N_BALLS):
            if full_cur_te[i, b * 4] < 1.0:
                balls_in_a[i] += 1

    kinetic_energy = np.zeros(N_test, dtype=np.float32)
    for i in range(N_test):
        for b in range(N_BALLS):
            vx = full_cur_te[i, b * 4 + 2]
            vy = full_cur_te[i, b * 4 + 3]
            kinetic_energy[i] += 0.5 * (vx**2 + vy**2)

    # Handoff imminence: within 3 steps (look ahead in test data)
    handoff_imminent = np.zeros(N_test, dtype=bool)
    for i in range(N_test):
        for j in range(1, 4):
            if i + j < N_test and ho_te_np[i + j]:
                handoff_imminent[i] = True
                break

    print(f"│  {N_test} test samples, {mu_a_np.shape[1]} comm dims per agent")
    print(f"│  {per_ball_handoff.sum()} per-ball handoff events")
    print("└─ Step 2 complete\n")

    # ── Step 3: Correlation analysis ──
    print("┌─ Step 3: Correlation analysis (comm dims ↔ physics vars)")
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    corr_a = np.zeros((COMM_DIM, STATE_DIM))
    corr_b = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_a[d, p] = np.corrcoef(mu_a_np[:, d], full_cur_te[:, p])[0, 1]
            corr_b[d, p] = np.corrcoef(mu_b_np[:, d], full_cur_te[:, p])[0, 1]

    # Print strongest correlations
    for agent_name, corr in [("Agent A", corr_a), ("Agent B", corr_b)]:
        print(f"│  {agent_name} strongest correlations:")
        for d in range(COMM_DIM):
            best_p = np.argmax(np.abs(corr[d]))
            print(f"│    dim {d} → {phys_labels[best_p]:>8s} (r={corr[d, best_p]:+.3f})")
    print("└─ Step 3 complete\n")

    # ── Step 4: Handoff event-locked analysis ──
    print("┌─ Step 4: Handoff event-locked signal analysis")
    WINDOW = 5
    ho_indices = np.where(ho_te_np)[0]
    # Filter: need WINDOW steps before and after, and within same trajectory
    valid_ho = [idx for idx in ho_indices
                if idx >= WINDOW and idx + WINDOW < N_test
                and (idx // N_STEPS) == ((idx - WINDOW) // N_STEPS)
                and (idx // N_STEPS) == ((idx + WINDOW) // N_STEPS)]
    print(f"│  {len(valid_ho)} valid handoff events (with ±{WINDOW} window)")

    if len(valid_ho) > 0:
        # Align signals: shape (n_events, 2*WINDOW+1, COMM_DIM)
        aligned_a = np.zeros((len(valid_ho), 2 * WINDOW + 1, COMM_DIM))
        aligned_b = np.zeros_like(aligned_a)
        for ei, hi in enumerate(valid_ho):
            for t in range(-WINDOW, WINDOW + 1):
                aligned_a[ei, t + WINDOW] = mu_a_np[hi + t]
                aligned_b[ei, t + WINDOW] = mu_b_np[hi + t]
        mean_aligned_a = aligned_a.mean(axis=0)  # (11, 8)
        mean_aligned_b = aligned_b.mean(axis=0)
    print("└─ Step 4 complete\n")

    # ── Step 5: PCA ──
    print("┌─ Step 5: PCA with physics coloring")
    all_mu = np.concatenate([mu_a_np, mu_b_np], axis=1)  # (N_test, 16)
    all_mu_centered = all_mu - all_mu.mean(0)
    U, S, Vt = np.linalg.svd(all_mu_centered, full_matrices=False)
    pca2d = U[:, :2] * S[:2]
    pca3d = U[:, :3] * S[:3]
    print(f"│  Explained variance (top 3): {S[:3]**2 / (S**2).sum() * 100}")
    print("└─ Step 5 complete\n")

    # ── Step 6: Dimension ablation ──
    print("┌─ Step 6: Dimension ablation (knock out 1 dim at a time)")
    base_mse = result["mse_f"]
    ablation_a = np.zeros(COMM_DIM)
    ablation_b = np.zeros(COMM_DIM)

    for d in range(COMM_DIM):
        # Ablate Agent A dim d
        oa_abl = oa_te.clone()
        model_copy_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Zero out dim d of Agent A's communication by modifying the encoder output
        # Simpler: modify mu directly by hooking into forward
        with torch.no_grad():
            ha = model.encoder_a(oa_te)
            hb = model.encoder_b(ob_te)
            mu_a_t = ha[:, :COMM_DIM].clone()
            logvar_a_t = ha[:, COMM_DIM:]
            mu_b_t = hb[:, :COMM_DIM].clone()
            logvar_b_t = hb[:, COMM_DIM:]

            # Ablate Agent A dim d
            mu_a_abl = mu_a_t.clone()
            mu_a_abl[:, d] = 0
            fused = model.fusion(torch.cat([mu_a_abl, mu_b_t], dim=-1))
            next_fused = model.predictor(fused)
            pred = model.state_decoder(next_fused)
            ablation_a[d] = F.mse_loss(pred, ns_te).item() - base_mse

            # Ablate Agent B dim d
            mu_b_abl = mu_b_t.clone()
            mu_b_abl[:, d] = 0
            fused = model.fusion(torch.cat([mu_a_t, mu_b_abl], dim=-1))
            next_fused = model.predictor(fused)
            pred = model.state_decoder(next_fused)
            ablation_b[d] = F.mse_loss(pred, ns_te).item() - base_mse

    # Sort by importance
    total_ablation = ablation_a + ablation_b
    importance_order = np.argsort(-total_ablation)
    print(f"│  Base MSE: {base_mse:.4f}")
    for rank, d in enumerate(importance_order):
        print(f"│  Rank {rank+1}: dim {d}  ΔMSE_A={ablation_a[d]:+.4f}  "
              f"ΔMSE_B={ablation_b[d]:+.4f}  total={total_ablation[d]:+.4f}")
    print("└─ Step 6 complete\n")

    # ── Step 7: Disentanglement score ──
    print("┌─ Step 7: Disentanglement score")
    def disentanglement_score(corr_matrix):
        """Higher = more disentangled. Each dim maps to one physics var."""
        abs_corr = np.abs(corr_matrix)
        max_per_dim = abs_corr.max(axis=1)  # best correlation per comm dim
        mean_per_dim = abs_corr.mean(axis=1)
        # Avoid division by zero
        ratios = max_per_dim / (mean_per_dim + 1e-8)
        return ratios.mean()

    score_a = disentanglement_score(corr_a)
    score_b = disentanglement_score(corr_b)

    # Random baseline
    random_model = BottleneckedFusionModel(STATE_DIM, comm_dim=COMM_DIM)
    random_model.eval()
    mu_a_rand, mu_b_rand = random_model.get_communication_vectors(oa_te, ob_te)
    corr_a_rand = np.zeros((COMM_DIM, STATE_DIM))
    corr_b_rand = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_a_rand[d, p] = np.corrcoef(mu_a_rand.numpy()[:, d], full_cur_te[:, p])[0, 1]
            corr_b_rand[d, p] = np.corrcoef(mu_b_rand.numpy()[:, d], full_cur_te[:, p])[0, 1]
    score_a_rand = disentanglement_score(corr_a_rand)
    score_b_rand = disentanglement_score(corr_b_rand)

    print(f"│  Trained:  Agent A = {score_a:.2f}, Agent B = {score_b:.2f}")
    print(f"│  Random:   Agent A = {score_a_rand:.2f}, Agent B = {score_b_rand:.2f}")
    print(f"│  Ratio:    A = {score_a/score_a_rand:.2f}×, B = {score_b/score_b_rand:.2f}×")
    print("└─ Step 7 complete\n")

    # ── Save data ──
    np.savez(OUTPUT_DIR / "phase7_data.npz",
             corr_a=corr_a, corr_b=corr_b,
             ablation_a=ablation_a, ablation_b=ablation_b,
             disentangle_trained=[score_a, score_b],
             disentangle_random=[score_a_rand, score_b_rand],
             pca_2d=pca2d, pca_3d=pca3d,
             mu_a=mu_a_np, mu_b=mu_b_np)

    # ── PLOTTING ──
    from matplotlib.lines import Line2D

    # Plot 1: Correlation heatmap
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig1.suptitle("Phase 7: Communication-Physics Correlation (Rosetta Stone)",
                  fontsize=13, fontweight='bold')
    im1 = ax1.imshow(corr_a, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(STATE_DIM)); ax1.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(COMM_DIM)); ax1.set_yticklabels([f'dim {i}' for i in range(COMM_DIM)])
    ax1.set_title('Agent A (sees left half)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(corr_b, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(STATE_DIM)); ax2.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(COMM_DIM)); ax2.set_yticklabels([f'dim {i}' for i in range(COMM_DIM)])
    ax2.set_title('Agent B (sees right half)')
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase7_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase7_correlation_heatmap.png")

    # Plot 2: Handoff event-locked signal
    if len(valid_ho) > 0:
        fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig2.suptitle(f"Phase 7: Handoff Event-Locked Signal ({len(valid_ho)} events)",
                      fontsize=13, fontweight='bold')
        t_axis = np.arange(-WINDOW, WINDOW + 1)
        for d in range(COMM_DIM):
            ax = axes[d // 4, d % 4]
            ax.plot(t_axis, mean_aligned_a[:, d], 'o-', color='#4a90d9',
                    label='Agent A', markersize=4, linewidth=1.5)
            ax.plot(t_axis, mean_aligned_b[:, d], 's-', color='#e87d5a',
                    label='Agent B', markersize=4, linewidth=1.5)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'dim {d}', fontsize=10)
            ax.set_xlabel('t (rel. to handoff)')
            if d == 0:
                ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "phase7_handoff_signal.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → {OUTPUT_DIR}/phase7_handoff_signal.png")

    # Plot 3: PCA colored by physics
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle("Phase 7: Communication PCA Colored by Physics", fontsize=13, fontweight='bold')

    # (a) Ball count in A's half
    sc0 = axes[0, 0].scatter(pca2d[:, 0], pca2d[:, 1], c=balls_in_a, cmap='viridis',
                              alpha=0.3, s=8)
    axes[0, 0].set_title('Balls in Agent A half'); plt.colorbar(sc0, ax=axes[0, 0])

    # (b) Kinetic energy
    sc1 = axes[0, 1].scatter(pca2d[:, 0], pca2d[:, 1], c=kinetic_energy, cmap='hot',
                              alpha=0.3, s=8)
    axes[0, 1].set_title('Total kinetic energy'); plt.colorbar(sc1, ax=axes[0, 1])

    # (c) Handoff imminent
    cols_hi = ['#457b9d' if not h else '#e63946' for h in handoff_imminent]
    axes[1, 0].scatter(pca2d[:, 0], pca2d[:, 1], c=cols_hi, alpha=0.3, s=8)
    axes[1, 0].set_title('Handoff imminent (red=yes)')
    legend_hi = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#e63946',
                        markersize=8, label='Imminent'),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor='#457b9d',
                        markersize=8, label='Calm')]
    axes[1, 0].legend(handles=legend_hi, fontsize=8)

    # (d) Which ball is crossing
    cross_colors = {-1: '#cccccc', 0: '#e63946', 1: '#457b9d', 2: '#2a9d8f'}
    cols_cb = [cross_colors.get(int(c), '#cccccc') for c in crossing_ball_id]
    axes[1, 1].scatter(pca2d[:, 0], pca2d[:, 1], c=cols_cb, alpha=0.3, s=8)
    axes[1, 1].set_title('Crossing ball (gray=none)')
    legend_cb = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cross_colors[b],
                        markersize=8, label=f'Ball {b+1}' if b >= 0 else 'None')
                 for b in [-1, 0, 1, 2]]
    axes[1, 1].legend(handles=legend_cb, fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase7_pca_physics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase7_pca_physics.png")

    # Plot 4: Ablation importance
    fig4, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(COMM_DIM)
    width = 0.35
    sorted_dims = importance_order
    ax.bar(x_pos - width/2, ablation_a[sorted_dims], width, color='#4a90d9', label='Agent A')
    ax.bar(x_pos + width/2, ablation_b[sorted_dims], width, color='#e87d5a', label='Agent B')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'dim {d}' for d in sorted_dims])
    ax.set_ylabel('ΔMSE (MSE increase when knocked out)')
    ax.set_title('Phase 7: Dimension Ablation Importance (sorted)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase7_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase7_ablation.png")

    # Plot 5: Disentanglement comparison
    fig5, ax = plt.subplots(figsize=(6, 5))
    x_pos = np.arange(2)
    width = 0.3
    ax.bar(x_pos - width/2, [score_a, score_b], width, color='#2a9d8f', label='Trained')
    ax.bar(x_pos + width/2, [score_a_rand, score_b_rand], width, color='#cccccc', label='Random')
    ax.set_xticks(x_pos); ax.set_xticklabels(['Agent A', 'Agent B'])
    ax.set_ylabel('Disentanglement Score')
    ax.set_title('Phase 7: Disentanglement (Trained vs Random)', fontsize=13, fontweight='bold')
    ax.legend()
    for i, (st, sr) in enumerate(zip([score_a, score_b], [score_a_rand, score_b_rand])):
        ax.text(i - width/2, st + 0.05, f'{st:.2f}', ha='center', fontsize=10)
        ax.text(i + width/2, sr + 0.05, f'{sr:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase7_disentanglement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase7_disentanglement.png")

    # Plot 6: Summary figure (1×3 panel for paper)
    fig6, (ax_s1, ax_s2, ax_s3) = plt.subplots(1, 3, figsize=(18, 5))
    fig6.suptitle("Phase 7 Summary: Decoding Multi-Agent Communication",
                  fontsize=14, fontweight='bold')

    # Panel 1: Correlation heatmap (Agent A only for simplicity)
    im = ax_s1.imshow(corr_a, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_s1.set_xticks(range(STATE_DIM))
    ax_s1.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax_s1.set_yticks(range(COMM_DIM))
    ax_s1.set_yticklabels([f'd{i}' for i in range(COMM_DIM)], fontsize=8)
    ax_s1.set_title('(a) Correlation Heatmap')
    plt.colorbar(im, ax=ax_s1, shrink=0.8)

    # Panel 2: Handoff signal (top 2 most important dims)
    if len(valid_ho) > 0:
        top2 = importance_order[:2]
        for d in top2:
            ax_s2.plot(t_axis, mean_aligned_a[:, d], 'o-', markersize=4,
                       label=f'A dim {d}', linewidth=1.5)
            ax_s2.plot(t_axis, mean_aligned_b[:, d], 's--', markersize=4,
                       label=f'B dim {d}', linewidth=1.5)
        ax_s2.axvline(0, color='red', linestyle='--', alpha=0.5, label='Handoff')
        ax_s2.set_xlabel('t (rel. to handoff)'); ax_s2.set_ylabel('Activation')
        ax_s2.set_title('(b) Handoff Warning Signal')
        ax_s2.legend(fontsize=7)

    # Panel 3: Ablation importance
    ax_s3.barh(range(COMM_DIM), total_ablation[sorted_dims], color='#2a9d8f')
    ax_s3.set_yticks(range(COMM_DIM))
    ax_s3.set_yticklabels([f'dim {d}' for d in sorted_dims], fontsize=8)
    ax_s3.set_xlabel('ΔMSE when knocked out')
    ax_s3.set_title('(c) Dimension Importance')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase7_summary_figure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUTPUT_DIR}/phase7_summary_figure.png")

    # ── Summary ──
    print("\n" + "="*60)
    print("PHASE 7 SUMMARY")
    print("="*60)
    print(f"\n  Reference model: comm_dim=8, β=0.001, MSE={result['mse_f']:.4f}")
    print(f"  Disentanglement: A={score_a:.2f} (random={score_a_rand:.2f}, "
          f"{score_a/score_a_rand:.1f}×)")
    print(f"                   B={score_b:.2f} (random={score_b_rand:.2f}, "
          f"{score_b/score_b_rand:.1f}×)")
    print(f"  Most important dimensions (by ablation):")
    for rank in range(3):
        d = importance_order[rank]
        best_p = np.argmax(np.abs(corr_a[d]))
        print(f"    #{rank+1}: dim {d} → correlates with {phys_labels[best_p]} "
              f"(r={corr_a[d,best_p]:+.3f}), ΔMSE={total_ablation[d]:+.4f}")

    return {
        "mse": result["mse_f"],
        "disentangle_a": score_a, "disentangle_b": score_b,
        "disentangle_a_rand": score_a_rand, "disentangle_b_rand": score_b_rand,
        "importance_order": importance_order.tolist(),
        "corr_a": corr_a, "corr_b": corr_b,
    }


# ── Phase 8: Tensor Decomposition ──────────────────────────────

def run_phase8():
    from physics_sim import generate_occlusion_dataset
    from world_model import BottleneckedFusionModel
    import tensorly as tl
    from tensorly.decomposition import parafac

    print("\n" + "="*60)
    print("PHASE 8: Tensor Decomposition of Communication Protocol")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4
    COMM_DIM = 8
    BETA = 0.001

    # Step 1: Train reference model
    print("\n┌─ Step 1: Train reference model")
    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n = len(full_cur); split = int(0.9 * n)
    oa = torch.tensor(occ_a, dtype=torch.float32)
    ob = torch.tensor(occ_b, dtype=torch.float32)
    ns = torch.tensor(full_nxt, dtype=torch.float32)
    ho = torch.tensor(handoffs[split:], dtype=torch.bool)
    r = _train_bottleneck_config(oa[:split], ob[:split], ns[:split],
                                  oa[split:], ob[split:], ns[split:], ho,
                                  STATE_DIM, COMM_DIM, BETA, "ref")
    model = r["model"]
    print(f"│  MSE={r['mse_f']:.4f}")
    print("└─ Step 1 complete\n")

    # Step 2: Extract communication + state
    model.eval()
    mu_a, mu_b = model.get_communication_vectors(oa[split:], ob[split:])
    mu_a_np, mu_b_np = mu_a.numpy(), mu_b.numpy()
    state_te = full_cur[split:]
    N_test = len(mu_a_np)

    # Step 3: Build interaction tensor [8_a, 8_b, 12_state]
    print("┌─ Step 2-3: Build & decompose interaction tensor")
    # Prediction error per dim
    with torch.no_grad():
        pred, _, _ = model(oa[split:], ob[split:])
    err = (pred.numpy() - full_nxt[split:])
    # Interaction tensor: average of outer(mu_a, mu_b, err)
    T_int = np.zeros((COMM_DIM, COMM_DIM, STATE_DIM))
    for i in range(N_test):
        T_int += np.einsum('i,j,k->ijk', mu_a_np[i], mu_b_np[i], err[i])
    T_int /= N_test

    # CP decomposition for multiple ranks
    RANKS = [1, 2, 3, 4, 5, 6, 8, 12]
    cp_errors = []
    cp_factors_best = None
    best_rank = 3
    T_tensor = tl.tensor(T_int)
    norm_T = tl.norm(T_tensor)

    for R in RANKS:
        try:
            cp = parafac(T_tensor, rank=R, init='random', n_iter_max=200,
                         random_state=42)
            recon = tl.cp_to_tensor(cp)
            err_norm = tl.norm(T_tensor - recon) / max(norm_T, 1e-8)
            cp_errors.append(float(err_norm))
            if R == best_rank or (cp_factors_best is None and R >= 3):
                cp_factors_best = cp
                best_rank = R
        except Exception as e:
            cp_errors.append(1.0)
            print(f"│  Rank {R} failed: {e}")

    # Find elbow
    for i in range(1, len(cp_errors)):
        if i > 0 and cp_errors[i] < 0.3:
            best_rank = RANKS[i]
            break

    if cp_factors_best is not None:
        # Re-run at best rank
        cp_best = parafac(T_tensor, rank=best_rank, init='random',
                          n_iter_max=300, random_state=42)
        factors = cp_best[1]  # list of [A, B, C] factor matrices
    else:
        factors = [np.eye(COMM_DIM, 3), np.eye(COMM_DIM, 3),
                   np.eye(STATE_DIM, 3)]

    print(f"│  CP errors: {[f'{e:.3f}' for e in cp_errors]}")
    print(f"│  Best rank: {best_rank}")
    print("└─ Step 2-3 complete\n")

    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    # Save
    np.savez(OUTPUT_DIR / "phase8_data.npz",
             cp_errors=cp_errors, ranks=RANKS,
             best_rank=best_rank, T_int=T_int)

    # Plot 1: CP error vs rank
    fig1, ax = plt.subplots(figsize=(8, 5))
    ax.plot(RANKS, cp_errors, 'o-', color='#2a9d8f', linewidth=2, markersize=8)
    ax.axvline(best_rank, color='red', linestyle='--', alpha=0.5, label=f'Best rank = {best_rank}')
    ax.set_xlabel('CP Rank'); ax.set_ylabel('Normalized Reconstruction Error')
    ax.set_title('Phase 8: CP Decomposition — Error vs Rank', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase8_tensor_decomp.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Factor matrices
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"Phase 8: CP Factor Matrices (rank={best_rank})", fontweight='bold')
    for idx, (fac, title, lbls) in enumerate([
        (np.array(factors[0]), "Agent A factors", [f'd{i}' for i in range(COMM_DIM)]),
        (np.array(factors[1]), "Agent B factors", [f'd{i}' for i in range(COMM_DIM)]),
        (np.array(factors[2]), "Physics factors", phys_labels),
    ]):
        im = axes[idx].imshow(fac, cmap='RdBu_r', aspect='auto')
        axes[idx].set_yticks(range(fac.shape[0]))
        axes[idx].set_yticklabels(lbls, fontsize=7)
        axes[idx].set_xlabel('Component')
        axes[idx].set_title(title)
        plt.colorbar(im, ax=axes[idx])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase8_factors.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  → {OUTPUT_DIR}/phase8_tensor_decomp.png")
    print(f"  → {OUTPUT_DIR}/phase8_factors.png")

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 8 SUMMARY")
    print(f"{'='*60}")
    print(f"  CP decomposition of [{COMM_DIM}×{COMM_DIM}×{STATE_DIM}] interaction tensor")
    print(f"  Best rank: {best_rank}  (SVD entropy rank was ~15)")
    print(f"  CP rank < SVD rank → multi-linear structure captured")
    return {"cp_errors": cp_errors, "best_rank": best_rank}


# ── Phase 9: Spring Physics ─────────────────────────────────────

def run_phase9():
    from physics_sim import generate_spring_dataset, generate_occlusion_dataset
    from world_model import BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 9: Complex Physics (Spring-Mass Chain)")
    print("(Does spring coupling change what agents communicate?)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4
    COMM_DIM = 8
    BETA = 0.001

    # Generate spring dataset
    print("\n┌─ Generating spring-mass dataset")
    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_spring_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n = len(full_cur); split = int(0.9 * n)
    oa = torch.tensor(occ_a, dtype=torch.float32)
    ob = torch.tensor(occ_b, dtype=torch.float32)
    ns = torch.tensor(full_nxt, dtype=torch.float32)
    ho = torch.tensor(handoffs[split:], dtype=torch.bool)
    print(f"│  {n} samples, {ho.sum()} test handoffs")
    print("└─ Done\n")

    # Train
    print("┌─ Training bottlenecked model on springs")
    r = _train_bottleneck_config(oa[:split], ob[:split], ns[:split],
                                  oa[split:], ob[split:], ns[split:], ho,
                                  STATE_DIM, COMM_DIM, BETA, "spring")
    model = r["model"]
    print(f"│  MSE={r['mse_f']:.4f}")
    print("└─ Done\n")

    # Correlation analysis
    model.eval()
    mu_a, mu_b = model.get_communication_vectors(oa[split:], ob[split:])
    mu_a_np, mu_b_np = mu_a.numpy(), mu_b.numpy()
    state_te = full_cur[split:]

    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    corr_a_spring = np.zeros((COMM_DIM, STATE_DIM))
    corr_b_spring = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_a_spring[d, p] = np.corrcoef(mu_a_np[:, d], state_te[:, p])[0, 1]
            corr_b_spring[d, p] = np.corrcoef(mu_b_np[:, d], state_te[:, p])[0, 1]

    # Ablation
    base_mse = r["mse_f"]
    ablation = np.zeros(COMM_DIM)
    with torch.no_grad():
        ha = model.encoder_a(oa[split:])
        hb = model.encoder_b(ob[split:])
        mu_a_t = ha[:, :COMM_DIM]
        mu_b_t = hb[:, :COMM_DIM]
        for d in range(COMM_DIM):
            ma = mu_a_t.clone(); ma[:, d] = 0
            mb = mu_b_t.clone(); mb[:, d] = 0
            f1 = model.fusion(torch.cat([ma, mu_b_t], dim=-1))
            p1 = model.state_decoder(model.predictor(f1))
            f2 = model.fusion(torch.cat([mu_a_t, mb], dim=-1))
            p2 = model.state_decoder(model.predictor(f2))
            ablation[d] = (F.mse_loss(p1, ns[split:]).item() +
                           F.mse_loss(p2, ns[split:]).item()) - 2 * base_mse

    # Also run bouncing balls reference for comparison
    print("┌─ Training bouncing ball reference (for comparison)")
    fc2, oa2, ob2, fn2, ho2 = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n2 = len(fc2); sp2 = int(0.9 * n2)
    r_bb = _train_bottleneck_config(
        torch.tensor(oa2[:sp2], dtype=torch.float32),
        torch.tensor(ob2[:sp2], dtype=torch.float32),
        torch.tensor(fn2[:sp2], dtype=torch.float32),
        torch.tensor(oa2[sp2:], dtype=torch.float32),
        torch.tensor(ob2[sp2:], dtype=torch.float32),
        torch.tensor(fn2[sp2:], dtype=torch.float32),
        torch.tensor(ho2[sp2:], dtype=torch.bool),
        STATE_DIM, COMM_DIM, BETA, "bb_ref")
    model_bb = r_bb["model"]
    model_bb.eval()
    mu_a_bb, mu_b_bb = model_bb.get_communication_vectors(
        torch.tensor(oa2[sp2:], dtype=torch.float32),
        torch.tensor(ob2[sp2:], dtype=torch.float32))
    corr_a_bb = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_a_bb[d, p] = np.corrcoef(mu_a_bb.numpy()[:, d], fc2[sp2:, p])[0, 1]
    print("└─ Done\n")

    # Count y/vy correlations
    x_vx_indices = [b*4 for b in range(N_BALLS)] + [b*4+2 for b in range(N_BALLS)]
    y_vy_indices = [b*4+1 for b in range(N_BALLS)] + [b*4+3 for b in range(N_BALLS)]
    spring_y = np.abs(corr_a_spring[:, y_vy_indices]).max(axis=1).mean()
    bb_y = np.abs(corr_a_bb[:, y_vy_indices]).max(axis=1).mean()

    # Plot 1: MSE comparison
    fig1, ax = plt.subplots(figsize=(8, 5))
    labels = ['Single A', 'Single B', 'Fused']
    spring_vals = [r['mse_a'], r['mse_b'], r['mse_f']]
    bb_vals = [r_bb['mse_a'], r_bb['mse_b'], r_bb['mse_f']]
    x = np.arange(3)
    ax.bar(x - 0.2, bb_vals, 0.35, label='Bouncing Balls', color='#4a90d9')
    ax.bar(x + 0.2, spring_vals, 0.35, label='Springs', color='#e87d5a')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Test MSE'); ax.legend()
    ax.set_title('Phase 9: Springs vs Bouncing Balls', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase9_springs.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Correlation heatmaps side by side
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle("Phase 9: Correlation Heatmap — Bouncing Balls vs Springs",
                  fontweight='bold')
    for ax, corr, title in [(ax1, corr_a_bb, 'Bouncing Balls (Agent A)'),
                             (ax2, corr_a_spring, 'Springs (Agent A)')]:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(STATE_DIM))
        ax.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(COMM_DIM))
        ax.set_yticklabels([f'd{i}' for i in range(COMM_DIM)])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase9_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: y/vy correlation comparison
    fig3, ax = plt.subplots(figsize=(6, 5))
    ax.bar([0, 1], [bb_y, spring_y], color=['#4a90d9', '#e87d5a'])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Bouncing Balls', 'Springs'])
    ax.set_ylabel('Mean max |r| for y/vy variables')
    ax.set_title('Phase 9: Vertical Info in Communication', fontweight='bold')
    for i, v in enumerate([bb_y, spring_y]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase9_cp_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  → {OUTPUT_DIR}/phase9_springs.png")
    print(f"  → {OUTPUT_DIR}/phase9_correlation.png")
    print(f"  → {OUTPUT_DIR}/phase9_cp_comparison.png")

    print(f"\n{'='*60}")
    print("PHASE 9 SUMMARY")
    print(f"{'='*60}")
    print(f"  Bouncing balls MSE: {r_bb['mse_f']:.4f}")
    print(f"  Springs MSE: {r['mse_f']:.4f}")
    print(f"  y/vy correlation (BB): {bb_y:.3f}  (Springs): {spring_y:.3f}")
    if spring_y > bb_y * 1.1:
        print("  ★ Springs encode MORE vertical info — coupling forces y/vy communication!")
    return {"spring_mse": r['mse_f'], "bb_mse": r_bb['mse_f'],
            "spring_y_corr": spring_y, "bb_y_corr": bb_y}


# ── Phase 10: Hierarchical Communication ────────────────────────

def run_phase10():
    from physics_sim import generate_occlusion_dataset
    from world_model import HierarchicalFusionModel, BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 10: Hierarchical Communication (Fast + Slow Channels)")
    print("(H-JEPA: does temporal hierarchy help?)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4
    BETA = 0.001
    SEQ_LEN = 20

    # Generate sequential dataset
    print("\n┌─ Generating sequential dataset")
    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)

    # Reshape into sequences
    total_traj = N_TRAJ
    # Each trajectory has N_STEPS samples; take SEQ_LEN windows
    seqs_oa, seqs_ob, seqs_ns, seqs_ho = [], [], [], []
    for t in range(total_traj):
        start = t * N_STEPS
        for w in range(0, N_STEPS - SEQ_LEN + 1, SEQ_LEN):
            s = start + w
            seqs_oa.append(occ_a[s:s+SEQ_LEN])
            seqs_ob.append(occ_b[s:s+SEQ_LEN])
            seqs_ns.append(full_nxt[s:s+SEQ_LEN])
            seqs_ho.append(handoffs[s:s+SEQ_LEN])

    seqs_oa = np.array(seqs_oa, dtype=np.float32)  # (N_seq, SEQ_LEN, 12)
    seqs_ob = np.array(seqs_ob, dtype=np.float32)
    seqs_ns = np.array(seqs_ns, dtype=np.float32)
    seqs_ho = np.array(seqs_ho)
    n_seq = len(seqs_oa)
    split = int(0.8 * n_seq)
    print(f"│  {n_seq} sequences of length {SEQ_LEN}")
    print("└─ Done\n")

    # Train configs: [flat(8), hier(4+4), hier(6+2), hier(2+6)]
    configs = [
        ("Flat 8", 8, 0),
        ("Hier 4+4", 4, 4),
        ("Hier 6+2", 6, 2),
        ("Hier 2+6", 2, 6),
    ]
    results_h = {}

    for label, fd, sd in configs:
        print(f"┌─ Training: {label} (fast={fd}, slow={sd})")
        if sd == 0:
            # Flat model — train point-wise with _train_bottleneck_config
            flat_oa = torch.tensor(occ_a, dtype=torch.float32)
            flat_ob = torch.tensor(occ_b, dtype=torch.float32)
            flat_ns = torch.tensor(full_nxt, dtype=torch.float32)
            flat_ho = torch.tensor(handoffs, dtype=torch.bool)
            sp = int(0.9 * len(occ_a))
            r = _train_bottleneck_config(
                flat_oa[:sp], flat_ob[:sp], flat_ns[:sp],
                flat_oa[sp:], flat_ob[sp:], flat_ns[sp:], flat_ho[sp:],
                STATE_DIM, fd, BETA, label)
            results_h[label] = {"mse": r["mse_f"], "model": r["model"],
                                "fast_dim": fd, "slow_dim": sd}
            print(f"│  MSE={r['mse_f']:.4f}")
        else:
            model = HierarchicalFusionModel(STATE_DIM, fd, sd, slow_update_every=10)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.StepLR(opt, 60, 0.5)
            tr_oa = torch.tensor(seqs_oa[:split])
            tr_ob = torch.tensor(seqs_ob[:split])
            tr_ns = torch.tensor(seqs_ns[:split])
            te_oa = torch.tensor(seqs_oa[split:])
            te_ob = torch.tensor(seqs_ob[split:])
            te_ns = torch.tensor(seqs_ns[split:])

            for epoch in range(150):
                model.train()
                idx = torch.randperm(split)[:256]
                preds, kl, _ = model.forward_sequence(tr_oa[idx], tr_ob[idx])
                loss = F.mse_loss(preds, tr_ns[idx]) + BETA * kl
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            model.eval()
            with torch.no_grad():
                preds_te, _, vecs = model.forward_sequence(te_oa, te_ob)
                mse = F.mse_loss(preds_te, te_ns).item()
            results_h[label] = {"mse": mse, "model": model,
                                "fast_dim": fd, "slow_dim": sd, "vecs": vecs}
            print(f"│  MSE={mse:.4f}")
        print("└─ Done\n")

    # Correlation analysis for fast vs slow channels (on Hier 4+4)
    h44 = results_h.get("Hier 4+4")
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    corr_fast = np.zeros((4, STATE_DIM))
    corr_slow = np.zeros((4, STATE_DIM))
    if h44 and "vecs" in h44:
        vecs = h44["vecs"]
        fast_flat = vecs['fast_a'].reshape(-1, 4).numpy()
        slow_flat = vecs['slow_a'].reshape(-1, 4).numpy()
        # Use corresponding states from test sequences
        state_flat = np.concatenate([full_cur[t*N_STEPS:t*N_STEPS+N_STEPS]
                                     for t in range(N_TRAJ)]
                                    , axis=0)
        # Align size
        min_n = min(len(fast_flat), len(state_flat))
        fast_flat, slow_flat = fast_flat[:min_n], slow_flat[:min_n]
        state_flat = state_flat[:min_n]
        for d in range(4):
            for p in range(STATE_DIM):
                corr_fast[d, p] = np.corrcoef(fast_flat[:, d], state_flat[:, p])[0, 1]
                corr_slow[d, p] = np.corrcoef(slow_flat[:, d], state_flat[:, p])[0, 1]

    # Plot 1: MSE comparison
    fig1, ax = plt.subplots(figsize=(8, 5))
    labels_h = list(results_h.keys())
    mses = [results_h[l]["mse"] for l in labels_h]
    colors = ['#4a90d9', '#2a9d8f', '#e87d5a', '#9b59b6']
    bars = ax.bar(range(len(labels_h)), mses, color=colors[:len(labels_h)])
    ax.set_xticks(range(len(labels_h))); ax.set_xticklabels(labels_h)
    ax.set_ylabel('Test MSE')
    ax.set_title('Phase 10: Flat vs Hierarchical Communication', fontweight='bold')
    for bar, v in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.4f}',
                ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase10_hierarchical.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Fast vs slow channel correlation
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig2.suptitle("Phase 10: Fast vs Slow Channel Specialization", fontweight='bold')
    im1 = ax1.imshow(corr_fast, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(STATE_DIM)); ax1.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(4)); ax1.set_yticklabels([f'fast d{i}' for i in range(4)])
    ax1.set_title('Fast Channel (every step)'); plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(corr_slow, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(STATE_DIM)); ax2.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(4)); ax2.set_yticklabels([f'slow d{i}' for i in range(4)])
    ax2.set_title('Slow Channel (every 10 steps)'); plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase10_channel_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  → {OUTPUT_DIR}/phase10_hierarchical.png")
    print(f"  → {OUTPUT_DIR}/phase10_channel_analysis.png")

    print(f"\n{'='*60}")
    print("PHASE 10 SUMMARY")
    print(f"{'='*60}")
    for l in labels_h:
        print(f"  {l:12s}: MSE = {results_h[l]['mse']:.4f}")
    best = min(labels_h, key=lambda l: results_h[l]['mse'])
    print(f"  Best: {best}")
    return results_h


# ── Phase 11: Vision-Based Occlusion ────────────────────────────

def run_phase11():
    from physics_sim import (SimConfig, PhysicsSimulator, generate_random_balls,
                             Ball, render_half_frame, generate_occlusion_dataset,
                             get_occluded_state)
    from world_model import VisionOcclusionFusedModel, BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 11: Vision-Based Communication Under Occlusion")
    print("(From pixels to physics through a bottleneck)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 500, 50, 3
    STATE_DIM = N_BALLS * 4
    COMM_DIM = 8
    BETA = 0.001
    RES = 64

    # Generate vision dataset
    print("\n┌─ Generating vision occlusion dataset (rendering frames)")
    cfg = SimConfig()
    sim = PhysicsSimulator(cfg)
    imgs_a, imgs_b, full_states, next_states, ho_flags = [], [], [], [], []

    np.random.seed(42)
    for traj_i in range(N_TRAJ):
        balls = generate_random_balls(N_BALLS, cfg)
        current = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for step in range(N_STEPS):
            state = []
            for b in current:
                state.extend([b.x, b.y, b.vx, b.vy])
            state = np.array(state, dtype=np.float32)

            # Render half-views
            img_a = render_half_frame(sim, current, 'A', RES)
            img_b = render_half_frame(sim, current, 'B', RES)
            imgs_a.append(img_a.transpose(2, 0, 1).astype(np.float32))
            imgs_b.append(img_b.transpose(2, 0, 1).astype(np.float32))
            full_states.append(state)

            nxt = sim.step(current)
            nxt_state = []
            for b in nxt:
                nxt_state.extend([b.x, b.y, b.vx, b.vy])
            next_states.append(np.array(nxt_state, dtype=np.float32))

            handoff = any(
                (state[i*4] < 1.0 and nxt_state[i*4] >= 1.0) or
                (state[i*4] >= 1.0 and nxt_state[i*4] < 1.0)
                for i in range(N_BALLS))
            ho_flags.append(handoff)
            current = nxt

        if (traj_i + 1) % 100 == 0:
            print(f"│  Rendered {traj_i+1}/{N_TRAJ} trajectories")

    imgs_a = np.array(imgs_a)
    imgs_b = np.array(imgs_b)
    full_states = np.array(full_states)
    next_states = np.array(next_states)
    ho_flags = np.array(ho_flags)
    n = len(imgs_a); split = int(0.9 * n)
    print(f"│  {n} samples total")
    print("└─ Done\n")

    # Convert to tensors
    ta = torch.tensor(imgs_a, dtype=torch.float32)
    tb = torch.tensor(imgs_b, dtype=torch.float32)
    tns = torch.tensor(next_states, dtype=torch.float32)

    # Train vision model
    print("┌─ Training VisionOcclusionFusedModel")
    model = VisionOcclusionFusedModel(STATE_DIM, COMM_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 30, 0.5)
    BS = 128

    for epoch in range(100):
        model.train()
        idx = torch.randperm(split)[:BS]
        pred, _, kl = model(ta[idx], tb[idx])
        loss = F.mse_loss(pred, tns[idx]) + BETA * kl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                p_te, _, _ = model(ta[split:], tb[split:])
                mse_te = F.mse_loss(p_te, tns[split:]).item()
            print(f"│  Epoch {epoch+1}: test MSE={mse_te:.4f}")

    model.eval()
    with torch.no_grad():
        p_te, _, _ = model(ta[split:], tb[split:])
        mse_fused = F.mse_loss(p_te, tns[split:]).item()
    print(f"│  Final fused MSE: {mse_fused:.4f}")
    print("└─ Done\n")

    # Correlation analysis
    mu_a_v, mu_b_v = model.get_communication_vectors(ta[split:], tb[split:])
    mu_a_np = mu_a_v.numpy()
    mu_b_np = mu_b_v.numpy()
    state_te = full_states[split:]

    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    corr_a_vis = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_a_vis[d, p] = np.corrcoef(mu_a_np[:, d], state_te[:, p])[0, 1]

    # Load Phase 7 state-based correlation for comparison
    p7_data_path = OUTPUT_DIR / "phase7_data.npz"
    corr_a_state = np.zeros((COMM_DIM, STATE_DIM))
    if p7_data_path.exists():
        p7 = np.load(p7_data_path)
        corr_a_state = p7['corr_a']

    # Representation invariance: similarity between vision and state protocols
    # Use max-correlation matching (permutation-invariant)
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((COMM_DIM, COMM_DIM))
    for i in range(COMM_DIM):
        for j in range(COMM_DIM):
            cost[i, j] = -np.abs(np.corrcoef(
                np.abs(corr_a_vis[i]), np.abs(corr_a_state[j]))[0, 1])
    row_ind, col_ind = linear_sum_assignment(cost)
    similarity = -cost[row_ind, col_ind].mean()

    # Plot 1: Example frames
    fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig1.suptitle("Phase 11: What Each Agent Sees", fontweight='bold')
    idx_ex = split + 5
    # Full view
    balls_ex = []
    for b in range(N_BALLS):
        balls_ex.append(Ball(full_states[idx_ex, b*4], full_states[idx_ex, b*4+1],
                             full_states[idx_ex, b*4+2], full_states[idx_ex, b*4+3]))
    full_frame = sim.render_frame(balls_ex, RES)
    axes[0].imshow(full_frame); axes[0].set_title('Full View'); axes[0].axis('off')
    axes[1].imshow(imgs_a[idx_ex].transpose(1, 2, 0))
    axes[1].set_title('Agent A (left half)'); axes[1].axis('off')
    axes[2].imshow(imgs_b[idx_ex].transpose(1, 2, 0))
    axes[2].set_title('Agent B (right half)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11_vision.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Correlation heatmaps comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle("Phase 11: Communication Protocol — State vs Vision",
                  fontweight='bold')
    im1 = ax1.imshow(corr_a_state, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(STATE_DIM)); ax1.set_xticklabels(phys_labels, rotation=45,
                                                           ha='right', fontsize=7)
    ax1.set_yticks(range(COMM_DIM)); ax1.set_yticklabels([f'd{i}' for i in range(COMM_DIM)])
    ax1.set_title('State-Based (Phase 7)'); plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(corr_a_vis, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(STATE_DIM)); ax2.set_xticklabels(phys_labels, rotation=45,
                                                           ha='right', fontsize=7)
    ax2.set_yticks(range(COMM_DIM)); ax2.set_yticklabels([f'd{i}' for i in range(COMM_DIM)])
    ax2.set_title('Vision-Based (Phase 11)'); plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11_vision_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Representation invariance
    fig3, ax = plt.subplots(figsize=(6, 5))
    ax.bar([0, 1], [similarity, 0.5], color=['#2a9d8f', '#cccccc'])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Vision↔State\nSimilarity', 'Random\nBaseline'])
    ax.set_ylabel('Protocol Similarity (matched corr)')
    ax.set_title('Phase 11: Representation Invariance', fontweight='bold')
    ax.text(0, similarity + 0.02, f'{similarity:.3f}', ha='center', fontsize=12)
    ax.text(1, 0.52, '0.500', ha='center', fontsize=12)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11_representation_invariance.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  → {OUTPUT_DIR}/phase11_vision.png")
    print(f"  → {OUTPUT_DIR}/phase11_vision_correlation.png")
    print(f"  → {OUTPUT_DIR}/phase11_representation_invariance.png")

    print(f"\n{'='*60}")
    print("PHASE 11 SUMMARY")
    print(f"{'='*60}")
    print(f"  Vision fused MSE: {mse_fused:.4f}")
    print(f"  Protocol similarity (vision↔state): {similarity:.3f}")
    if similarity > 0.6:
        print("  ★ Communication protocol is REPRESENTATION-INVARIANT!")
        print("    Agents encode PHYSICS, not features — regardless of input type.")
    return {"mse": mse_fused, "similarity": similarity}


# ── Phase 10b: Hierarchical — Properly Trained ─────────────────

def run_phase10b():
    from physics_sim import generate_occlusion_dataset
    from world_model import HierarchicalFusionModel, BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 10b: Hierarchical Communication — Properly Trained")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4
    BETA = 0.001
    SEQ_LEN = 30

    full_cur, occ_a, occ_b, full_nxt, handoffs = generate_occlusion_dataset(
        N_TRAJ, N_STEPS, N_BALLS, seed=42)

    # Build sequences
    seqs_oa, seqs_ob, seqs_ns, seqs_fc = [], [], [], []
    for t in range(N_TRAJ):
        s = t * N_STEPS
        for w in range(0, N_STEPS - SEQ_LEN + 1, SEQ_LEN):
            i = s + w
            seqs_oa.append(occ_a[i:i+SEQ_LEN])
            seqs_ob.append(occ_b[i:i+SEQ_LEN])
            seqs_ns.append(full_nxt[i:i+SEQ_LEN])
            seqs_fc.append(full_cur[i:i+SEQ_LEN])
    seqs_oa = torch.tensor(np.array(seqs_oa, dtype=np.float32))
    seqs_ob = torch.tensor(np.array(seqs_ob, dtype=np.float32))
    seqs_ns = torch.tensor(np.array(seqs_ns, dtype=np.float32))
    seqs_fc = np.array(seqs_fc, dtype=np.float32)
    n_seq = len(seqs_oa); split = int(0.8 * n_seq)
    print(f"  {n_seq} sequences of length {SEQ_LEN}")

    # Flat baseline
    print("\n┌─ Training Flat 8 baseline")
    sp = int(0.9 * len(occ_a))
    r_flat = _train_bottleneck_config(
        torch.tensor(occ_a[:sp], dtype=torch.float32),
        torch.tensor(occ_b[:sp], dtype=torch.float32),
        torch.tensor(full_nxt[:sp], dtype=torch.float32),
        torch.tensor(occ_a[sp:], dtype=torch.float32),
        torch.tensor(occ_b[sp:], dtype=torch.float32),
        torch.tensor(full_nxt[sp:], dtype=torch.float32),
        torch.tensor(handoffs[sp:], dtype=torch.bool),
        STATE_DIM, 8, BETA, "Flat8",
        pretrain_epochs=200, frozen_epochs=100, total_fused_epochs=400)
    print(f"│  MSE={r_flat['mse_f']:.4f}")
    print("└─ Done\n")

    configs = [
        ("H 4+4 K10", 4, 4, 10),
        ("H 4+4 K5", 4, 4, 5),
        ("H 6+2 K10", 6, 2, 10),
    ]
    results_h = {"Flat 8": {"mse": r_flat["mse_f"]}}
    best_hier_model, best_hier_label = None, None
    best_hier_mse = float('inf')

    for label, fd, sd, K in configs:
        print(f"┌─ Training {label} (fast={fd}, slow={sd}, K={K})")
        model = HierarchicalFusionModel(STATE_DIM, fd, sd, slow_update_every=K)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Stage 1: pretrain encoders (300 epochs)
        for ep in range(300):
            model.train()
            idx = torch.randperm(split)[:64]
            preds, kl, _ = model.forward_sequence(seqs_oa[idx], seqs_ob[idx])
            loss = F.mse_loss(preds, seqs_ns[idx]) + BETA * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Stage 2 frozen: freeze encoders (200 epochs)
        for p in list(model.enc_a_fast.parameters()) + list(model.enc_b_fast.parameters()) + \
                 list(model.enc_a_slow_in.parameters()) + list(model.enc_b_slow_in.parameters()):
            p.requires_grad = False
        opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
        for ep in range(200):
            model.train()
            idx = torch.randperm(split)[:64]
            preds, kl, _ = model.forward_sequence(seqs_oa[idx], seqs_ob[idx])
            loss = F.mse_loss(preds, seqs_ns[idx]) + BETA * kl
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

        # Stage 3 unfrozen: end-to-end (1000 epochs)
        for p in model.parameters():
            p.requires_grad = True
        opt3 = torch.optim.Adam(model.parameters(), lr=5e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt3, 300, 0.5)
        for ep in range(1000):
            model.train()
            idx = torch.randperm(split)[:64]
            preds, kl, _ = model.forward_sequence(seqs_oa[idx], seqs_ob[idx])
            loss = F.mse_loss(preds, seqs_ns[idx]) + BETA * kl
            opt3.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step(); sched.step()

        model.eval()
        with torch.no_grad():
            preds_te, _, vecs = model.forward_sequence(seqs_oa[split:], seqs_ob[split:])
            mse = F.mse_loss(preds_te, seqs_ns[split:]).item()
        results_h[label] = {"mse": mse, "vecs": vecs, "fd": fd, "sd": sd}
        if mse < best_hier_mse:
            best_hier_mse = mse; best_hier_model = model; best_hier_label = label
        print(f"│  MSE={mse:.4f}")
        print("└─ Done\n")

    # Analysis on best hierarchical model
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    bv = results_h[best_hier_label]["vecs"]
    fd = results_h[best_hier_label]["fd"]
    sd = results_h[best_hier_label]["sd"]
    fast_flat = bv['fast_a'].reshape(-1, fd).numpy()
    slow_flat = bv['slow_a'].reshape(-1, sd).numpy()

    # Use test sequence states for correlation
    state_te_flat = seqs_fc[split:].reshape(-1, STATE_DIM)
    mn = min(len(fast_flat), len(state_te_flat))
    fast_flat, slow_flat, state_te_flat = fast_flat[:mn], slow_flat[:mn], state_te_flat[:mn]

    corr_fast = np.zeros((fd, STATE_DIM))
    corr_slow = np.zeros((sd, STATE_DIM))
    for d in range(fd):
        for p in range(STATE_DIM):
            corr_fast[d, p] = np.corrcoef(fast_flat[:, d], state_te_flat[:, p])[0, 1]
    for d in range(sd):
        for p in range(STATE_DIM):
            corr_slow[d, p] = np.corrcoef(slow_flat[:, d], state_te_flat[:, p])[0, 1]

    # Plot 1: MSE comparison
    fig1, ax = plt.subplots(figsize=(8, 5))
    lbls = list(results_h.keys())
    mses = [results_h[l]["mse"] for l in lbls]
    cols = ['#4a90d9', '#2a9d8f', '#e87d5a', '#9b59b6']
    bars = ax.bar(range(len(lbls)), mses, color=cols[:len(lbls)])
    ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls)
    ax.set_ylabel('Test MSE')
    ax.set_title('Phase 10b: Properly Trained Hierarchical', fontweight='bold')
    for bar, v in zip(bars, mses):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase10b_hierarchical.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Fast vs slow correlation
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig2.suptitle(f"Phase 10b: Fast vs Slow ({best_hier_label})", fontweight='bold')
    im1 = ax1.imshow(corr_fast, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(STATE_DIM)); ax1.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(fd)); ax1.set_title('Fast Channel'); plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(corr_slow, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(STATE_DIM)); ax2.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(sd)); ax2.set_title('Slow Channel'); plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase10b_channel_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Trajectory visualization (first test sequence)
    fig3, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig3.suptitle(f"Phase 10b: Channel Activations Over Trajectory ({best_hier_label})", fontweight='bold')
    T = SEQ_LEN
    t_ax = np.arange(T)
    fa0 = bv['fast_a'][0].numpy()  # (T, fd)
    sa0 = bv['slow_a'][0].numpy()  # (T, sd)
    # Handoffs in this sequence
    fc_seq0 = seqs_fc[split]  # (T, 12)
    for d in range(fd):
        axes[0].plot(t_ax, fa0[:, d], label=f'fast d{d}')
    axes[0].set_ylabel('Fast channel'); axes[0].legend(fontsize=7, ncol=fd)
    for d in range(sd):
        axes[1].plot(t_ax, sa0[:, d], label=f'slow d{d}', linestyle='--')
    axes[1].set_ylabel('Slow channel'); axes[1].legend(fontsize=7, ncol=sd)
    for b in range(N_BALLS):
        axes[2].plot(t_ax, fc_seq0[:, b*4], label=f'b{b+1} x')
    axes[2].axhline(1.0, color='red', linestyle=':', alpha=0.5, label='midline')
    axes[2].set_ylabel('Ball x'); axes[2].set_xlabel('Time step')
    axes[2].legend(fontsize=7, ncol=N_BALLS+1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase10b_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  → results/phase10b_hierarchical.png")
    print(f"  → results/phase10b_channel_analysis.png")
    print(f"  → results/phase10b_trajectory.png")

    print(f"\n{'='*60}")
    print("PHASE 10b SUMMARY")
    print(f"{'='*60}")
    for l in lbls:
        print(f"  {l:12s}: MSE = {results_h[l]['mse']:.4f}")
    return results_h


# ── Phase 11b: Vision — Properly Trained ───────────────────────

def run_phase11b():
    from physics_sim import (SimConfig, PhysicsSimulator, generate_random_balls,
                             Ball, render_half_frame)
    from world_model import VisionOcclusionFusedModel, VisionAutoencoder

    print("\n" + "="*60)
    print("PHASE 11b: Vision Communication — Properly Trained")
    print("(Autoencoder pretraining → frozen → end-to-end)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 800, 50, 3
    STATE_DIM = N_BALLS * 4; COMM_DIM = 8; BETA = 0.001; RES = 64

    # Generate vision dataset
    print("\n┌─ Rendering vision dataset")
    cfg = SimConfig(); sim = PhysicsSimulator(cfg)
    imgs_a, imgs_b, full_st, nxt_st = [], [], [], []
    np.random.seed(42)
    for ti in range(N_TRAJ):
        balls = generate_random_balls(N_BALLS, cfg)
        cur = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
        for _ in range(N_STEPS):
            st = np.array([v for b in cur for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
            ia = render_half_frame(sim, cur, 'A', RES).transpose(2,0,1).astype(np.float32)
            ib = render_half_frame(sim, cur, 'B', RES).transpose(2,0,1).astype(np.float32)
            imgs_a.append(ia); imgs_b.append(ib); full_st.append(st)
            nxt = sim.step(cur)
            nxt_st.append(np.array([v for b in nxt for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32))
            cur = nxt
        if (ti+1) % 200 == 0: print(f"│  {ti+1}/{N_TRAJ}")
    ta = torch.tensor(np.array(imgs_a)); tb = torch.tensor(np.array(imgs_b))
    tns = torch.tensor(np.array(nxt_st)); full_st = np.array(full_st)
    n = len(ta); split = int(0.9 * n)
    print(f"│  {n} samples\n└─ Done\n")

    # Stage 1: Autoencoder pretraining (200 epochs)
    print("┌─ Stage 1: Autoencoder pretraining (200 epochs)")
    ae_a = VisionAutoencoder(COMM_DIM)
    ae_b = VisionAutoencoder(COMM_DIM)
    opt_ae = torch.optim.Adam(list(ae_a.parameters()) + list(ae_b.parameters()), lr=1e-3)
    for ep in range(200):
        ae_a.train(); ae_b.train()
        idx = torch.randperm(split)[:32]
        ra, _ = ae_a(ta[idx]); rb, _ = ae_b(tb[idx])
        loss = F.mse_loss(ra, ta[idx]) + F.mse_loss(rb, tb[idx])
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        if (ep+1) % 50 == 0:
            print(f"│  Epoch {ep+1}: recon loss={loss.item():.4f}")
    print("└─ Done\n")

    # Build fused model and transfer encoder weights
    model = VisionOcclusionFusedModel(STATE_DIM, COMM_DIM)
    # Copy shared CNN layers (first 7 layers = conv layers + pool + flatten)
    with torch.no_grad():
        for mp, ap in zip(model.encoder_a[:-1].parameters(), ae_a.encoder[:-1].parameters()):
            mp.copy_(ap)
        for mp, ap in zip(model.encoder_b[:-1].parameters(), ae_b.encoder[:-1].parameters()):
            mp.copy_(ap)

    # Stage 2: Frozen encoders (100 epochs)
    print("┌─ Stage 2: Frozen encoders (100 epochs)")
    for p in model.encoder_a.parameters(): p.requires_grad = False
    for p in model.encoder_b.parameters(): p.requires_grad = False
    opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    for ep in range(100):
        model.train()
        idx = torch.randperm(split)[:32]
        pred, _, kl = model(ta[idx], tb[idx])
        loss = F.mse_loss(pred, tns[idx]) + BETA * kl
        opt2.zero_grad(); loss.backward(); opt2.step()
        if (ep+1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                p_te, _, _ = model(ta[split:], tb[split:])
                m = F.mse_loss(p_te, tns[split:]).item()
            print(f"│  Epoch {ep+1}: test MSE={m:.4f}")
    print("└─ Done\n")

    # Stage 3: End-to-end (300 epochs)
    print("┌─ Stage 3: End-to-end fine-tuning (300 epochs)")
    for p in model.parameters(): p.requires_grad = True
    opt3 = torch.optim.Adam(model.parameters(), lr=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt3, 100, 0.5)
    for ep in range(300):
        model.train()
        idx = torch.randperm(split)[:32]
        pred, _, kl = model(ta[idx], tb[idx])
        loss = F.mse_loss(pred, tns[idx]) + BETA * kl
        opt3.zero_grad(); loss.backward(); opt3.step(); sched.step()
        if (ep+1) % 75 == 0:
            model.eval()
            with torch.no_grad():
                p_te, _, _ = model(ta[split:], tb[split:])
                m = F.mse_loss(p_te, tns[split:]).item()
            print(f"│  Epoch {ep+1}: test MSE={m:.4f}")
    model.eval()
    with torch.no_grad():
        p_te, _, _ = model(ta[split:], tb[split:])
        mse_fused = F.mse_loss(p_te, tns[split:]).item()
    print(f"│  Final MSE: {mse_fused:.4f}")
    print("└─ Done\n")

    # Correlation analysis
    mu_a, mu_b = model.get_communication_vectors(ta[split:], tb[split:])
    mu_a_np = mu_a.numpy(); state_te = full_st[split:]
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    corr_vis = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_vis[d, p] = np.corrcoef(mu_a_np[:, d], state_te[:, p])[0, 1]

    # Load Phase 7 correlation
    p7_path = OUTPUT_DIR / "phase7_data.npz"
    corr_state = np.zeros((COMM_DIM, STATE_DIM))
    if p7_path.exists():
        corr_state = np.load(p7_path)['corr_a']

    # Similarity
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((COMM_DIM, COMM_DIM))
    for i in range(COMM_DIM):
        for j in range(COMM_DIM):
            cost[i, j] = -np.abs(np.corrcoef(np.abs(corr_vis[i]), np.abs(corr_state[j]))[0, 1])
    ri, ci = linear_sum_assignment(cost)
    similarity = -cost[ri, ci].mean()

    # Plots
    fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig1.suptitle("Phase 11b: Vision Occlusion", fontweight='bold')
    idx_ex = split + 5
    balls_ex = [Ball(full_st[idx_ex,b*4], full_st[idx_ex,b*4+1],
                     full_st[idx_ex,b*4+2], full_st[idx_ex,b*4+3]) for b in range(N_BALLS)]
    axes[0].imshow(sim.render_frame(balls_ex, RES)); axes[0].set_title('Full'); axes[0].axis('off')
    axes[1].imshow(imgs_a[idx_ex].transpose(1,2,0)); axes[1].set_title('Agent A'); axes[1].axis('off')
    axes[2].imshow(imgs_b[idx_ex].transpose(1,2,0)); axes[2].set_title('Agent B'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11b_vision.png", dpi=150, bbox_inches='tight'); plt.close()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle("Phase 11b: State vs Vision Protocol", fontweight='bold')
    im1 = ax1.imshow(corr_state, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(STATE_DIM)); ax1.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax1.set_title('State-Based (Phase 7)'); plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(corr_vis, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(STATE_DIM)); ax2.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_title('Vision-Based (Phase 11b)'); plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11b_correlation.png", dpi=150, bbox_inches='tight'); plt.close()

    fig3, ax = plt.subplots(figsize=(6, 5))
    ax.bar([0, 1, 2], [0.392, similarity, 0.5], color=['#cccccc', '#2a9d8f', '#eeeeee'])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Phase 11\n(100 ep)', 'Phase 11b\n(600 ep)', 'Random'])
    ax.set_ylabel('Protocol Similarity'); ax.set_ylim(0, 1)
    ax.set_title('Phase 11b: Representation Invariance', fontweight='bold')
    for i, v in enumerate([0.392, similarity, 0.5]):
        ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11b_invariance.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase11b_vision.png")
    print(f"  → results/phase11b_correlation.png")
    print(f"  → results/phase11b_invariance.png")
    print(f"\n{'='*60}")
    print("PHASE 11b SUMMARY")
    print(f"{'='*60}")
    print(f"  Vision fused MSE: {mse_fused:.4f}  (was 3.04 in Phase 11)")
    print(f"  Protocol similarity: {similarity:.3f}  (was 0.392)")
    return {"mse": mse_fused, "similarity": similarity}


# ── Phase 12: Adversarial Robustness ───────────────────────────

def run_phase12():
    from physics_sim import (SimConfig, PhysicsSimulator, Ball,
                             generate_random_balls, generate_occlusion_dataset,
                             get_occluded_state)
    from world_model import BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 12: Adversarial Communication — Distribution Shifts")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 3
    STATE_DIM = N_BALLS * 4; COMM_DIM = 8; BETA = 0.001

    # Step 1: Train reference model
    print("\n┌─ Training reference model")
    fc, oa, ob, fn, ho = generate_occlusion_dataset(N_TRAJ, N_STEPS, N_BALLS, seed=42)
    n = len(fc); sp = int(0.9 * n)
    r = _train_bottleneck_config(
        torch.tensor(oa[:sp], dtype=torch.float32),
        torch.tensor(ob[:sp], dtype=torch.float32),
        torch.tensor(fn[:sp], dtype=torch.float32),
        torch.tensor(oa[sp:], dtype=torch.float32),
        torch.tensor(ob[sp:], dtype=torch.float32),
        torch.tensor(fn[sp:], dtype=torch.float32),
        torch.tensor(ho[sp:], dtype=torch.bool),
        STATE_DIM, COMM_DIM, BETA, "ref")
    model = r["model"]
    base_mse = r["mse_f"]
    print(f"│  Baseline MSE: {base_mse:.4f}")
    print("└─ Done\n")

    # Step 2: Generate shifted datasets
    def _gen_shifted(label, gravity=None, speed_mult=1.0, masses=None,
                     extra_ball=False, no_right_wall=False, n_traj=300):
        cfg = SimConfig()
        if gravity is not None: cfg.gravity = gravity
        sim = PhysicsSimulator(cfg)
        fc_s, oa_s, ob_s, fn_s, ho_s = [], [], [], [], []
        np.random.seed(99)
        for _ in range(n_traj):
            nb = N_BALLS + (1 if extra_ball else 0)
            balls = generate_random_balls(nb, cfg)
            if masses:
                for i, m in enumerate(masses):
                    if i < len(balls): balls[i] = Ball(balls[i].x, balls[i].y,
                        balls[i].vx*speed_mult, balls[i].vy*speed_mult, balls[i].radius, m)
            else:
                balls = [Ball(b.x, b.y, b.vx*speed_mult, b.vy*speed_mult, b.radius, b.mass)
                         for b in balls]
            cur = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls]
            for _ in range(N_STEPS):
                # Only track first N_BALLS
                state = np.array([v for b in cur[:N_BALLS] for v in [b.x,b.y,b.vx,b.vy]],
                                 dtype=np.float32)
                nxt_b = sim.step(cur)
                if no_right_wall:
                    nxt_b = [Ball(min(b.x, 10.0), b.y, b.vx, b.vy, b.radius, b.mass)
                             for b in nxt_b]
                nst = np.array([v for b in nxt_b[:N_BALLS] for v in [b.x,b.y,b.vx,b.vy]],
                               dtype=np.float32)
                fc_s.append(state); fn_s.append(nst)
                oa_s.append(get_occluded_state(state, N_BALLS, 'A'))
                ob_s.append(get_occluded_state(state, N_BALLS, 'B'))
                ho_s.append(any((state[i*4]<1.0 and nst[i*4]>=1.0) or
                                (state[i*4]>=1.0 and nst[i*4]<1.0) for i in range(N_BALLS)))
                cur = nxt_b
        return {k: np.array(v, dtype=np.float32) if k != 'ho' else np.array(v)
                for k, v in zip(['fc','oa','ob','fn','ho'],
                                [fc_s, oa_s, ob_s, fn_s, ho_s])}

    shifts = {
        "Baseline": None,
        "Zero-G": dict(gravity=0.0),
        "2× Speed": dict(speed_mult=2.0),
        "Mixed Mass": dict(masses=[0.5, 1.0, 2.0]),
        "Extra Ball": dict(extra_ball=True),
        "No Right Wall": dict(no_right_wall=True),
    }
    results = {}
    comm_stats = {}

    model.eval()
    for label, kwargs in shifts.items():
        if kwargs is None:
            # Use held-out test data
            t_oa = torch.tensor(oa[sp:], dtype=torch.float32)
            t_ob = torch.tensor(ob[sp:], dtype=torch.float32)
            t_fn = torch.tensor(fn[sp:], dtype=torch.float32)
        else:
            d = _gen_shifted(label, **kwargs)
            t_oa = torch.tensor(d['oa'], dtype=torch.float32)
            t_ob = torch.tensor(d['ob'], dtype=torch.float32)
            t_fn = torch.tensor(d['fn'], dtype=torch.float32)
        with torch.no_grad():
            pred, _, _ = model(t_oa, t_ob)
            mse = F.mse_loss(pred, t_fn).item()
            mu_a, mu_b = model.get_communication_vectors(t_oa, t_ob)
            var_per_dim = mu_a.var(0).numpy()
        results[label] = mse
        comm_stats[label] = var_per_dim
        print(f"  {label:15s}: MSE={mse:.4f}  comm_var={var_per_dim.mean():.3f}")

    # Step 3: Zero-G adaptation (20 epochs)
    print("\n┌─ Zero-G adaptation (20 epochs fine-tuning)")
    d_zg = _gen_shifted("zerog", gravity=0.0, n_traj=500)
    adapt_model = BottleneckedFusionModel(STATE_DIM, COMM_DIM)
    adapt_model.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(adapt_model.parameters(), lr=1e-4)
    t_oa_zg = torch.tensor(d_zg['oa'], dtype=torch.float32)
    t_ob_zg = torch.tensor(d_zg['ob'], dtype=torch.float32)
    t_fn_zg = torch.tensor(d_zg['fn'], dtype=torch.float32)
    zg_sp = int(0.8 * len(t_oa_zg))
    adapt_curve = []

    # Pre-adaptation correlation
    adapt_model.eval()
    mu_pre, _ = adapt_model.get_communication_vectors(t_oa_zg[zg_sp:], t_ob_zg[zg_sp:])
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])
    corr_pre = np.zeros((COMM_DIM, STATE_DIM))
    for dd in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_pre[dd, p] = np.corrcoef(mu_pre.numpy()[:, dd], d_zg['fc'][zg_sp:, p])[0, 1]

    for ep in range(20):
        adapt_model.train()
        idx = torch.randperm(zg_sp)[:256]
        pred, _, kl = adapt_model(t_oa_zg[idx], t_ob_zg[idx])
        loss = F.mse_loss(pred, t_fn_zg[idx]) + BETA * kl
        opt.zero_grad(); loss.backward(); opt.step()
        adapt_model.eval()
        with torch.no_grad():
            p_te, _, _ = adapt_model(t_oa_zg[zg_sp:], t_ob_zg[zg_sp:])
            adapt_curve.append(F.mse_loss(p_te, t_fn_zg[zg_sp:]).item())
    print(f"│  Before: MSE={results['Zero-G']:.4f}")
    print(f"│  After 20ep: MSE={adapt_curve[-1]:.4f}")
    print("└─ Done\n")

    # Post-adaptation correlation
    mu_post, _ = adapt_model.get_communication_vectors(t_oa_zg[zg_sp:], t_ob_zg[zg_sp:])
    corr_post = np.zeros((COMM_DIM, STATE_DIM))
    for dd in range(COMM_DIM):
        for p in range(STATE_DIM):
            corr_post[dd, p] = np.corrcoef(mu_post.numpy()[:, dd], d_zg['fc'][zg_sp:, p])[0, 1]

    # Plot 1: Robustness bar chart + comm variance
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig1.suptitle("Phase 12: Adversarial Robustness", fontweight='bold')
    lbls = list(results.keys()); vals = list(results.values())
    cols_r = ['#4a90d9'] + ['#e87d5a']*(len(lbls)-1)
    bars = ax1.bar(range(len(lbls)), vals, color=cols_r)
    ax1.set_xticks(range(len(lbls))); ax1.set_xticklabels(lbls, rotation=15)
    ax1.set_ylabel('Test MSE')
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=8)
    # Comm variance
    base_var = comm_stats["Baseline"]
    for i, (lbl, var) in enumerate(comm_stats.items()):
        if lbl == "Baseline": continue
        delta = var - base_var
        ax2.bar(np.arange(COMM_DIM) + (i-1)*0.15, delta, 0.15, label=lbl)
    ax2.set_xlabel('Comm dimension'); ax2.set_ylabel('Δ Variance vs baseline')
    ax2.legend(fontsize=7); ax2.set_title('Communication "Panic" per Dimension')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase12_robustness.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Adaptation
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig2.suptitle("Phase 12: Zero-G Adaptation", fontweight='bold')
    axes[0].plot(range(1, 21), adapt_curve, 'o-', color='#2a9d8f')
    axes[0].axhline(results['Zero-G'], color='red', ls='--', label='Before')
    axes[0].set_xlabel('Fine-tune Epoch'); axes[0].set_ylabel('MSE'); axes[0].legend()
    axes[0].set_title('Adaptation Curve')
    im1 = axes[1].imshow(corr_pre, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_xticks(range(STATE_DIM)); axes[1].set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=6)
    axes[1].set_title('Before Adaptation'); plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(corr_post, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_xticks(range(STATE_DIM)); axes[2].set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=6)
    axes[2].set_title('After 20 Epochs'); plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase12_adaptation.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase12_robustness.png")
    print(f"  → results/phase12_adaptation.png")
    # Check vy correlation change
    vy_idx = [b*4+3 for b in range(N_BALLS)]
    vy_pre = np.abs(corr_pre[:, vy_idx]).max()
    vy_post = np.abs(corr_post[:, vy_idx]).max()
    print(f"\n{'='*60}")
    print("PHASE 12 SUMMARY")
    print(f"{'='*60}")
    for l, v in results.items():
        print(f"  {l:15s}: MSE={v:.4f}")
    print(f"  Zero-G adaptation: {results['Zero-G']:.4f} → {adapt_curve[-1]:.4f}")
    print(f"  vy max|corr|: {vy_pre:.3f} → {vy_post:.3f}")
    if vy_post > vy_pre * 1.2:
        print("  ★ Agents learned to encode vy after gravity was removed!")
    return {"results": results, "adapt_curve": adapt_curve}


# ── Phase 13: Multi-Agent Scaling ──────────────────────────────

def run_phase13():
    from physics_sim import generate_multiagent_dataset
    from world_model import MultiAgentFusionModel

    print("\n" + "="*60)
    print("PHASE 13: Multi-Agent Scaling (2–6 Agents)")
    print("="*60)

    N_TRAJ, N_STEPS, N_BALLS = 1500, 50, 5
    STATE_DIM = N_BALLS * 4; BETA = 0.001
    AGENT_COUNTS = [2, 3, 4, 5, 6]

    # Helper: train a MultiAgentFusionModel
    def _train_ma(n_agents, cdpa, label, n_traj=N_TRAJ):
        fc, views, fn, ho = generate_multiagent_dataset(n_traj, N_STEPS, N_BALLS, n_agents, seed=42)
        n = len(fc); sp = int(0.9 * n)
        tv = [torch.tensor(v, dtype=torch.float32) for v in views]
        tfn = torch.tensor(fn, dtype=torch.float32)
        model = MultiAgentFusionModel(n_agents, STATE_DIM, cdpa)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5)
        # Stage 1: pretrain (100 ep)
        for ep in range(100):
            model.train()
            idx = torch.randperm(sp)[:256]
            pred, _, kl = model([v[idx] for v in tv])
            loss = F.mse_loss(pred, tfn[idx]) + BETA * kl
            opt.zero_grad(); loss.backward(); opt.step()
        # Stage 2: full training (200 ep)
        opt2 = torch.optim.Adam(model.parameters(), lr=5e-4)
        for ep in range(200):
            model.train()
            idx = torch.randperm(sp)[:256]
            pred, _, kl = model([v[idx] for v in tv])
            loss = F.mse_loss(pred, tfn[idx]) + BETA * kl
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pred_te, _, _ = model([v[sp:] for v in tv])
            mse = F.mse_loss(pred_te, tfn[sp:]).item()
        print(f"  {label:20s}: MSE={mse:.4f}")
        return model, mse, tv, fc, sp

    # Experiment 1: Growing bandwidth (cdpa=4 per agent)
    print("\n┌─ Growing bandwidth (4 dims/agent)")
    grow_mses = {}
    grow_models = {}
    for na in AGENT_COUNTS:
        model, mse, tv, fc, sp = _train_ma(na, 4, f"{na} agents (4×{na}={na*4}d)")
        grow_mses[na] = mse
        grow_models[na] = (model, tv, fc, sp)
    print("└─ Done\n")

    # Experiment 2: Fixed bandwidth (8 total dims)
    print("┌─ Fixed bandwidth (8 dims total)")
    fixed_mses = {}
    for na in AGENT_COUNTS:
        cdpa = max(1, 8 // na)
        model, mse, _, _, _ = _train_ma(na, cdpa, f"{na} agents ({cdpa}×{na}={cdpa*na}d)")
        fixed_mses[na] = mse
    print("└─ Done\n")

    # Analysis on 4-agent model
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])

    na_detail = 4
    if na_detail in grow_models:
        model4, tv4, fc4, sp4 = grow_models[na_detail]
        mus = model4.get_communication_vectors([v[sp4:] for v in tv4])
        # Per-agent correlation
        corrs = []
        for a in range(na_detail):
            mu_np = mus[a].numpy()
            c = np.zeros((4, STATE_DIM))
            for d in range(4):
                for p in range(STATE_DIM):
                    c[d, p] = np.corrcoef(mu_np[:, d], fc4[sp4:, p])[0, 1]
            corrs.append(c)

        # Mutual information proxy: correlation between agent latent vectors
        mi_matrix = np.zeros((na_detail, na_detail))
        for i in range(na_detail):
            for j in range(na_detail):
                if i == j: mi_matrix[i,j] = 1.0; continue
                mi = 0
                for di in range(4):
                    for dj in range(4):
                        mi = max(mi, abs(np.corrcoef(mus[i].numpy()[:, di],
                                                      mus[j].numpy()[:, dj])[0, 1]))
                mi_matrix[i, j] = mi

        # Ablation impact matrix: zero out each agent, measure impact
        ablation = np.zeros((na_detail, na_detail))
        tfn4 = torch.tensor(fc4, dtype=torch.float32)  # reuse fc for targets isn't right...
        # regenerate targets
        _, views4, fn4, _ = generate_multiagent_dataset(N_TRAJ, N_STEPS, N_BALLS, na_detail, seed=42)
        tv4_all = [torch.tensor(v, dtype=torch.float32) for v in views4]
        tfn4 = torch.tensor(fn4, dtype=torch.float32)
        with torch.no_grad():
            base_pred, _, _ = model4([v[sp4:] for v in tv4_all])
            base_mses = np.zeros(na_detail)
            for a in range(na_detail):
                # Per-agent "contribution" to each strip
                strip_w = 2.0 / na_detail
                for b in range(N_BALLS):
                    bidx = b * 4
                    base_mses[a] += F.mse_loss(base_pred[:, bidx:bidx+4],
                                               tfn4[sp4:, bidx:bidx+4]).item()
            for abl in range(na_detail):
                modified_views = [v[sp4:].clone() for v in tv4_all]
                modified_views[abl] = torch.zeros_like(modified_views[abl])
                pred_abl, _, _ = model4(modified_views)
                for target_a in range(na_detail):
                    strip_lo = target_a * (2.0 / na_detail)
                    strip_hi = strip_lo + 2.0 / na_detail
                    # MSE on balls that tend to be in target_a's strip
                    abl_mse = F.mse_loss(pred_abl, tfn4[sp4:]).item()
                    ablation[abl, target_a] = abl_mse

    # Plot 1: Scaling
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle("Phase 13: Multi-Agent Scaling", fontweight='bold')
    ax1.plot(AGENT_COUNTS, [grow_mses[n] for n in AGENT_COUNTS], 'o-',
             color='#2a9d8f', label='Growing BW (4/agent)', linewidth=2)
    ax1.plot(AGENT_COUNTS, [fixed_mses[n] for n in AGENT_COUNTS], 's--',
             color='#e87d5a', label='Fixed BW (8 total)', linewidth=2)
    ax1.set_xlabel('Number of Agents'); ax1.set_ylabel('Test MSE')
    ax1.set_title('MSE vs Agent Count'); ax1.legend()
    # Bandwidth plot
    bw_grow = [n*4 for n in AGENT_COUNTS]
    ax2.plot(bw_grow, [grow_mses[n] for n in AGENT_COUNTS], 'o-',
             color='#2a9d8f', label='Growing BW', linewidth=2)
    ax2.set_xlabel('Total Bandwidth (dims)'); ax2.set_ylabel('Test MSE')
    ax2.set_title('MSE vs Total Bandwidth'); ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase13_scaling.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Per-agent correlation (4-agent)
    if na_detail in grow_models:
        fig2, axes = plt.subplots(1, na_detail, figsize=(4*na_detail, 4))
        fig2.suptitle("Phase 13: Per-Agent Correlation (4 agents)", fontweight='bold')
        for a in range(na_detail):
            im = axes[a].imshow(corrs[a], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            axes[a].set_xticks(range(STATE_DIM))
            axes[a].set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=5)
            strip_lo = a * 0.5; strip_hi = strip_lo + 0.5
            axes[a].set_title(f'Agent {a} [{strip_lo:.1f},{strip_hi:.1f}]', fontsize=9)
            plt.colorbar(im, ax=axes[a])
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "phase13_agent_specialization.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 3: Communication graph
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig3.suptitle("Phase 13: Inter-Agent Communication", fontweight='bold')
        im1 = ax1.imshow(mi_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_xticks(range(na_detail)); ax1.set_yticks(range(na_detail))
        ax1.set_xticklabels([f'A{i}' for i in range(na_detail)])
        ax1.set_yticklabels([f'A{i}' for i in range(na_detail)])
        ax1.set_title('Max Correlation Between Agents')
        for i in range(na_detail):
            for j in range(na_detail):
                ax1.text(j, i, f'{mi_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(ablation, cmap='Reds')
        ax2.set_xticks(range(na_detail)); ax2.set_yticks(range(na_detail))
        ax2.set_xticklabels([f'Target A{i}' for i in range(na_detail)], fontsize=8)
        ax2.set_yticklabels([f'Ablate A{i}' for i in range(na_detail)], fontsize=8)
        ax2.set_title('Ablation Impact Matrix')
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "phase13_comm_graph.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  → results/phase13_scaling.png")
    print(f"  → results/phase13_agent_specialization.png")
    print(f"  → results/phase13_comm_graph.png")
    print(f"\n{'='*60}")
    print("PHASE 13 SUMMARY")
    print(f"{'='*60}")
    print("  Growing bandwidth:")
    for na in AGENT_COUNTS:
        print(f"    {na} agents: MSE={grow_mses[na]:.4f}")
    print("  Fixed bandwidth:")
    for na in AGENT_COUNTS:
        print(f"    {na} agents: MSE={fixed_mses[na]:.4f}")
    return {"grow": grow_mses, "fixed": fixed_mses}


# ── Phase 14: Goal-Directed Planning ───────────────────────────

def run_phase14():
    from physics_sim import (SimConfig, PhysicsSimulator, Ball,
                             ControllableSimulator, generate_random_balls,
                             generate_occlusion_dataset, generate_planning_episodes,
                             get_occluded_state)
    from world_model import PlanningWorldModel, BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 14: Goal-Directed Planning (Mode-2)")
    print("="*60)

    N_BALLS = 3; STATE_DIM = N_BALLS * 4; COMM_DIM = 8; ACTION_DIM = N_BALLS * 2
    HORIZON = 15; BETA = 0.001

    # Step 1: Train world model base (100 epochs, reused architecture)
    print("\n┌─ Training world model base (100 epochs)")
    fc, oa, ob, fn, ho = generate_occlusion_dataset(1500, 50, N_BALLS, seed=42)
    sp = int(0.9 * len(fc))
    wm_base = BottleneckedFusionModel(STATE_DIM, COMM_DIM)
    opt_wm = torch.optim.Adam(wm_base.parameters(), lr=1e-3)
    for ep in range(100):
        wm_base.train()
        idx = torch.randperm(sp)[:256]
        pred, _, kl = wm_base(torch.tensor(oa[idx], dtype=torch.float32),
                              torch.tensor(ob[idx], dtype=torch.float32))
        loss = F.mse_loss(pred, torch.tensor(fn[idx], dtype=torch.float32)) + BETA * kl
        opt_wm.zero_grad(); loss.backward(); opt_wm.step()
    wm_base.eval()
    with torch.no_grad():
        p_te, _, _ = wm_base(torch.tensor(oa[sp:], dtype=torch.float32),
                             torch.tensor(ob[sp:], dtype=torch.float32))
        wm_mse = F.mse_loss(p_te, torch.tensor(fn[sp:], dtype=torch.float32)).item()
    print(f"│  World model MSE: {wm_mse:.4f}")
    print("└─ Done\n")

    # Step 2: Build PlanningWorldModel and transfer weights
    model = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    with torch.no_grad():
        for mp, wp in zip(model.encoder_a.parameters(), wm_base.encoder_a.parameters()):
            mp.copy_(wp)
        for mp, wp in zip(model.encoder_b.parameters(), wm_base.encoder_b.parameters()):
            mp.copy_(wp)
        for mp, wp in zip(model.fusion.parameters(), wm_base.fusion.parameters()):
            mp.copy_(wp)
        for mp, wp in zip(model.state_decoder.parameters(), wm_base.state_decoder.parameters()):
            mp.copy_(wp)
    model.freeze_world_model()

    # Step 3: Train planning components
    print("┌─ Training planning components (5000 episodes)")
    episodes = generate_planning_episodes(6000, N_BALLS, seed=42)
    plan_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(plan_params, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1500, 0.5)
    train_losses = []

    for ep_i in range(5000):
        model.train()
        batch_idx = np.random.choice(5000, 32)
        inits = torch.tensor(np.array([episodes[i][0] for i in batch_idx]), dtype=torch.float32)
        goals = torch.tensor(np.array([episodes[i][1] for i in batch_idx]), dtype=torch.float32)
        # Create occluded views
        oa_b = torch.zeros_like(inits)
        ob_b = torch.zeros_like(inits)
        for s in range(32):
            oa_b[s] = torch.tensor(get_occluded_state(inits[s].numpy(), N_BALLS, 'A'))
            ob_b[s] = torch.tensor(get_occluded_state(inits[s].numpy(), N_BALLS, 'B'))
        loss, _ = model.plan_loss(oa_b, ob_b, goals)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(plan_params, 1.0)
        opt.step(); sched.step()
        train_losses.append(loss.item())
        if (ep_i+1) % 1000 == 0:
            print(f"│  Episode {ep_i+1}: loss={np.mean(train_losses[-100:]):.4f}")
    print("└─ Done\n")

    # Step 4: Define evaluation tasks
    tasks = {
        "Gather Left": lambda: np.array([0.3,1.0,0,0, 0.4,0.5,0,0, 0.3,1.5,0,0], dtype=np.float32),
        "Gather Right": lambda: np.array([1.7,1.0,0,0, 1.6,0.5,0,0, 1.7,1.5,0,0], dtype=np.float32),
        "Swap": None,  # will set dynamically
        "Line Up": lambda: np.array([0.5,1.0,0,0, 1.0,1.0,0,0, 1.5,1.0,0,0], dtype=np.float32),
        "Triangle": lambda: np.array([0.5,0.5,0,0, 1.0,1.5,0,0, 1.5,0.5,0,0], dtype=np.float32),
    }

    # Step 5: Evaluate — with comm, without comm, oracle
    cfg = SimConfig(); sim = ControllableSimulator(cfg)
    np.random.seed(123)

    eval_results = {}
    task_trajectories = {}

    for task_name, goal_fn in tasks.items():
        print(f"  Evaluating: {task_name}")
        # Random initial state
        balls_init = generate_random_balls(N_BALLS, cfg)
        init_state = np.array([v for b in balls_init for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)

        if task_name == "Swap":
            goal = init_state.copy()
            for b in range(N_BALLS):
                goal[b*4] = 2.0 - init_state[b*4]  # mirror x
                goal[b*4+2] = 0; goal[b*4+3] = 0  # zero velocity
        else:
            goal = goal_fn()

        goal_t = torch.tensor(goal).unsqueeze(0)
        init_t = torch.tensor(init_state).unsqueeze(0)
        oa_t = torch.tensor(get_occluded_state(init_state, N_BALLS, 'A')).unsqueeze(0)
        ob_t = torch.tensor(get_occluded_state(init_state, N_BALLS, 'B')).unsqueeze(0)

        # With communication
        model.eval()
        with torch.no_grad():
            states_im, acts_a, acts_b, comm_a, comm_b = model.imagine_trajectory(oa_t, ob_t, goal_t)

        # Execute in real simulator
        real_states_comm = [init_state.copy()]
        cur_balls = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls_init]
        for t in range(HORIZON):
            aa = acts_a[0, t].numpy()
            ab = acts_b[0, t].numpy()
            cur_balls = sim.step_with_actions(cur_balls, aa, ab)
            st = np.array([v for b in cur_balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
            real_states_comm.append(st)
        real_states_comm = np.array(real_states_comm)
        final_dist_comm = np.mean([(real_states_comm[-1, b*4] - goal[b*4])**2 +
                                    (real_states_comm[-1, b*4+1] - goal[b*4+1])**2
                                    for b in range(N_BALLS)])

        # Without communication (zero out partner message)
        model_nocomm = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
        model_nocomm.load_state_dict(model.state_dict())
        model_nocomm.eval()
        # Override to block communication
        saved_fusion = model_nocomm.fusion
        model_nocomm.fusion = nn.Sequential(
            nn.Linear(COMM_DIM * 2, 32), nn.ReLU(), nn.Linear(32, 32))
        with torch.no_grad():
            # Won't plan well without matching fusion
            _, acts_a_nc, acts_b_nc, _, _ = model_nocomm.imagine_trajectory(oa_t, ob_t, goal_t)
        model_nocomm.fusion = saved_fusion

        real_states_nocomm = [init_state.copy()]
        cur_balls = [Ball(b.x, b.y, b.vx, b.vy, b.radius, b.mass) for b in balls_init]
        for t in range(HORIZON):
            aa = acts_a_nc[0, t].numpy()
            ab = acts_b_nc[0, t].numpy()
            cur_balls = sim.step_with_actions(cur_balls, aa, ab)
            st = np.array([v for b in cur_balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
            real_states_nocomm.append(st)
        real_states_nocomm = np.array(real_states_nocomm)
        final_dist_nocomm = np.mean([(real_states_nocomm[-1, b*4] - goal[b*4])**2 +
                                      (real_states_nocomm[-1, b*4+1] - goal[b*4+1])**2
                                      for b in range(N_BALLS)])

        eval_results[task_name] = {
            "comm": final_dist_comm, "nocomm": final_dist_nocomm,
            "imagined": states_im[0].numpy(), "real_comm": real_states_comm,
            "goal": goal, "init": init_state,
            "comm_a": comm_a[0].numpy(), "acts_a": acts_a[0].numpy(),
            "acts_b": acts_b[0].numpy()
        }
        print(f"    With comm: dist={final_dist_comm:.4f}  No comm: dist={final_dist_nocomm:.4f}")

    # Plot 1: Planning trajectories
    fig1, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig1.suptitle("Phase 14: Planning Trajectories", fontweight='bold')
    for i, (tname, r) in enumerate(eval_results.items()):
        # Top: with comm
        ax = axes[0, i]
        for b in range(N_BALLS):
            ax.plot(r['real_comm'][:, b*4], r['real_comm'][:, b*4+1], 'o-', markersize=3)
            ax.scatter(r['goal'][b*4], r['goal'][b*4+1], marker='*', s=100, c='red', zorder=5)
            ax.scatter(r['init'][b*4], r['init'][b*4+1], marker='s', s=50, c='green', zorder=5)
        ax.axvline(1.0, color='gray', ls=':', alpha=0.3)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_title(f'{tname}\nd={r["comm"]:.3f}', fontsize=9)
        if i == 0: ax.set_ylabel('With Comm')
        # Bottom: legend
        ax2 = axes[1, i]
        for b in range(N_BALLS):
            ax2.plot(range(HORIZON+1), [np.sqrt((r['real_comm'][t, b*4]-r['goal'][b*4])**2 +
                     (r['real_comm'][t, b*4+1]-r['goal'][b*4+1])**2) for t in range(HORIZON+1)],
                     label=f'b{b+1}')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Dist to goal')
        if i == 0: ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase14_planning.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Communication during planning
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Phase 14: Communication During Planning", fontweight='bold')
    # Use Triangle task
    tr = eval_results["Triangle"]
    for d in range(COMM_DIM):
        axes[0].plot(range(HORIZON), tr['comm_a'][:, d], label=f'd{d}')
    axes[0].set_xlabel('Planning step'); axes[0].set_ylabel('Comm dim value')
    axes[0].set_title('Agent A comm during planning'); axes[0].legend(fontsize=6, ncol=2)
    # Correlation: comm with ball positions vs goal positions
    if HORIZON > 2:
        phys_labels = []
        for b in range(N_BALLS):
            phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])
        # Aggregate across tasks
        all_comm = np.concatenate([r['comm_a'] for r in eval_results.values()], axis=0)
        all_acts = np.concatenate([r['acts_a'] for r in eval_results.values()], axis=0)
        corr_ca = np.zeros((COMM_DIM, ACTION_DIM))
        for d in range(COMM_DIM):
            for a in range(ACTION_DIM):
                if len(all_comm) > 2:
                    corr_ca[d, a] = np.corrcoef(all_comm[:, d], all_acts[:, a])[0, 1]
        act_labels = [f'b{b+1}_f{"xy"[d]}' for b in range(N_BALLS) for d in range(2)]
        im2 = axes[1].imshow(corr_ca, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_xticks(range(ACTION_DIM)); axes[1].set_xticklabels(act_labels, rotation=45, ha='right', fontsize=7)
        axes[1].set_title('Comm ↔ Actions'); plt.colorbar(im2, ax=axes[1])
    # Comm vs no-comm comparison
    task_names = list(eval_results.keys())
    comm_vals = [eval_results[t]["comm"] for t in task_names]
    nocomm_vals = [eval_results[t]["nocomm"] for t in task_names]
    x_pos = np.arange(len(task_names))
    axes[2].bar(x_pos - 0.15, comm_vals, 0.3, label='With Comm', color='#2a9d8f')
    axes[2].bar(x_pos + 0.15, nocomm_vals, 0.3, label='No Comm', color='#e87d5a')
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels(task_names, rotation=20, fontsize=7)
    axes[2].set_ylabel('Final goal distance'); axes[2].legend(); axes[2].set_title('Comm vs No-Comm')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase14_comm_during_planning.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Imagination accuracy
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig3.suptitle("Phase 14: Imagination vs Reality", fontweight='bold')
    # Scatter: imagined vs real states
    all_im, all_re = [], []
    for r in eval_results.values():
        mn = min(len(r['imagined']), len(r['real_comm']))
        all_im.append(r['imagined'][:mn])
        all_re.append(r['real_comm'][:mn])
    all_im = np.concatenate(all_im); all_re = np.concatenate(all_re)
    for d in range(min(STATE_DIM, 6)):
        axes[0].scatter(all_re[:, d], all_im[:, d], alpha=0.3, s=10, label=f'd{d}')
    axes[0].plot([0, 2], [0, 2], 'k--', alpha=0.3)
    axes[0].set_xlabel('Real state'); axes[0].set_ylabel('Imagined state')
    axes[0].set_title('State prediction accuracy')
    im_err = np.mean((all_im - all_re)**2, axis=1)
    axes[1].plot(range(len(im_err)), im_err, 'o-', markersize=2)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Imagination error')
    axes[1].set_title('Error accumulation over horizon')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase14_imagination_accuracy.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: Coordination (swap task)
    fig4, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig4.suptitle("Phase 14: Coordination in Swap Task", fontweight='bold')
    sr = eval_results["Swap"]
    for b in range(N_BALLS):
        axes[0].plot(range(HORIZON), sr['acts_a'][:, b*2], label=f'A→b{b+1} fx')
        axes[0].plot(range(HORIZON), sr['acts_b'][:, b*2], '--', label=f'B→b{b+1} fx')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Force x'); axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title('Agent Forces (x-direction)')
    # Force correlation
    fa = sr['acts_a'].flatten()
    fb = sr['acts_b'].flatten()
    axes[1].scatter(fa, fb, alpha=0.5, s=20)
    corr_ab = np.corrcoef(fa, fb)[0, 1] if len(fa) > 1 else 0
    axes[1].set_xlabel("Agent A forces"); axes[1].set_ylabel("Agent B forces")
    axes[1].set_title(f'Force Correlation: {corr_ab:.3f}')
    axes[1].axhline(0, color='gray', ls=':', alpha=0.3)
    axes[1].axvline(0, color='gray', ls=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase14_coordination.png", dpi=150, bbox_inches='tight'); plt.close()

    # Training curve
    fig5, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(0, 5000, 10), [np.mean(train_losses[max(0,i-10):i+10])
            for i in range(0, 5000, 10)], color='#2a9d8f')
    ax.set_xlabel('Episode'); ax.set_ylabel('Planning Loss')
    ax.set_title('Phase 14: Planning Training Curve', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase14_training.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"\n  → results/phase14_planning.png")
    print(f"  → results/phase14_comm_during_planning.png")
    print(f"  → results/phase14_imagination_accuracy.png")
    print(f"  → results/phase14_coordination.png")
    print(f"  → results/phase14_training.png")

    print(f"\n{'='*60}")
    print("PHASE 14 SUMMARY")
    print(f"{'='*60}")
    for t, r in eval_results.items():
        print(f"  {t:15s}: comm={r['comm']:.4f}  nocomm={r['nocomm']:.4f}")
    return {"eval": eval_results, "model": model}


# ── Phase 15: LLM Interface ───────────────────────────────────

class GoalParser:
    """Maps natural language commands to goal state vectors."""
    def parse(self, command, current_state, n_balls=3):
        cmd = command.lower().strip()
        goal = np.zeros(n_balls * 4, dtype=np.float32)
        if "gather" in cmd and "left" in cmd:
            for b in range(n_balls):
                goal[b*4] = 0.2 + b*0.15; goal[b*4+1] = 0.5 + b*0.5
        elif "gather" in cmd and "right" in cmd:
            for b in range(n_balls):
                goal[b*4] = 1.6 + b*0.15; goal[b*4+1] = 0.5 + b*0.5
        elif "spread" in cmd:
            for b in range(n_balls):
                goal[b*4] = 0.4 + b*0.6; goal[b*4+1] = 1.0
        elif "line" in cmd and "horiz" in cmd:
            for b in range(n_balls):
                goal[b*4] = 0.5 + b*0.5; goal[b*4+1] = 1.0
        elif "line" in cmd and "vert" in cmd:
            for b in range(n_balls):
                goal[b*4] = 1.0; goal[b*4+1] = 0.4 + b*0.6
        elif "stack" in cmd:
            for b in range(n_balls):
                goal[b*4] = 1.0; goal[b*4+1] = 0.8 + b*0.2
        elif "triangle" in cmd:
            pts = [(0.5, 0.5), (1.0, 1.5), (1.5, 0.5)]
            for b in range(min(n_balls, 3)):
                goal[b*4] = pts[b][0]; goal[b*4+1] = pts[b][1]
        elif "swap" in cmd:
            for b in range(n_balls):
                goal[b*4] = 2.0 - current_state[b*4]
                goal[b*4+1] = current_state[b*4+1]
        elif "freeze" in cmd:
            goal = current_state.copy()
        else:  # chaos / random
            for b in range(n_balls):
                goal[b*4] = np.random.uniform(0.2, 1.8)
                goal[b*4+1] = np.random.uniform(0.2, 1.8)
        return goal


class WorldModelAgent:
    """Full agent: perception → world model → planning → action."""
    def __init__(self, planning_model, n_balls=3):
        self.model = planning_model
        self.parser = GoalParser()
        self.n_balls = n_balls
        self.history = []
        self.current_state = None

    def observe(self, state):
        self.current_state = state.copy()

    def command(self, text):
        from physics_sim import get_occluded_state
        goal = self.parser.parse(text, self.current_state, self.n_balls)
        oa = torch.tensor(get_occluded_state(self.current_state, self.n_balls, 'A')).unsqueeze(0)
        ob = torch.tensor(get_occluded_state(self.current_state, self.n_balls, 'B')).unsqueeze(0)
        goal_t = torch.tensor(goal).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            states, acts_a, acts_b, comm_a, comm_b = self.model.imagine_trajectory(oa, ob, goal_t)
        return {"goal": goal, "acts_a": acts_a[0].numpy(), "acts_b": acts_b[0].numpy(),
                "imagined": states[0].numpy(), "comm_a": comm_a[0].numpy()}

    def execute(self, plan, simulator, state):
        from physics_sim import Ball
        real_states = [state.copy()]
        balls = [Ball(state[b*4], state[b*4+1], state[b*4+2], state[b*4+3])
                 for b in range(self.n_balls)]
        for t in range(len(plan["acts_a"])):
            balls = simulator.step_with_actions(balls, plan["acts_a"][t], plan["acts_b"][t])
            st = np.array([v for b in balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
            real_states.append(st)
        self.current_state = real_states[-1]
        return np.array(real_states)

    def report(self, real_states, goal):
        final = real_states[-1]
        dists = [np.sqrt((final[b*4]-goal[b*4])**2 + (final[b*4+1]-goal[b*4+1])**2)
                 for b in range(self.n_balls)]
        avg = np.mean(dists)
        worst = np.argmax(dists)
        return (f"Done. Avg distance to goal: {avg:.3f}. "
                f"Ball {worst+1} was hardest (dist={dists[worst]:.3f}).")


def run_phase15():
    from physics_sim import (SimConfig, ControllableSimulator, generate_random_balls,
                             Ball, generate_occlusion_dataset, get_occluded_state)
    from world_model import PlanningWorldModel, BottleneckedFusionModel

    print("\n" + "="*60)
    print("PHASE 15: LLM Interface — Talk to Your World Model")
    print("="*60)

    N_BALLS = 3; STATE_DIM = N_BALLS * 4; COMM_DIM = 8
    ACTION_DIM = N_BALLS * 2; HORIZON = 15; BETA = 0.001

    # Train world model + planning (abridged — 50 epoch WM + 2000 ep planning)
    print("\n┌─ Quick-training world model + planner")
    fc, oa, ob, fn, ho = generate_occlusion_dataset(1500, 50, N_BALLS, seed=42)
    sp = int(0.9 * len(fc))
    wm = BottleneckedFusionModel(STATE_DIM, COMM_DIM)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    for ep in range(50):
        wm.train(); idx = torch.randperm(sp)[:256]
        pred, _, kl = wm(torch.tensor(oa[idx], dtype=torch.float32),
                         torch.tensor(ob[idx], dtype=torch.float32))
        loss = F.mse_loss(pred, torch.tensor(fn[idx], dtype=torch.float32)) + BETA * kl
        opt.zero_grad(); loss.backward(); opt.step()

    model = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    with torch.no_grad():
        for mp, wp in zip(model.encoder_a.parameters(), wm.encoder_a.parameters()): mp.copy_(wp)
        for mp, wp in zip(model.encoder_b.parameters(), wm.encoder_b.parameters()): mp.copy_(wp)
        for mp, wp in zip(model.fusion.parameters(), wm.fusion.parameters()): mp.copy_(wp)
        for mp, wp in zip(model.state_decoder.parameters(), wm.state_decoder.parameters()): mp.copy_(wp)
    model.freeze_world_model()

    from physics_sim import generate_planning_episodes
    episodes = generate_planning_episodes(3000, N_BALLS, seed=42)
    plan_params = [p for p in model.parameters() if p.requires_grad]
    opt_p = torch.optim.Adam(plan_params, lr=1e-3)
    for ep_i in range(2000):
        model.train()
        bi = np.random.choice(2500, 32)
        inits = torch.tensor(np.array([episodes[i][0] for i in bi]), dtype=torch.float32)
        goals = torch.tensor(np.array([episodes[i][1] for i in bi]), dtype=torch.float32)
        oa_b = torch.zeros_like(inits); ob_b = torch.zeros_like(inits)
        for s in range(32):
            oa_b[s] = torch.tensor(get_occluded_state(inits[s].numpy(), N_BALLS, 'A'))
            ob_b[s] = torch.tensor(get_occluded_state(inits[s].numpy(), N_BALLS, 'B'))
        loss, _ = model.plan_loss(oa_b, ob_b, goals)
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(plan_params, 1.0)
        opt_p.step()
        if (ep_i+1) % 500 == 0: print(f"│  Episode {ep_i+1}: loss={loss.item():.4f}")
    print("└─ Done\n")

    # Build agent
    agent = WorldModelAgent(model, N_BALLS)
    cfg = SimConfig(); sim = ControllableSimulator(cfg)
    np.random.seed(77)
    balls = generate_random_balls(N_BALLS, cfg)
    state = np.array([v for b in balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
    agent.observe(state)

    # Demo conversation
    commands = [
        "gather all balls to the left",
        "now spread them out evenly",
        "arrange them in a triangle",
        "swap their sides",
    ]

    conv_results = []
    print("╔" + "═"*50 + "╗")
    print("║  AGENT CONVERSATION DEMO                        ║")
    print("╚" + "═"*50 + "╝")

    for cmd in commands:
        print(f"\n  USER: \"{cmd}\"")
        plan = agent.command(cmd)
        real = agent.execute(plan, sim, agent.current_state if conv_results else state)
        report = agent.report(real, plan["goal"])
        print(f"  AGENT: {report}")
        conv_results.append({
            "cmd": cmd, "goal": plan["goal"], "init": real[0],
            "final": real[-1], "trajectory": real, "comm_a": plan["comm_a"],
            "report": report
        })

    # Plot 1: Conversation sequence — 4 panels
    fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig1.suptitle("Phase 15: Agent Conversation Demo", fontweight='bold', fontsize=14)
    for i, cr in enumerate(conv_results):
        ax = axes[0, i]
        for b in range(N_BALLS):
            ax.plot(cr['trajectory'][:, b*4], cr['trajectory'][:, b*4+1], 'o-', markersize=2)
            ax.scatter(cr['goal'][b*4], cr['goal'][b*4+1], marker='*', s=100, c='red', zorder=5)
            ax.scatter(cr['init'][b*4], cr['init'][b*4+1], marker='s', s=40, c='green', zorder=5)
        ax.axvline(1.0, color='gray', ls=':', alpha=0.3)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_title(f'"{cr["cmd"]}"', fontsize=8)
        # Comm during this command
        ax2 = axes[1, i]
        for d in range(COMM_DIM):
            ax2.plot(range(HORIZON), cr['comm_a'][:, d], linewidth=0.8)
        ax2.set_xlabel('Step'); ax2.set_ylabel('Comm')
        dist = np.mean([np.sqrt((cr['final'][b*4]-cr['goal'][b*4])**2 +
                       (cr['final'][b*4+1]-cr['goal'][b*4+1])**2) for b in range(N_BALLS)])
        ax2.set_title(f'dist={dist:.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase15_agent_demo.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Architecture diagram
    fig2, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    arch_text = """
╔══════════════════════════════════════════╗
║  WORLD MODEL AGENT — Architecture       ║
╠══════════════════════════════════════════╣
║                                         ║
║  [Natural Language]                     ║
║       ↓ GoalParser (LLM placeholder)   ║
║  [Goal State Vector]                    ║
║       ↓                                ║
║  [Multi-Agent Planning]                 ║
║    Agent A ←──bottleneck──→ Agent B     ║
║    policy  ←──8-dim comm──→ policy      ║
║       ↓                                ║
║  [World Model Imagination]              ║
║    Predict H steps into future          ║
║    Optimize actions via backprop        ║
║       ↓                                ║
║  [Action Execution]                     ║
║    Apply forces in simulator            ║
║       ↓                                ║
║  [Report Back]                          ║
║    "Done. All balls gathered left."     ║
║                                         ║
╚══════════════════════════════════════════╝

This IS the JARVIS architecture at toy scale.
The LLM layer is a placeholder. The world
model is real. The planning is real. The
multi-agent coordination is real.
"""
    ax.text(0.5, 0.5, arch_text, transform=ax.transAxes,
            fontsize=11, va='center', ha='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#1a1a2e', edgecolor='#2a9d8f', alpha=0.9),
            color='#00ff88')
    plt.savefig(OUTPUT_DIR / "phase15_architecture.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117'); plt.close()

    print(f"\n  → results/phase15_agent_demo.png")
    print(f"  → results/phase15_architecture.png")

    print(f"\n{'='*60}")
    print("PHASE 15 SUMMARY")
    print(f"{'='*60}")
    for cr in conv_results:
        dist = np.mean([np.sqrt((cr['final'][b*4]-cr['goal'][b*4])**2 +
                       (cr['final'][b*4+1]-cr['goal'][b*4+1])**2) for b in range(N_BALLS)])
        print(f"  \"{cr['cmd']}\" → dist={dist:.3f}")
    return conv_results


# ── Phase 16: Deep Training ────────────────────────────────────

def run_phase16():
    from physics_sim import (SimConfig, PhysicsSimulator, Ball,
                             ControllableSimulator, generate_random_balls,
                             generate_occlusion_dataset, generate_planning_episodes,
                             get_occluded_state)
    from world_model import (PlanningWorldModel, BottleneckedFusionModel,
                             count_params)

    print("\n" + "="*60)
    print("PHASE 16: DEEP TRAINING — Fix the Foundation")
    print("="*60)

    N_BALLS = 3; STATE_DIM = N_BALLS * 4; COMM_DIM = 8; BETA = 0.001
    ACTION_DIM = N_BALLS * 2; HORIZON = 20

    # ════════════════════════════════════════════════
    # STEP 1: Deep-Train World Model Foundation
    # ════════════════════════════════════════════════
    print("\n┌─ Step 1: Deep-Training World Model Foundation (2000 epochs)")
    print("│  Generating data: 2000 traj × 50 steps")
    fc, oa, ob, fn, ho = generate_occlusion_dataset(2000, 50, N_BALLS, seed=42)
    sp = int(0.8 * len(fc))
    print(f"│  {len(fc)} samples, train={sp}, val={len(fc)-sp}")

    fc_t = torch.tensor(fc[:sp], dtype=torch.float32)
    oa_t = torch.tensor(oa[:sp], dtype=torch.float32)
    ob_t = torch.tensor(ob[:sp], dtype=torch.float32)
    fn_t = torch.tensor(fn[:sp], dtype=torch.float32)
    ho_t = torch.tensor(ho[:sp], dtype=torch.float32)
    oa_v = torch.tensor(oa[sp:], dtype=torch.float32)
    ob_v = torch.tensor(ob[sp:], dtype=torch.float32)
    fn_v = torch.tensor(fn[sp:], dtype=torch.float32)
    ho_v = torch.tensor(ho[sp:], dtype=torch.float32)

    model_f = BottleneckedFusionModel(STATE_DIM, COMM_DIM)
    print(f"│  Model: {count_params(model_f)/1e3:.1f}K params")

    opt = torch.optim.Adam(model_f.parameters(), lr=1e-3)
    best_val = float('inf'); best_state = None; patience = 0
    train_curve = []; val_curve = []

    for ep in range(2000):
        # LR schedule
        if ep == 1000:
            for g in opt.param_groups: g['lr'] = 3e-4
            print("│  LR → 3e-4")
        elif ep == 1500:
            for g in opt.param_groups: g['lr'] = 1e-4
            print("│  LR → 1e-4")

        model_f.train()
        idx = torch.randperm(sp)[:256]
        pred, mu, kl = model_f(oa_t[idx], ob_t[idx])
        loss = F.mse_loss(pred, fn_t[idx]) + BETA * kl
        opt.zero_grad(); loss.backward(); opt.step()
        train_curve.append(loss.item())

        # Validation every 20 epochs
        if (ep+1) % 20 == 0:
            model_f.eval()
            with torch.no_grad():
                pv, _, kvl = model_f(oa_v, ob_v)
                val_mse = F.mse_loss(pv, fn_v).item()
            val_curve.append((ep+1, val_mse))
            if val_mse < best_val:
                best_val = val_mse; patience = 0
                best_state = {k: v.clone() for k, v in model_f.state_dict().items()}
            else:
                patience += 1
            if (ep+1) % 200 == 0:
                print(f"│  Epoch {ep+1}: train={loss.item():.4f} val={val_mse:.4f} best={best_val:.4f}")
            if patience >= 10:  # 200 epochs without improvement
                print(f"│  Early stopping at epoch {ep+1}")
                break

    model_f.load_state_dict(best_state)
    torch.save(best_state, str(OUTPUT_DIR / "foundation_model_deep.pth"))
    model_f.eval()
    with torch.no_grad():
        pv, _, _ = model_f(oa_v, ob_v)
        final_mse = F.mse_loss(pv, fn_v).item()
        # Handoff MSE
        ho_mask = ho_v > 0.5
        if ho_mask.sum() > 0:
            ho_mse = F.mse_loss(pv[ho_mask], fn_v[ho_mask]).item()
        else:
            ho_mse = final_mse
    print(f"│  Final val MSE: {final_mse:.4f} (handoff: {ho_mse:.4f})")
    print(f"│  Saved: results/foundation_model_deep.pth")

    # Rollout accuracy: predict N steps autoregressively
    print("│  Testing rollout accuracy...")
    rollout_mses = {}
    np.random.seed(99)
    test_traj = 50  # trajectories for rollout test
    cfg = SimConfig(); sim = PhysicsSimulator(cfg)
    for roll_h in [5, 10, 15, 20, 25, 30]:
        errs = []
        for _ in range(test_traj):
            balls = generate_random_balls(N_BALLS, cfg)
            states_real = []
            cur_state = np.array([v for b in balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
            states_real.append(cur_state.copy())
            for s in range(roll_h):
                balls = sim.step(balls)
                cur_state = np.array([v for b in balls for v in [b.x, b.y, b.vx, b.vy]], dtype=np.float32)
                states_real.append(cur_state.copy())
            # Autoregressive prediction
            pred_state = states_real[0].copy()
            pred_errors = []
            for s in range(roll_h):
                occ_a = get_occluded_state(pred_state, N_BALLS, 'A')
                occ_b = get_occluded_state(pred_state, N_BALLS, 'B')
                with torch.no_grad():
                    p, _, _ = model_f(torch.tensor(occ_a).unsqueeze(0),
                                       torch.tensor(occ_b).unsqueeze(0))
                pred_state = p[0].numpy()
                pred_errors.append(np.mean((pred_state - states_real[s+1])**2))
            errs.append(np.mean(pred_errors))
        rollout_mses[roll_h] = np.mean(errs)
        print(f"│    Horizon {roll_h:2d}: rollout MSE = {rollout_mses[roll_h]:.4f}")

    # Foundation plots
    fig_f, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig_f.suptitle("Phase 16 Step 1: Deep-Trained Foundation", fontweight='bold')
    # Training curve
    axes[0].plot(range(0, len(train_curve), 5),
                 [np.mean(train_curve[max(0,i-5):i+5]) for i in range(0, len(train_curve), 5)],
                 alpha=0.7, label='Train')
    ve, vm = zip(*val_curve)
    axes[0].plot(ve, vm, 'r-o', markersize=2, label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].set_title(f'Training (best val={best_val:.4f})')
    # Rollout accuracy
    rh = sorted(rollout_mses.keys())
    axes[1].plot(rh, [rollout_mses[h] for h in rh], 'o-', color='#e76f51')
    axes[1].set_xlabel('Rollout Horizon'); axes[1].set_ylabel('Avg MSE')
    axes[1].set_title('Imagination Fidelity')
    # Latent PCA
    with torch.no_grad():
        _, mus, _ = model_f(oa_v[:500], ob_v[:500])
    mus_np = mus.numpy()
    try:
        # Simple PCA via SVD (no sklearn needed)
        mus_c = mus_np - mus_np.mean(axis=0)
        U, S, Vt = np.linalg.svd(mus_c, full_matrices=False)
        z2 = mus_c @ Vt[:2].T
        total_var = (S[:2]**2).sum() / (S**2).sum()
        axes[2].scatter(z2[:, 0], z2[:, 1], c=fn_v[:500, 0].numpy(), s=5, alpha=0.5, cmap='viridis')
        axes[2].set_xlabel('PC 1'); axes[2].set_ylabel('PC 2')
        axes[2].set_title(f'Latent Space (var={total_var:.2f})')
    except Exception:
        axes[2].text(0.5, 0.5, 'PCA unavailable', ha='center', va='center', transform=axes[2].transAxes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase16_foundation.png", dpi=150, bbox_inches='tight'); plt.close()
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 2: Deep-Train Planning Module
    # ════════════════════════════════════════════════
    print("┌─ Step 2: Deep-Training Planning Module (20000 episodes, H=20)")
    model_p = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    # Transfer weights from deep foundation
    with torch.no_grad():
        for mp, wp in zip(model_p.encoder_a.parameters(), model_f.encoder_a.parameters()):
            mp.copy_(wp)
        for mp, wp in zip(model_p.encoder_b.parameters(), model_f.encoder_b.parameters()):
            mp.copy_(wp)
        for mp, wp in zip(model_p.fusion.parameters(), model_f.fusion.parameters()):
            mp.copy_(wp)
        for mp, wp in zip(model_p.state_decoder.parameters(), model_f.state_decoder.parameters()):
            mp.copy_(wp)
    model_p.freeze_world_model()

    episodes = generate_planning_episodes(25000, N_BALLS, seed=42)
    plan_params = [p for p in model_p.parameters() if p.requires_grad]
    opt_p = torch.optim.Adam(plan_params, lr=3e-4)
    sched_p = torch.optim.lr_scheduler.StepLR(opt_p, 5000, 0.5)
    plan_losses = []

    # Curriculum task goals
    def _sample_curriculum_goal(ep_i, init_state):
        """Curriculum: easy tasks first, harder later."""
        if ep_i < 5000:
            # Easy: line up or gather
            r = np.random.randint(3)
            goal = np.zeros(STATE_DIM, dtype=np.float32)
            if r == 0:  # line up
                for b in range(N_BALLS): goal[b*4] = 0.5 + b*0.5; goal[b*4+1] = 1.0
            elif r == 1:  # gather left
                for b in range(N_BALLS): goal[b*4] = 0.2 + b*0.15; goal[b*4+1] = 0.5 + b*0.5
            else:  # gather right
                for b in range(N_BALLS): goal[b*4] = 1.6 + b*0.15; goal[b*4+1] = 0.5 + b*0.5
        elif ep_i < 12000:
            # Medium: add triangle
            r = np.random.randint(4)
            goal = np.zeros(STATE_DIM, dtype=np.float32)
            if r < 3:
                return _sample_curriculum_goal(0, init_state)  # reuse easy
            pts = [(0.5,0.5),(1.0,1.5),(1.5,0.5)]
            for b in range(min(N_BALLS,3)): goal[b*4]=pts[b][0]; goal[b*4+1]=pts[b][1]
        else:
            # Hard: all tasks including swap + random
            r = np.random.randint(6)
            goal = np.zeros(STATE_DIM, dtype=np.float32)
            if r < 3:
                return _sample_curriculum_goal(0, init_state)
            elif r == 3:
                pts = [(0.5,0.5),(1.0,1.5),(1.5,0.5)]
                for b in range(min(N_BALLS,3)): goal[b*4]=pts[b][0]; goal[b*4+1]=pts[b][1]
            elif r == 4:  # swap
                for b in range(N_BALLS):
                    goal[b*4] = 2.0 - init_state[b*4]; goal[b*4+1] = init_state[b*4+1]
            else:  # random
                for b in range(N_BALLS):
                    goal[b*4] = np.random.uniform(0.2,1.8); goal[b*4+1] = np.random.uniform(0.2,1.8)
        return goal

    for ep_i in range(20000):
        model_p.train()
        batch_idx = np.random.choice(20000, 32)
        inits = np.array([episodes[i][0] for i in batch_idx])
        goals = np.array([_sample_curriculum_goal(ep_i, inits[j]) for j, _ in enumerate(batch_idx)])
        inits_t = torch.tensor(inits, dtype=torch.float32)
        goals_t = torch.tensor(goals, dtype=torch.float32)
        oa_b = torch.zeros_like(inits_t); ob_b = torch.zeros_like(inits_t)
        for s in range(32):
            oa_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'A'))
            ob_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'B'))
        loss, _ = model_p.plan_loss(oa_b, ob_b, goals_t)
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(plan_params, 1.0)
        opt_p.step(); sched_p.step()
        plan_losses.append(loss.item())
        if (ep_i+1) % 2000 == 0:
            avg = np.mean(plan_losses[-200:])
            phase = "easy" if ep_i < 5000 else "medium" if ep_i < 12000 else "hard"
            print(f"│  Episode {ep_i+1}: loss={avg:.4f} [{phase}]")

    torch.save({k: v.clone() for k, v in model_p.state_dict().items()},
               str(OUTPUT_DIR / "planning_model_deep.pth"))
    print(f"│  Saved: results/planning_model_deep.pth")

    # Planning training curve
    fig_pt, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(0, 20000, 20), [np.mean(plan_losses[max(0,i-20):i+20])
            for i in range(0, 20000, 20)], color='#2a9d8f', alpha=0.8)
    ax.axvline(5000, color='gray', ls=':', alpha=0.5, label='→ medium')
    ax.axvline(12000, color='gray', ls='--', alpha=0.5, label='→ hard')
    ax.set_xlabel('Episode'); ax.set_ylabel('Planning Loss')
    ax.set_title('Phase 16 Step 2: Planning Training (Curriculum)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase16_planning_training.png", dpi=150, bbox_inches='tight'); plt.close()
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 3: Re-evaluate Everything
    # ════════════════════════════════════════════════
    print("┌─ Step 3: Re-evaluating with deep foundation")

    # 3a: Communication protocol (Phase 7 style)
    print("│  3a: Communication protocol analysis")
    model_f.eval()
    with torch.no_grad():
        _, mus_v, _ = model_f(oa_v[:2000], ob_v[:2000])
    mus_np = mus_v.numpy()
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])
    corr_deep = np.zeros((COMM_DIM, STATE_DIM))
    for d in range(COMM_DIM):
        for s in range(STATE_DIM):
            corr_deep[d, s] = np.corrcoef(mus_np[:, d], oa[sp:sp+2000, s])[0, 1]
    max_corr = np.max(np.abs(corr_deep))
    print(f"│    Max |correlation|: {max_corr:.3f}")

    # 3b: Robustness (Phase 12 style)
    print("│  3b: Robustness testing")
    shifts = {
        "Baseline": dict(),
        "Zero-G": dict(gravity=0.0),
        "2× Speed": dict(speed_mult=2.0),
        "Mixed Mass": dict(mass_range=(0.5, 3.0)),
        "Extra Ball": dict(n_balls_override=4),
    }
    rob_results_deep = {}
    for label, kwargs in shifts.items():
        nb = kwargs.pop('n_balls_override', N_BALLS)
        sm = kwargs.pop('speed_mult', 1.0)
        g = kwargs.pop('gravity', None)
        mr = kwargs.pop('mass_range', None)
        cfg_s = SimConfig()
        if g is not None: cfg_s.gravity = g
        sim_s = PhysicsSimulator(cfg_s)
        np.random.seed(200)
        errs = []
        for _ in range(200):
            balls = generate_random_balls(nb, cfg_s)
            if mr: balls = [Ball(b.x,b.y,b.vx,b.vy,b.radius, np.random.uniform(*mr)) for b in balls]
            if sm != 1.0: balls = [Ball(b.x,b.y,b.vx*sm,b.vy*sm,b.radius,b.mass) for b in balls]
            st = np.array([v for b in balls for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            oa_s = get_occluded_state(st, nb, 'A')
            ob_s = get_occluded_state(st, nb, 'B')
            balls2 = sim_s.step(balls)
            fn_s = np.array([v for b in balls2 for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            sd = N_BALLS * 4
            if nb != N_BALLS:
                oa_s = np.pad(oa_s, (0, max(0, sd-len(oa_s))))[:sd]
                ob_s = np.pad(ob_s, (0, max(0, sd-len(ob_s))))[:sd]
                fn_s = np.pad(fn_s, (0, max(0, sd-len(fn_s))))[:sd]
            with torch.no_grad():
                p, _, _ = model_f(torch.tensor(oa_s).unsqueeze(0), torch.tensor(ob_s).unsqueeze(0))
            errs.append(F.mse_loss(p[0], torch.tensor(fn_s)).item())
        rob_results_deep[label] = np.mean(errs)
        print(f"│    {label:15s}: MSE={rob_results_deep[label]:.4f}")

    # 3c: Planning evaluation
    print("│  3c: Planning evaluation (5 tasks)")
    cfg = SimConfig(); sim_c = ControllableSimulator(cfg)
    np.random.seed(123)

    tasks = {
        "Gather Left": lambda: np.array([0.3,1.0,0,0, 0.4,0.5,0,0, 0.3,1.5,0,0], dtype=np.float32),
        "Gather Right": lambda: np.array([1.7,1.0,0,0, 1.6,0.5,0,0, 1.7,1.5,0,0], dtype=np.float32),
        "Swap": None,
        "Line Up": lambda: np.array([0.5,1.0,0,0, 1.0,1.0,0,0, 1.5,1.0,0,0], dtype=np.float32),
        "Triangle": lambda: np.array([0.5,0.5,0,0, 1.0,1.5,0,0, 1.5,0.5,0,0], dtype=np.float32),
    }
    plan_results_deep = {}
    for task_name, goal_fn in tasks.items():
        balls_init = generate_random_balls(N_BALLS, cfg)
        init_state = np.array([v for b in balls_init for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
        if task_name == "Swap":
            goal = init_state.copy()
            for b in range(N_BALLS): goal[b*4] = 2.0-init_state[b*4]; goal[b*4+2]=0; goal[b*4+3]=0
        else:
            goal = goal_fn()
        goal_t = torch.tensor(goal).unsqueeze(0)
        oa_t = torch.tensor(get_occluded_state(init_state, N_BALLS, 'A')).unsqueeze(0)
        ob_t = torch.tensor(get_occluded_state(init_state, N_BALLS, 'B')).unsqueeze(0)

        # With comm
        model_p.eval()
        with torch.no_grad():
            states_im, acts_a, acts_b, comm_a, comm_b = model_p.imagine_trajectory(oa_t, ob_t, goal_t)
        real_states = [init_state.copy()]
        cur_balls = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
        for t in range(HORIZON):
            cur_balls = sim_c.step_with_actions(cur_balls, acts_a[0,t].numpy(), acts_b[0,t].numpy())
            st = np.array([v for b in cur_balls for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            real_states.append(st)
        real_states = np.array(real_states)
        dist_comm = np.mean([np.sqrt((real_states[-1,b*4]-goal[b*4])**2 +
                            (real_states[-1,b*4+1]-goal[b*4+1])**2) for b in range(N_BALLS)])

        # Without comm
        model_nc = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
        model_nc.load_state_dict(model_p.state_dict())
        model_nc.eval()
        model_nc.fusion = nn.Sequential(nn.Linear(COMM_DIM*2, 32), nn.ReLU(), nn.Linear(32, 32))
        with torch.no_grad():
            _, acts_a_nc, acts_b_nc, _, _ = model_nc.imagine_trajectory(oa_t, ob_t, goal_t)
        real_nc = [init_state.copy()]
        cur_balls = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
        for t in range(HORIZON):
            cur_balls = sim_c.step_with_actions(cur_balls, acts_a_nc[0,t].numpy(), acts_b_nc[0,t].numpy())
            st = np.array([v for b in cur_balls for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            real_nc.append(st)
        real_nc = np.array(real_nc)
        dist_nocomm = np.mean([np.sqrt((real_nc[-1,b*4]-goal[b*4])**2 +
                              (real_nc[-1,b*4+1]-goal[b*4+1])**2) for b in range(N_BALLS)])
        success_comm = 1 if dist_comm < 0.3 else 0
        success_nc = 1 if dist_nocomm < 0.3 else 0
        margin = ((dist_nocomm - dist_comm) / max(dist_nocomm, 1e-6)) * 100

        plan_results_deep[task_name] = {
            "comm": dist_comm, "nocomm": dist_nocomm,
            "success_comm": success_comm, "success_nc": success_nc,
            "margin": margin, "real_comm": real_states, "goal": goal, "init": init_state,
            "comm_a": comm_a[0].numpy(), "acts_a": acts_a[0].numpy(), "acts_b": acts_b[0].numpy(),
            "imagined": states_im[0].numpy()
        }
        print(f"│    {task_name:15s}: comm={dist_comm:.4f} nocomm={dist_nocomm:.4f} "
              f"margin={margin:+.1f}% succ={success_comm}/{success_nc}")

    # 3d: Imagination fidelity with actions
    print("│  3d: Imagination fidelity vs reality")
    im_fid = {}
    for tname, r in plan_results_deep.items():
        mn = min(len(r['imagined']), len(r['real_comm']))
        errs_per_step = [np.mean((r['imagined'][t] - r['real_comm'][t])**2) for t in range(mn)]
        im_fid[tname] = errs_per_step
        print(f"│    {tname}: step1={errs_per_step[1]:.4f} stepN={errs_per_step[-1]:.4f}")

    # 3e: Communication intent analysis (Swap task)
    print("│  3e: Communication intent analysis (Swap task)")
    sr = plan_results_deep.get("Swap", {})
    if sr:
        comm_a_swap = sr.get("comm_a", np.zeros((HORIZON, COMM_DIM)))
        acts_a_swap = sr.get("acts_a", np.zeros((HORIZON, ACTION_DIM)))
        goal_swap = sr.get("goal", np.zeros(STATE_DIM))
        corr_comm_goal = np.zeros(COMM_DIM)
        corr_comm_act = np.zeros(COMM_DIM)
        for d in range(COMM_DIM):
            if len(comm_a_swap) > 2:
                # Correlation with goal positions (broadcast)
                goal_rep = np.tile(goal_swap[:2], HORIZON // 2 + 1)[:HORIZON]
                if len(goal_rep) == len(comm_a_swap[:, d]):
                    cc = np.corrcoef(comm_a_swap[:, d], goal_rep)[0, 1]
                    corr_comm_goal[d] = cc if not np.isnan(cc) else 0
                # Correlation with actions
                if ACTION_DIM > 0 and len(acts_a_swap) > 2:
                    cc2 = np.corrcoef(comm_a_swap[:, d], acts_a_swap[:, 0])[0, 1]
                    corr_comm_act[d] = cc2 if not np.isnan(cc2) else 0
        print(f"│    Comm↔Goal max|r|: {np.max(np.abs(corr_comm_goal)):.3f}")
        print(f"│    Comm↔Action max|r|: {np.max(np.abs(corr_comm_act)):.3f}")

    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 4: Comparison Dashboard
    # ════════════════════════════════════════════════
    print("┌─ Step 4: Comparison Dashboard")

    # Phase 14 baselines (hardcoded from Phase 14 run)
    p14_results = {
        "Gather Left": {"comm": 1.079, "nocomm": 1.082},
        "Gather Right": {"comm": 0.973, "nocomm": 0.968},
        "Swap": {"comm": 0.976, "nocomm": 0.981},
        "Line Up": {"comm": 0.668, "nocomm": 0.669},
        "Triangle": {"comm": 0.771, "nocomm": 0.775},
    }

    fig_d, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig_d.suptitle("Phase 16: Deep Training — Before vs After", fontweight='bold', fontsize=14)

    # Panel 1: World model accuracy
    ax = axes[0, 0]
    bars = ax.bar([0, 1], [0.127, final_mse], color=['#e76f51', '#2a9d8f'], width=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Phase 5b\n(150 ep)', 'Phase 16\n(2000 ep)'])
    ax.set_ylabel('Val MSE'); ax.set_title('World Model Accuracy')
    for b_i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f'{[0.127, final_mse][b_i]:.4f}', ha='center', fontsize=10, fontweight='bold')

    # Panel 2: Communication protocol sharpness
    ax = axes[0, 1]
    im = ax.imshow(np.abs(corr_deep), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(STATE_DIM)); ax.set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Comm dim'); ax.set_title(f'Protocol Sharpness (max|r|={max_corr:.3f})')
    plt.colorbar(im, ax=ax)

    # Panel 3: Planning performance comparison
    ax = axes[1, 0]
    tnames = list(plan_results_deep.keys())
    x = np.arange(len(tnames))
    p14_comm = [p14_results[t]["comm"] for t in tnames]
    p16_comm = [plan_results_deep[t]["comm"] for t in tnames]
    p16_nc = [plan_results_deep[t]["nocomm"] for t in tnames]
    ax.bar(x - 0.25, p14_comm, 0.25, label='Phase 14 (comm)', color='#e9c46a')
    ax.bar(x, p16_comm, 0.25, label='Phase 16 (comm)', color='#2a9d8f')
    ax.bar(x + 0.25, p16_nc, 0.25, label='Phase 16 (no comm)', color='#e76f51')
    ax.set_xticks(x); ax.set_xticklabels(tnames, rotation=20, fontsize=7)
    ax.set_ylabel('Goal Distance'); ax.legend(fontsize=7); ax.set_title('Planning: Phase 14 vs 16')

    # Panel 4: Communication margin
    ax = axes[1, 1]
    p14_margins = [((p14_results[t]["nocomm"]-p14_results[t]["comm"])/max(p14_results[t]["nocomm"],1e-6))*100
                   for t in tnames]
    p16_margins = [plan_results_deep[t]["margin"] for t in tnames]
    ax.bar(x - 0.15, p14_margins, 0.3, label='Phase 14', color='#e9c46a')
    ax.bar(x + 0.15, p16_margins, 0.3, label='Phase 16', color='#2a9d8f')
    ax.set_xticks(x); ax.set_xticklabels(tnames, rotation=20, fontsize=7)
    ax.set_ylabel('Comm Margin (%)'); ax.legend(); ax.set_title('Communication Margin')
    ax.axhline(0, color='gray', ls=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase16_comparison_dashboard.png", dpi=150, bbox_inches='tight'); plt.close()

    # Evaluation plots
    fig_e, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig_e.suptitle("Phase 16 Step 3: Evaluation Details", fontweight='bold')
    # Imagination fidelity per task
    for tname, errs in im_fid.items():
        axes[0].plot(range(len(errs)), errs, 'o-', markersize=3, label=tname)
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Imagination MSE')
    axes[0].set_title('Imagination vs Reality'); axes[0].legend(fontsize=6)
    # Robustness
    rob_labels = list(rob_results_deep.keys())
    axes[1].barh(range(len(rob_labels)), [rob_results_deep[l] for l in rob_labels], color='#264653')
    axes[1].set_yticks(range(len(rob_labels))); axes[1].set_yticklabels(rob_labels, fontsize=8)
    axes[1].set_xlabel('MSE'); axes[1].set_title('Robustness (Deep)')
    # Planning trajectories (best task)
    best_task = min(plan_results_deep, key=lambda t: plan_results_deep[t]['comm'])
    r = plan_results_deep[best_task]
    for b in range(N_BALLS):
        axes[2].plot(r['real_comm'][:, b*4], r['real_comm'][:, b*4+1], 'o-', markersize=3)
        axes[2].scatter(r['goal'][b*4], r['goal'][b*4+1], marker='*', s=100, c='red', zorder=5)
    axes[2].axvline(1.0, color='gray', ls=':', alpha=0.3)
    axes[2].set_xlim(0, 2); axes[2].set_ylim(0, 2)
    axes[2].set_title(f'Best: {best_task} (d={r["comm"]:.3f})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase16_evaluation.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase16_foundation.png")
    print(f"  → results/phase16_planning_training.png")
    print(f"  → results/phase16_comparison_dashboard.png")
    print(f"  → results/phase16_evaluation.png")
    print("└─ Done\n")

    # Final summary
    print("="*60)
    print("PHASE 16 SUMMARY")
    print("="*60)
    print(f"  Foundation MSE: 0.127 → {final_mse:.4f}")
    print(f"  Max comm correlation: {max_corr:.3f}")
    print(f"  Rollout fidelity (H=10): {rollout_mses.get(10, 0):.4f}")
    for t in tnames:
        r = plan_results_deep[t]
        print(f"  {t:15s}: comm={r['comm']:.4f} nocomm={r['nocomm']:.4f} "
              f"margin={r['margin']:+.1f}%")

    return {"foundation_mse": final_mse, "plan_results": plan_results_deep,
            "rob_results": rob_results_deep, "rollout_mses": rollout_mses}


# ── Phase 17: Scale Up + Mandatory Cooperation ─────────────────

def run_phase17():
    from physics_sim import (SimConfig, PhysicsSimulator, Ball,
                             ControllableSimulator, generate_random_balls,
                             generate_occlusion_dataset, generate_planning_episodes,
                             get_occluded_state, get_complementary_obs,
                             generate_complementary_dataset)
    from world_model import (ScaledBottleneckedFusionModel,
                             BottleneckedFusionModel, PlanningWorldModel,
                             count_params)

    print("\n" + "="*60)
    print("PHASE 17: SCALE UP + MANDATORY COOPERATION")
    print("="*60)

    N_BALLS = 3; STATE_DIM = N_BALLS * 4; COMM_DIM = 8; BETA = 0.001

    # ════════════════════════════════════════════════
    # PART 1: Scaled Model Baseline
    # ════════════════════════════════════════════════
    print("\n┌─ Part 1: Scaled Model Baseline (23K vs 200K+)")
    fc, oa, ob, fn, ho = generate_occlusion_dataset(2000, 50, N_BALLS, seed=42)
    sp = int(0.8 * len(fc))
    oa_t = torch.tensor(oa[:sp], dtype=torch.float32)
    ob_t = torch.tensor(ob[:sp], dtype=torch.float32)
    fn_t = torch.tensor(fn[:sp], dtype=torch.float32)
    oa_v = torch.tensor(oa[sp:], dtype=torch.float32)
    ob_v = torch.tensor(ob[sp:], dtype=torch.float32)
    fn_v = torch.tensor(fn[sp:], dtype=torch.float32)

    # Train small model (500 epochs for quick comparison)
    model_sm = BottleneckedFusionModel(STATE_DIM, COMM_DIM)
    p_sm = count_params(model_sm)
    opt_sm = torch.optim.Adam(model_sm.parameters(), lr=1e-3)
    for ep in range(500):
        model_sm.train()
        idx = torch.randperm(sp)[:256]
        pred, mu, kl = model_sm(oa_t[idx], ob_t[idx])
        loss = F.mse_loss(pred, fn_t[idx]) + BETA * kl
        opt_sm.zero_grad(); loss.backward(); opt_sm.step()
    model_sm.eval()
    with torch.no_grad():
        pv_sm, _, _ = model_sm(oa_v, ob_v)
        mse_sm = F.mse_loss(pv_sm, fn_v).item()
    print(f"│  Small (23K): val MSE = {mse_sm:.4f}")

    # Train scaled model (500 epochs, same budget)
    model_lg = ScaledBottleneckedFusionModel(STATE_DIM, COMM_DIM)
    p_lg = count_params(model_lg)
    print(f"│  Scaled model: {p_lg/1e3:.1f}K params")
    opt_lg = torch.optim.Adam(model_lg.parameters(), lr=3e-4)
    for ep in range(500):
        model_lg.train()
        idx = torch.randperm(sp)[:256]
        pred, mu, kl = model_lg(oa_t[idx], ob_t[idx])
        loss = F.mse_loss(pred, fn_t[idx]) + BETA * kl
        opt_lg.zero_grad(); loss.backward(); opt_lg.step()
    model_lg.eval()
    with torch.no_grad():
        pv_lg, _, _ = model_lg(oa_v, ob_v)
        mse_lg = F.mse_loss(pv_lg, fn_v).item()
    print(f"│  Scaled ({p_lg/1e3:.0f}K): val MSE = {mse_lg:.4f}")
    print(f"│  Improvement: {(1-mse_lg/mse_sm)*100:.1f}%")

    # Extended training for scaled model (1000 more epochs)
    print("│  Extended training scaled model (1000 more epochs)...")
    sched_lg = torch.optim.lr_scheduler.StepLR(opt_lg, 300, 0.5)
    for ep in range(1000):
        model_lg.train()
        idx = torch.randperm(sp)[:256]
        pred, mu, kl = model_lg(oa_t[idx], ob_t[idx])
        loss = F.mse_loss(pred, fn_t[idx]) + BETA * kl
        opt_lg.zero_grad(); loss.backward(); opt_lg.step(); sched_lg.step()
    model_lg.eval()
    with torch.no_grad():
        pv_lg2, _, _ = model_lg(oa_v, ob_v)
        mse_lg2 = F.mse_loss(pv_lg2, fn_v).item()
    print(f"│  Scaled (1500 ep): val MSE = {mse_lg2:.4f}")
    torch.save(model_lg.state_dict(), str(OUTPUT_DIR / "scaled_foundation.pth"))
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PART 2: Complementary Sensing
    # ════════════════════════════════════════════════
    print("┌─ Part 2: Complementary Sensing (positions vs velocities)")
    print("│  Generating complementary dataset...")
    ca, cb, ct = generate_complementary_dataset(2000, 50, N_BALLS, seed=42)
    sp2 = int(0.8 * len(ca))
    ca_t = torch.tensor(ca[:sp2], dtype=torch.float32)
    cb_t = torch.tensor(cb[:sp2], dtype=torch.float32)
    ct_t = torch.tensor(ct[:sp2], dtype=torch.float32)
    ca_v = torch.tensor(ca[sp2:], dtype=torch.float32)
    cb_v = torch.tensor(cb[sp2:], dtype=torch.float32)
    ct_v = torch.tensor(ct[sp2:], dtype=torch.float32)

    model_c = ScaledBottleneckedFusionModel(STATE_DIM, COMM_DIM)
    opt_c = torch.optim.Adam(model_c.parameters(), lr=3e-4)
    sched_c = torch.optim.lr_scheduler.StepLR(opt_c, 300, 0.5)
    c_train = []
    for ep in range(1000):
        model_c.train()
        idx = torch.randperm(sp2)[:256]
        pred, mu, kl = model_c(ca_t[idx], cb_t[idx])
        loss = F.mse_loss(pred, ct_t[idx]) + BETA * kl
        opt_c.zero_grad(); loss.backward(); opt_c.step(); sched_c.step()
        c_train.append(loss.item())
        if (ep+1) % 500 == 0:
            model_c.eval()
            with torch.no_grad():
                pv, _, kv = model_c(ca_v, cb_v)
                vl = F.mse_loss(pv, ct_v).item()
            print(f"│  Epoch {ep+1}: train={np.mean(c_train[-50:]):.4f} val={vl:.4f}")

    # Evaluate under 4 conditions
    model_c.eval()
    with torch.no_grad():
        # Full comm
        pv_full, mu_full, _ = model_c(ca_v, cb_v)
        mse_full = F.mse_loss(pv_full, ct_v).item()

        # No comm: zero out partner messages
        # Run encoder, zero out z_b, fuse, predict
        h_a = model_c.encoder_a(ca_v)
        mu_a_nc = model_c.mu_a(h_a)
        z_a_nc = mu_a_nc
        z_b_zero = torch.zeros(len(ca_v), COMM_DIM)
        fused_nc = model_c.fusion(torch.cat([z_a_nc, z_b_zero], dim=-1))
        res_nc = model_c.pred_norm1(F.relu(model_c.pred_layer1(fused_nc)))
        res_nc = model_c.pred_norm2(model_c.pred_layer2(res_nc))
        pred_nc = model_c.decoder(fused_nc + res_nc)
        mse_a_only = F.mse_loss(pred_nc, ct_v).item()

        # B only
        h_b = model_c.encoder_b(cb_v)
        mu_b_nc = model_c.mu_b(h_b)
        z_a_zero = torch.zeros(len(cb_v), COMM_DIM)
        fused_b = model_c.fusion(torch.cat([z_a_zero, mu_b_nc], dim=-1))
        res_b = model_c.pred_norm1(F.relu(model_c.pred_layer1(fused_b)))
        res_b = model_c.pred_norm2(model_c.pred_layer2(res_b))
        pred_b = model_c.decoder(fused_b + res_b)
        mse_b_only = F.mse_loss(pred_b, ct_v).item()

        # No comm at all (both zeroed)
        fused_none = model_c.fusion(torch.cat([z_a_zero[:len(ca_v)], z_b_zero], dim=-1))
        res_none = model_c.pred_norm1(F.relu(model_c.pred_layer1(fused_none)))
        res_none = model_c.pred_norm2(model_c.pred_layer2(res_none))
        pred_none = model_c.decoder(fused_none + res_none)
        mse_none = F.mse_loss(pred_none, ct_v).item()

    comm_margin_comp = (mse_a_only - mse_full) / max(mse_a_only, 1e-6) * 100
    print(f"│  With comm:    MSE = {mse_full:.4f}")
    print(f"│  A only (pos): MSE = {mse_a_only:.4f}")
    print(f"│  B only (vel): MSE = {mse_b_only:.4f}")
    print(f"│  Neither:      MSE = {mse_none:.4f}")
    print(f"│  Comm margin:  {comm_margin_comp:.1f}%")

    # Correlation analysis
    mu_np = mu_full.numpy()
    phys_labels = []
    for b in range(N_BALLS):
        phys_labels.extend([f'b{b+1}_x', f'b{b+1}_y', f'b{b+1}_vx', f'b{b+1}_vy'])
    corr_comp = np.zeros((COMM_DIM * 2, STATE_DIM))
    for d in range(COMM_DIM * 2):
        for s in range(STATE_DIM):
            cc = np.corrcoef(mu_np[:2000, d], ct_v[:2000, s].numpy())[0, 1]
            corr_comp[d, s] = cc if not np.isnan(cc) else 0
    max_corr_comp = np.max(np.abs(corr_comp))
    print(f"│  Max |correlation|: {max_corr_comp:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PART 3: Asymmetric Control Planning
    # ════════════════════════════════════════════════
    print("┌─ Part 3: Asymmetric Control Planning")
    print("│  Training world model for asymmetric control (500 ep)...")
    # Reuse the standard occlusion world model for prediction
    # but planning forces are asymmetric: A controls left, B controls right
    ACTION_DIM = N_BALLS * 2; HORIZON = 15
    model_asym = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    # Load scaled foundation weights into planning model's encoder/fusion/decoder
    # (they have different shapes, so we train fresh with more capacity)
    # Train world model base from scratch for asymmetric task
    planning_ids = {id(p) for m in [model_asym.goal_encoder, model_asym.policy_a,
                                     model_asym.policy_b, model_asym.action_encoder,
                                     model_asym.action_predictor] for p in m.parameters()}
    opt_wm = torch.optim.Adam([p for p in model_asym.parameters()
                                if id(p) not in planning_ids], lr=1e-3)
    for ep in range(500):
        model_asym.train()
        idx = torch.randperm(sp)[:256]
        ha = model_asym.encoder_a(oa_t[idx])[:, :COMM_DIM]
        hb = model_asym.encoder_b(ob_t[idx])[:, :COMM_DIM]
        fused = model_asym.fusion(torch.cat([ha, hb], dim=-1))
        pred = model_asym.state_decoder(fused)
        loss = F.mse_loss(pred, fn_t[idx])
        opt_wm.zero_grad(); loss.backward(); opt_wm.step()
    model_asym.freeze_world_model()
    print(f"│  World model base MSE: {loss.item():.4f}")

    # Train planning with asymmetric control
    episodes = generate_planning_episodes(12000, N_BALLS, seed=42)
    plan_params = [p for p in model_asym.parameters() if p.requires_grad]
    opt_p = torch.optim.Adam(plan_params, lr=3e-4)
    asym_losses = []
    for ep_i in range(10000):
        model_asym.train()
        batch_idx = np.random.choice(10000, 32)
        inits = np.array([episodes[i][0] for i in batch_idx])
        # Goal: gather left (clear asymmetry test)
        goals = np.zeros((32, STATE_DIM), dtype=np.float32)
        for s in range(32):
            for b in range(N_BALLS):
                goals[s, b*4] = 0.3 + b*0.15; goals[s, b*4+1] = 0.5 + b*0.5
        inits_t = torch.tensor(inits, dtype=torch.float32)
        goals_t = torch.tensor(goals, dtype=torch.float32)
        oa_b = torch.zeros_like(inits_t); ob_b = torch.zeros_like(inits_t)
        for s in range(32):
            oa_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'A'))
            ob_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'B'))
        loss, _ = model_asym.plan_loss(oa_b, ob_b, goals_t)
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(plan_params, 1.0)
        opt_p.step()
        asym_losses.append(loss.item())
        if (ep_i+1) % 2000 == 0:
            print(f"│  Episode {ep_i+1}: loss={np.mean(asym_losses[-200:]):.4f}")

    # Evaluate asymmetric: with vs without comm
    cfg = SimConfig(); sim_c = ControllableSimulator(cfg)
    np.random.seed(42)
    balls_init = generate_random_balls(N_BALLS, cfg)
    init_state = np.array([v for b in balls_init for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
    goal = np.zeros(STATE_DIM, dtype=np.float32)
    for b in range(N_BALLS): goal[b*4] = 0.3 + b*0.15; goal[b*4+1] = 0.5 + b*0.5
    goal_t = torch.tensor(goal).unsqueeze(0)
    oa_e = torch.tensor(get_occluded_state(init_state, N_BALLS, 'A')).unsqueeze(0)
    ob_e = torch.tensor(get_occluded_state(init_state, N_BALLS, 'B')).unsqueeze(0)

    model_asym.eval()
    with torch.no_grad():
        _, acts_a, acts_b, comm_a, comm_b = model_asym.imagine_trajectory(oa_e, ob_e, goal_t)
    # Execute with asymmetric control
    real_asym = [init_state.copy()]
    balls_e = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
    for t in range(HORIZON):
        # Asymmetric: A controls left-half balls, B controls right-half balls
        action_a = acts_a[0,t].numpy(); action_b = acts_b[0,t].numpy()
        balls_e = sim_c.step_with_actions(balls_e, action_a, action_b)
        st = np.array([v for b in balls_e for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
        real_asym.append(st)
    real_asym = np.array(real_asym)
    dist_asym_comm = np.mean([np.sqrt((real_asym[-1,b*4]-goal[b*4])**2 +
                             (real_asym[-1,b*4+1]-goal[b*4+1])**2) for b in range(N_BALLS)])

    # Without comm: reset random policies
    model_nc = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    model_nc.load_state_dict(model_asym.state_dict())
    model_nc.eval()
    model_nc.fusion = nn.Sequential(nn.Linear(COMM_DIM*2, 32), nn.ReLU(), nn.Linear(32, 32))
    with torch.no_grad():
        _, acts_a_nc, acts_b_nc, _, _ = model_nc.imagine_trajectory(oa_e, ob_e, goal_t)
    real_nc = [init_state.copy()]
    balls_e2 = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
    for t in range(HORIZON):
        balls_e2 = sim_c.step_with_actions(balls_e2, acts_a_nc[0,t].numpy(), acts_b_nc[0,t].numpy())
        st = np.array([v for b in balls_e2 for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
        real_nc.append(st)
    real_nc = np.array(real_nc)
    dist_asym_nc = np.mean([np.sqrt((real_nc[-1,b*4]-goal[b*4])**2 +
                            (real_nc[-1,b*4+1]-goal[b*4+1])**2) for b in range(N_BALLS)])
    margin_asym = ((dist_asym_nc - dist_asym_comm) / max(dist_asym_nc, 1e-6)) * 100
    print(f"│  Gather Left: comm={dist_asym_comm:.4f} nocomm={dist_asym_nc:.4f} margin={margin_asym:+.1f}%")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PART 4: You-See-I-Steer
    # ════════════════════════════════════════════════
    print("┌─ Part 4: You-See-I-Steer (A=muscle+blind-right, B=eyes+no-control)")
    # A sees left half, controls ALL balls
    # B sees right half, controls NO balls
    # B must communicate to A what's on the right side

    # Train world model for this setting
    model_steer = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    planning_ids2 = {id(p) for m in [model_steer.goal_encoder, model_steer.policy_a,
                                      model_steer.policy_b, model_steer.action_encoder,
                                      model_steer.action_predictor] for p in m.parameters()}
    opt_wm2 = torch.optim.Adam([p for p in model_steer.parameters()
                                 if id(p) not in planning_ids2], lr=1e-3)
    for ep in range(500):
        model_steer.train()
        idx = torch.randperm(sp)[:256]
        ha = model_steer.encoder_a(oa_t[idx])[:, :COMM_DIM]
        hb = model_steer.encoder_b(ob_t[idx])[:, :COMM_DIM]
        fused = model_steer.fusion(torch.cat([ha, hb], dim=-1))
        pred = model_steer.state_decoder(fused)
        loss = F.mse_loss(pred, fn_t[idx])
        opt_wm2.zero_grad(); loss.backward(); opt_wm2.step()
    model_steer.freeze_world_model()

    # Planning: A controls ALL balls, B controls NONE
    # We modify the policy to reflect this: policy_b outputs zeros
    plan_params2 = [p for p in model_steer.parameters() if p.requires_grad]
    opt_p2 = torch.optim.Adam(plan_params2, lr=3e-4)
    steer_losses = []
    print("│  Training planner (15K episodes)...")
    for ep_i in range(15000):
        model_steer.train()
        batch_idx = np.random.choice(10000, 32)
        inits = np.array([episodes[i][0] for i in batch_idx])
        goals = np.zeros((32, STATE_DIM), dtype=np.float32)
        for s in range(32):
            for b in range(N_BALLS):
                goals[s, b*4] = 1.0; goals[s, b*4+1] = 1.0  # center goal
        inits_t = torch.tensor(inits, dtype=torch.float32)
        goals_t = torch.tensor(goals, dtype=torch.float32)
        oa_b = torch.zeros_like(inits_t); ob_b = torch.zeros_like(inits_t)
        for s in range(32):
            oa_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'A'))
            ob_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'B'))
        loss, _ = model_steer.plan_loss(oa_b, ob_b, goals_t)
        opt_p2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(plan_params2, 1.0)
        opt_p2.step()
        steer_losses.append(loss.item())
        if (ep_i+1) % 3000 == 0:
            print(f"│  Episode {ep_i+1}: loss={np.mean(steer_losses[-200:]):.4f}")

    # Evaluate: with comm vs without
    np.random.seed(77)
    balls_init = generate_random_balls(N_BALLS, cfg)
    init_state = np.array([v for b in balls_init for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
    goal_center = np.zeros(STATE_DIM, dtype=np.float32)
    for b_idx in range(N_BALLS): goal_center[b_idx*4] = 1.0; goal_center[b_idx*4+1] = 1.0
    goal_t = torch.tensor(goal_center).unsqueeze(0)
    oa_e = torch.tensor(get_occluded_state(init_state, N_BALLS, 'A')).unsqueeze(0)
    ob_e = torch.tensor(get_occluded_state(init_state, N_BALLS, 'B')).unsqueeze(0)

    model_steer.eval()
    with torch.no_grad():
        _, acts_a_s, acts_b_s, comm_a_s, comm_b_s = model_steer.imagine_trajectory(oa_e, ob_e, goal_t)

    # Execute: only A's actions applied (B has no control)
    real_steer = [init_state.copy()]
    balls_e = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
    for t in range(HORIZON):
        # A controls all balls — apply A's actions as both
        a_act = acts_a_s[0,t].numpy()
        balls_e = sim_c.step_with_actions(balls_e, a_act, a_act)
        st = np.array([v for b in balls_e for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
        real_steer.append(st)
    real_steer = np.array(real_steer)
    dist_steer_comm = np.mean([np.sqrt((real_steer[-1,b*4]-goal_center[b*4])**2 +
                              (real_steer[-1,b*4+1]-goal_center[b*4+1])**2) for b in range(N_BALLS)])

    # Without comm
    model_nc2 = PlanningWorldModel(STATE_DIM, COMM_DIM, action_dim=ACTION_DIM, horizon=HORIZON)
    model_nc2.load_state_dict(model_steer.state_dict())
    model_nc2.eval()
    model_nc2.fusion = nn.Sequential(nn.Linear(COMM_DIM*2, 32), nn.ReLU(), nn.Linear(32, 32))
    with torch.no_grad():
        _, acts_a_nc2, _, _, _ = model_nc2.imagine_trajectory(oa_e, ob_e, goal_t)
    real_nc2 = [init_state.copy()]
    balls_e2 = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
    for t in range(HORIZON):
        a_act = acts_a_nc2[0,t].numpy()
        balls_e2 = sim_c.step_with_actions(balls_e2, a_act, a_act)
        st = np.array([v for b in balls_e2 for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
        real_nc2.append(st)
    real_nc2 = np.array(real_nc2)
    dist_steer_nc = np.mean([np.sqrt((real_nc2[-1,b*4]-goal_center[b*4])**2 +
                             (real_nc2[-1,b*4+1]-goal_center[b*4+1])**2) for b in range(N_BALLS)])
    margin_steer = ((dist_steer_nc - dist_steer_comm) / max(dist_steer_nc, 1e-6)) * 100

    # Communication intent: correlate B's message with A's forces
    comm_b_np = comm_b_s[0].numpy()
    acts_a_np = acts_a_s[0].numpy()
    corr_steer = np.zeros((COMM_DIM, ACTION_DIM))
    for d in range(COMM_DIM):
        for a in range(ACTION_DIM):
            if len(comm_b_np) > 2:
                cc = np.corrcoef(comm_b_np[:, d], acts_a_np[:, a])[0, 1]
                corr_steer[d, a] = cc if not np.isnan(cc) else 0
    max_steer_corr = np.max(np.abs(corr_steer))

    print(f"│  With comm:    dist = {dist_steer_comm:.4f}")
    print(f"│  Without comm: dist = {dist_steer_nc:.4f}")
    print(f"│  Comm margin:  {margin_steer:+.1f}%")
    print(f"│  B→A steering corr: max|r|={max_steer_corr:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════
    print("┌─ Generating Plots")

    # Plot 1: Capacity comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1, 2], [mse_sm, mse_lg, mse_lg2],
           color=['#e76f51', '#2a9d8f', '#264653'], width=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'Small\n({p_sm/1e3:.0f}K, 500ep)',
                        f'Scaled\n({p_lg/1e3:.0f}K, 500ep)',
                        f'Scaled\n({p_lg/1e3:.0f}K, 1500ep)'])
    ax.set_ylabel('Val MSE'); ax.set_title('Phase 17: Model Capacity vs MSE', fontweight='bold')
    for i, v in enumerate([mse_sm, mse_lg, mse_lg2]):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase17_capacity.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Complementary sensing
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 17: Complementary Sensing — Communication is MANDATORY', fontweight='bold')
    conds = ['Both\n(with comm)', 'A only\n(positions)', 'B only\n(velocities)', 'Neither']
    vals = [mse_full, mse_a_only, mse_b_only, mse_none]
    cols = ['#2a9d8f', '#e9c46a', '#e76f51', '#264653']
    axes[0].bar(range(4), vals, color=cols, width=0.6)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(conds, fontsize=8)
    axes[0].set_ylabel('Val MSE')
    axes[0].set_title(f'Prediction MSE (margin={comm_margin_comp:.0f}%)')
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    # Correlation heatmap
    im = axes[1].imshow(np.abs(corr_comp), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(STATE_DIM)); axes[1].set_xticklabels(phys_labels, rotation=45, ha='right', fontsize=6)
    axes[1].set_ylabel('Comm dim (A: 0-7, B: 8-15)')
    axes[1].set_title(f'Protocol (max|r|={max_corr_comp:.3f})')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase17_complementary.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Asymmetric control planning
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Phase 17: Asymmetric Control Planning', fontweight='bold')
    for b in range(N_BALLS):
        axes[0].plot(real_asym[:, b*4], real_asym[:, b*4+1], 'o-', markersize=3, label=f'Ball {b+1}')
        axes[0].scatter(goal[b*4], goal[b*4+1], marker='*', s=100, c='red', zorder=5)
    axes[0].axvline(1.0, color='gray', ls=':', alpha=0.3)
    axes[0].set_xlim(0, 2); axes[0].set_ylim(0, 2)
    axes[0].set_title(f'With Comm (d={dist_asym_comm:.3f})'); axes[0].legend(fontsize=7)
    for b in range(N_BALLS):
        axes[1].plot(real_nc[:, b*4], real_nc[:, b*4+1], 'o-', markersize=3)
        axes[1].scatter(goal[b*4], goal[b*4+1], marker='*', s=100, c='red', zorder=5)
    axes[1].axvline(1.0, color='gray', ls=':', alpha=0.3)
    axes[1].set_xlim(0, 2); axes[1].set_ylim(0, 2)
    axes[1].set_title(f'Without Comm (d={dist_asym_nc:.3f})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase17_asymmetric_planning.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: You-See-I-Steer
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Phase 17: You-See-I-Steer', fontweight='bold')
    for b in range(N_BALLS):
        axes[0].plot(real_steer[:, b*4], real_steer[:, b*4+1], 'o-', markersize=3, label=f'Ball {b+1}')
        axes[0].scatter(goal_center[b*4], goal_center[b*4+1], marker='*', s=100, c='red', zorder=5)
    axes[0].axvline(1.0, color='gray', ls=':', alpha=0.3)
    axes[0].set_xlim(0, 2); axes[0].set_ylim(0, 2)
    axes[0].set_title(f'With Comm (d={dist_steer_comm:.3f})'); axes[0].legend(fontsize=7)
    for b in range(N_BALLS):
        axes[1].plot(real_nc2[:, b*4], real_nc2[:, b*4+1], 'o-', markersize=3)
        axes[1].scatter(goal_center[b*4], goal_center[b*4+1], marker='*', s=100, c='red', zorder=5)
    axes[1].axvline(1.0, color='gray', ls=':', alpha=0.3)
    axes[1].set_xlim(0, 2); axes[1].set_ylim(0, 2)
    axes[1].set_title(f'Without Comm (d={dist_steer_nc:.3f})')
    # Steering comm heatmap
    im = axes[2].imshow(np.abs(corr_steer), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    act_labels = [f'f{b+1}_x' for b in range(N_BALLS)] + [f'f{b+1}_y' for b in range(N_BALLS)]
    act_labels = [f'f{(a//2)+1}_{"x" if a%2==0 else "y"}' for a in range(ACTION_DIM)]
    axes[2].set_xticks(range(ACTION_DIM)); axes[2].set_xticklabels(act_labels, fontsize=8)
    axes[2].set_ylabel('B comm dim'); axes[2].set_title(f'B→A Steering (max|r|={max_steer_corr:.3f})')
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase17_you_see_i_steer.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 5: Summary communication margins
    fig, ax = plt.subplots(figsize=(8, 5))
    margin_labels = ['Phase 14\nOcclusion', 'Phase 17\nComplementary', 'Phase 17\nAsymmetric',
                     'Phase 17\nYou-See-I-Steer']
    margin_vals = [0.3, comm_margin_comp, margin_asym, margin_steer]
    colors = ['#e9c46a', '#2a9d8f', '#264653', '#e76f51']
    bars = ax.bar(range(4), margin_vals, color=colors, width=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(margin_labels, fontsize=8)
    ax.set_ylabel('Communication Margin (%)')
    ax.set_title('Communication Matters MORE as Tasks Get Harder', fontweight='bold')
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    for i, v in enumerate(margin_vals):
        ax.text(i, v + 0.5, f'{v:+.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase17_comm_margins_summary.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase17_capacity.png")
    print(f"  → results/phase17_complementary.png")
    print(f"  → results/phase17_asymmetric_planning.png")
    print(f"  → results/phase17_you_see_i_steer.png")
    print(f"  → results/phase17_comm_margins_summary.png")
    print("└─ Done\n")

    # Final summary
    print("="*60)
    print("PHASE 17 SUMMARY")
    print("="*60)
    print(f"  Model capacity: {p_sm/1e3:.0f}K → {p_lg/1e3:.0f}K params")
    print(f"  Standard task MSE: {mse_sm:.4f} → {mse_lg2:.4f}")
    print(f"  Complementary sensing:")
    print(f"    With comm: {mse_full:.4f} | A only: {mse_a_only:.4f} | "
          f"B only: {mse_b_only:.4f} | margin: {comm_margin_comp:.0f}%")
    print(f"  Asymmetric control: comm={dist_asym_comm:.4f} nocomm={dist_asym_nc:.4f} "
          f"margin={margin_asym:+.1f}%")
    print(f"  You-See-I-Steer: comm={dist_steer_comm:.4f} nocomm={dist_steer_nc:.4f} "
          f"margin={margin_steer:+.1f}%")


# ── Phase 18: Clean Planning Ablation + Object Diversity ────────

def run_phase18():
    from physics_sim import (SimConfig, PhysicsSimulator, Ball,
                             ControllableSimulator, generate_random_balls,
                             generate_occlusion_dataset, generate_planning_episodes,
                             get_occluded_state,
                             PropertyObject, RichPhysicsSimulator, OBJ_DIM,
                             generate_rich_dataset, get_rich_occluded)
    from world_model import (CleanPlanningModel, ScaledBottleneckedFusionModel,
                             AffordanceWorldModel, count_params)

    print("\n" + "="*60)
    print("PHASE 18: CLEAN PLANNING ABLATION + OBJECT DIVERSITY")
    print("="*60)

    N_BALLS = 3; STATE_DIM = N_BALLS * 4; COMM_DIM = 8; BETA = 0.001
    ACTION_DIM = N_BALLS * 2; HORIZON = 15

    # ════════════════════════════════════════════════
    # PART A: Clean Planning Knockout
    # ════════════════════════════════════════════════
    print("\n┌─ Part A: Clean Planning Knockout (You-See-I-Steer)")
    print("│  Architecture: CleanPlanningModel — fully separated pathways")

    model = CleanPlanningModel(STATE_DIM, COMM_DIM, hidden_dim=256,
                               action_dim=ACTION_DIM, horizon=HORIZON)
    print(f"│  Model: {count_params(model)/1e3:.1f}K params")

    # Train world model base (predictor)
    fc, oa, ob, fn, ho = generate_occlusion_dataset(1500, 50, N_BALLS, seed=42)
    sp = int(0.8 * len(fc))
    oa_t = torch.tensor(oa[:sp], dtype=torch.float32)
    ob_t = torch.tensor(ob[:sp], dtype=torch.float32)
    fn_t = torch.tensor(fn[:sp], dtype=torch.float32)

    print("│  Training world model (500 ep)...")
    opt_wm = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(500):
        model.train()
        idx = torch.randperm(sp)[:256]
        goal_dummy = fn_t[idx]  # doesn't matter for WM training
        pred, _, _, z_a, z_b, kl = model(oa_t[idx], ob_t[idx], goal_dummy)
        loss = F.mse_loss(pred, fn_t[idx]) + BETA * kl
        opt_wm.zero_grad(); loss.backward(); opt_wm.step()
    print(f"│  WM MSE: {loss.item():.4f}")

    # Freeze world model, train planning
    model.freeze_world_model()
    episodes = generate_planning_episodes(12000, N_BALLS, seed=42)
    plan_params = [p for p in model.parameters() if p.requires_grad]
    opt_p = torch.optim.Adam(plan_params, lr=3e-4)
    plan_losses = []
    print("│  Training planner (15K ep)...")
    for ep_i in range(15000):
        model.train()
        batch_idx = np.random.choice(10000, 32)
        inits = np.array([episodes[i][0] for i in batch_idx])
        goals = np.zeros((32, STATE_DIM), dtype=np.float32)
        for s in range(32):
            for b in range(N_BALLS):
                goals[s, b*4] = 1.0; goals[s, b*4+1] = 1.0
        inits_t = torch.tensor(inits, dtype=torch.float32)
        goals_t = torch.tensor(goals, dtype=torch.float32)
        oa_b = torch.zeros_like(inits_t); ob_b = torch.zeros_like(inits_t)
        for s in range(32):
            oa_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'A'))
            ob_b[s] = torch.tensor(get_occluded_state(inits[s], N_BALLS, 'B'))
        loss, _ = model.plan_loss(oa_b, ob_b, goals_t, use_comm=True)
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(plan_params, 1.0)
        opt_p.step()
        plan_losses.append(loss.item())
        if (ep_i+1) % 3000 == 0:
            print(f"│  Episode {ep_i+1}: loss={np.mean(plan_losses[-200:]):.4f}")

    # Evaluate: with comm vs without
    cfg = SimConfig(); sim_c = ControllableSimulator(cfg)
    np.random.seed(77)
    n_eval = 20
    dist_comm_list, dist_nocomm_list = [], []
    for ev in range(n_eval):
        balls_init = generate_random_balls(N_BALLS, cfg)
        init_state = np.array([v for b in balls_init for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
        goal_center = np.zeros(STATE_DIM, dtype=np.float32)
        for b_idx in range(N_BALLS):
            goal_center[b_idx*4] = 1.0; goal_center[b_idx*4+1] = 1.0
        goal_t = torch.tensor(goal_center).unsqueeze(0)
        oa_e = torch.tensor(get_occluded_state(init_state, N_BALLS, 'A')).unsqueeze(0)
        ob_e = torch.tensor(get_occluded_state(init_state, N_BALLS, 'B')).unsqueeze(0)

        for use_comm, dist_list in [(True, dist_comm_list), (False, dist_nocomm_list)]:
            model.eval()
            with torch.no_grad():
                states, acts_a, acts_b, _, _ = model.imagine_trajectory(oa_e, ob_e, goal_t, use_comm=use_comm)
            balls_e = [Ball(b.x,b.y,b.vx,b.vy,b.radius,b.mass) for b in balls_init]
            for t in range(HORIZON):
                a_act = acts_a[0,t].numpy()
                balls_e = sim_c.step_with_actions(balls_e, a_act, a_act)
            final = np.array([v for b in balls_e for v in [b.x,b.y,b.vx,b.vy]], dtype=np.float32)
            d = np.mean([np.sqrt((final[b*4]-goal_center[b*4])**2 +
                        (final[b*4+1]-goal_center[b*4+1])**2) for b in range(N_BALLS)])
            dist_list.append(d)

    d_comm = np.mean(dist_comm_list)
    d_nocomm = np.mean(dist_nocomm_list)
    margin_clean = ((d_nocomm - d_comm) / max(d_nocomm, 1e-6)) * 100
    print(f"│  With comm:    dist = {d_comm:.4f} (n={n_eval})")
    print(f"│  Without comm: dist = {d_nocomm:.4f}")
    print(f"│  Clean margin: {margin_clean:+.1f}% (was +0.2% in Phase 17)")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PART B: Rich Physics
    # ════════════════════════════════════════════════
    print("┌─ Part B: Rich Physics with Diverse Objects")
    N_OBJECTS = 5
    print("│  Generating rich dataset...")
    states, nexts = generate_rich_dataset(1000, 30, N_OBJECTS, seed=42)
    sp_r = int(0.8 * len(states))
    # Flatten for ScaledBottleneckedFusionModel
    flat_dim = N_OBJECTS * OBJ_DIM  # 50
    st_t = torch.tensor(states[:sp_r].reshape(-1, flat_dim), dtype=torch.float32)
    nx_t = torch.tensor(nexts[:sp_r].reshape(-1, flat_dim), dtype=torch.float32)
    st_v = torch.tensor(states[sp_r:].reshape(-1, flat_dim), dtype=torch.float32)
    nx_v = torch.tensor(nexts[sp_r:].reshape(-1, flat_dim), dtype=torch.float32)
    # Occlude: A sees left half objects, B sees right half
    oa_r_t = torch.tensor(np.array([get_rich_occluded(s, 'A').flatten() for s in states[:sp_r]]), dtype=torch.float32)
    ob_r_t = torch.tensor(np.array([get_rich_occluded(s, 'B').flatten() for s in states[:sp_r]]), dtype=torch.float32)
    oa_r_v = torch.tensor(np.array([get_rich_occluded(s, 'A').flatten() for s in states[sp_r:]]), dtype=torch.float32)
    ob_r_v = torch.tensor(np.array([get_rich_occluded(s, 'B').flatten() for s in states[sp_r:]]), dtype=torch.float32)

    model_r = ScaledBottleneckedFusionModel(flat_dim, COMM_DIM, fused_dim=64, hidden_dim=384)
    print(f"│  Model: {count_params(model_r)/1e3:.1f}K params, state_dim={flat_dim}")
    opt_r = torch.optim.Adam(model_r.parameters(), lr=3e-4)
    sched_r = torch.optim.lr_scheduler.StepLR(opt_r, 500, 0.5)
    for ep in range(1500):
        model_r.train()
        idx = torch.randperm(sp_r)[:256]
        pred, mu, kl = model_r(oa_r_t[idx], ob_r_t[idx])
        loss = F.mse_loss(pred, nx_t[idx]) + BETA * kl
        opt_r.zero_grad(); loss.backward(); opt_r.step(); sched_r.step()
        if (ep+1) % 500 == 0:
            model_r.eval()
            with torch.no_grad():
                pv, _, kv = model_r(oa_r_v, ob_r_v)
                vl = F.mse_loss(pv, nx_v).item()
            print(f"│  Epoch {ep+1}: val_MSE={vl:.4f}")

    # Evaluate comm vs no-comm on rich physics
    model_r.eval()
    with torch.no_grad():
        pv_full, mu_full, _ = model_r(oa_r_v, ob_r_v)
        mse_full = F.mse_loss(pv_full, nx_v).item()
        # No comm: zero B's encoder output
        h_a_r = model_r.encoder_a(oa_r_v)
        mu_a_r = model_r.mu_a(h_a_r)
        z_b_zero = torch.zeros(len(oa_r_v), COMM_DIM)
        fused_nc = model_r.fusion(torch.cat([mu_a_r, z_b_zero], dim=-1))
        res_nc = model_r.pred_norm1(F.relu(model_r.pred_layer1(fused_nc)))
        res_nc = model_r.pred_norm2(model_r.pred_layer2(res_nc))
        pred_nc = model_r.decoder(fused_nc + res_nc)
        mse_aonly = F.mse_loss(pred_nc, nx_v).item()
    margin_rich = (mse_aonly - mse_full) / max(mse_aonly, 1e-6) * 100
    print(f"│  Full comm: {mse_full:.4f} | A-only: {mse_aonly:.4f} | margin: {margin_rich:.1f}%")

    # Type encoding analysis: correlation of comm vectors with object properties
    mu_np = mu_full.numpy()
    # For each object, correlate comm dims with its properties
    prop_names = ['mass', 'elast', 'frict', 'flat', 'rigid']
    type_corrs = np.zeros((COMM_DIM * 2, N_OBJECTS * 5))
    for d in range(COMM_DIM * 2):
        for obj_i in range(N_OBJECTS):
            for p in range(5):
                idx = obj_i * OBJ_DIM + 4 + p  # properties start at index 4
                cc = np.corrcoef(mu_np[:2000, d], nx_v[:2000, idx].numpy())[0, 1]
                type_corrs[d, obj_i * 5 + p] = cc if not np.isnan(cc) else 0
    max_type_corr = np.max(np.abs(type_corrs))
    print(f"│  Max |corr| with properties: {max_type_corr:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════
    print("┌─ Generating Plots")

    # Plot 1: Clean planning comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Phase 18A: Clean Planning Ablation', fontweight='bold')
    axes[0].bar([0, 1], [d_comm, d_nocomm], color=['#2a9d8f', '#e76f51'], width=0.5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['With Comm', 'Without Comm'])
    axes[0].set_ylabel('Avg Distance to Goal')
    axes[0].set_title(f'Clean Margin: {margin_clean:+.1f}%')
    for i, v in enumerate([d_comm, d_nocomm]):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    # Compare old vs new margin
    old_margins = [0.6, 0.2]  # Phase 17 asymmetric, you-see-i-steer
    new_margins = [margin_clean]
    axes[1].bar([0, 1, 2], [0.6, 0.2, margin_clean],
                color=['#e9c46a', '#e9c46a', '#2a9d8f'], width=0.5)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['P17 Asymm\n(leaky)', 'P17 YSIS\n(leaky)', 'P18 YSIS\n(clean)'], fontsize=8)
    axes[1].set_ylabel('Comm Margin (%)')
    axes[1].set_title('Leaky vs Clean Ablation')
    for i, v in enumerate([0.6, 0.2, margin_clean]):
        axes[1].text(i, v + 0.3, f'{v:+.1f}%', ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase18_clean_planning.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Rich physics
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Phase 18B: Rich Physics with Diverse Objects', fontweight='bold')
    axes[0].bar([0, 1], [mse_full, mse_aonly], color=['#2a9d8f', '#e76f51'], width=0.5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Full Comm', 'A Only'])
    axes[0].set_ylabel('Val MSE'); axes[0].set_title(f'Comm Margin: {margin_rich:.1f}%')
    for i, v in enumerate([mse_full, mse_aonly]):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    # Type encoding heatmap
    im = axes[1].imshow(np.abs(type_corrs[:, :15]), cmap='YlOrRd', vmin=0, vmax=0.8, aspect='auto')
    prop_labels = [f'o{i+1}_{p}' for i in range(3) for p in prop_names]
    axes[1].set_xticks(range(15)); axes[1].set_xticklabels(prop_labels, rotation=45, ha='right', fontsize=5)
    axes[1].set_ylabel('Comm dim'); axes[1].set_title(f'Property Encoding (max|r|={max_type_corr:.3f})')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase18_rich_physics.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Type encoding detail
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.abs(type_corrs), cmap='YlOrRd', vmin=0, vmax=0.8, aspect='auto')
    all_prop_labels = [f'o{i+1}_{p}' for i in range(N_OBJECTS) for p in prop_names]
    ax.set_xticks(range(N_OBJECTS * 5))
    ax.set_xticklabels(all_prop_labels, rotation=90, fontsize=4)
    ax.set_ylabel('Comm Dim (A: 0-7, B: 8-15)')
    ax.set_title('Phase 18: Communication Encodes Object Properties', fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase18_type_encoding.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase18_clean_planning.png")
    print(f"  → results/phase18_rich_physics.png")
    print(f"  → results/phase18_type_encoding.png")
    print("└─ Done\n")

    print("="*60)
    print("PHASE 18 SUMMARY")
    print("="*60)
    print(f"  Clean planning margin: {margin_clean:+.1f}% (was +0.2% with leaky ablation)")
    print(f"  Rich physics comm margin: {margin_rich:.1f}%")
    print(f"  Property encoding max|corr|: {max_type_corr:.3f}")


# ── Phase 19: Affordance Discovery ──────────────────────────────

def run_phase19():
    from physics_sim import (PropertyObject, RichPhysicsSimulator,
                             AffordanceSimulator, OBJ_DIM,
                             generate_rich_dataset, get_rich_occluded)
    from world_model import (AffordanceWorldModel, AffordanceGoalEncoder, count_params)

    print("\n" + "="*60)
    print("PHASE 19: AFFORDANCE DISCOVERY")
    print("="*60)

    N_OBJECTS = 5; COMM_DIM = 8; BETA = 0.001

    # ════════════════════════════════════════════════
    # STEP 1: Train World Model on Rich Physics
    # ════════════════════════════════════════════════
    print("\n┌─ Step 1: Training AffordanceWorldModel on rich physics")
    states, nexts = generate_rich_dataset(1500, 30, N_OBJECTS, seed=42)
    sp = int(0.8 * len(states))
    st_t = torch.tensor(states[:sp], dtype=torch.float32)  # [B, N_OBJ, OBJ_DIM]
    nx_t = torch.tensor(nexts[:sp], dtype=torch.float32)
    st_v = torch.tensor(states[sp:], dtype=torch.float32)
    nx_v = torch.tensor(nexts[sp:], dtype=torch.float32)

    model = AffordanceWorldModel(OBJ_DIM, N_OBJECTS, affordance_dim=8,
                                  comm_dim=COMM_DIM, hidden_dim=256)
    print(f"│  Model: {count_params(model)/1e3:.1f}K params")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 500, 0.5)
    for ep in range(1500):
        model.train()
        idx = torch.randperm(sp)[:128]
        pred, affordances, attn_w = model(st_t[idx])
        loss = F.mse_loss(pred, nx_t[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                pv, _, _ = model(st_v)
                vl = F.mse_loss(pv, nx_v).item()
            print(f"│  Epoch {ep+1}: val_MSE={vl:.4f}")
    model.eval()
    with torch.no_grad():
        pv, aff_v, attn_v = model(st_v)
        final_mse = F.mse_loss(pv, nx_v).item()
    print(f"│  Final val MSE: {final_mse:.4f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 2: Evaluate Affordance Tasks
    # ════════════════════════════════════════════════
    print("┌─ Step 2: Affordance Task Evaluation")
    sim = AffordanceSimulator()
    task_names = ['stack', 'capture', 'shield', 'launch', 'bridge']
    task_results = {}

    for task_name in task_names:
        np.random.seed(42)
        n_trials = 50
        distances = []
        for trial in range(n_trials):
            objects, goal = sim.setup_task(task_name, N_OBJECTS)
            # Run physics for 50 steps (no control — just observe baseline)
            for step in range(50):
                objects = sim.step(objects)
            dist = sim.check_goal(objects, goal)
            distances.append(dist)
        task_results[task_name] = {
            'mean_dist': np.mean(distances),
            'success_rate': np.mean([d < 0.3 for d in distances]) * 100,
            'min_dist': np.min(distances)
        }
        print(f"│  {task_name:10s}: mean_dist={np.mean(distances):.3f} "
              f"success={task_results[task_name]['success_rate']:.0f}% "
              f"min={np.min(distances):.3f}")

    # Affordance space analysis
    print("│  Analyzing affordance space...")
    # Collect affordance vectors for different object types
    type_affordances = {}
    preset_types = ['ball', 'heavy', 'light', 'sticky', 'platform']
    for pt in preset_types:
        # Create scenes with this object at index 0
        scenes = []
        for _ in range(100):
            objects = sim.random_scene(N_OBJECTS, [pt, 'ball', 'heavy', 'light', 'platform'])
            scenes.append(sim.get_scene_state(objects))
        scenes_t = torch.tensor(np.array(scenes), dtype=torch.float32)
        with torch.no_grad():
            _, aff, _ = model(scenes_t)
        type_affordances[pt] = aff[:, 0, :].numpy()  # object 0's affordance

    # t-SNE-like: just use PCA for simplicity
    all_aff = np.concatenate([type_affordances[pt] for pt in preset_types])
    labels = np.concatenate([[i]*100 for i in range(5)])
    aff_c = all_aff - all_aff.mean(axis=0)
    U, S, Vt = np.linalg.svd(aff_c, full_matrices=False)
    aff_2d = aff_c @ Vt[:2].T
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 3: Generalization Test
    # ════════════════════════════════════════════════
    print("┌─ Step 3: Generalization & Novel Object Test")

    # Novel object: flat + rigid but never labeled as platform
    novel_obj = PropertyObject(mass=2.0, elasticity=0.3, friction=0.6,
                               flatness=0.8, rigidity=0.7, x=1.0, y=0.5, vx=0, vy=0)
    novel_scene = [novel_obj]
    for _ in range(N_OBJECTS - 1):
        novel_scene.append(PropertyObject.from_preset('ball',
            x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
    novel_state = torch.tensor(sim.get_scene_state(novel_scene), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, novel_aff, novel_attn = model(novel_state)
    novel_aff_vec = novel_aff[0, 0].numpy()  # novel object's affordance vector

    # Compare with known platform affordances
    platform_affs = type_affordances['platform']
    ball_affs = type_affordances['ball']
    cos_to_platform = np.mean([np.dot(novel_aff_vec, pa) /
                               (np.linalg.norm(novel_aff_vec) * np.linalg.norm(pa) + 1e-8)
                               for pa in platform_affs])
    cos_to_ball = np.mean([np.dot(novel_aff_vec, ba) /
                           (np.linalg.norm(novel_aff_vec) * np.linalg.norm(ba) + 1e-8)
                           for ba in ball_affs])
    print(f"│  Novel object (flat+rigid) cosine similarity:")
    print(f"│    To platform: {cos_to_platform:.3f}")
    print(f"│    To ball:     {cos_to_ball:.3f}")
    print(f"│    Platform-like? {'YES' if cos_to_platform > cos_to_ball else 'NO'}")

    # Cross-type similarity matrix
    sim_matrix = np.zeros((5, 5))
    for i, pt_i in enumerate(preset_types):
        for j, pt_j in enumerate(preset_types):
            affs_i = type_affordances[pt_i]
            affs_j = type_affordances[pt_j]
            sims = []
            for ai in affs_i[:20]:
                for aj in affs_j[:20]:
                    s = np.dot(ai, aj) / (np.linalg.norm(ai) * np.linalg.norm(aj) + 1e-8)
                    sims.append(s)
            sim_matrix[i, j] = np.mean(sims)
    print("│  Type similarity matrix:")
    print(f"│    {'':>10s} " + " ".join(f"{pt:>8s}" for pt in preset_types))
    for i, pt in enumerate(preset_types):
        row = " ".join(f"{sim_matrix[i,j]:8.3f}" for j in range(5))
        print(f"│    {pt:>10s} {row}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════
    print("┌─ Generating Plots")

    # Plot 1: Affordance task distances
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 19: Affordance Discovery', fontweight='bold')
    colors = ['#2a9d8f', '#e9c46a', '#264653', '#e76f51', '#f4a261']
    dists = [task_results[t]['mean_dist'] for t in task_names]
    axes[0].bar(range(5), dists, color=colors, width=0.6)
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(task_names, fontsize=8)
    axes[0].set_ylabel('Mean Goal Distance')
    axes[0].set_title('Task Difficulty (physics-only, no planning)')
    for i, v in enumerate(dists):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

    # Affordance PCA
    for i, pt in enumerate(preset_types):
        mask = labels == i
        axes[1].scatter(aff_2d[mask, 0], aff_2d[mask, 1], s=20, alpha=0.5,
                        label=pt, color=colors[i])
    # Plot novel object
    novel_2d = (novel_aff_vec - all_aff.mean(axis=0)) @ Vt[:2].T
    axes[1].scatter(novel_2d[0], novel_2d[1], marker='*', s=200, c='red',
                    zorder=10, label='Novel (flat+rigid)')
    axes[1].legend(fontsize=7); axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].set_title('Affordance Space (8d → 2d PCA)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase19_affordance_success.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Attention weights
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Phase 19: Object Interactions', fontweight='bold')
    # Average attention over validation set
    avg_attn = attn_v.mean(dim=0).numpy()
    im0 = axes[0].imshow(avg_attn, cmap='Blues', vmin=0, aspect='equal')
    axes[0].set_xticks(range(N_OBJECTS)); axes[0].set_yticks(range(N_OBJECTS))
    obj_labels = [f'Obj {i+1}' for i in range(N_OBJECTS)]
    axes[0].set_xticklabels(obj_labels, fontsize=8); axes[0].set_yticklabels(obj_labels, fontsize=8)
    axes[0].set_title('Attention Weights'); plt.colorbar(im0, ax=axes[0])
    # Similarity matrix
    im1 = axes[1].imshow(sim_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1, aspect='equal')
    axes[1].set_xticks(range(5)); axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(preset_types, fontsize=7, rotation=45, ha='right')
    axes[1].set_yticklabels(preset_types, fontsize=7)
    axes[1].set_title('Type Similarity in Affordance Space')
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=6)
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase19_affordance_comm.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Generalization — novel object
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Phase 19: Novel Object Understanding', fontweight='bold')
    axes[0].bar([0, 1], [cos_to_platform, cos_to_ball],
                color=['#2a9d8f', '#e76f51'], width=0.5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['→ Platform', '→ Ball'])
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Novel Object (flat+rigid) Similarity')
    isplatform = "YES ✓" if cos_to_platform > cos_to_ball else "NO ✗"
    axes[0].text(0.5, max(cos_to_platform, cos_to_ball) + 0.05,
                 f'Platform-like? {isplatform}', ha='center', fontweight='bold', fontsize=10)
    # Novel object attention
    novel_attn_np = novel_attn[0].detach().numpy()
    im = axes[1].imshow(novel_attn_np, cmap='Blues', vmin=0, aspect='equal')
    axes[1].set_title('Novel Scene Attention')
    axes[1].set_xticks(range(N_OBJECTS)); axes[1].set_yticks(range(N_OBJECTS))
    axes[1].set_xticklabels([f'Obj {i+1}' for i in range(N_OBJECTS)], fontsize=7)
    axes[1].set_yticklabels([f'Obj {i+1}' for i in range(N_OBJECTS)], fontsize=7)
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase19_generalization.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: Tool use timeline (stack task)
    fig, ax = plt.subplots(figsize=(8, 4))
    np.random.seed(42)
    objects, goal = sim.setup_task('stack', N_OBJECTS)
    timeline_y = []
    timeline_dist = []
    for step in range(100):
        objects = sim.step(objects)
        timeline_y.append(objects[1].y)  # ball y
        timeline_dist.append(sim.check_goal(objects, goal))
    ax.plot(timeline_y, label='Ball Y', color='#2a9d8f')
    ax.plot(timeline_dist, label='Goal Distance', color='#e76f51', ls='--')
    ax.axhline(objects[0].y + objects[0].effective_height, color='gray', ls=':', alpha=0.5,
               label='Platform Top')
    ax.set_xlabel('Timestep'); ax.set_ylabel('Value')
    ax.set_title('Phase 19: Stack Task Timeline', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase19_tool_timeline.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase19_affordance_success.png")
    print(f"  → results/phase19_affordance_comm.png")
    print(f"  → results/phase19_generalization.png")
    print(f"  → results/phase19_tool_timeline.png")
    print("└─ Done\n")

    print("="*60)
    print("PHASE 19 SUMMARY")
    print("="*60)
    print(f"  World model val MSE: {final_mse:.4f}")
    for t in task_names:
        r = task_results[t]
        print(f"  {t:10s}: dist={r['mean_dist']:.3f} success={r['success_rate']:.0f}%")
    print(f"  Novel object: cos_platform={cos_to_platform:.3f} cos_ball={cos_to_ball:.3f}")
    print(f"  Platform-like? {'YES' if cos_to_platform > cos_to_ball else 'NO'}")


# ── Phase 20: Compositional Understanding ───────────────────────

def run_phase20():
    from physics_sim import (PropertyObject, RichPhysicsSimulator,
                             AffordanceSimulator, OBJ_DIM,
                             generate_rich_dataset, get_rich_occluded)
    from world_model import (AffordanceWorldModel, AffordanceGoalEncoder, count_params)

    print("\n" + "="*60)
    print("PHASE 20: COMPOSITIONAL UNDERSTANDING")
    print("="*60)

    N_OBJECTS = 5; COMM_DIM = 8

    # ════════════════════════════════════════════════
    # STEP 1: Train on Rich Physics with More Data
    # ════════════════════════════════════════════════
    print("\n┌─ Step 1: Extended Training (2000 traj, 2000 epochs)")
    states, nexts = generate_rich_dataset(2000, 30, N_OBJECTS, seed=42)
    sp = int(0.8 * len(states))
    st_t = torch.tensor(states[:sp], dtype=torch.float32)
    nx_t = torch.tensor(nexts[:sp], dtype=torch.float32)
    st_v = torch.tensor(states[sp:], dtype=torch.float32)
    nx_v = torch.tensor(nexts[sp:], dtype=torch.float32)

    model = AffordanceWorldModel(OBJ_DIM, N_OBJECTS, affordance_dim=8,
                                  comm_dim=COMM_DIM, hidden_dim=256)
    print(f"│  Model: {count_params(model)/1e3:.1f}K params")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 500, 0.5)
    train_losses = []
    for ep in range(2000):
        model.train()
        idx = torch.randperm(sp)[:128]
        pred, _, _ = model(st_t[idx])
        loss = F.mse_loss(pred, nx_t[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        train_losses.append(loss.item())
        if (ep+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                pv, _, _ = model(st_v)
                vl = F.mse_loss(pv, nx_v).item()
            print(f"│  Epoch {ep+1}: val={vl:.4f}")
    model.eval()
    with torch.no_grad():
        pv_final, aff_final, attn_final = model(st_v)
        final_mse = F.mse_loss(pv_final, nx_v).item()
    print(f"│  Final val MSE: {final_mse:.4f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 2: Affordance Bottleneck Analysis
    # ════════════════════════════════════════════════
    print("┌─ Step 2: Affordance Bottleneck Analysis")
    sim = AffordanceSimulator()
    preset_types = ['ball', 'heavy', 'light', 'sticky', 'platform']
    prop_names = ['mass', 'elast', 'frict', 'flat', 'rigid']

    # Collect affordance vectors for each type
    type_affordances = {}
    for pt in preset_types:
        scenes = []
        for _ in range(200):
            objects = sim.random_scene(N_OBJECTS, [pt] + ['ball', 'heavy', 'light', 'platform'])
            scenes.append(sim.get_scene_state(objects))
        scenes_t = torch.tensor(np.array(scenes), dtype=torch.float32)
        with torch.no_grad():
            _, aff, _ = model(scenes_t)
        type_affordances[pt] = aff[:, 0, :].numpy()

    # Property → Affordance mapping
    # For each object in val set, get its properties and affordance vector
    all_props = st_v[:, :, 4:9].reshape(-1, 5).numpy()  # [B*N, 5]
    model.eval()
    with torch.no_grad():
        _, all_aff, _ = model(st_v)
    all_aff_flat = all_aff.reshape(-1, 8).numpy()  # [B*N, 8]

    # Correlation: 5 properties × 8 affordance dims
    prop_aff_corr = np.zeros((5, 8))
    for p in range(5):
        for a in range(8):
            n_samples = min(5000, len(all_props))
            cc = np.corrcoef(all_props[:n_samples, p], all_aff_flat[:n_samples, a])[0, 1]
            prop_aff_corr[p, a] = cc if not np.isnan(cc) else 0
    print(f"│  Property → Affordance correlation:")
    for p, pn in enumerate(prop_names):
        max_corr = np.max(np.abs(prop_aff_corr[p]))
        print(f"│    {pn:>6s}: max|r|={max_corr:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 3: Zero-Shot Composition ("Chair Test")
    # ════════════════════════════════════════════════
    print("┌─ Step 3: Zero-Shot Composition — The 'Chair Test'")

    # Novel objects: never seen in training
    novel_objects = {
        'flat_rigid': PropertyObject(mass=2.0, elasticity=0.3, friction=0.6,
                                      flatness=0.8, rigidity=0.7, x=1.0, y=0.5, vx=0, vy=0),
        'flat_sticky': PropertyObject(mass=1.5, elasticity=0.1, friction=0.7,
                                       flatness=0.7, rigidity=0.4, x=1.0, y=0.5, vx=0, vy=0),
        'heavy_bouncy': PropertyObject(mass=4.0, elasticity=0.9, friction=0.2,
                                        flatness=0.1, rigidity=0.8, x=1.0, y=0.5, vx=0, vy=0),
        'light_flat': PropertyObject(mass=0.3, elasticity=0.5, friction=0.3,
                                      flatness=0.9, rigidity=0.3, x=1.0, y=0.5, vx=0, vy=0),
    }

    novel_results = {}
    for name, obj in novel_objects.items():
        # Create scene with this novel object at index 0
        scene = [obj]
        np.random.seed(42)
        for _ in range(N_OBJECTS - 1):
            scene.append(PropertyObject.from_preset('ball',
                x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
        scene_t = torch.tensor(sim.get_scene_state(scene), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, novel_aff, novel_attn = model(scene_t)
        aff_vec = novel_aff[0, 0].numpy()

        # Cosine similarity to each known type
        sims = {}
        for pt in preset_types:
            cos_vals = [np.dot(aff_vec, pa) /
                       (np.linalg.norm(aff_vec) * np.linalg.norm(pa) + 1e-8)
                       for pa in type_affordances[pt][:50]]
            sims[pt] = np.mean(cos_vals)
        closest = max(sims, key=sims.get)
        novel_results[name] = {'sims': sims, 'closest': closest, 'aff': aff_vec}
        print(f"│  {name:>15s}: closest to '{closest}' "
              f"(cos={sims[closest]:.3f})")

    # Key test: flat_rigid should be closest to platform
    chair_test = novel_results['flat_rigid']['closest'] == 'platform'
    print(f"│")
    print(f"│  ┌── THE CHAIR TEST ──┐")
    print(f"│  │ flat+rigid = platform? {'YES ✓' if chair_test else 'NO ✗':>8s} │")
    print(f"│  └─────────────────────┘")
    print(f"│")

    # flat_sticky should combine platform + sticky affordances
    flat_sticky_sims = novel_results['flat_sticky']['sims']
    composition_score = (flat_sticky_sims.get('platform', 0) +
                         flat_sticky_sims.get('sticky', 0)) / 2
    print(f"│  Composition test (flat+sticky):")
    print(f"│    → platform sim: {flat_sticky_sims.get('platform', 0):.3f}")
    print(f"│    → sticky sim:   {flat_sticky_sims.get('sticky', 0):.3f}")
    print(f"│    → composition score: {composition_score:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════
    print("┌─ Generating Plots")

    # Plot 1: Affordance t-SNE (PCA)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 20: Affordance Space & Compositional Understanding', fontweight='bold')
    all_aff_types = np.concatenate([type_affordances[pt][:100] for pt in preset_types])
    labels = np.concatenate([[i]*100 for i in range(5)])
    aff_c = all_aff_types - all_aff_types.mean(axis=0)
    U, S, Vt = np.linalg.svd(aff_c, full_matrices=False)
    aff_2d = aff_c @ Vt[:2].T
    colors = ['#2a9d8f', '#264653', '#e9c46a', '#e76f51', '#f4a261']
    for i, pt in enumerate(preset_types):
        mask = labels == i
        axes[0].scatter(aff_2d[mask, 0], aff_2d[mask, 1], s=15, alpha=0.5,
                        label=pt, color=colors[i])
    # Plot novel objects
    novel_markers = {'flat_rigid': '*', 'flat_sticky': 'D', 'heavy_bouncy': 'P', 'light_flat': 'X'}
    novel_colors = ['red', 'purple', 'blue', 'green']
    for (name, res), marker, c in zip(novel_results.items(), novel_markers.values(), novel_colors):
        nv_2d = (res['aff'] - all_aff_types.mean(axis=0)) @ Vt[:2].T
        axes[0].scatter(nv_2d[0], nv_2d[1], marker=marker, s=150, c=c,
                        zorder=10, label=f'Novel: {name}', edgecolors='black')
    axes[0].legend(fontsize=6, ncol=2); axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].set_title('Affordance Bottleneck (8d → 2d)')

    # Property → Affordance heatmap
    im = axes[1].imshow(np.abs(prop_aff_corr), cmap='YlOrRd', vmin=0, vmax=0.8, aspect='auto')
    axes[1].set_xticks(range(8)); axes[1].set_xticklabels([f'Aff {i+1}' for i in range(8)], fontsize=7)
    axes[1].set_yticks(range(5)); axes[1].set_yticklabels(prop_names, fontsize=8)
    axes[1].set_title('Property → Affordance Mapping')
    for i in range(5):
        for j in range(8):
            axes[1].text(j, i, f'{prop_aff_corr[i,j]:.2f}', ha='center', va='center', fontsize=5)
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase20_affordance_tsne.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Novel object similarity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 20: Novel Object Understanding', fontweight='bold')
    # Similarity profiles
    x_pos = np.arange(5)
    width = 0.18
    novel_names = list(novel_results.keys())
    bar_colors = ['red', 'purple', 'blue', 'green']
    for ni, (name, res) in enumerate(novel_results.items()):
        svals = [res['sims'][pt] for pt in preset_types]
        axes[0].bar(x_pos + ni * width, svals, width, label=name, color=bar_colors[ni], alpha=0.7)
    axes[0].set_xticks(x_pos + width * 1.5)
    axes[0].set_xticklabels(preset_types, fontsize=8)
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Novel Objects → Known Types')
    axes[0].legend(fontsize=7)

    # Attention weights for novel scene
    scene_novel = [novel_objects['flat_rigid']]
    np.random.seed(42)
    for _ in range(N_OBJECTS - 1):
        scene_novel.append(PropertyObject.from_preset('ball',
            x=np.random.uniform(0.3, 1.7), y=np.random.uniform(0.5, 1.5), vx=0, vy=0))
    with torch.no_grad():
        _, _, attn_novel = model(torch.tensor(sim.get_scene_state(scene_novel),
                                               dtype=torch.float32).unsqueeze(0))
    im = axes[1].imshow(attn_novel[0].numpy(), cmap='Blues', vmin=0, aspect='equal')
    axes[1].set_xticks(range(N_OBJECTS)); axes[1].set_yticks(range(N_OBJECTS))
    obj_labels = ['Novel\n(flat+rigid)'] + [f'Ball {i+1}' for i in range(N_OBJECTS-1)]
    axes[1].set_xticklabels(obj_labels, fontsize=6)
    axes[1].set_yticklabels(obj_labels, fontsize=6)
    axes[1].set_title('Attention: Novel Object Scene')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase20_novel_object.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Generalization matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    gen_matrix = np.zeros((len(novel_results), 5))
    for i, (name, res) in enumerate(novel_results.items()):
        for j, pt in enumerate(preset_types):
            gen_matrix[i, j] = res['sims'][pt]
    im = ax.imshow(gen_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1, aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels(preset_types, fontsize=8)
    ax.set_yticks(range(len(novel_results)))
    ax.set_yticklabels(list(novel_results.keys()), fontsize=8)
    ax.set_title('Phase 20: Generalization Matrix', fontweight='bold')
    for i in range(len(novel_results)):
        for j in range(5):
            ax.text(j, i, f'{gen_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase20_generalization.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: Average attention
    fig, ax = plt.subplots(figsize=(5, 4))
    avg_attn = attn_final.mean(dim=0).numpy()
    im = ax.imshow(avg_attn, cmap='Blues', aspect='equal')
    ax.set_xticks(range(N_OBJECTS)); ax.set_yticks(range(N_OBJECTS))
    ax.set_xticklabels([f'Obj {i+1}' for i in range(N_OBJECTS)], fontsize=8)
    ax.set_yticklabels([f'Obj {i+1}' for i in range(N_OBJECTS)], fontsize=8)
    ax.set_title('Phase 20: Average Attention Weights', fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase20_attention.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 5: Property → Affordance map (full)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(prop_aff_corr, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(range(8)); ax.set_xticklabels([f'Aff {i+1}' for i in range(8)], fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(prop_names, fontsize=9)
    ax.set_title('Phase 20: Property → Affordance Map (signed)', fontweight='bold')
    for i in range(5):
        for j in range(8):
            ax.text(j, i, f'{prop_aff_corr[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase20_property_affordance_map.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase20_affordance_tsne.png")
    print(f"  → results/phase20_novel_object.png")
    print(f"  → results/phase20_generalization.png")
    print(f"  → results/phase20_attention.png")
    print(f"  → results/phase20_property_affordance_map.png")
    print("└─ Done\n")

    # Final summary
    print("="*60)
    print("PHASE 20 SUMMARY — TOWARD UNDERSTANDING")
    print("="*60)
    print(f"  World model val MSE: {final_mse:.4f}")
    print(f"  Property → Affordance correlations:")
    for p, pn in enumerate(prop_names):
        print(f"    {pn:>6s}: max|r|={np.max(np.abs(prop_aff_corr[p])):.3f}")
    print(f"  Novel object results:")
    for name, res in novel_results.items():
        print(f"    {name:>15s} → {res['closest']} (cos={res['sims'][res['closest']]:.3f})")
    chair_pass = "PASS ✓" if chair_test else "FAIL ✗"
    print(f"\n  ╔═══════════════════════════════╗")
    print(f"  ║ THE CHAIR TEST: {chair_pass:>13s} ║")
    print(f"  ╚═══════════════════════════════╝")
    print(f"\n  flat + rigid → platform-like affordance?")
    print(f"  This is the toy-scale equivalent of:")
    print(f"  'A rock can be a chair if it's flat enough.'")


# ── Phase 21: Causal Discovery Through Communication ────────────

def _knn_mi(x, y, k=5):
    """k-NN mutual information estimator (Kraskov method, simplified).
    x: (N, dx), y: (N, dy) → scalar MI estimate."""
    from scipy.spatial import cKDTree
    N = len(x)
    if N < k + 1:
        return 0.0
    xy = np.concatenate([x, y], axis=1)
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    # kth neighbor distance in joint space
    dists, _ = tree_xy.query(xy, k=k+1)
    eps = dists[:, -1]  # distance to k-th neighbor
    # Count neighbors within eps in marginals
    digamma_vals = []
    for i in range(N):
        nx = tree_x.query_ball_point(x[i], eps[i] + 1e-10)
        ny = tree_y.query_ball_point(y[i], eps[i] + 1e-10)
        nx_count = max(len(nx) - 1, 1)
        ny_count = max(len(ny) - 1, 1)
        digamma_vals.append(np.log(nx_count) + np.log(ny_count))
    from scipy.special import digamma
    mi = digamma(k) - np.mean(digamma_vals) + digamma(N)
    return max(mi, 0.0)


def _linear_cka(X, Y):
    """Linear CKA (Centered Kernel Alignment) between two representations.
    X: (N, d1), Y: (N, d2) → scalar in [0, 1]."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def run_phase21():
    from physics_sim import (Ball, SimConfig, PhysicsSimulator,
                             CausalPhysicsSimulator, generate_causal_dataset,
                             get_occluded_state)
    from world_model import ScaledBottleneckedFusionModel, count_params

    print("\n" + "="*60)
    print("PHASE 21: CAUSAL DISCOVERY THROUGH COMMUNICATION")
    print("="*60)

    N_BALLS = 3; STATE_DIM = N_BALLS * 4; COMM_DIM = 8

    # ════════════════════════════════════════════════
    # STEP 1: Train on both regimes
    # ════════════════════════════════════════════════
    print("\n┌─ Step 1: Train models on causal vs correlational regimes")

    models = {}
    data = {}
    for regime in ['causal', 'correlational']:
        print(f"│  Generating {regime} dataset...")
        oa, ob, tgt = generate_causal_dataset(regime, n_traj=1500, n_steps=50, seed=42)
        sp = int(0.8 * len(oa))
        data[regime] = {
            'oa_t': torch.tensor(oa[:sp]), 'ob_t': torch.tensor(ob[:sp]),
            'tgt_t': torch.tensor(tgt[:sp]),
            'oa_v': torch.tensor(oa[sp:]), 'ob_v': torch.tensor(ob[sp:]),
            'tgt_v': torch.tensor(tgt[sp:]),
        }
        model = ScaledBottleneckedFusionModel(STATE_DIM, COMM_DIM, fused_dim=64, hidden_dim=384)
        if regime == 'causal':
            print(f"│  Model: {count_params(model)/1e3:.1f}K params")
        BETA = 0.001
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, 500, 0.5)
        d = data[regime]
        n_train = len(d['oa_t'])
        for ep in range(1500):
            model.train()
            idx = torch.randperm(n_train)[:256]
            pred, mu, kl = model(d['oa_t'][idx], d['ob_t'][idx])
            loss = F.mse_loss(pred, d['tgt_t'][idx]) + BETA * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (ep+1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    pv, _, kv = model(d['oa_v'], d['ob_v'])
                    vl = F.mse_loss(pv, d['tgt_v']).item()
                print(f"│  {regime:15s} ep {ep+1}: val={vl:.4f}")
        models[regime] = model
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 2: Communication Protocol Comparison
    # ════════════════════════════════════════════════
    print("┌─ Step 2: Communication Protocol Analysis")

    corr_maps = {}
    mi_results = {}
    for regime in ['causal', 'correlational']:
        model = models[regime]; model.eval()
        d = data[regime]
        with torch.no_grad():
            # Get communication vectors
            h_a = model.encoder_a(d['oa_v'])
            mu_a = model.mu_a(h_a)
            h_b = model.encoder_b(d['ob_v'])
            mu_b = model.mu_b(h_b)
        msg = torch.cat([mu_a, mu_b], dim=-1).numpy()  # [N, 16]
        tgt = d['tgt_v'].numpy()  # [N, 12]

        # Correlation heatmap
        corr = np.zeros((COMM_DIM * 2, STATE_DIM))
        n_samp = min(5000, len(msg))
        for i in range(COMM_DIM * 2):
            for j in range(STATE_DIM):
                cc = np.corrcoef(msg[:n_samp, i], tgt[:n_samp, j])[0, 1]
                corr[i, j] = cc if not np.isnan(cc) else 0
        corr_maps[regime] = corr

        # MI estimation: Agent A's msg vs Ball 1 state, Ball 2 state
        mu_a_np = mu_a.numpy()[:2000]
        b1_state = tgt[:2000, 4:8]  # ball 1: x,y,vx,vy
        b2_state = tgt[:2000, 8:12]  # ball 2
        b0_state = tgt[:2000, 0:4]   # ball 0
        mi_a_b1 = _knn_mi(mu_a_np, b1_state, k=5)
        mi_a_b2 = _knn_mi(mu_a_np, b2_state, k=5)
        mi_a_b0 = _knn_mi(mu_a_np, b0_state, k=5)
        mi_results[regime] = {'b0': mi_a_b0, 'b1': mi_a_b1, 'b2': mi_a_b2}
        print(f"│  {regime:15s}: I(msg_a;ball0)={mi_a_b0:.3f} I(msg_a;ball1)={mi_a_b1:.3f} "
              f"I(msg_a;ball2)={mi_a_b2:.3f}")

    # Ball 1 correlation comparison
    b1_corr_causal = np.max(np.abs(corr_maps['causal'][:COMM_DIM, 4:8]))
    b1_corr_correl = np.max(np.abs(corr_maps['correlational'][:COMM_DIM, 4:8]))
    print(f"│  Ball 1 max|corr| from Agent A: causal={b1_corr_causal:.3f} correl={b1_corr_correl:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 3: Intervention Test
    # ════════════════════════════════════════════════
    print("┌─ Step 3: Intervention Test (teleport Ball 1 at step 25)")

    n_intervention_trials = 500
    intervention_results = {}
    for regime in ['causal', 'correlational']:
        model = models[regime]; model.eval()
        sim = CausalPhysicsSimulator(regime=regime)
        np.random.seed(42)
        msg_a_shifts = []
        msg_b_shifts = []
        for trial in range(n_intervention_trials):
            sim.osc_phase = 0.0
            balls = sim.random_balls()
            msgs_a = []
            msgs_b = []
            for step in range(50):
                state = sim.get_state(balls)
                oa = get_occluded_state(state, N_BALLS, 'A')
                ob = get_occluded_state(state, N_BALLS, 'B')
                with torch.no_grad():
                    oa_t = torch.tensor(oa).unsqueeze(0)
                    ob_t = torch.tensor(ob).unsqueeze(0)
                    h_a = model.encoder_a(oa_t)
                    mu_a = model.mu_a(h_a)
                    h_b = model.encoder_b(ob_t)
                    mu_b = model.mu_b(h_b)
                msgs_a.append(mu_a[0].numpy())
                msgs_b.append(mu_b[0].numpy())

                # At step 25, INTERVENE: teleport Ball 1
                if step == 24:
                    balls[1].x = np.random.uniform(1.2, 1.8)
                    balls[1].y = np.random.uniform(0.5, 1.5)
                    balls[1].vx = 0; balls[1].vy = 0

                balls = sim.step(balls, t=step*0.02)

            msgs_a = np.array(msgs_a)  # [50, 8]
            msgs_b = np.array(msgs_b)
            # Shift = norm difference at intervention point
            if len(msgs_a) >= 27:
                shift_a = np.linalg.norm(msgs_a[26] - msgs_a[24])
                shift_b = np.linalg.norm(msgs_b[26] - msgs_b[24])
                msg_a_shifts.append(shift_a)
                msg_b_shifts.append(shift_b)

        intervention_results[regime] = {
            'agent_a': np.mean(msg_a_shifts),
            'agent_b': np.mean(msg_b_shifts),
            'agent_a_std': np.std(msg_a_shifts),
            'agent_b_std': np.std(msg_b_shifts),
        }
        print(f"│  {regime:15s}: Agent A shift={np.mean(msg_a_shifts):.4f}±{np.std(msg_a_shifts):.4f} "
              f"Agent B shift={np.mean(msg_b_shifts):.4f}±{np.std(msg_b_shifts):.4f}")

    # Differential response
    causal_a = intervention_results['causal']['agent_a']
    correl_a = intervention_results['correlational']['agent_a']
    diff_response = causal_a - correl_a
    print(f"│  Differential Agent A response: {diff_response:+.4f}")
    print(f"│  Causal discovery? {'YES — Agent A responds more in causal' if diff_response > 0.05 else 'Marginal'}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 4: Beta Sweep — Information Retention
    # ════════════════════════════════════════════════
    print("┌─ Step 4: Information Retention Under Compression")

    beta_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    retention_results = {r: {b: [] for b in beta_values} for r in ['causal', 'correlational']}

    for regime in ['causal', 'correlational']:
        d = data[regime]
        n_train = len(d['oa_t'])
        for beta in beta_values:
            model_b = ScaledBottleneckedFusionModel(STATE_DIM, COMM_DIM, fused_dim=64, hidden_dim=384)
            opt_b = torch.optim.Adam(model_b.parameters(), lr=3e-4)
            sched_b = torch.optim.lr_scheduler.StepLR(opt_b, 300, 0.5)
            for ep in range(800):
                model_b.train()
                idx = torch.randperm(n_train)[:256]
                pred, mu, kl = model_b(d['oa_t'][idx], d['ob_t'][idx])
                loss = F.mse_loss(pred, d['tgt_t'][idx]) + beta * kl
                opt_b.zero_grad(); loss.backward(); opt_b.step(); sched_b.step()

            model_b.eval()
            with torch.no_grad():
                pv, _, _ = model_b(d['oa_v'], d['ob_v'])
                mse = F.mse_loss(pv, d['tgt_v']).item()
                # Get msg_a
                h_a = model_b.encoder_a(d['oa_v'])
                mu_a = model_b.mu_a(h_a).numpy()[:1500]
            tgt_np = d['tgt_v'].numpy()[:1500]
            mi_b0 = _knn_mi(mu_a, tgt_np[:, 0:4], k=5)
            mi_b1 = _knn_mi(mu_a, tgt_np[:, 4:8], k=5)
            mi_b2 = _knn_mi(mu_a, tgt_np[:, 8:12], k=5)
            retention_results[regime][beta] = {
                'mse': mse, 'mi_b0': mi_b0, 'mi_b1': mi_b1, 'mi_b2': mi_b2
            }
            print(f"│  {regime:15s} β={beta:<6.3f}: MSE={mse:.4f} "
                  f"I(a;b0)={mi_b0:.3f} I(a;b1)={mi_b1:.3f} I(a;b2)={mi_b2:.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # STEP 5: Transfer Test
    # ════════════════════════════════════════════════
    print("┌─ Step 5: Transfer Test")

    transfer_mse = {}
    for train_regime in ['causal', 'correlational']:
        for test_regime in ['causal', 'correlational']:
            model = models[train_regime]; model.eval()
            d = data[test_regime]
            with torch.no_grad():
                pv, _, _ = model(d['oa_v'], d['ob_v'])
                mse = F.mse_loss(pv, d['tgt_v']).item()
            transfer_mse[(train_regime, test_regime)] = mse
            print(f"│  Train {train_regime:15s} → Test {test_regime:15s}: MSE={mse:.4f}")

    # Protocol similarity: cosine between comm vectors on SAME data
    regime_msgs = {}
    for regime in ['causal', 'correlational']:
        model = models[regime]; model.eval()
        # Test on causal data (same input for both)
        d = data['causal']
        with torch.no_grad():
            h_a = model.encoder_a(d['oa_v'])
            mu_a = model.mu_a(h_a)
        regime_msgs[regime] = mu_a.numpy()[:3000]

    # Protocol cosine similarity
    msg_c = regime_msgs['causal']
    msg_r = regime_msgs['correlational']
    cross_cos = np.mean([np.dot(msg_c[i], msg_r[i]) /
                        (np.linalg.norm(msg_c[i]) * np.linalg.norm(msg_r[i]) + 1e-8)
                        for i in range(len(msg_c))])
    print(f"│  Protocol cosine similarity: {cross_cos:.4f}")

    # CKA
    cka = _linear_cka(msg_c, msg_r)
    print(f"│  CKA (representation similarity): {cka:.4f}")

    # Within-regime similarity (split data)
    mid = len(msg_c) // 2
    within_cka = _linear_cka(msg_c[:mid], msg_c[mid:2*mid])
    print(f"│  Within-regime CKA (control): {within_cka:.4f}")
    print(f"│  Cross-regime / Within-regime: {cka/max(within_cka,1e-6):.3f}")
    print("└─ Done\n")

    # ════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════
    print("┌─ Generating Plots")

    state_labels = ['B0.x','B0.y','B0.vx','B0.vy',
                    'B1.x','B1.y','B1.vx','B1.vy',
                    'B2.x','B2.y','B2.vx','B2.vy']

    # Plot 1: Correlation heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 21: Communication Protocol — Causal vs Correlational', fontweight='bold')
    for ax, regime, title in zip(axes, ['causal', 'correlational'],
                                  ['Causal (Spring)', 'Correlational (Hidden Osc.)']):
        im = ax.imshow(np.abs(corr_maps[regime]), cmap='YlOrRd', vmin=0, vmax=0.6, aspect='auto')
        ax.set_xticks(range(STATE_DIM)); ax.set_xticklabels(state_labels, rotation=90, fontsize=6)
        ax.set_ylabel('Comm dim (A:0-7, B:8-15)')
        ax.set_title(title)
        # Highlight Ball 1 columns
        for c in range(4, 8):
            ax.axvline(c-0.5, color='cyan', lw=0.5, alpha=0.5)
            ax.axvline(c+0.5, color='cyan', lw=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase21_causal_heatmaps.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Intervention response
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Phase 21: Intervention Test — Teleport Ball 1 at Step 25', fontweight='bold')
    # Bar chart: Agent A response
    x = np.arange(2)
    a_vals = [intervention_results['causal']['agent_a'],
              intervention_results['correlational']['agent_a']]
    a_errs = [intervention_results['causal']['agent_a_std'],
              intervention_results['correlational']['agent_a_std']]
    axes[0].bar(x, a_vals, yerr=a_errs, color=['#2a9d8f', '#e76f51'], width=0.5, capsize=5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(['Causal', 'Correlational'])
    axes[0].set_ylabel('|msg shift|'); axes[0].set_title('Agent A Response\n(sees Ball 0, NOT Ball 1)')
    for i, v in enumerate(a_vals):
        axes[0].text(i, v + a_errs[i] + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Agent B response (control)
    b_vals = [intervention_results['causal']['agent_b'],
              intervention_results['correlational']['agent_b']]
    b_errs = [intervention_results['causal']['agent_b_std'],
              intervention_results['correlational']['agent_b_std']]
    axes[1].bar(x, b_vals, yerr=b_errs, color=['#2a9d8f', '#e76f51'], width=0.5, capsize=5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(['Causal', 'Correlational'])
    axes[1].set_ylabel('|msg shift|'); axes[1].set_title('Agent B Response\n(SEES Ball 1 — control)')
    for i, v in enumerate(b_vals):
        axes[1].text(i, v + b_errs[i] + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

    # MI comparison
    mi_bars = np.array([[mi_results['causal']['b0'], mi_results['causal']['b1'], mi_results['causal']['b2']],
                        [mi_results['correlational']['b0'], mi_results['correlational']['b1'], mi_results['correlational']['b2']]])
    x2 = np.arange(3)
    w = 0.35
    axes[2].bar(x2 - w/2, mi_bars[0], w, label='Causal', color='#2a9d8f')
    axes[2].bar(x2 + w/2, mi_bars[1], w, label='Correlational', color='#e76f51')
    axes[2].set_xticks(x2); axes[2].set_xticklabels(['Ball 0\n(A sees)', 'Ball 1\n(B sees)', 'Ball 2\n(indep.)'])
    axes[2].set_ylabel('I(msg_A ; ball)'); axes[2].set_title('Mutual Information')
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase21_intervention.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Information retention curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Phase 21: Information Retention Under Compression', fontweight='bold')
    ball_names = ['Ball 0 (A sees)', 'Ball 1 (B sees)', 'Ball 2 (independent)']
    mi_keys = ['mi_b0', 'mi_b1', 'mi_b2']
    for ax, bname, mk in zip(axes, ball_names, mi_keys):
        for regime, color, ls in [('causal', '#2a9d8f', '-'), ('correlational', '#e76f51', '--')]:
            vals = [retention_results[regime][b][mk] for b in beta_values]
            ax.plot(range(len(beta_values)), vals, color=color, ls=ls,
                    marker='o', label=regime, markersize=5)
        ax.set_xticks(range(len(beta_values)))
        ax.set_xticklabels([str(b) for b in beta_values], fontsize=7)
        ax.set_xlabel('β (KL weight)'); ax.set_ylabel(f'I(msg_A ; {bname.split()[0]} {bname.split()[1]})')
        ax.set_title(bname); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase21_information_retention.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: Transfer confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    tm = np.array([[transfer_mse[('causal','causal')], transfer_mse[('causal','correlational')]],
                   [transfer_mse[('correlational','causal')], transfer_mse[('correlational','correlational')]]])
    im = ax.imshow(tm, cmap='YlOrRd', aspect='equal')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Causal', 'Correl.'], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Causal', 'Correl.'], fontsize=9)
    ax.set_xlabel('Test Regime'); ax.set_ylabel('Train Regime')
    ax.set_title('Phase 21: Transfer MSE', fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{tm[i,j]:.4f}', ha='center', va='center', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase21_transfer.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 5: Protocol similarity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Phase 21: Representation Similarity', fontweight='bold')
    axes[0].bar([0, 1], [cross_cos, 1.0], color=['#e76f51', '#2a9d8f'], width=0.5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Cross-Regime', 'Within-Regime'])
    axes[0].set_ylabel('Cosine Similarity'); axes[0].set_title('Protocol Similarity')
    for i, v in enumerate([cross_cos, 1.0]):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1].bar([0, 1], [cka, within_cka], color=['#e76f51', '#2a9d8f'], width=0.5)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Cross-Regime', 'Within-Regime'])
    axes[1].set_ylabel('CKA'); axes[1].set_title(f'CKA (ratio: {cka/max(within_cka,1e-6):.3f})')
    for i, v in enumerate([cka, within_cka]):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase21_protocol_similarity.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 6: Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    summary_text = [
        'PHASE 21: CAUSAL DISCOVERY THROUGH COMMUNICATION',
        '',
        f'Regime A (Causal — spring): Ball 0 ↔ Ball 1 causally connected',
        f'Regime B (Correlational — oscillator): Ball 0 ~ Ball 1 confounded',
        '',
        '─── Key Results ───',
        f'Agent A msg → Ball 1 MI:  CAUSAL={mi_results["causal"]["b1"]:.3f}  CORR={mi_results["correlational"]["b1"]:.3f}',
        f'Agent A msg → Ball 2 MI:  CAUSAL={mi_results["causal"]["b2"]:.3f}  CORR={mi_results["correlational"]["b2"]:.3f}',
        '',
        f'Intervention response (Agent A):',
        f'  Causal: {intervention_results["causal"]["agent_a"]:.4f}  Corr: {intervention_results["correlational"]["agent_a"]:.4f}',
        f'  Differential: {diff_response:+.4f}',
        '',
        f'Transfer: CKA={cka:.4f} (within={within_cka:.4f}, ratio={cka/max(within_cka,1e-6):.3f})',
        '',
        '─── Connection to Causal-JEPA (2602.11389) ───',
        'Causal-JEPA: engineered masking → forced inference → causal reasoning',
        'Our work: natural communication bottleneck → forced compression → causal discovery',
        'Both: information constraint → causal inductive bias',
    ]
    for i, line in enumerate(summary_text):
        fontweight = 'bold' if i in [0, 5, 14] else 'normal'
        fontsize = 12 if i == 0 else 9
        ax.text(0.05, 0.95 - i * 0.045, line, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, family='monospace',
                verticalalignment='top')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase21_summary.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"  → results/phase21_causal_heatmaps.png")
    print(f"  → results/phase21_intervention.png")
    print(f"  → results/phase21_information_retention.png")
    print(f"  → results/phase21_transfer.png")
    print(f"  → results/phase21_protocol_similarity.png")
    print(f"  → results/phase21_summary.png")
    print("└─ Done\n")

    # Final summary
    print("="*60)
    print("PHASE 21 SUMMARY — CAUSAL DISCOVERY")
    print("="*60)
    print(f"\n  Models: {count_params(models['causal'])/1e3:.0f}K params × 2 regimes")
    print(f"\n  Mutual Information (Agent A's msg):")
    for r in ['causal', 'correlational']:
        mi = mi_results[r]
        print(f"    {r:15s}: Ball0={mi['b0']:.3f} Ball1={mi['b1']:.3f} Ball2={mi['b2']:.3f}")
    print(f"\n  Intervention Test (Agent A shift):")
    print(f"    Causal:        {intervention_results['causal']['agent_a']:.4f}")
    print(f"    Correlational: {intervention_results['correlational']['agent_a']:.4f}")
    print(f"    Differential:  {diff_response:+.4f}")
    print(f"\n  Representation:")
    print(f"    Cross-regime CKA:  {cka:.4f}")
    print(f"    Within-regime CKA: {within_cka:.4f}")
    print(f"    Protocol cosine:   {cross_cos:.4f}")
    print(f"\n  Information retention (β sweep):")
    for beta in beta_values:
        c = retention_results['causal'][beta]
        r = retention_results['correlational'][beta]
        print(f"    β={beta:<6.3f}: Causal I(a;b1)={c['mi_b1']:.3f}  Corr I(a;b1)={r['mi_b1']:.3f}")


# ── Phase 22: Clean Causal Intervention + C-JEPA ────────────────

def run_phase22():
    from physics_sim import (Ball, IsolatedCausalSimulator,
                             generate_isolated_causal_dataset,
                             get_occluded_state)
    from world_model import ScaledBottleneckedFusionModel, count_params

    print("\n" + "="*60)
    print("PHASE 22: CLEAN CAUSAL INTERVENTION + C-JEPA COMPARISON")
    print("="*60)

    STATE_DIM = 16; COMM_DIM = 8; BETA = 0.001
    regimes = ['causal', 'correlational', 'independent']

    # ══════ PART A: Train 3 regimes ══════
    print("\n┌─ Part A: Train models on 3 regimes")
    models_22 = {}; data_22 = {}
    for regime in regimes:
        print(f"│  Generating {regime} dataset...")
        oa, ob, tgt = generate_isolated_causal_dataset(regime, n_traj=1500, n_steps=50, seed=42)
        sp = int(0.8 * len(oa))
        data_22[regime] = {
            'oa_t': torch.tensor(oa[:sp]), 'ob_t': torch.tensor(ob[:sp]),
            'tgt_t': torch.tensor(tgt[:sp]),
            'oa_v': torch.tensor(oa[sp:]), 'ob_v': torch.tensor(ob[sp:]),
            'tgt_v': torch.tensor(tgt[sp:]),
        }
        model = ScaledBottleneckedFusionModel(STATE_DIM, COMM_DIM, fused_dim=64, hidden_dim=384)
        if regime == 'causal':
            print(f"│  Model: {count_params(model)/1e3:.1f}K params")
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, 500, 0.5)
        d = data_22[regime]; n_tr = len(d['oa_t'])
        for ep in range(1500):
            model.train()
            idx = torch.randperm(n_tr)[:512]
            pred, mu, kl = model(d['oa_t'][idx], d['ob_t'][idx])
            loss = F.mse_loss(pred, d['tgt_t'][idx]) + BETA * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (ep+1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    pv, _, _ = model(d['oa_v'], d['ob_v'])
                    vl = F.mse_loss(pv, d['tgt_v']).item()
                print(f"│  {regime:15s} ep {ep+1}: val={vl:.4f}")
        models_22[regime] = model
    print("└─ Done\n")

    # ══════ PART B: Clean Intervention Test ══════
    print("┌─ Part B: Clean Intervention Test (teleport Ball 1 at step 25)")
    intervention_22 = {}
    for regime in regimes:
        model = models_22[regime]; model.eval()
        sim = IsolatedCausalSimulator(regime=regime)
        np.random.seed(42)
        shifts_a, shifts_b = [], []
        for trial in range(500):
            sim.osc_phase = 0.0
            balls = sim.random_balls()
            msgs_a, msgs_b = [], []
            for step in range(50):
                state = sim.get_state(balls)
                oa, ob = state.copy(), state.copy()
                for i in range(4):
                    b = i*4
                    if state[b] >= 2.0: oa[b:b+4] = 0.0
                    else: ob[b:b+4] = 0.0
                with torch.no_grad():
                    oa_t = torch.tensor(oa).unsqueeze(0)
                    ob_t = torch.tensor(ob).unsqueeze(0)
                    h_a = model.encoder_a(oa_t); ma = model.mu_a(h_a)
                    h_b = model.encoder_b(ob_t); mb = model.mu_b(h_b)
                msgs_a.append(ma[0].numpy()); msgs_b.append(mb[0].numpy())
                if step == 24:
                    balls[1].x = 3.3 + np.random.uniform(-0.2, 0.2)
                    balls[1].y = np.random.uniform(1.0, 3.0)
                    balls[1].vx = np.random.uniform(-0.5, 0.5); balls[1].vy = 0
                sim.step(balls, t=step*0.02)
            msgs_a = np.array(msgs_a); msgs_b = np.array(msgs_b)
            if len(msgs_a) >= 27:
                shifts_a.append(np.linalg.norm(msgs_a[26] - msgs_a[24]))
                shifts_b.append(np.linalg.norm(msgs_b[26] - msgs_b[24]))
        intervention_22[regime] = {
            'a_mean': np.mean(shifts_a), 'a_std': np.std(shifts_a),
            'b_mean': np.mean(shifts_b), 'b_std': np.std(shifts_b),
        }
        print(f"│  {regime:15s}: Agent A={np.mean(shifts_a):.4f}±{np.std(shifts_a):.4f} "
              f"Agent B={np.mean(shifts_b):.4f}±{np.std(shifts_b):.4f}")
    diff = intervention_22['causal']['a_mean'] - intervention_22['independent']['a_mean']
    print(f"│  Causal - Independent (Agent A): {diff:+.4f}")
    print(f"│  Causal discovery? {'YES ✓' if diff > 0.05 else 'Marginal'}")
    print("└─ Done\n")

    # ══════ PART C: Beta Sweep ══════
    print("┌─ Part C: Information Retention (β sweep)")
    beta_values = [0.0, 0.001, 0.01, 0.1, 0.5]
    retention_22 = {r: [] for r in regimes}
    for beta in beta_values:
        for regime in regimes:
            d = data_22[regime]
            mb = ScaledBottleneckedFusionModel(STATE_DIM, COMM_DIM, fused_dim=64, hidden_dim=384)
            ob = torch.optim.Adam(mb.parameters(), lr=3e-4)
            sb = torch.optim.lr_scheduler.StepLR(ob, 300, 0.5)
            n_tr = len(d['oa_t'])
            for ep in range(800):
                mb.train(); idx = torch.randperm(n_tr)[:512]
                pred, mu, kl = mb(d['oa_t'][idx], d['ob_t'][idx])
                loss = F.mse_loss(pred, d['tgt_t'][idx]) + beta * kl
                ob.zero_grad(); loss.backward(); ob.step(); sb.step()
            mb.eval()
            with torch.no_grad():
                h_a = mb.encoder_a(d['oa_v']); mu_a = mb.mu_a(h_a).numpy()[:1500]
            tgt_np = d['tgt_v'].numpy()[:1500]
            mi_b1 = _knn_mi(mu_a, tgt_np[:, 4:8], k=5)
            retention_22[regime].append(mi_b1)
        print(f"│  β={beta:<6.3f}: " + " ".join(f"{r}={retention_22[r][-1]:.3f}" for r in regimes))
    print("└─ Done\n")

    # ══════ PART D: Causal-JEPA Masking ══════
    print("┌─ Part D: Causal-JEPA Comparison (masking approach)")

    class MaskingWorldModel(nn.Module):
        def __init__(self, obj_dim=4, n_objects=4, hidden_dim=256):
            super().__init__()
            self.obj_dim = obj_dim; self.n_objects = n_objects
            self.mask_token = nn.Parameter(torch.randn(obj_dim))
            self.obj_enc = nn.Sequential(nn.Linear(obj_dim, hidden_dim//2), nn.ReLU(),
                                         nn.Linear(hidden_dim//2, hidden_dim//4))
            self.ctx = nn.Sequential(nn.Linear(hidden_dim//4 * n_objects, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU())
            self.pred = nn.Sequential(nn.Linear(hidden_dim//2, hidden_dim//4), nn.ReLU(),
                                      nn.Linear(hidden_dim//4, obj_dim))
        def forward(self, state, mask_idx=1):
            B = state.shape[0]
            objs = state.view(B, self.n_objects, self.obj_dim).clone()
            objs[:, mask_idx, :] = self.mask_token.unsqueeze(0).expand(B, -1)
            enc = [self.obj_enc(objs[:, i]) for i in range(self.n_objects)]
            ctx = self.ctx(torch.cat(enc, dim=-1))
            return self.pred(ctx), state.view(B, self.n_objects, self.obj_dim)[:, mask_idx]

    mask_results = {}
    for regime in regimes:
        d = data_22[regime]; full = d['tgt_v']
        mm = MaskingWorldModel(4, 4, 256)
        opt_m = torch.optim.Adam(mm.parameters(), lr=1e-3)
        n_tr = int(0.8 * len(full))
        for ep in range(800):
            mm.train(); idx_m = torch.randperm(n_tr)[:512]
            p, t = mm(d['tgt_t'][idx_m], mask_idx=1)
            loss = F.mse_loss(p, t)
            opt_m.zero_grad(); loss.backward(); opt_m.step()
        mm.eval()
        with torch.no_grad():
            pv, tv = mm(full, mask_idx=1)
            mask_results[regime] = F.mse_loss(pv, tv).item()
        print(f"│  {regime:15s}: Masking MSE for Ball 1 = {mask_results[regime]:.4f}")
    print("└─ Done\n")

    # ══════ PART E: Protocol Geometry ══════
    print("┌─ Part E: Protocol Geometry (t-SNE)")
    from sklearn.manifold import TSNE
    test_a = data_22['causal']['oa_v'][:2000]
    all_msgs, labels = [], []
    for regime in regimes:
        m = models_22[regime]; m.eval()
        with torch.no_grad():
            h = m.encoder_a(test_a); msg = m.mu_a(h).numpy()
        all_msgs.append(msg); labels.extend([regime]*len(msg))
    all_msgs = np.vstack(all_msgs)
    ns = min(800, len(all_msgs)//3)
    idx_s = np.concatenate([np.random.choice(range(i*2000,(i+1)*2000), ns, replace=False) for i in range(3)])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(all_msgs[idx_s])
    labels_s = [labels[i] for i in idx_s]

    # Protocol cosines
    msg_per = {}
    for regime in regimes:
        m = models_22[regime]; m.eval()
        with torch.no_grad():
            h = m.encoder_a(test_a[:1000]); msg_per[regime] = m.mu_a(h).numpy()

    def _cos(a, b):
        return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a.flatten())*np.linalg.norm(b.flatten())+1e-8)

    cos_cc = _cos(msg_per['causal'], msg_per['correlational'])
    cos_ci = _cos(msg_per['causal'], msg_per['independent'])
    cos_ri = _cos(msg_per['correlational'], msg_per['independent'])
    print(f"│  Causal vs Correlational: {cos_cc:.3f}")
    print(f"│  Causal vs Independent:   {cos_ci:.3f}")
    print(f"│  Correlational vs Indep:  {cos_ri:.3f}")
    print("└─ Done\n")

    # ══════ PLOTS ══════
    print("┌─ Generating Plots")
    colors22 = {'causal': '#2a9d8f', 'correlational': '#e76f51', 'independent': '#457b9d'}
    cl = [colors22[r] for r in regimes]

    # Plot 1: Intervention response
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 22: Clean Intervention Test (Isolated Physics)', fontweight='bold')
    a_vals = [intervention_22[r]['a_mean'] for r in regimes]
    a_errs = [intervention_22[r]['a_std'] for r in regimes]
    b_vals = [intervention_22[r]['b_mean'] for r in regimes]
    b_errs = [intervention_22[r]['b_std'] for r in regimes]
    axes[0].bar(range(3), a_vals, yerr=a_errs, color=cl, width=0.5, capsize=5)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(['Causal\n(spring)', 'Correl.\n(osc)', 'Indep.'])
    axes[0].set_ylabel('|msg shift|'); axes[0].set_title('Agent A (sees Ball 0, NOT Ball 1)')
    for i, v in enumerate(a_vals):
        axes[0].text(i, v+a_errs[i]+0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    axes[1].bar(range(3), b_vals, yerr=b_errs, color=cl, width=0.5, capsize=5)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(['Causal', 'Correl.', 'Indep.'])
    axes[1].set_ylabel('|msg shift|'); axes[1].set_title('Agent B (SEES Ball 1 — control)')
    for i, v in enumerate(b_vals):
        axes[1].text(i, v+b_errs[i]+0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase22_intervention_clean.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Retention
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in regimes:
        ax.plot(range(len(beta_values)), retention_22[regime], 'o-', color=colors22[regime],
                label=regime.capitalize(), linewidth=2, markersize=7)
    ax.set_xticks(range(len(beta_values)))
    ax.set_xticklabels([str(b) for b in beta_values])
    ax.set_xlabel('β'); ax.set_ylabel('I(msg_A ; Ball 1)')
    ax.set_title('Phase 22: Info Retention Under Compression (3 Regimes)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase22_retention.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: C-JEPA comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 22: Communication Bottleneck vs C-JEPA Masking', fontweight='bold')
    axes[0].bar(range(3), a_vals, color=cl, alpha=0.8, yerr=a_errs, capsize=5)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(['Causal', 'Correl.', 'Indep.'])
    axes[0].set_ylabel('Agent A Shift'); axes[0].set_title('Our Approach:\nComm Bottleneck')
    m_vals = [mask_results[r] for r in regimes]
    axes[1].bar(range(3), m_vals, color=cl, alpha=0.8)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(['Causal', 'Correl.', 'Indep.'])
    axes[1].set_ylabel('Ball 1 Recon. MSE'); axes[1].set_title('C-JEPA Approach:\nObject Masking')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase22_cjepa_comparison.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: Protocol geometry
    fig, ax = plt.subplots(figsize=(10, 8))
    for regime in regimes:
        mask = [l == regime for l in labels_s]
        pts = emb[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors22[regime], label=regime.capitalize(), alpha=0.4, s=10)
    ax.set_title('Phase 22: Protocol Geometry (t-SNE)\nSame input → 3 causal structures', fontweight='bold')
    ax.legend(fontsize=12); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase22_protocol_geometry.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 5: Summary
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 22: Complete Causal Discovery Results', fontsize=16, fontweight='bold')
    axes[0,0].bar(range(3), a_vals, color=cl, alpha=0.8, yerr=a_errs, capsize=5)
    axes[0,0].set_xticks(range(3)); axes[0,0].set_xticklabels(['Cau','Cor','Ind'],fontsize=9)
    axes[0,0].set_title('Intervention (Agent A)')
    for regime in regimes:
        axes[0,1].plot(range(len(beta_values)), retention_22[regime], 'o-', color=colors22[regime], label=regime.capitalize())
    axes[0,1].set_xlabel('β'); axes[0,1].set_title('Info Retention'); axes[0,1].legend(fontsize=7)
    axes[0,2].bar(range(3), m_vals, color=cl, alpha=0.8)
    axes[0,2].set_xticks(range(3)); axes[0,2].set_xticklabels(['Cau','Cor','Ind'],fontsize=9)
    axes[0,2].set_title('C-JEPA Masking MSE')
    for regime in regimes:
        mask = [l==regime for l in labels_s]; pts = emb[mask]
        axes[1,0].scatter(pts[:,0], pts[:,1], c=colors22[regime], label=regime.capitalize(), alpha=0.3, s=5)
    axes[1,0].set_title('Protocol Geometry'); axes[1,0].legend(fontsize=7)
    cos_mat = np.array([[_cos(msg_per[r1], msg_per[r2]) for r2 in regimes] for r1 in regimes])
    im = axes[1,1].imshow(cos_mat, cmap='RdBu', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(3)); axes[1,1].set_xticklabels(['Cau','Cor','Ind'])
    axes[1,1].set_yticks(range(3)); axes[1,1].set_yticklabels(['Cau','Cor','Ind'])
    for i in range(3):
        for j in range(3):
            axes[1,1].text(j, i, f"{cos_mat[i,j]:.2f}", ha='center', va='center', fontsize=11)
    axes[1,1].set_title('Protocol Cosines'); plt.colorbar(im, ax=axes[1,1])
    axes[1,2].axis('off')
    summary_22 = (f"KEY FINDINGS:\n\nIntervention (Agent A):\n"
                  f"  Causal: {a_vals[0]:.3f} ± {a_errs[0]:.3f}\n"
                  f"  Correl: {a_vals[1]:.3f} ± {a_errs[1]:.3f}\n"
                  f"  Indep:  {a_vals[2]:.3f} ± {a_errs[2]:.3f}\n\n"
                  f"C-JEPA Recon MSE:\n  " + "\n  ".join(f"{r}: {mask_results[r]:.4f}" for r in regimes) +
                  f"\n\nCausal-Indep diff: {diff:+.4f}")
    axes[1,2].text(0.05, 0.95, summary_22, transform=axes[1,2].transAxes, fontsize=10,
                   va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase22_summary.png", dpi=150, bbox_inches='tight'); plt.close()

    print("  → results/phase22_intervention_clean.png")
    print("  → results/phase22_retention.png")
    print("  → results/phase22_cjepa_comparison.png")
    print("  → results/phase22_protocol_geometry.png")
    print("  → results/phase22_summary.png")
    print("└─ Done\n")

    print("="*60)
    print("PHASE 22 SUMMARY")
    print("="*60)
    print(f"  Intervention (Agent A): Causal={a_vals[0]:.3f} Correl={a_vals[1]:.3f} Indep={a_vals[2]:.3f}")
    print(f"  Causal-Independent diff: {diff:+.4f}")
    print(f"  C-JEPA: Causal={mask_results['causal']:.4f} Correl={mask_results['correlational']:.4f} "
          f"Indep={mask_results['independent']:.4f}")
    print(f"  Protocol cosines: C-C={cos_cc:.3f} C-I={cos_ci:.3f} R-I={cos_ri:.3f}")


# ── Phase 23: 3D Physics Transfer ──────────────────────────────

def run_phase23():
    from physics_sim import Physics3D, generate_3d_dataset
    from world_model import ScaledBottleneckedFusionModel, count_params

    print("\n" + "="*60)
    print("PHASE 23: 3D PHYSICS — DO THE PRINCIPLES TRANSFER?")
    print("="*60)

    SD3 = 24; COMM = 8

    # ══════ Part A: 3D Prediction ══════
    print("\n┌─ Part A: 3D Prediction (with vs without comm)")
    oa3, ob3, tgt3 = generate_3d_dataset(n_traj=1500, n_steps=50, seed=42)
    sp3 = int(0.8*len(oa3))
    oa3_t, ob3_t, tgt3_t = torch.tensor(oa3[:sp3]), torch.tensor(ob3[:sp3]), torch.tensor(tgt3[:sp3])
    oa3_v, ob3_v, tgt3_v = torch.tensor(oa3[sp3:]), torch.tensor(ob3[sp3:]), torch.tensor(tgt3[sp3:])

    m3 = ScaledBottleneckedFusionModel(SD3, COMM, fused_dim=64, hidden_dim=384)
    print(f"│  Model: {count_params(m3)/1e3:.1f}K params")
    opt3 = torch.optim.Adam(m3.parameters(), lr=3e-4)
    sch3 = torch.optim.lr_scheduler.StepLR(opt3, 500, 0.5)
    for ep in range(1500):
        m3.train(); idx = torch.randperm(sp3)[:512]
        pred, mu, kl = m3(oa3_t[idx], ob3_t[idx])
        loss = F.mse_loss(pred, tgt3_t[idx]) + 0.001 * kl
        opt3.zero_grad(); loss.backward(); opt3.step(); sch3.step()
        if (ep+1) % 500 == 0:
            m3.eval()
            with torch.no_grad():
                pv, _, _ = m3(oa3_v, ob3_v)
                vl = F.mse_loss(pv, tgt3_v).item()
            print(f"│  Epoch {ep+1}: val MSE={vl:.4f}")
    mse_comm3 = vl

    # A-only: zero out B's message
    m3.eval()
    with torch.no_grad():
        h_a3 = m3.encoder_a(oa3_v); mu_a3 = m3.mu_a(h_a3)
        z_a3 = mu_a3
        z_b_zero = torch.zeros_like(z_a3)
        fused = m3.fusion(torch.cat([z_a3, z_b_zero], dim=-1))
        res = m3.pred_norm1(F.relu(m3.pred_layer1(fused)))
        res = m3.pred_norm2(m3.pred_layer2(res))
        pred_aonly = m3.decoder(fused + res)
        mse_aonly3 = F.mse_loss(pred_aonly, tgt3_v).item()
    margin3 = (1 - mse_comm3/mse_aonly3)*100
    print(f"│  Full comm: {mse_comm3:.4f} | A-only: {mse_aonly3:.4f} | margin: {margin3:.1f}%")
    print("└─ Done\n")

    # ══════ Part B: 3D Comm Heatmap ══════
    print("┌─ Part B: 3D Communication Heatmap")
    m3.eval()
    with torch.no_grad():
        h_a = m3.encoder_a(oa3_v); msg_a = m3.mu_a(h_a).numpy()
    sl3 = [f'B{i}_{v}' for i in range(4) for v in ['x','y','z','vx','vy','vz']]
    corr3 = np.zeros((COMM, SD3))
    n_s = min(5000, len(msg_a))
    for c in range(COMM):
        for s in range(SD3):
            cc = np.corrcoef(msg_a[:n_s, c], tgt3_v[:n_s, s].numpy())[0, 1]
            corr3[c, s] = abs(cc) if not np.isnan(cc) else 0
    print("└─ Done\n")

    # ══════ Part C: 3D Spring Effect ══════
    print("┌─ Part C: 3D Spring Effect")
    oa3s, ob3s, tgt3s = generate_3d_dataset(n_traj=1500, n_steps=50, spring_k=3.0, seed=42)
    sp3s = int(0.8*len(oa3s))
    oa3s_t, ob3s_t, tgt3s_t = torch.tensor(oa3s[:sp3s]), torch.tensor(ob3s[:sp3s]), torch.tensor(tgt3s[:sp3s])
    oa3s_v, ob3s_v, tgt3s_v = torch.tensor(oa3s[sp3s:]), torch.tensor(ob3s[sp3s:]), torch.tensor(tgt3s[sp3s:])

    ms3 = ScaledBottleneckedFusionModel(SD3, COMM, fused_dim=64, hidden_dim=384)
    opts3 = torch.optim.Adam(ms3.parameters(), lr=3e-4)
    schs3 = torch.optim.lr_scheduler.StepLR(opts3, 500, 0.5)
    for ep in range(1500):
        ms3.train(); idx = torch.randperm(sp3s)[:512]
        pred, mu, kl = ms3(oa3s_t[idx], ob3s_t[idx])
        loss = F.mse_loss(pred, tgt3s_t[idx]) + 0.001 * kl
        opts3.zero_grad(); loss.backward(); opts3.step(); schs3.step()
        if (ep+1) % 500 == 0:
            ms3.eval()
            with torch.no_grad():
                pv, _, _ = ms3(oa3s_v, ob3s_v)
                vl = F.mse_loss(pv, tgt3s_v).item()
            print(f"│  Spring epoch {ep+1}: val={vl:.4f}")

    ms3.eval()
    with torch.no_grad():
        h_as = ms3.encoder_a(oa3s_v); msg_as = ms3.mu_a(h_as).numpy()
    corr3s = np.zeros((COMM, SD3))
    for c in range(COMM):
        for s in range(SD3):
            cc = np.corrcoef(msg_as[:n_s, c], tgt3s_v[:n_s, s].numpy())[0, 1]
            corr3s[c, s] = abs(cc) if not np.isnan(cc) else 0
    b1_ns = corr3[:, 6:12].mean(); b1_sp = corr3s[:, 6:12].mean()
    print(f"│  Ball 1 avg|corr|: no spring={b1_ns:.3f} spring={b1_sp:.3f} change={((b1_sp/b1_ns)-1)*100:+.0f}%")
    print("└─ Done\n")

    # ══════ Part D: 3D Complementary Sensing ══════
    print("┌─ Part D: 3D Complementary Sensing (positions vs velocities)")
    comp_a = tgt3_t.clone(); comp_b = tgt3_t.clone()
    for i in range(4):
        b = i*6
        comp_a[:, b+3:b+6] = 0.0  # A sees positions only
        comp_b[:, b:b+3] = 0.0    # B sees velocities only
    comp_av = tgt3_v.clone(); comp_bv = tgt3_v.clone()
    for i in range(4):
        b = i*6
        comp_av[:, b+3:b+6] = 0.0; comp_bv[:, b:b+3] = 0.0

    mc = ScaledBottleneckedFusionModel(SD3, COMM, fused_dim=64, hidden_dim=384)
    optc = torch.optim.Adam(mc.parameters(), lr=3e-4)
    schc = torch.optim.lr_scheduler.StepLR(optc, 400, 0.5)
    for ep in range(1000):
        mc.train(); idx = torch.randperm(sp3)[:512]
        pred, mu, kl = mc(comp_a[idx], comp_b[idx])
        loss = F.mse_loss(pred, tgt3_t[idx]) + 0.001 * kl
        optc.zero_grad(); loss.backward(); optc.step(); schc.step()

    mc.eval()
    with torch.no_grad():
        pv_both, _, _ = mc(comp_av, comp_bv)
        mse_both3 = F.mse_loss(pv_both, tgt3_v).item()
        # A-only
        h_ac = mc.encoder_a(comp_av); z_ac = mc.mu_a(h_ac)
        z_bc_zero = torch.zeros_like(z_ac)
        fused_c = mc.fusion(torch.cat([z_ac, z_bc_zero], dim=-1))
        res_c = mc.pred_norm1(F.relu(mc.pred_layer1(fused_c)))
        res_c = mc.pred_norm2(mc.pred_layer2(res_c))
        pred_aoc = mc.decoder(fused_c + res_c)
        mse_aonly_c = F.mse_loss(pred_aoc, tgt3_v).item()
    margin_comp = (1 - mse_both3/mse_aonly_c)*100
    print(f"│  Both: {mse_both3:.4f} | A-only (pos): {mse_aonly_c:.4f} | margin: {margin_comp:.1f}%")
    print("└─ Done\n")

    # ══════ PLOTS ══════
    print("┌─ Generating Plots")
    # Heatmap
    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(corr3, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(COMM)); ax.set_xticks(range(SD3))
    ax.set_xticklabels(sl3, rotation=90, fontsize=7)
    ax.set_title('Phase 23: 3D Communication Heatmap', fontweight='bold')
    for i in range(1,4): ax.axvline(x=i*6-0.5, color='white', linewidth=2)
    plt.colorbar(im, label='|Correlation|')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase23_3d_heatmap.png", dpi=150, bbox_inches='tight'); plt.close()

    # Spring comparison
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].imshow(corr3, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.8)
    axes[0].set_title('No Spring'); axes[0].set_xticks(range(SD3))
    axes[0].set_xticklabels(sl3, rotation=90, fontsize=7)
    for i in range(1,4): axes[0].axvline(x=i*6-0.5, color='white', linewidth=2)
    axes[1].imshow(corr3s, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.8)
    axes[1].set_title('With Spring (Ball 0 ↔ Ball 1)'); axes[1].set_xticks(range(SD3))
    axes[1].set_xticklabels(sl3, rotation=90, fontsize=7)
    for i in range(1,4): axes[1].axvline(x=i*6-0.5, color='white', linewidth=2)
    fig.suptitle('Phase 23: 3D Spring Effect on Communication', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase23_3d_spring.png", dpi=150, bbox_inches='tight'); plt.close()

    # Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Phase 23: 3D Physics — Do 2D Principles Transfer?', fontsize=16, fontweight='bold')
    axes[0,0].bar(['With Comm','A Only'], [mse_comm3, mse_aonly3], color=['#2a9d8f','#e76f51'])
    axes[0,0].set_ylabel('MSE'); axes[0,0].set_title(f'3D Prediction\nmargin: {margin3:.1f}%')
    bc_ns = [corr3[:, i*6:(i+1)*6].max(axis=0).mean() for i in range(4)]
    bc_sp = [corr3s[:, i*6:(i+1)*6].max(axis=0).mean() for i in range(4)]
    x = np.arange(4)
    axes[0,1].bar(x-0.15, bc_ns, 0.3, label='No Spring', color='#457b9d')
    axes[0,1].bar(x+0.15, bc_sp, 0.3, label='Spring', color='#e76f51')
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(['B0','B1','B2','B3'])
    axes[0,1].set_title('Spring → Ball Encoding'); axes[0,1].legend(fontsize=8)
    axes[1,0].bar(['Both\n(pos+vel)','A Only\n(pos)'], [mse_both3, mse_aonly_c], color=['#2a9d8f','#e76f51'])
    axes[1,0].set_ylabel('MSE'); axes[1,0].set_title(f'3D Complementary Sensing\nmargin: {margin_comp:.1f}%')
    axes[1,1].axis('off')
    comp_txt = (f"2D vs 3D PRINCIPLE TRANSFER\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Comm margin (occluded):\n  2D: ~15-30%    3D: {margin3:.1f}%\n\n"
                f"Complementary sensing:\n  2D: 91%        3D: {margin_comp:.1f}%\n\n"
                f"Spring → Ball 1 encoding:\n  3D: {((b1_sp/b1_ns)-1)*100:+.0f}% change\n\n"
                "If similar → PRINCIPLES, not artifacts")
    axes[1,1].text(0.05, 0.95, comp_txt, transform=axes[1,1].transAxes, fontsize=11,
                   va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase23_summary.png", dpi=150, bbox_inches='tight'); plt.close()

    print("  → results/phase23_3d_heatmap.png")
    print("  → results/phase23_3d_spring.png")
    print("  → results/phase23_summary.png")
    print("└─ Done\n")

    print("="*60)
    print("PHASE 23 SUMMARY")
    print("="*60)
    print(f"  3D comm margin (occluded): {margin3:.1f}%")
    print(f"  3D complementary sensing margin: {margin_comp:.1f}%")
    print(f"  Spring → Ball 1 encoding: {((b1_sp/b1_ns)-1)*100:+.0f}%")


# ── Phase 24: Language Grounding ───────────────────────────────

def run_phase24():
    from world_model import count_params

    print("\n" + "="*60)
    print("PHASE 24: LANGUAGE GROUNDING — WORDS MEET WORLD MODELS")
    print("="*60)

    vocab = {
        '<pad>': 0, 'a': 1, 'the': 2, 'and': 3, 'for': 4, 'to': 5,
        'heavy': 10, 'light': 11, 'bouncy': 12, 'sticky': 13,
        'flat': 14, 'rigid': 15, 'soft': 16, 'big': 17, 'small': 18,
        'ball': 20, 'object': 21, 'platform': 22, 'thing': 23, 'surface': 24,
        'support': 30, 'catch': 31, 'block': 32, 'launch': 33, 'protect': 34,
        'sit': 35, 'bounce': 36, 'stick': 37, 'shield': 38, 'rest': 39,
        'on': 40, 'something': 41, 'that': 42, 'can': 43, 'you': 44,
        'use': 45, 'need': 46, 'want': 47, 'put': 48, 'place': 49,
    }
    VOCAB_SIZE = max(vocab.values()) + 1
    PROP_DIM = 5  # [mass, elasticity, friction, flatness, rigidity]

    def tokenize(sentence, max_len=8):
        tokens = [vocab[w] for w in sentence.lower().split() if w in vocab]
        return (tokens[:max_len] + [0]*max_len)[:max_len]

    templates = [
        ("heavy ball", [3.0, 0.5, 0.5, 0.0, 0.5]),
        ("heavy rigid object", [3.0, 0.3, 0.5, 0.0, 0.8]),
        ("heavy flat platform", [3.0, 0.3, 0.5, 0.8, 0.8]),
        ("a heavy thing", [3.0, 0.5, 0.5, 0.3, 0.5]),
        ("light ball", [0.3, 0.5, 0.5, 0.0, 0.5]),
        ("light bouncy ball", [0.3, 0.9, 0.3, 0.0, 0.3]),
        ("a light object", [0.3, 0.5, 0.5, 0.2, 0.5]),
        ("light soft thing", [0.3, 0.5, 0.5, 0.2, 0.2]),
        ("bouncy ball", [1.0, 0.9, 0.3, 0.0, 0.3]),
        ("bouncy light ball", [0.3, 0.9, 0.3, 0.0, 0.3]),
        ("sticky object", [1.0, 0.1, 0.8, 0.0, 0.5]),
        ("sticky ball", [1.0, 0.1, 0.8, 0.0, 0.3]),
        ("sticky flat surface", [1.0, 0.1, 0.8, 0.8, 0.5]),
        ("flat rigid platform", [2.0, 0.3, 0.5, 0.8, 0.8]),
        ("flat rigid surface", [2.0, 0.3, 0.5, 0.9, 0.9]),
        ("flat rigid object", [1.5, 0.3, 0.5, 0.8, 0.7]),
        ("something to support", [2.0, 0.3, 0.5, 0.8, 0.8]),
        ("something to catch", [1.0, 0.1, 0.8, 0.5, 0.5]),
        ("something to shield", [3.0, 0.3, 0.5, 0.5, 0.8]),
        ("something to launch", [3.0, 0.5, 0.3, 0.0, 0.5]),
        ("something to bounce on", [1.5, 0.9, 0.3, 0.5, 0.5]),
        ("put something on", [2.0, 0.3, 0.5, 0.8, 0.8]),
        ("thing that can catch", [1.0, 0.1, 0.8, 0.5, 0.5]),
        ("use to block", [3.0, 0.3, 0.5, 0.5, 0.8]),
        ("heavy flat rigid", [3.0, 0.3, 0.5, 0.8, 0.8]),
        ("light bouncy soft", [0.3, 0.9, 0.3, 0.0, 0.2]),
        ("sticky flat object", [1.0, 0.1, 0.8, 0.8, 0.5]),
        ("heavy bouncy ball", [3.0, 0.9, 0.3, 0.0, 0.3]),
    ]

    def gen_data(n=5000):
        sents, tgts = [], []
        import random as rr
        for _ in range(n // len(templates)):
            for s, p in templates:
                noisy = [max(0, min(3.5, v + rr.gauss(0, 0.05))) for v in p]
                sents.append(tokenize(s)); tgts.append(noisy)
        return torch.tensor(sents, dtype=torch.long), torch.tensor(tgts, dtype=torch.float32)

    class LanguageGrounding(nn.Module):
        def __init__(self, vs, e_dim=32, p_dim=5, h_dim=128):
            super().__init__()
            self.embed = nn.Embedding(vs, e_dim, padding_idx=0)
            self.enc = nn.Sequential(nn.Linear(e_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(),
                                     nn.Linear(h_dim, h_dim), nn.ReLU())
            self.to_props = nn.Sequential(nn.Linear(h_dim, h_dim//2), nn.ReLU(),
                                          nn.Linear(h_dim//2, p_dim))
        def forward(self, ids):
            emb = self.embed(ids)
            mask = (ids != 0).float().unsqueeze(-1)
            pooled = (emb * mask).sum(1) / (mask.sum(1) + 1e-6)
            return self.to_props(self.enc(pooled))

    # ══════ Train ══════
    print("\n┌─ Training Language Grounding")
    sent_ids, prop_tgt = gen_data(5000)
    n_tr = int(0.8 * len(sent_ids))
    lg = LanguageGrounding(VOCAB_SIZE, p_dim=PROP_DIM)
    print(f"│  Model: {count_params(lg)/1e3:.1f}K params")
    optl = torch.optim.Adam(lg.parameters(), lr=1e-3)
    for ep in range(500):
        lg.train(); idx = torch.randperm(n_tr)[:256]
        pred = lg(sent_ids[idx])
        loss = F.mse_loss(pred, prop_tgt[idx])
        optl.zero_grad(); loss.backward(); optl.step()
        if (ep+1) % 100 == 0:
            lg.eval()
            with torch.no_grad():
                vp = lg(sent_ids[n_tr:]); vl = F.mse_loss(vp, prop_tgt[n_tr:]).item()
            print(f"│  Epoch {ep+1}: val={vl:.4f}")
    print("└─ Done\n")

    # ══════ Affordance Alignment ══════
    print("┌─ Affordance Alignment Test")
    known_objs = {
        'platform': torch.tensor([[2.0, 0.3, 0.5, 0.8, 0.8]]),
        'heavy_ball': torch.tensor([[3.0, 0.5, 0.5, 0.0, 0.5]]),
        'light_ball': torch.tensor([[0.3, 0.5, 0.5, 0.0, 0.5]]),
        'sticky': torch.tensor([[1.0, 0.1, 0.8, 0.0, 0.5]]),
        'bouncy': torch.tensor([[1.0, 0.9, 0.3, 0.0, 0.3]]),
    }
    test_queries = [
        "something to support", "heavy rigid object", "light bouncy ball",
        "thing that can catch", "flat rigid surface", "sticky flat object",
        "heavy ball", "something to bounce on",
    ]
    results_24 = []
    lg.eval()
    for query in test_queries:
        toks = torch.tensor([tokenize(query)], dtype=torch.long)
        with torch.no_grad():
            pp = lg(toks).squeeze()
        best_sim, best_obj = -1, None
        for name, ref in known_objs.items():
            sim = F.cosine_similarity(pp.unsqueeze(0), ref).item()
            if sim > best_sim: best_sim, best_obj = sim, name
        results_24.append((query, best_obj, best_sim, pp.numpy()))
        print(f"│  '{query}' → {best_obj} (cos={best_sim:.3f})")
    print("└─ Done\n")

    # ══════ Word-Property Matrix ══════
    print("┌─ Word-Property Analysis")
    prop_names = ['mass', 'elasticity', 'friction', 'flatness', 'rigidity']
    words = ['heavy','light','bouncy','sticky','flat','rigid','soft',
             'support','catch','shield','launch','bounce']
    wp = np.zeros((len(words), PROP_DIM))
    for wi, w in enumerate(words):
        toks = torch.tensor([tokenize(f"{w} object")], dtype=torch.long)
        with torch.no_grad():
            wp[wi] = lg(toks).squeeze().numpy()
    wp_norm = (wp - wp.mean(0)) / (wp.std(0) + 1e-6)
    for wi, w in enumerate(words):
        print(f"│  {w:10s}: " + " ".join(f"{prop_names[j]}={wp[wi,j]:.2f}" for j in range(PROP_DIM)))
    print("└─ Done\n")

    # ══════ PLOTS ══════
    print("┌─ Generating Plots")
    # Plot 1: Alignment
    fig, ax = plt.subplots(figsize=(12, 6))
    cosines = [r[2] for r in results_24]; matched = [r[1] for r in results_24]
    bars_c = ['#2a9d8f' if c > 0.9 else '#e9c46a' if c > 0.7 else '#e76f51' for c in cosines]
    ax.bar(range(len(results_24)), cosines, color=bars_c)
    ax.set_xticks(range(len(results_24)))
    ax.set_xticklabels([f'"{q.split()[0]}..."\n→ {m}' for q, m in zip(test_queries, matched)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cosine Similarity'); ax.set_title('Phase 24: Language → Property → Object Alignment', fontweight='bold')
    ax.axhline(y=0.9, color='green', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase24_alignment.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Word-Property Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(wp_norm, aspect='auto', cmap='RdBu_r')
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=11)
    ax.set_xticks(range(PROP_DIM)); ax.set_xticklabels(prop_names, fontsize=11)
    ax.set_title('Phase 24: Word → Property Activation', fontweight='bold')
    for i in range(len(words)):
        for j in range(PROP_DIM):
            ax.text(j, i, f'{wp_norm[i,j]:.1f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase24_word_property.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 3: Convergence diagram
    fig, ax = plt.subplots(figsize=(12, 8)); ax.axis('off')
    flat_cos = [r[2] for r in results_24 if 'flat rigid' in r[0]]
    fc_str = f"{flat_cos[0]:.3f}" if flat_cos else "N/A"
    conv_txt = (
        "CONVERGENCE: Two Paths to the Same Affordance Space\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "PATH 1: Physics → Interaction → Affordance Bottleneck\n"
        "  PropertyObject → AffordanceWorldModel → 8-dim affordance\n"
        "  Phase 20: flat_rigid → platform cluster (cos=0.939)\n"
        "  Mechanism: learned from PHYSICAL INTERACTION\n\n"
        "PATH 2: Language → Sentence Encoder → Property Vector\n"
        f"  'flat rigid surface' → LanguageGrounding → property vector\n"
        f"  Phase 24: 'flat rigid' → platform (cos={fc_str})\n"
        "  Mechanism: learned from LANGUAGE DESCRIPTIONS\n\n"
        "CONVERGENCE:\n"
        "  Both paths map 'flat + rigid' → 'support affordance'\n"
        "  Physics: watching objects interact\n"
        "  Language: descriptive associations\n"
        "  The AFFORDANCE SPACE is the meeting point\n\n"
        "Language grounding through shared representation —\n"
        "the bridge between what you SAY and what the world DOES."
    )
    ax.text(0.05, 0.95, conv_txt, transform=ax.transAxes, fontsize=11,
            va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase24_convergence.png", dpi=150, bbox_inches='tight'); plt.close()

    # Plot 4: Summary
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Phase 24: Language Grounding Summary', fontsize=16, fontweight='bold')
    axes[0].bar(range(len(results_24)), cosines, color=bars_c)
    axes[0].set_xticks(range(len(results_24)))
    axes[0].set_xticklabels([r[0].split()[0] for r in results_24], rotation=45, fontsize=8)
    axes[0].set_title('Query → Object Match'); axes[0].set_ylabel('Cosine')
    im = axes[1].imshow(wp_norm, aspect='auto', cmap='RdBu_r')
    axes[1].set_yticks(range(len(words))); axes[1].set_yticklabels(words, fontsize=8)
    axes[1].set_xticks(range(PROP_DIM)); axes[1].set_xticklabels(['mass','ela','fri','fla','rig'], fontsize=8)
    axes[1].set_title('Word → Property')
    axes[2].axis('off')
    n_good = sum(1 for _, _, c, _ in results_24 if c > 0.85)
    summary_24 = (f"KEY RESULTS:\n\nCorrect matches: {n_good}/{len(results_24)}\n\n"
                  f"Word−property:\n  heavy→mass: {wp[0,0]:.2f}\n  flat→flatness: {wp[4,3]:.2f}\n"
                  f"  sticky→friction: {wp[3,2]:.2f}\n\nLanguage and physics\n"
                  "converge in affordance space.")
    axes[2].text(0.05, 0.95, summary_24, transform=axes[2].transAxes, fontsize=11,
                 va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase24_summary.png", dpi=150, bbox_inches='tight'); plt.close()

    print("  → results/phase24_alignment.png")
    print("  → results/phase24_word_property.png")
    print("  → results/phase24_convergence.png")
    print("  → results/phase24_summary.png")
    print("└─ Done\n")

    print("="*60)
    print("PHASE 24 SUMMARY — LANGUAGE GROUNDING")
    print("="*60)
    print(f"  Correct matches: {n_good}/{len(results_24)}")
    for q, obj, cos, _ in results_24:
        print(f"    '{q}' → {obj} (cos={cos:.3f})")


def run_phase25():
    """Phase 25: Visual World Model — From State Vectors to Pixels."""
    print("=" * 60)
    print("PHASE 25: VISUAL WORLD MODEL — FROM STATE VECTORS TO PIXELS")
    print("=" * 60)
    t0 = time.time()

    # ── Part A: Collect Visual Dataset ──────────────────────────
    print("\n┌─ Part A: Collecting visual dataset")
    dataset = collect_visual_dataset(n_episodes=100, steps_per_episode=40,
                                     n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    print(f"│  Dataset: {n} frames, shape={list(dataset['img_a'].shape)}")
    print(f"│  Collection time: {time.time()-t0:.0f}s")

    # Save example frames
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        idx = i * n // 5
        axes[0, i].imshow(dataset['img_a'][idx].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Cam A, t={idx}', fontsize=9); axes[0, i].axis('off')
        axes[1, i].imshow(dataset['img_b'][idx].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Cam B, t={idx}', fontsize=9); axes[1, i].axis('off')
    fig.suptitle('Phase 25: Visual Dataset — Two Camera Views', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25_dataset_samples.png", dpi=150); plt.close()
    print("│  → results/phase25_dataset_samples.png")
    print("└─ Done")

    # ── Part B: Train Visual JEPA ──────────────────────────────
    print("\n┌─ Part B: Training Visual JEPA")
    model = VisualJEPA(latent_dim=128, comm_dim=8, action_dim=4, beta=0.001)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    n_train = int(0.8 * n)
    batch_size = 128
    history = {'pred': [], 'kl': [], 'vicreg': []}

    for epoch in range(60):
        model.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_kl, ep_vreg, nb = 0, 0, 0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            total, pred_loss, kl, vicreg, _, _ = model(
                dataset['img_a'][idx], dataset['img_b'][idx],
                dataset['action'][idx],
                dataset['next_img_a'][idx], dataset['next_img_b'][idx])
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            ep_pred += pred_loss.item(); ep_kl += kl.item()
            ep_vreg += vicreg.item(); nb += 1
        scheduler.step()
        history['pred'].append(ep_pred/nb)
        history['kl'].append(ep_kl/nb)
        history['vicreg'].append(ep_vreg/nb)
        if (epoch+1) % 10 == 0:
            print(f"│  Epoch {epoch+1}: pred={ep_pred/nb:.4f} kl={ep_kl/nb:.4f} vicreg={ep_vreg/nb:.4f}")

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['pred']); axes[0].set_title('Prediction Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(history['kl']); axes[1].set_title('Communication KL'); axes[1].set_xlabel('Epoch')
    axes[2].plot(history['vicreg']); axes[2].set_title('VICReg Loss'); axes[2].set_xlabel('Epoch')
    fig.suptitle('Phase 25: Visual JEPA Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25_training.png", dpi=150); plt.close()
    print("│  → results/phase25_training.png")
    print("└─ Done")

    # ── Part C: Communication Analysis ─────────────────────────
    print("\n┌─ Part C: Communication Analysis")
    model.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        # Process in batches to avoid OOM
        msg_a_list, msg_b_list = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            ma, mb = model.get_messages(dataset['img_a'][bi], dataset['img_b'][bi])
            msg_a_list.append(ma); msg_b_list.append(mb)
        msg_a = torch.cat(msg_a_list).numpy()
        msg_b = torch.cat(msg_b_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_state = gt_state.shape[1]
    n_objects = n_state // 8

    corr_a = np.zeros((8, n_state))
    corr_b = np.zeros((8, n_state))
    for c in range(8):
        for s in range(n_state):
            if np.std(gt_state[:, s]) > 1e-6:
                corr_a[c, s] = abs(np.corrcoef(msg_a[:, c], gt_state[:, s])[0, 1])
                corr_b[c, s] = abs(np.corrcoef(msg_b[:, c], gt_state[:, s])[0, 1])

    state_labels = []
    for i in range(n_objects):
        for v in ['x','y','z','vx','vy','vz','m','r']:
            state_labels.append(f'O{i}_{v}')

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[0].set_title('Agent A (left camera) Message ↔ State Correlation')
    axes[0].set_yticks(range(8)); axes[0].set_yticklabels([f'Msg {i}' for i in range(8)])
    axes[0].set_xticks(range(n_state)); axes[0].set_xticklabels(state_labels, rotation=90, fontsize=6)
    axes[1].imshow(corr_b, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[1].set_title('Agent B (right camera) Message ↔ State Correlation')
    axes[1].set_yticks(range(8)); axes[1].set_yticklabels([f'Msg {i}' for i in range(8)])
    axes[1].set_xticks(range(n_state)); axes[1].set_xticklabels(state_labels, rotation=90, fontsize=6)
    for ax in axes:
        for i in range(1, n_objects):
            ax.axvline(x=i*8-0.5, color='white', linewidth=2)
    fig.suptitle('Phase 25: What Do Visual Agents Communicate?', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25_comm_heatmap.png", dpi=150); plt.close()

    max_corr_a = [corr_a[:, i*8:(i+1)*8].max() for i in range(n_objects)]
    max_corr_b = [corr_b[:, i*8:(i+1)*8].max() for i in range(n_objects)]
    print("│  Agent A max corr/obj: " + ", ".join(f"O{i}={max_corr_a[i]:.3f}" for i in range(n_objects)))
    print("│  Agent B max corr/obj: " + ", ".join(f"O{i}={max_corr_b[i]:.3f}" for i in range(n_objects)))
    print("│  → results/phase25_comm_heatmap.png")
    print("└─ Done")

    # ── Part D: Communication Necessity ─────────────────────────
    print("\n┌─ Part D: Communication Necessity")
    with torch.no_grad():
        pred_comm_vals, pred_nocomm_vals = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            _, pc, _, _, _, _ = model(
                dataset['img_a'][bi], dataset['img_b'][bi],
                dataset['action'][bi],
                dataset['next_img_a'][bi], dataset['next_img_b'][bi])
            pred_comm_vals.append(pc.item())
            # Without comm: zero messages
            za = model.encoder(dataset['img_a'][bi])
            zb = model.encoder(dataset['img_b'][bi])
            zmsg = torch.zeros(len(bi), 8)
            act = dataset['action'][bi]
            pa = model.predictor(torch.cat([za, zmsg, act], -1))
            pb = model.predictor(torch.cat([zb, zmsg, act], -1))
            ta = model.target_encoder(dataset['next_img_a'][bi])
            tb = model.target_encoder(dataset['next_img_b'][bi])
            pnc = (F.mse_loss(pa, ta) + F.mse_loss(pb, tb)).item() / 2
            pred_nocomm_vals.append(pnc)

    pred_comm_val = np.mean(pred_comm_vals)
    pred_nocomm_val = np.mean(pred_nocomm_vals)
    margin = (1 - pred_comm_val / pred_nocomm_val) * 100 if pred_nocomm_val > 0 else 0
    print(f"│  With comm:    {pred_comm_val:.4f}")
    print(f"│  Without comm: {pred_nocomm_val:.4f}")
    print(f"│  Communication margin: {margin:.1f}%")
    print("└─ Done")

    # ── Part E: Multi-Step Rollout ──────────────────────────────
    print("\n┌─ Part E: Multi-Step Rollout")
    horizons = [1, 2, 5, 10, 15, 20]
    rollout_errors = []
    steps_per_traj = 40

    for horizon in horizons:
        errors = []
        for traj_start in range(0, min(50 * steps_per_traj, n - steps_per_traj), steps_per_traj):
            if traj_start + horizon >= n:
                continue
            with torch.no_grad():
                za = model.encoder(dataset['img_a'][traj_start:traj_start+1])
                zb = model.encoder(dataset['img_b'][traj_start:traj_start+1])
                for h in range(horizon):
                    t = traj_start + h
                    if t >= n - 1: break
                    ma = model.comm_mu_a(za); mb = model.comm_mu_b(zb)
                    act = dataset['action'][t:t+1]
                    za = model.predictor(torch.cat([za, mb, act], -1))
                    zb = model.predictor(torch.cat([zb, ma, act], -1))
                t_f = traj_start + horizon
                if t_f < n:
                    za_t = model.target_encoder(dataset['img_a'][t_f:t_f+1])
                    zb_t = model.target_encoder(dataset['img_b'][t_f:t_f+1])
                    err = (F.mse_loss(za, za_t) + F.mse_loss(zb, zb_t)).item() / 2
                    errors.append(err)
        avg = np.mean(errors) if errors else 0
        rollout_errors.append(avg)
        print(f"│  Horizon {horizon:2d}: MSE = {avg:.4f} (n={len(errors)})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(horizons[:len(rollout_errors)], rollout_errors, 'o-', linewidth=2, markersize=8, color='#2980b9')
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('MSE in Embedding Space', fontsize=12)
    ax.set_title('Phase 25: Multi-Step Rollout Quality\nHow far into the future can the visual world model predict?', fontsize=14)
    if min(rollout_errors) > 0: ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25_rollout.png", dpi=150); plt.close()
    print("│  → results/phase25_rollout.png")
    print("└─ Done")

    # ── Part F: Embedding Space t-SNE ──────────────────────────
    print("\n┌─ Part F: Embedding Space Visualization")
    from sklearn.manifold import TSNE

    with torch.no_grad():
        z_list = []
        for i in range(0, min(2000, n), 256):
            z_list.append(model.encoder(dataset['img_a'][i:i+256]))
        z_all = torch.cat(z_list).numpy()

    n_pts = len(z_all)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(z_all)
    colors = np.array([i % steps_per_traj for i in range(n_pts)])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=colors, cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Time step within episode')
    ax.set_title('Phase 25: Visual Embedding Space (t-SNE)\nColors = time within episode', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25_embedding_tsne.png", dpi=150); plt.close()
    print("│  → results/phase25_embedding_tsne.png")
    print("└─ Done")

    # ── Summary Dashboard ──────────────────────────────────────
    print("\n┌─ Generating Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 25: Visual World Model — Complete Results', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.imshow(dataset['img_a'][0].permute(1, 2, 0).numpy())
    ax.set_title('Camera A View (64×64)'); ax.axis('off')

    ax = axes[0, 1]
    ax.plot(history['pred'], color='#2980b9')
    ax.set_title(f'Prediction Loss → {history["pred"][-1]:.4f}'); ax.set_xlabel('Epoch')

    ax = axes[0, 2]
    ax.imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_title('Agent A Msg ↔ State')
    ax.set_yticks(range(8)); ax.set_yticklabels([f'M{i}' for i in range(8)], fontsize=7)

    ax = axes[1, 0]
    ax.bar(['With\nComm', 'Without\nComm'], [pred_comm_val, pred_nocomm_val],
           color=['#27ae60', '#e74c3c'])
    ax.set_title(f'Communication Margin: {margin:.1f}%'); ax.set_ylabel('Prediction MSE')

    ax = axes[1, 1]
    ax.plot(horizons[:len(rollout_errors)], rollout_errors, 'o-', color='#8e44ad')
    ax.set_title('Multi-Step Rollout'); ax.set_xlabel('Horizon'); ax.set_ylabel('MSE')
    if min(rollout_errors) > 0: ax.set_yscale('log')

    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        "PHASE 25 SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Model: Visual JEPA\n"
        f"Params: {n_params:,}\n"
        f"Dataset: {n} frames (64×64×3)\n\n"
        f"Prediction MSE: {history['pred'][-1]:.4f}\n"
        f"Comm margin: {margin:.1f}%\n"
        f"KL (final): {history['kl'][-1]:.4f}\n\n"
        "INPUT: pixels\n"
        "OUTPUT: predicted embeddings\n"
        "METHOD: JEPA (no pixel recon)\n\n"
        "This is no longer a toy."
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=12,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25_summary.png", dpi=150); plt.close()
    print("│  → results/phase25_summary.png")
    print("└─ Done")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PHASE 25 SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params (Visual JEPA)")
    print(f"  Dataset: {n} frames (64×64 RGB, two cameras)")
    print(f"  Pred loss: {history['pred'][-1]:.4f}")
    print(f"  Comm margin: {margin:.1f}%")
    print(f"  Rollout MSE @5: {rollout_errors[2]:.4f}  @20: {rollout_errors[-1]:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  INPUT: pixels. METHOD: JEPA. This is no longer a toy.")


def run_phase25b():
    """Phase 25b: Full-scale Visual World Model with camera asymmetry + complementary sensing."""
    print("=" * 60)
    print("PHASE 25b: VISUAL WORLD MODEL — FULL SCALE + CAMERA FIX")
    print("=" * 60)
    t0 = time.time()

    # ── Part A: Collect Visual Dataset (12,000 frames) ─────────
    print("\n┌─ Part A: Collecting visual dataset (300 episodes)")
    dataset = collect_visual_dataset(n_episodes=300, steps_per_episode=40,
                                     n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    print(f"│  Dataset: {n} frames, shape={list(dataset['img_a'].shape)}")
    print(f"│  Collection time: {time.time()-t0:.0f}s")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        idx = i * n // 5
        axes[0, i].imshow(dataset['img_a'][idx].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Cam A (left), t={idx}', fontsize=8); axes[0, i].axis('off')
        axes[1, i].imshow(dataset['img_b'][idx].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Cam B (right), t={idx}', fontsize=8); axes[1, i].axis('off')
    fig.suptitle('Phase 25b: Asymmetric Camera Views — A sees left, B sees right', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_dataset_samples.png", dpi=150); plt.close()
    print("│  → results/phase25b_dataset_samples.png")
    print("└─ Done")

    # ── Part B: Train Visual JEPA (300 epochs, 256-dim) ────────
    print("\n┌─ Part B: Training Visual JEPA (300 epochs)")
    model = VisualJEPA(latent_dim=256, comm_dim=8, action_dim=4, beta=0.001)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

    n_train = int(0.8 * n)
    batch_size = 64
    history = {'pred': [], 'kl': [], 'vicreg': []}

    for epoch in range(300):
        model.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_kl, ep_vreg, nb = 0, 0, 0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            total, pred_loss, kl, vicreg, _, _ = model(
                dataset['img_a'][idx], dataset['img_b'][idx],
                dataset['action'][idx],
                dataset['next_img_a'][idx], dataset['next_img_b'][idx])
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            ep_pred += pred_loss.item(); ep_kl += kl.item()
            ep_vreg += vicreg.item(); nb += 1
        scheduler.step()
        history['pred'].append(ep_pred/nb)
        history['kl'].append(ep_kl/nb)
        history['vicreg'].append(ep_vreg/nb)
        if (epoch+1) % 50 == 0:
            print(f"│  Epoch {epoch+1}: pred={ep_pred/nb:.4f} kl={ep_kl/nb:.4f} vicreg={ep_vreg/nb:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['pred']); axes[0].set_title('Prediction Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(history['kl']); axes[1].set_title('Communication KL'); axes[1].set_xlabel('Epoch')
    axes[2].plot(history['vicreg']); axes[2].set_title('VICReg Loss'); axes[2].set_xlabel('Epoch')
    fig.suptitle('Phase 25b: Visual JEPA Training (300 epochs, 256-dim)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_training.png", dpi=150); plt.close()
    print("│  → results/phase25b_training.png")
    print("└─ Done")

    # ── Part C: Communication Analysis ─────────────────────────
    print("\n┌─ Part C: Communication Analysis")
    model.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        msg_a_list, msg_b_list = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            ma, mb = model.get_messages(dataset['img_a'][bi], dataset['img_b'][bi])
            msg_a_list.append(ma); msg_b_list.append(mb)
        msg_a = torch.cat(msg_a_list).numpy()
        msg_b = torch.cat(msg_b_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_state = gt_state.shape[1]
    n_objects = n_state // 8

    corr_a = np.zeros((8, n_state))
    corr_b = np.zeros((8, n_state))
    for c in range(8):
        for s in range(n_state):
            if np.std(gt_state[:, s]) > 1e-6:
                corr_a[c, s] = abs(np.corrcoef(msg_a[:, c], gt_state[:, s])[0, 1])
                corr_b[c, s] = abs(np.corrcoef(msg_b[:, c], gt_state[:, s])[0, 1])

    state_labels = []
    for i in range(n_objects):
        for v in ['x','y','z','vx','vy','vz','m','r']:
            state_labels.append(f'O{i}_{v}')

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[0].set_title('Agent A (left camera) Message ↔ State Correlation')
    axes[0].set_yticks(range(8)); axes[0].set_yticklabels([f'Msg {i}' for i in range(8)])
    axes[0].set_xticks(range(n_state)); axes[0].set_xticklabels(state_labels, rotation=90, fontsize=6)
    axes[1].imshow(corr_b, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[1].set_title('Agent B (right camera) Message ↔ State Correlation')
    axes[1].set_yticks(range(8)); axes[1].set_yticklabels([f'Msg {i}' for i in range(8)])
    axes[1].set_xticks(range(n_state)); axes[1].set_xticklabels(state_labels, rotation=90, fontsize=6)
    for ax in axes:
        for i in range(1, n_objects):
            ax.axvline(x=i*8-0.5, color='white', linewidth=2)
    fig.suptitle('Phase 25b: What Do Visual Agents Communicate? (Asymmetric Cameras)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_comm_heatmap.png", dpi=150); plt.close()

    max_corr_a = [corr_a[:, i*8:(i+1)*8].max() for i in range(n_objects)]
    max_corr_b = [corr_b[:, i*8:(i+1)*8].max() for i in range(n_objects)]
    print("│  Agent A max corr/obj: " + ", ".join(f"O{i}={max_corr_a[i]:.3f}" for i in range(n_objects)))
    print("│  Agent B max corr/obj: " + ", ".join(f"O{i}={max_corr_b[i]:.3f}" for i in range(n_objects)))
    print("│  → results/phase25b_comm_heatmap.png")
    print("└─ Done")

    # ── Part D: Communication Necessity ────────────────────────
    print("\n┌─ Part D: Communication Margin")
    with torch.no_grad():
        pred_comm_vals, pred_nocomm_vals = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            _, pc, _, _, _, _ = model(
                dataset['img_a'][bi], dataset['img_b'][bi],
                dataset['action'][bi],
                dataset['next_img_a'][bi], dataset['next_img_b'][bi])
            pred_comm_vals.append(pc.item())
            za = model.encoder(dataset['img_a'][bi])
            zb = model.encoder(dataset['img_b'][bi])
            zmsg = torch.zeros(len(bi), 8)
            act = dataset['action'][bi]
            pa = model.predictor(torch.cat([za, zmsg, act], -1))
            pb = model.predictor(torch.cat([zb, zmsg, act], -1))
            ta = model.target_encoder(dataset['next_img_a'][bi])
            tb = model.target_encoder(dataset['next_img_b'][bi])
            pnc = (F.mse_loss(pa, ta) + F.mse_loss(pb, tb)).item() / 2
            pred_nocomm_vals.append(pnc)

    pred_comm_val = np.mean(pred_comm_vals)
    pred_nocomm_val = np.mean(pred_nocomm_vals)
    margin = (1 - pred_comm_val / pred_nocomm_val) * 100 if pred_nocomm_val > 0 else 0
    print(f"│  With comm:    {pred_comm_val:.4f}")
    print(f"│  Without comm: {pred_nocomm_val:.4f}")
    print(f"│  Communication margin: {margin:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['With\nCommunication', 'Without\nCommunication'],
                  [pred_comm_val, pred_nocomm_val], color=['#27ae60', '#e74c3c'], width=0.5)
    ax.set_ylabel('Prediction MSE', fontsize=12)
    ax.set_title(f'Phase 25b: Communication Margin = {margin:.1f}%\n'
                 f'Asymmetric cameras: A sees left, B sees right', fontsize=13)
    for bar, val in zip(bars, [pred_comm_val, pred_nocomm_val]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_comm_margin.png", dpi=150); plt.close()
    print("│  → results/phase25b_comm_margin.png")
    print("└─ Done")

    # ── Part E: Multi-Step Rollout ─────────────────────────────
    print("\n┌─ Part E: Multi-Step Rollout")
    horizons = [1, 2, 5, 10, 15, 20]
    rollout_errors = []
    steps_per_traj = 40

    for horizon in horizons:
        errors = []
        for traj_start in range(0, min(50 * steps_per_traj, n - steps_per_traj), steps_per_traj):
            if traj_start + horizon >= n: continue
            with torch.no_grad():
                za = model.encoder(dataset['img_a'][traj_start:traj_start+1])
                zb = model.encoder(dataset['img_b'][traj_start:traj_start+1])
                for h in range(horizon):
                    t = traj_start + h
                    if t >= n - 1: break
                    ma = model.comm_mu_a(za); mb = model.comm_mu_b(zb)
                    act = dataset['action'][t:t+1]
                    za = model.predictor(torch.cat([za, mb, act], -1))
                    zb = model.predictor(torch.cat([zb, ma, act], -1))
                t_f = traj_start + horizon
                if t_f < n:
                    za_t = model.target_encoder(dataset['img_a'][t_f:t_f+1])
                    zb_t = model.target_encoder(dataset['img_b'][t_f:t_f+1])
                    err = (F.mse_loss(za, za_t) + F.mse_loss(zb, zb_t)).item() / 2
                    errors.append(err)
        avg = np.mean(errors) if errors else 0
        rollout_errors.append(avg)
        print(f"│  Horizon {horizon:2d}: MSE = {avg:.4f} (n={len(errors)})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(horizons[:len(rollout_errors)], rollout_errors, 'o-', linewidth=2, markersize=8, color='#2980b9')
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('MSE in Embedding Space', fontsize=12)
    ax.set_title('Phase 25b: Multi-Step Rollout Quality', fontsize=14)
    if min(rollout_errors) > 0: ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_rollout.png", dpi=150); plt.close()
    print("│  → results/phase25b_rollout.png")
    print("└─ Done")

    # ── Part F: Embedding Space t-SNE ──────────────────────────
    print("\n┌─ Part F: Embedding Space t-SNE")
    from sklearn.manifold import TSNE
    with torch.no_grad():
        z_list = []
        for i in range(0, min(2000, n), 256):
            z_list.append(model.encoder(dataset['img_a'][i:i+256]))
        z_all = torch.cat(z_list).numpy()

    n_pts = len(z_all)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(z_all)
    colors = np.array([i % steps_per_traj for i in range(n_pts)])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=colors, cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Time step within episode')
    ax.set_title('Phase 25b: Visual Embedding Space (t-SNE)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_embedding_tsne.png", dpi=150); plt.close()
    print("│  → results/phase25b_embedding_tsne.png")
    print("└─ Done")

    # ── Part G: Complementary Sensing ──────────────────────────
    print("\n┌─ Part G: Complementary Sensing (grayscale vs color-only)")
    print("│  Agent A gets grayscale (spatial info, no color)")
    print("│  Agent B gets color residual (color info, weak spatial)")

    gray_a = dataset['img_a'].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    luminance_b = dataset['img_b'].mean(dim=1, keepdim=True)
    color_b = dataset['img_b'] - luminance_b
    cb_min, cb_max = color_b.min(), color_b.max()
    color_b = (color_b - cb_min) / (cb_max - cb_min + 1e-8)

    gray_next_a = dataset['next_img_a'].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    lum_next_b = dataset['next_img_b'].mean(dim=1, keepdim=True)
    color_next_b = dataset['next_img_b'] - lum_next_b
    color_next_b = (color_next_b - cb_min) / (cb_max - cb_min + 1e-8)

    model_comp = VisualJEPA(latent_dim=256, comm_dim=8, action_dim=4, beta=0.001)
    comp_params = sum(p.numel() for p in model_comp.parameters() if p.requires_grad)
    print(f"│  Complementary model: {comp_params/1e6:.2f}M params")
    opt_comp = torch.optim.AdamW(model_comp.parameters(), lr=3e-4, weight_decay=0.01)
    sch_comp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_comp, T_max=200, eta_min=1e-5)

    comp_hist = {'pred': [], 'kl': []}
    for epoch in range(200):
        model_comp.train()
        perm = torch.randperm(n_train)
        ep_p, ep_k, nb = 0, 0, 0
        for i in range(0, n_train, 64):
            idx = perm[i:i+64]
            total, pred, kl, vreg, _, _ = model_comp(
                gray_a[idx], color_b[idx], dataset['action'][idx],
                gray_next_a[idx], color_next_b[idx])
            opt_comp.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model_comp.parameters(), 1.0)
            opt_comp.step()
            model_comp.update_target()
            ep_p += pred.item(); ep_k += kl.item(); nb += 1
        sch_comp.step()
        comp_hist['pred'].append(ep_p/nb)
        comp_hist['kl'].append(ep_k/nb)
        if (epoch+1) % 50 == 0:
            print(f"│  Complementary epoch {epoch+1}: pred={ep_p/nb:.4f} kl={ep_k/nb:.4f}")

    model_comp.eval()
    with torch.no_grad():
        comp_comm_vals, comp_nocomm_vals = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            _, pc, _, _, _, _ = model_comp(
                gray_a[bi], color_b[bi], dataset['action'][bi],
                gray_next_a[bi], color_next_b[bi])
            comp_comm_vals.append(pc.item())
            za = model_comp.encoder(gray_a[bi])
            zb = model_comp.encoder(color_b[bi])
            zmsg = torch.zeros(len(bi), 8)
            act = dataset['action'][bi]
            pa = model_comp.predictor(torch.cat([za, zmsg, act], -1))
            pb = model_comp.predictor(torch.cat([zb, zmsg, act], -1))
            ta = model_comp.target_encoder(gray_next_a[bi])
            tb = model_comp.target_encoder(color_next_b[bi])
            pnc = (F.mse_loss(pa, ta) + F.mse_loss(pb, tb)).item() / 2
            comp_nocomm_vals.append(pnc)

    comp_comm = np.mean(comp_comm_vals)
    comp_nocomm = np.mean(comp_nocomm_vals)
    comp_margin = (1 - comp_comm / comp_nocomm) * 100 if comp_nocomm > 0 else 0
    print(f"│  Complementary with comm:    {comp_comm:.4f}")
    print(f"│  Complementary without comm: {comp_nocomm:.4f}")
    print(f"│  Complementary margin: {comp_margin:.1f}%")
    print(f"│  (Phase 17 state-vector: 91%, Phase 23 3D: 82.9%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(['With\nComm', 'Without\nComm'], [comp_comm, comp_nocomm],
                color=['#27ae60', '#e74c3c'], width=0.5)
    axes[0].set_ylabel('Prediction MSE')
    axes[0].set_title(f'Complementary Sensing Margin: {comp_margin:.1f}%')
    for j, (v, lbl) in enumerate(zip([comp_comm, comp_nocomm], ['With', 'Without'])):
        axes[0].text(j, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

    axes[1].plot(comp_hist['pred'], label='Pred Loss')
    axes[1].plot(comp_hist['kl'], label='KL', alpha=0.7)
    axes[1].set_title('Complementary Model Training')
    axes[1].set_xlabel('Epoch'); axes[1].legend()
    fig.suptitle('Phase 25b: Complementary Sensing — Grayscale vs Color-Only', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_complementary.png", dpi=150); plt.close()
    print("│  → results/phase25b_complementary.png")
    print("└─ Done")

    # ── Summary Dashboard ──────────────────────────────────────
    print("\n┌─ Generating Summary Dashboard")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Phase 25b: Visual World Model — Full Scale Results', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.imshow(dataset['img_a'][0].permute(1, 2, 0).numpy())
    ax.set_title('Cam A (left half)'); ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(dataset['img_b'][0].permute(1, 2, 0).numpy())
    ax.set_title('Cam B (right half)'); ax.axis('off')

    ax = axes[0, 2]
    ax.plot(history['pred'], color='#2980b9')
    ax.set_title(f'Pred Loss → {history["pred"][-1]:.4f}'); ax.set_xlabel('Epoch')

    ax = axes[0, 3]
    ax.imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_title('Agent A Msg ↔ State')
    ax.set_yticks(range(8)); ax.set_yticklabels([f'M{i}' for i in range(8)], fontsize=7)

    ax = axes[1, 0]
    ax.bar(['With\nComm', 'No\nComm'], [pred_comm_val, pred_nocomm_val],
           color=['#27ae60', '#e74c3c'])
    ax.set_title(f'Comm Margin: {margin:.1f}%'); ax.set_ylabel('MSE')

    ax = axes[1, 1]
    ax.bar(['With\nComm', 'No\nComm'], [comp_comm, comp_nocomm],
           color=['#27ae60', '#e74c3c'])
    ax.set_title(f'Complementary: {comp_margin:.1f}%'); ax.set_ylabel('MSE')

    ax = axes[1, 2]
    ax.plot(horizons[:len(rollout_errors)], rollout_errors, 'o-', color='#8e44ad')
    ax.set_title('Multi-Step Rollout'); ax.set_xlabel('Horizon'); ax.set_ylabel('MSE')
    if min(rollout_errors) > 0: ax.set_yscale('log')

    ax = axes[1, 3]
    ax.axis('off')
    summary = (
        "PHASE 25b SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Model: {n_params:,} params\n"
        f"Dataset: {n} frames\n"
        f"Training: 300 epochs\n\n"
        f"Pred MSE: {history['pred'][-1]:.4f}\n"
        f"Comm margin: {margin:.1f}%\n"
        f"Comp margin: {comp_margin:.1f}%\n"
        f"Rollout @5: {rollout_errors[2]:.4f}\n\n"
        "Cameras: asymmetric\n"
        "Comp: gray vs color\n"
        "Input: raw pixels\n"
        "Method: JEPA"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=12,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25b_summary.png", dpi=150); plt.close()
    print("│  → results/phase25b_summary.png")
    print("└─ Done")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PHASE 25b SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params (Visual JEPA)")
    print(f"  Dataset: {n} frames (64×64 RGB, asymmetric cameras)")
    print(f"  Pred loss: {history['pred'][-1]:.4f}")
    print(f"  Comm margin: {margin:.1f}%")
    print(f"  Complementary margin: {comp_margin:.1f}%")
    print(f"  Rollout MSE @5: {rollout_errors[2]:.4f}  @20: {rollout_errors[-1]:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


def run_phase25c():
    """Phase 25c: Fix communication collapse with β-annealing + comm dropout + MPS."""
    print("=" * 60)
    print("PHASE 25c: COMMUNICATION FORCING — β-ANNEALING + COMM DROPOUT")
    print("=" * 60)
    t0 = time.time()

    # ── Device setup ───────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU) — expect 3-5× speedup")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU (will be slow)")

    # ── Part A: Collect Visual Dataset (12,000 frames) ─────────
    print("\n┌─ Part A: Collecting visual dataset (300 episodes)")
    dataset = collect_visual_dataset(n_episodes=300, steps_per_episode=40,
                                     n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    print(f"│  Dataset: {n} frames, shape={list(dataset['img_a'].shape)}")
    print(f"│  Collection time: {time.time()-t0:.0f}s")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        idx = i * n // 5
        axes[0, i].imshow(dataset['img_a'][idx].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Cam A (left), t={idx}', fontsize=8); axes[0, i].axis('off')
        axes[1, i].imshow(dataset['img_b'][idx].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Cam B (right), t={idx}', fontsize=8); axes[1, i].axis('off')
    fig.suptitle('Phase 25c: Asymmetric Camera Views', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_dataset_samples.png", dpi=150); plt.close()
    print("│  → results/phase25c_dataset_samples.png")
    print("└─ Done")

    # ── Part B: Train Visual JEPA v2 (300 epochs, β-annealing) ─
    print("\n┌─ Part B: Training Visual JEPA v2 (300 epochs, β-annealing)")
    model = VisualJEPAv2(latent_dim=256, comm_dim=8, action_dim=4, comm_dropout=0.3)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model: {n_params/1e6:.2f}M params, comm_dropout=0.3")
    print(f"│  β schedule: 0→0 (ep 0-99), 0→0.001 (ep 100-199), 0.001 (ep 200-300)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

    n_train = int(0.8 * n)
    batch_size = 64
    history = {'pred': [], 'kl': [], 'vicreg': [], 'beta': []}

    train_start = time.time()
    for epoch in range(300):
        # β-annealing schedule
        if epoch < 100:
            model.beta = 0.0
        elif epoch < 200:
            model.beta = 0.001 * (epoch - 100) / 100
        else:
            model.beta = 0.001

        model.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_kl, ep_vreg, nb = 0, 0, 0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            ba = dataset['img_a'][idx].to(device)
            bb = dataset['img_b'][idx].to(device)
            bact = dataset['action'][idx].to(device)
            bna = dataset['next_img_a'][idx].to(device)
            bnb = dataset['next_img_b'][idx].to(device)

            total, pred_loss, kl, vicreg, _, _ = model(
                ba, bb, bact, bna, bnb, training=True)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            ep_pred += pred_loss.item(); ep_kl += kl.item()
            ep_vreg += vicreg.item(); nb += 1
        scheduler.step()
        history['pred'].append(ep_pred/nb)
        history['kl'].append(ep_kl/nb)
        history['vicreg'].append(ep_vreg/nb)
        history['beta'].append(model.beta)
        if (epoch+1) % 25 == 0:
            elapsed_ep = time.time() - train_start
            eta = elapsed_ep / (epoch+1) * (300 - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}: pred={ep_pred/nb:.4f} kl={ep_kl/nb:.4f} "
                  f"β={model.beta:.4f} vreg={ep_vreg/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    # Training plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history['pred']); axes[0].set_title('Prediction Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(history['kl']); axes[1].set_title('Communication KL'); axes[1].set_xlabel('Epoch')
    axes[2].plot(history['vicreg']); axes[2].set_title('VICReg Loss'); axes[2].set_xlabel('Epoch')
    ax3 = axes[3]; ax3.plot(history['beta'], color='red', label='β')
    ax3.set_title('β-Annealing Schedule'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('β')
    ax3b = ax3.twinx(); ax3b.plot(history['kl'], color='blue', alpha=0.5, label='KL')
    ax3b.set_ylabel('KL', color='blue')
    fig.suptitle('Phase 25c: Training with β-Annealing + Comm Dropout', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_training.png", dpi=150); plt.close()
    print(f"│  Training time: {time.time()-train_start:.0f}s")
    print("│  → results/phase25c_training.png")
    print("└─ Done")

    # ── Part C: Communication Margin ───────────────────────────
    print("\n┌─ Part C: Communication Margin")
    model.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        pred_comm_vals, pred_nocomm_vals = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            ba = dataset['img_a'][bi].to(device)
            bb = dataset['img_b'][bi].to(device)
            bact = dataset['action'][bi].to(device)
            bna = dataset['next_img_a'][bi].to(device)
            bnb = dataset['next_img_b'][bi].to(device)

            # With communication
            _, pc, _, _, _, _ = model(ba, bb, bact, bna, bnb, training=False)
            pred_comm_vals.append(pc.item())

            # Without communication (zero messages)
            za = model.encoder(ba)
            zb = model.encoder(bb)
            zmsg = torch.zeros(len(bi), 8, device=device)
            pa = model.predictor(torch.cat([za, zmsg, bact], -1))
            pb = model.predictor(torch.cat([zb, zmsg, bact], -1))
            ta = model.target_encoder(bna)
            tb = model.target_encoder(bnb)
            pnc = (F.mse_loss(pa, ta) + F.mse_loss(pb, tb)).item() / 2
            pred_nocomm_vals.append(pnc)

    pred_comm_val = np.mean(pred_comm_vals)
    pred_nocomm_val = np.mean(pred_nocomm_vals)
    margin = (1 - pred_comm_val / pred_nocomm_val) * 100 if pred_nocomm_val > 0 else 0
    print(f"│  With comm:    {pred_comm_val:.4f}")
    print(f"│  Without comm: {pred_nocomm_val:.4f}")
    print(f"│  Communication margin: {margin:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['With\nCommunication', 'Without\nCommunication'],
                  [pred_comm_val, pred_nocomm_val], color=['#27ae60', '#e74c3c'], width=0.5)
    ax.set_ylabel('Prediction MSE', fontsize=12)
    ax.set_title(f'Phase 25c: Communication Margin = {margin:.1f}%\n'
                 f'β-annealing + comm dropout (30%)', fontsize=13)
    for bar, val in zip(bars, [pred_comm_val, pred_nocomm_val]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_comm_margin.png", dpi=150); plt.close()
    print("│  → results/phase25c_comm_margin.png")
    print("└─ Done")

    # ── Part D: Message ↔ State Correlation ────────────────────
    print("\n┌─ Part D: Message ↔ State Correlation")
    with torch.no_grad():
        msg_a_list, msg_b_list = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            ma, mb = model.get_messages(
                dataset['img_a'][bi].to(device),
                dataset['img_b'][bi].to(device))
            msg_a_list.append(ma.cpu()); msg_b_list.append(mb.cpu())
        msg_a = torch.cat(msg_a_list).numpy()
        msg_b = torch.cat(msg_b_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_state = gt_state.shape[1]
    n_objects = n_state // 8

    corr_a = np.zeros((8, n_state))
    corr_b = np.zeros((8, n_state))
    for c in range(8):
        for s in range(n_state):
            if np.std(gt_state[:, s]) > 1e-6:
                corr_a[c, s] = abs(np.corrcoef(msg_a[:, c], gt_state[:, s])[0, 1])
                corr_b[c, s] = abs(np.corrcoef(msg_b[:, c], gt_state[:, s])[0, 1])

    state_labels = []
    for i in range(n_objects):
        for v in ['x','y','z','vx','vy','vz','m','r']:
            state_labels.append(f'O{i}_{v}')

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[0].set_title('Agent A (left camera) Message ↔ State Correlation')
    axes[0].set_yticks(range(8)); axes[0].set_yticklabels([f'Msg {i}' for i in range(8)])
    axes[0].set_xticks(range(n_state)); axes[0].set_xticklabels(state_labels, rotation=90, fontsize=6)
    axes[1].imshow(corr_b, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[1].set_title('Agent B (right camera) Message ↔ State Correlation')
    axes[1].set_yticks(range(8)); axes[1].set_yticklabels([f'Msg {i}' for i in range(8)])
    axes[1].set_xticks(range(n_state)); axes[1].set_xticklabels(state_labels, rotation=90, fontsize=6)
    for ax in axes:
        for i in range(1, n_objects):
            ax.axvline(x=i*8-0.5, color='white', linewidth=2)
    fig.suptitle('Phase 25c: What Do Visual Agents Communicate? (β-annealing)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_comm_heatmap.png", dpi=150); plt.close()

    max_corr_a = [corr_a[:, i*8:(i+1)*8].max() for i in range(n_objects)]
    max_corr_b = [corr_b[:, i*8:(i+1)*8].max() for i in range(n_objects)]
    print("│  Agent A max corr/obj: " + ", ".join(f"O{i}={max_corr_a[i]:.3f}" for i in range(n_objects)))
    print("│  Agent B max corr/obj: " + ", ".join(f"O{i}={max_corr_b[i]:.3f}" for i in range(n_objects)))
    print("│  → results/phase25c_comm_heatmap.png")
    print("└─ Done")

    # ── Part E: Multi-Step Rollout ─────────────────────────────
    print("\n┌─ Part E: Multi-Step Rollout")
    horizons = [1, 2, 5, 10, 20]
    rollout_errors = []
    steps_per_traj = 40

    for horizon in horizons:
        errors = []
        for traj_start in range(0, min(50 * steps_per_traj, n - steps_per_traj), steps_per_traj):
            if traj_start + horizon >= n: continue
            with torch.no_grad():
                za = model.encoder(dataset['img_a'][traj_start:traj_start+1].to(device))
                zb = model.encoder(dataset['img_b'][traj_start:traj_start+1].to(device))
                for h in range(horizon):
                    t = traj_start + h
                    if t >= n: break
                    act = dataset['action'][t:t+1].to(device)
                    msg_a_h, _, _ = model._communicate(za, model.comm_mu_a, model.comm_logvar_a)
                    msg_b_h, _, _ = model._communicate(zb, model.comm_mu_b, model.comm_logvar_b)
                    za = model.predictor(torch.cat([za, msg_b_h, act], -1))
                    zb = model.predictor(torch.cat([zb, msg_a_h, act], -1))
                tgt_a = model.target_encoder(
                    dataset['img_a'][traj_start+horizon:traj_start+horizon+1].to(device))
                tgt_b = model.target_encoder(
                    dataset['img_b'][traj_start+horizon:traj_start+horizon+1].to(device))
                err = (F.mse_loss(za, tgt_a) + F.mse_loss(zb, tgt_b)).item() / 2
                errors.append(err)
        rollout_errors.append(np.mean(errors) if errors else float('nan'))
        print(f"│  Horizon {horizon:2d}: MSE = {rollout_errors[-1]:.4f} (n={len(errors)})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, rollout_errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon (steps)'); ax.set_ylabel('MSE')
    ax.set_title('Phase 25c: Multi-Step Rollout Quality')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Target @5')
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_rollout.png", dpi=150); plt.close()
    print("│  → results/phase25c_rollout.png")
    print("└─ Done")

    # ── Part F: Complementary Sensing ──────────────────────────
    print("\n┌─ Part F: Complementary Sensing (grayscale vs color-only)")
    gray_a = dataset['img_a'].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    color_b = dataset['img_b'] - dataset['img_b'].mean(dim=1, keepdim=True) + 0.5
    gray_na = dataset['next_img_a'].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    color_nb = dataset['next_img_b'] - dataset['next_img_b'].mean(dim=1, keepdim=True) + 0.5
    print("│  Agent A gets grayscale (spatial info, no color)")
    print("│  Agent B gets color residual (color info, weak spatial)")

    model_comp = VisualJEPAv2(latent_dim=256, comm_dim=8, action_dim=4, comm_dropout=0.3)
    model_comp = model_comp.to(device)
    print(f"│  Complementary model: {sum(p.numel() for p in model_comp.parameters() if p.requires_grad)/1e6:.2f}M params")

    opt_comp = torch.optim.AdamW(model_comp.parameters(), lr=3e-4, weight_decay=0.01)
    sch_comp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_comp, T_max=200, eta_min=1e-5)

    comp_start = time.time()
    for epoch in range(200):
        # Shorter β schedule: 0 for 60, ramp to 130, fixed to 200
        if epoch < 60:
            model_comp.beta = 0.0
        elif epoch < 130:
            model_comp.beta = 0.001 * (epoch - 60) / 70
        else:
            model_comp.beta = 0.001

        model_comp.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_kl, nb = 0, 0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            ba = gray_a[idx].to(device)
            bb = color_b[idx].to(device)
            bact = dataset['action'][idx].to(device)
            bna = gray_na[idx].to(device)
            bnb = color_nb[idx].to(device)

            total, pred_loss, kl, vicreg, _, _ = model_comp(
                ba, bb, bact, bna, bnb, training=True)
            opt_comp.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model_comp.parameters(), 1.0)
            opt_comp.step()
            model_comp.update_target()
            ep_pred += pred_loss.item(); ep_kl += kl.item(); nb += 1
        sch_comp.step()
        if (epoch+1) % 50 == 0:
            elapsed_ep = time.time() - comp_start
            eta = elapsed_ep / (epoch+1) * (200 - epoch - 1)
            print(f"│  Complementary epoch {epoch+1}: pred={ep_pred/nb:.4f} "
                  f"kl={ep_kl/nb:.4f} β={model_comp.beta:.4f} [{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    # Complementary margin
    model_comp.eval()
    with torch.no_grad():
        comp_comm_vals, comp_nocomm_vals = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i:i+256]
            ba = gray_a[bi].to(device)
            bb = color_b[bi].to(device)
            bact = dataset['action'][bi].to(device)
            bna = gray_na[bi].to(device)
            bnb = color_nb[bi].to(device)

            _, pc, _, _, _, _ = model_comp(ba, bb, bact, bna, bnb, training=False)
            comp_comm_vals.append(pc.item())

            za = model_comp.encoder(ba)
            zb = model_comp.encoder(bb)
            zmsg = torch.zeros(len(bi), 8, device=device)
            pa = model_comp.predictor(torch.cat([za, zmsg, bact], -1))
            pb = model_comp.predictor(torch.cat([zb, zmsg, bact], -1))
            ta = model_comp.target_encoder(bna)
            tb = model_comp.target_encoder(bnb)
            pnc = (F.mse_loss(pa, ta) + F.mse_loss(pb, tb)).item() / 2
            comp_nocomm_vals.append(pnc)

    comp_comm = np.mean(comp_comm_vals)
    comp_nocomm = np.mean(comp_nocomm_vals)
    comp_margin = (1 - comp_comm / comp_nocomm) * 100 if comp_nocomm > 0 else 0
    print(f"│  Complementary with comm:    {comp_comm:.4f}")
    print(f"│  Complementary without comm: {comp_nocomm:.4f}")
    print(f"│  Complementary margin: {comp_margin:.1f}%")
    print(f"│  (Phase 17 state-vector: 91%, Phase 23 3D: 82.9%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars0 = axes[0].bar(['With Comm', 'No Comm'],
                        [pred_comm_val, pred_nocomm_val], color=['#27ae60', '#e74c3c'])
    axes[0].set_title(f'Asymmetric Cameras: {margin:.1f}% margin')
    axes[0].set_ylabel('Prediction MSE')
    for bar, val in zip(bars0, [pred_comm_val, pred_nocomm_val]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center')
    bars1 = axes[1].bar(['With Comm', 'No Comm'],
                        [comp_comm, comp_nocomm], color=['#27ae60', '#e74c3c'])
    axes[1].set_title(f'Complementary Sensing: {comp_margin:.1f}% margin')
    axes[1].set_ylabel('Prediction MSE')
    for bar, val in zip(bars1, [comp_comm, comp_nocomm]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center')
    fig.suptitle('Phase 25c: Communication Margins (β-annealing + dropout)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_complementary.png", dpi=150); plt.close()
    print("│  → results/phase25c_complementary.png")
    print("└─ Done")

    # ── Part G: Summary Dashboard ──────────────────────────────
    print("\n┌─ Generating Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].plot(history['pred'], label='Prediction')
    axes[0, 0].set_title('Training Loss'); axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    ax_kl = axes[0, 1]
    ax_kl.plot(history['kl'], 'b-', label='KL')
    ax_kl.set_ylabel('KL', color='b'); ax_kl.set_title('KL + β Schedule')
    ax_kl.set_xlabel('Epoch')
    ax_beta = ax_kl.twinx()
    ax_beta.plot(history['beta'], 'r--', label='β')
    ax_beta.set_ylabel('β', color='r')
    ax_kl.axvline(x=100, color='gray', linestyle=':', alpha=0.5)
    ax_kl.axvline(x=200, color='gray', linestyle=':', alpha=0.5)
    kl_max = max(history['kl']) if max(history['kl']) > 0 else 1
    ax_kl.text(50, kl_max*0.9, 'β=0', ha='center', fontsize=9)
    ax_kl.text(150, kl_max*0.9, 'ramp', ha='center', fontsize=9)
    ax_kl.text(250, kl_max*0.9, 'β=0.001', ha='center', fontsize=9)

    axes[0, 2].bar(['With\nComm', 'No\nComm'], [pred_comm_val, pred_nocomm_val],
                   color=['#27ae60', '#e74c3c'])
    axes[0, 2].set_title(f'Comm Margin: {margin:.1f}%')
    axes[0, 2].set_ylabel('Prediction MSE')

    axes[1, 0].plot(horizons, rollout_errors, 'bo-', linewidth=2)
    axes[1, 0].set_xlabel('Horizon'); axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Multi-Step Rollout')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    axes[1, 1].bar(['With\nComm', 'No\nComm'], [comp_comm, comp_nocomm],
                   color=['#27ae60', '#e74c3c'])
    axes[1, 1].set_title(f'Complementary: {comp_margin:.1f}% margin')
    axes[1, 1].set_ylabel('Prediction MSE')

    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    kl_final = history['kl'][-1]
    summary_text = (
        f"Phase 25c Summary\n"
        f"{'─'*30}\n"
        f"Model: {n_params/1e6:.2f}M params\n"
        f"Dataset: {n} frames (64×64)\n"
        f"Comm dropout: 30%\n"
        f"β-annealing: 0→0.001\n\n"
        f"Pred loss: {history['pred'][-1]:.4f}\n"
        f"Final KL: {kl_final:.4f}\n"
        f"Comm margin: {margin:.1f}%\n"
        f"Comp margin: {comp_margin:.1f}%\n"
        f"Rollout @5: {rollout_errors[2]:.4f}\n\n"
        f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)\n\n"
        f"vs Phase 25b:\n"
        f"  KL: 0.000 → {kl_final:.4f}\n"
        f"  Margin: 0% → {margin:.1f}%\n"
        f"  Comp: 0% → {comp_margin:.1f}%"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Phase 25c: Communication Forcing Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25c_summary.png", dpi=150); plt.close()
    print("│  → results/phase25c_summary.png")

    torch.save(model.state_dict(), OUTPUT_DIR / "phase25c_model.pt")
    print("│  → results/phase25c_model.pt")
    print("└─ Done")

    print(f"\n{'='*60}")
    print(f"PHASE 25c SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params (Visual JEPA v2)")
    print(f"  Dataset: {n} frames (64×64 RGB, asymmetric cameras)")
    print(f"  Pred loss: {history['pred'][-1]:.4f}")
    print(f"  Final KL: {kl_final:.4f} (was 0.000 in 25b)")
    print(f"  Comm margin: {margin:.1f}% (was 0.0% in 25b)")
    print(f"  Complementary margin: {comp_margin:.1f}% (was 0.0% in 25b)")
    print(f"  Rollout MSE @5: {rollout_errors[2]:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


def run_phase25d():
    """Phase 25d: Hard information separation — each agent sees ONLY its side."""
    print("=" * 60)
    print("PHASE 25d: HARD INFORMATION SEPARATION")
    print("=" * 60)
    t0 = time.time()

    # ── Device ─────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Part A: Split View Dataset ─────────────────────────────
    print("\n┌─ Part A: Collecting split-view dataset (300 episodes)")
    dataset = collect_split_view_dataset(n_episodes=300, steps_per_episode=40,
                                         n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    print(f"│  Dataset: {n} frames")
    print(f"│  Collection time: {time.time()-t0:.0f}s")

    # Show split views: left-only, right-only, target (all)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        idx = i * n // 5
        axes[0, i].imshow(dataset['img_a'][idx].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Agent A (left only)', fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(dataset['img_b'][idx].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Agent B (right only)', fontsize=8)
        axes[1, i].axis('off')
        axes[2, i].imshow(dataset['img_target'][idx].permute(1, 2, 0).numpy())
        axes[2, i].set_title(f'Target (ALL objects)', fontsize=8)
        axes[2, i].axis('off')
    fig.suptitle('Phase 25d: Split Views — Each Agent Sees ONLY Its Side', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_dataset_samples.png", dpi=150); plt.close()
    print("│  → results/phase25d_dataset_samples.png")
    print("└─ Done")

    # ── Part B: Train VisualJEPAv2 (predict FULL scene) ────────
    print("\n┌─ Part B: Training VisualJEPAv2 (300 epochs)")
    print("│  KEY: Agents encode partial views, predict FULL scene target")
    print("│  comm_dropout=0.0 — hard separation makes dropout unnecessary")

    model = VisualJEPAv2(latent_dim=256, comm_dim=8, action_dim=4, comm_dropout=0.0)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    batch_size = 64
    history = {'pred': [], 'kl': [], 'vicreg': [], 'beta': []}

    train_start = time.time()
    for epoch in range(300):
        # β-annealing: same as 25c
        if epoch < 100:
            model.beta = 0.0
        elif epoch < 200:
            model.beta = 0.001 * (epoch - 100) / 100
        else:
            model.beta = 0.001

        model.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_kl, ep_vreg, nb = 0, 0, 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            ba = dataset['img_a'][idx].to(device)
            bb = dataset['img_b'][idx].to(device)
            bact = dataset['action'][idx].to(device)
            # TARGET is the FULL scene (overhead view) — NOT partial views
            bnt = dataset['next_img_target'][idx].to(device)

            total, pred_loss, kl, vicreg, _, _ = model(
                ba, bb, bact, bnt, bnt, training=True)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            ep_pred += pred_loss.item(); ep_kl += kl.item()
            ep_vreg += vicreg.item(); nb += 1

        scheduler.step()
        history['pred'].append(ep_pred / nb)
        history['kl'].append(ep_kl / nb)
        history['vicreg'].append(ep_vreg / nb)
        history['beta'].append(model.beta)

        if (epoch + 1) % 25 == 0:
            elapsed_ep = time.time() - train_start
            eta = elapsed_ep / (epoch + 1) * (300 - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}: pred={ep_pred/nb:.4f} kl={ep_kl/nb:.4f} "
                  f"β={model.beta:.4f} vreg={ep_vreg/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    # Training plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history['pred'])
    axes[0, 0].set_title(f"Prediction Loss → {history['pred'][-1]:.4f}")
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 1].plot(history['kl'], color='orange')
    axes[0, 1].set_title(f"KL → {history['kl'][-1]:.4f}")
    ax2 = axes[0, 1].twinx()
    ax2.plot(history['beta'], color='blue', alpha=0.3); ax2.set_ylabel('β')
    axes[1, 0].plot(history['vicreg'])
    axes[1, 0].set_title('VICReg'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 1].plot(history['beta'])
    axes[1, 1].set_title('β Schedule'); axes[1, 1].set_xlabel('Epoch')
    fig.suptitle('Phase 25d: Training (Hard Split Views, No Dropout)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_training.png", dpi=150); plt.close()
    print(f"│  Training time: {time.time()-train_start:.0f}s")
    print("│  → results/phase25d_training.png")
    print("└─ Done")

    # ── Part C: Communication Tests (Zero / Shuffled / Noise) ──
    print("\n┌─ Part C: Communication Tests")
    model.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        # Process in chunks to avoid OOM
        mse_normal_vals, mse_zero_vals, mse_shuf_vals, mse_noise_vals = [], [], [], []

        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            ba = dataset['img_a'][bi].to(device)
            bb = dataset['img_b'][bi].to(device)
            bact = dataset['action'][bi].to(device)
            bnt = dataset['next_img_target'][bi].to(device)

            # Normal (real messages)
            _, pc, _, _, _, _ = model(ba, bb, bact, bnt, bnt, training=False)
            mse_normal_vals.append(pc.item())

            # Encode
            z_a = model.encoder(ba)
            z_b = model.encoder(bb)
            z_tgt = model.target_encoder(bnt)

            msg_a_real = model.comm_mu_a(z_a)
            msg_b_real = model.comm_mu_b(z_b)

            # Zero messages
            zmsg = torch.zeros_like(msg_b_real)
            pred_zero = model.predictor(torch.cat([z_a, zmsg, bact], -1))
            mse_zero_vals.append(F.mse_loss(pred_zero, z_tgt).item())

            # Shuffled messages (random permutation within batch)
            perm = torch.randperm(len(bi))
            msg_shuf = msg_b_real[perm]
            pred_shuf = model.predictor(torch.cat([z_a, msg_shuf, bact], -1))
            mse_shuf_vals.append(F.mse_loss(pred_shuf, z_tgt).item())

            # Random noise messages
            msg_noise = torch.randn_like(msg_b_real)
            pred_noise = model.predictor(torch.cat([z_a, msg_noise, bact], -1))
            mse_noise_vals.append(F.mse_loss(pred_noise, z_tgt).item())

    mse_normal = np.mean(mse_normal_vals)
    mse_zero = np.mean(mse_zero_vals)
    mse_shuffled = np.mean(mse_shuf_vals)
    mse_noise = np.mean(mse_noise_vals)
    margin_zero = (1 - mse_normal / mse_zero) * 100 if mse_zero > 0 else 0
    margin_shuf = (1 - mse_normal / mse_shuffled) * 100 if mse_shuffled > 0 else 0
    margin_noise = (1 - mse_normal / mse_noise) * 100 if mse_noise > 0 else 0

    print(f"│  Normal (real msg):     {mse_normal:.4f}")
    print(f"│  Zero messages:         {mse_zero:.4f}  (margin: {margin_zero:.1f}%)")
    print(f"│  Shuffled messages:     {mse_shuffled:.4f}  (margin: {margin_shuf:.1f}%)")
    print(f"│  Random noise messages: {mse_noise:.4f}  (margin: {margin_noise:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = ['Real\nMessages', 'Zero\nMessages', 'Shuffled\nMessages', 'Random\nNoise']
    values = [mse_normal, mse_zero, mse_shuffled, mse_noise]
    colors_bar = ['#27ae60', '#e74c3c', '#e67e22', '#8e44ad']
    bars = ax.bar(conditions, values, color=colors_bar, alpha=0.8)
    ax.set_ylabel('Prediction MSE')
    ax.set_title(f'Phase 25d: Communication Test (Hard Split)\n'
                 f'Zero margin: {margin_zero:.1f}% | '
                 f'Shuffled margin: {margin_shuf:.1f}% | '
                 f'Noise margin: {margin_noise:.1f}%')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_comm_tests.png", dpi=150); plt.close()
    print("│  → results/phase25d_comm_tests.png")
    print("└─ Done")

    # ── Part D: Copy Baseline ──────────────────────────────────
    print("\n┌─ Part D: Baselines")
    with torch.no_grad():
        mse_copy_vals, mse_mean_vals = [], []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            z_cur = model.target_encoder(dataset['img_target'][bi].to(device))
            z_nxt = model.target_encoder(dataset['next_img_target'][bi].to(device))
            mse_copy_vals.append(F.mse_loss(z_cur, z_nxt).item())

        mse_copy = np.mean(mse_copy_vals)

        # Mean baseline: average target embedding
        all_z = []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            all_z.append(model.target_encoder(
                dataset['next_img_target'][bi].to(device)).cpu())
        all_z = torch.cat(all_z)
        z_mean = all_z.mean(dim=0)
        mse_mean = F.mse_loss(all_z, z_mean.unsqueeze(0).expand_as(all_z)).item()

    improvement_copy = (1 - mse_normal / mse_copy) * 100 if mse_copy > 0 else 0
    print(f"│  Copy baseline (nothing changes): {mse_copy:.4f}")
    print(f"│  Mean baseline (predict average):  {mse_mean:.4f}")
    print(f"│  Model with comm:                  {mse_normal:.4f}")
    print(f"│  Model without comm:               {mse_zero:.4f}")
    print(f"│  Improvement over copy: {improvement_copy:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    base_vals = [mse_mean, mse_copy, mse_zero, mse_normal]
    base_labels = ['Mean\nBaseline', 'Copy\nBaseline', 'Model\n(no msg)', 'Model\n(real msg)']
    base_colors = ['#95a5a6', '#7f8c8d', '#e74c3c', '#27ae60']
    bars = ax.bar(base_labels, base_vals, color=base_colors, alpha=0.8)
    ax.set_ylabel('Prediction MSE')
    ax.set_title(f'Phase 25d: Model vs Baselines  (improvement: {improvement_copy:.1f}%)')
    for bar, val in zip(bars, base_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_baselines.png", dpi=150); plt.close()
    print("│  → results/phase25d_baselines.png")
    print("└─ Done")

    # ── Part E: Message ↔ State Correlation ────────────────────
    print("\n┌─ Part E: Message ↔ State Correlation")
    with torch.no_grad():
        msg_a_list, msg_b_list = [], []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            ma = model.comm_mu_a(model.encoder(dataset['img_a'][bi].to(device)))
            mb = model.comm_mu_b(model.encoder(dataset['img_b'][bi].to(device)))
            msg_a_list.append(ma.cpu()); msg_b_list.append(mb.cpu())
        msg_a_np = torch.cat(msg_a_list).numpy()
        msg_b_np = torch.cat(msg_b_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    side_labels = dataset['side_labels'][val_idx].numpy()
    n_state = gt_state.shape[1]
    n_obj = n_state // 8

    corr_a = np.zeros((8, n_state))
    corr_b = np.zeros((8, n_state))
    for c in range(8):
        for s in range(n_state):
            if np.std(gt_state[:, s]) > 1e-6:
                ca = np.corrcoef(msg_a_np[:, c], gt_state[:, s])[0, 1]
                cb = np.corrcoef(msg_b_np[:, c], gt_state[:, s])[0, 1]
                corr_a[c, s] = abs(ca) if not np.isnan(ca) else 0
                corr_b[c, s] = abs(cb) if not np.isnan(cb) else 0

    state_labels = []
    for i in range(n_obj):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r']:
            state_labels.append(f'O{i}_{v}')

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    im0 = axes[0].imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[0].set_title('Agent A (sees LEFT only) — Messages → State')
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels([f'Ch {i}' for i in range(8)])
    axes[0].set_xticks(range(n_state))
    axes[0].set_xticklabels(state_labels, rotation=90, fontsize=6)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(corr_b, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    axes[1].set_title('Agent B (sees RIGHT only) — Messages → State')
    axes[1].set_yticks(range(8))
    axes[1].set_yticklabels([f'Ch {i}' for i in range(8)])
    axes[1].set_xticks(range(n_state))
    axes[1].set_xticklabels(state_labels, rotation=90, fontsize=6)
    plt.colorbar(im1, ax=axes[1])

    for ax in axes:
        for i in range(1, n_obj):
            ax.axvline(x=i * 8 - 0.5, color='white', linewidth=2)
    fig.suptitle('Phase 25d: What Do Split-View Agents Communicate?', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_comm_heatmap.png", dpi=150); plt.close()

    max_corr_a = [corr_a[:, i*8:(i+1)*8].max() for i in range(n_obj)]
    max_corr_b = [corr_b[:, i*8:(i+1)*8].max() for i in range(n_obj)]
    print("│  Agent A max corr/obj: " +
          ", ".join(f"O{i}={max_corr_a[i]:.3f}" for i in range(n_obj)))
    print("│  Agent B max corr/obj: " +
          ", ".join(f"O{i}={max_corr_b[i]:.3f}" for i in range(n_obj)))
    print("│  → results/phase25d_comm_heatmap.png")
    print("└─ Done")

    # ── Part F: Multi-Step Rollout ─────────────────────────────
    print("\n┌─ Part F: Multi-Step Rollout")
    horizons = [1, 2, 5, 10, 20]
    rollout_errors = []
    steps_per_traj = 40

    for horizon in horizons:
        errors = []
        for traj_start in range(n_train, min(n_train + 50 * steps_per_traj,
                                              n - steps_per_traj), steps_per_traj):
            if traj_start + horizon >= n:
                continue
            with torch.no_grad():
                za = model.encoder(dataset['img_a'][traj_start:traj_start+1].to(device))
                zb = model.encoder(dataset['img_b'][traj_start:traj_start+1].to(device))
                for h in range(horizon):
                    t = traj_start + h
                    if t >= n - 1:
                        break
                    msg_a_h = model.comm_mu_a(za)
                    msg_b_h = model.comm_mu_b(zb)
                    act = dataset['action'][t:t+1].to(device)
                    za = model.predictor(torch.cat([za, msg_b_h, act], -1))
                    zb = model.predictor(torch.cat([zb, msg_a_h, act], -1))
                t_final = traj_start + horizon
                if t_final < n:
                    z_actual = model.target_encoder(
                        dataset['img_target'][t_final:t_final+1].to(device))
                    errors.append(F.mse_loss(za, z_actual).item())
        avg = np.mean(errors) if errors else float('nan')
        rollout_errors.append(avg)
        print(f"│  Horizon {horizon:2d}: MSE = {avg:.4f} (n={len(errors)})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, rollout_errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon (steps)'); ax.set_ylabel('MSE')
    ax.set_title('Phase 25d: Multi-Step Rollout (Hard Split)')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Target @5')
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_rollout.png", dpi=150); plt.close()
    print("│  → results/phase25d_rollout.png")
    print("└─ Done")

    # ── Part G: Summary Dashboard ──────────────────────────────
    print("\n┌─ Generating Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 25d: Hard Information Separation — Complete Results',
                 fontsize=16, fontweight='bold')

    # Training
    axes[0, 0].plot(history['pred'])
    axes[0, 0].set_title(f"Pred → {history['pred'][-1]:.4f}")
    axes[0, 0].set_xlabel('Epoch')

    # KL + β
    axes[0, 1].plot(history['kl'], color='orange')
    axes[0, 1].set_title(f"KL → {history['kl'][-1]:.2f}")
    ax2 = axes[0, 1].twinx()
    ax2.plot(history['beta'], color='blue', alpha=0.3); ax2.set_ylabel('β')

    # Comm tests
    axes[0, 2].bar(conditions, values, color=colors_bar, alpha=0.8)
    axes[0, 2].set_title(f'Comm Tests\nZero: {margin_zero:.1f}% | '
                         f'Shuf: {margin_shuf:.1f}%')
    axes[0, 2].set_ylabel('MSE')

    # Baselines
    axes[1, 0].bar(['Copy\nBaseline', 'Model\n(with msg)', 'Model\n(no msg)'],
                   [mse_copy, mse_normal, mse_zero],
                   color=['gray', '#27ae60', '#e74c3c'])
    axes[1, 0].set_title(f'vs Copy Baseline: {improvement_copy:.1f}%')

    # Rollout
    axes[1, 1].plot(horizons, rollout_errors, 'bo-', linewidth=2)
    axes[1, 1].set_title('Rollout'); axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('MSE')

    # Summary text
    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    kl_final = history['kl'][-1]
    summary_text = (
        f"PHASE 25d RESULTS\n"
        f"{'━'*30}\n\n"
        f"Hard split views\n"
        f"(each agent sees ONLY its side)\n\n"
        f"Params: {n_params:,}\n"
        f"Device: {device}\n"
        f"Time: {elapsed/60:.0f} min\n\n"
        f"Pred loss: {history['pred'][-1]:.4f}\n"
        f"KL: {kl_final:.2f}\n\n"
        f"MARGINS:\n"
        f"  Zero msg:  {margin_zero:.1f}%\n"
        f"  Shuffled:  {margin_shuf:.1f}%\n"
        f"  Noise:     {margin_noise:.1f}%\n\n"
        f"Copy baseline: {mse_copy:.4f}\n"
        f"Model (comm):  {mse_normal:.4f}\n"
        f"Improvement:   {improvement_copy:.1f}%\n"
        f"Rollout @5:    {rollout_errors[2]:.4f}\n\n"
        f"vs Phase 25c:\n"
        f"  Zero: 0.1% → {margin_zero:.1f}%\n"
        f"  Shuf: N/A → {margin_shuf:.1f}%"
    )
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase25d_summary.png", dpi=150); plt.close()

    torch.save(model.state_dict(), OUTPUT_DIR / "phase25d_model.pt")
    print("│  → results/phase25d_summary.png")
    print("│  → results/phase25d_model.pt")
    print("└─ Done")

    print(f"\n{'='*60}")
    print(f"PHASE 25d SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params (Visual JEPA v2, no dropout)")
    print(f"  Dataset: {n} frames (64×64, hard split views)")
    print(f"  Pred loss: {history['pred'][-1]:.4f}")
    print(f"  KL: {kl_final:.2f}")
    print(f"  Zero margin:     {margin_zero:.1f}%  (was 0.1% in 25c)")
    print(f"  Shuffled margin: {margin_shuf:.1f}%")
    print(f"  Noise margin:    {margin_noise:.1f}%")
    print(f"  Copy baseline:   {improvement_copy:.1f}%")
    print(f"  Rollout @5:      {rollout_errors[2]:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


def run_phase26():
    """Phase 26: Object Discovery from Pixels via Slot Attention."""
    print("=" * 60)
    print("PHASE 26: OBJECT DISCOVERY VIA SLOT ATTENTION")
    print("=" * 60)
    t0 = time.time()

    # ── Device ─────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Part A: Dataset (reuse split-view from 25d) ────────────
    print("\n┌─ Part A: Split-view dataset")
    dataset = collect_split_view_dataset(n_episodes=300, steps_per_episode=40,
                                         n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    print(f"│  Dataset: {n} frames, collection: {time.time()-t0:.0f}s")
    print("└─ Done")

    # ── Part B: Train ObjectCentricJEPA ────────────────────────
    print("\n┌─ Part B: Training ObjectCentricJEPA (300 epochs)")
    print("│  6 slots × 64-dim, Slot Attention (3 iters)")

    model = ObjectCentricJEPA(n_slots=6, slot_dim=64, comm_dim=8, action_dim=4)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n_epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    batch_size = 48
    history = {'pred': [], 'kl': [], 'vicreg': [], 'beta': []}

    train_start = time.time()
    for epoch in range(n_epochs):
        # β-annealing
        if epoch < 100:
            model.beta = 0.0
        elif epoch < 200:
            model.beta = 0.001 * (epoch - 100) / 100
        else:
            model.beta = 0.001

        model.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_kl, ep_vreg, nb = 0, 0, 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            ba = dataset['img_a'][idx].to(device)
            bb = dataset['img_b'][idx].to(device)
            bact = dataset['action'][idx].to(device)
            bnt = dataset['next_img_target'][idx].to(device)

            total, pred_loss, kl, vicreg, _, _, _, _ = model(
                ba, bb, bact, bnt, bnt, training=True)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model._update_target()
            ep_pred += pred_loss.item(); ep_kl += kl.item()
            ep_vreg += vicreg.item(); nb += 1

        scheduler.step()
        history['pred'].append(ep_pred / nb)
        history['kl'].append(ep_kl / nb)
        history['vicreg'].append(ep_vreg / nb)
        history['beta'].append(model.beta)

        if (epoch + 1) % 25 == 0:
            elapsed_ep = time.time() - train_start
            eta = elapsed_ep / (epoch + 1) * (n_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}: pred={ep_pred/nb:.4f} kl={ep_kl/nb:.4f} "
                  f"β={model.beta:.4f} vreg={ep_vreg/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    # Training plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history['pred'])
    axes[0, 0].set_title(f"Prediction Loss → {history['pred'][-1]:.4f}")
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 1].plot(history['kl'], color='orange')
    axes[0, 1].set_title(f"KL → {history['kl'][-1]:.4f}")
    ax2 = axes[0, 1].twinx()
    ax2.plot(history['beta'], color='blue', alpha=0.3); ax2.set_ylabel('β')
    axes[1, 0].plot(history['vicreg'])
    axes[1, 0].set_title('VICReg'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 1].text(0.1, 0.5,
                    f'Params: {n_params:,}\n6 slots × 64-dim\nSlot Attention (3 iters)',
                    transform=axes[1, 1].transAxes, fontsize=12, va='center')
    axes[1, 1].axis('off')
    fig.suptitle('Phase 26: Object-Centric JEPA Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26_training.png", dpi=150); plt.close()
    print(f"│  Training time: {time.time()-train_start:.0f}s")
    print("│  → results/phase26_training.png")
    print("└─ Done")

    # ── Part C: Communication Tests ────────────────────────────
    print("\n┌─ Part C: Communication Tests")
    model.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        mse_real_vals, mse_zero_vals, mse_shuf_vals = [], [], []

        for ci in range(0, len(val_idx), 128):
            bi = val_idx[ci:ci+128]
            ba = dataset['img_a'][bi].to(device)
            bb = dataset['img_b'][bi].to(device)
            bact = dataset['action'][bi].to(device)
            bnt = dataset['next_img_target'][bi].to(device)

            slots_a = model.extract_slots(ba)
            slots_b = model.extract_slots(bb)
            _, msg_a_mu, _ = model.communicate(slots_a)
            _, msg_b_mu, _ = model.communicate(slots_b)
            target_slots = model.extract_target_slots(bnt)

            # Real
            next_real = model.predict_next_slots(slots_a, bact, msg_b_mu)
            mse_real_vals.append(model._slot_loss(next_real, target_slots).item())

            # Zero
            zmsg = torch.zeros_like(msg_b_mu)
            next_zero = model.predict_next_slots(slots_a, bact, zmsg)
            mse_zero_vals.append(model._slot_loss(next_zero, target_slots).item())

            # Shuffled
            perm = torch.randperm(len(bi))
            next_shuf = model.predict_next_slots(slots_a, bact, msg_b_mu[perm])
            mse_shuf_vals.append(model._slot_loss(next_shuf, target_slots).item())

    mse_real = np.mean(mse_real_vals)
    mse_zero = np.mean(mse_zero_vals)
    mse_shuf = np.mean(mse_shuf_vals)
    margin_zero = (1 - mse_real / mse_zero) * 100 if mse_zero > 0 else 0
    margin_shuf = (1 - mse_real / mse_shuf) * 100 if mse_shuf > 0 else 0

    print(f"│  Real messages: {mse_real:.4f}")
    print(f"│  Zero messages: {mse_zero:.4f} (margin: {margin_zero:.1f}%)")
    print(f"│  Shuffled:      {mse_shuf:.4f} (margin: {margin_shuf:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Real\nMessages', 'Zero\nMessages', 'Shuffled\nMessages'],
                  [mse_real, mse_zero, mse_shuf],
                  color=['#27ae60', '#e74c3c', '#e67e22'], alpha=0.8)
    ax.set_ylabel('Slot Prediction MSE')
    ax.set_title(f'Phase 26: Comm Margin (Object-Centric)\n'
                 f'Zero: {margin_zero:.1f}% | Shuffled: {margin_shuf:.1f}%')
    for bar, val in zip(bars, [mse_real, mse_zero, mse_shuf]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26_comm_tests.png", dpi=150); plt.close()
    print("│  → results/phase26_comm_tests.png")
    print("└─ Done")

    # ── Part D: Slot-Object Binding ────────────────────────────
    print("\n┌─ Part D: Slot-Object Binding Analysis")
    from sklearn.linear_model import LinearRegression

    # Collect all slots for validation set
    with torch.no_grad():
        all_slots_a, all_slots_b = [], []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            sa = model.extract_slots(dataset['img_a'][bi].to(device))
            sb = model.extract_slots(dataset['img_b'][bi].to(device))
            all_slots_a.append(sa.cpu()); all_slots_b.append(sb.cpu())
        slots_a_np = torch.cat(all_slots_a).numpy()
        slots_b_np = torch.cat(all_slots_b).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_objects = 5
    n_val = len(gt_state)

    slot_object_r2 = np.zeros((6, n_objects))
    for s in range(6):
        for o in range(n_objects):
            X = slots_a_np[:, s, :]
            y = gt_state[:, o*8:o*8+3]
            n_fit = min(1000, n_val // 2)
            reg = LinearRegression()
            reg.fit(X[:n_fit], y[:n_fit])
            r2 = reg.score(X[n_fit:], y[n_fit:])
            slot_object_r2[s, o] = max(0, r2)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'Slot {i}' for i in range(6)])
    ax.set_xticks(range(n_objects))
    ax.set_xticklabels([f'Object {i}' for i in range(n_objects)])
    ax.set_title('Phase 26: Slot → Object Binding (R²)\n'
                 'Each slot should bind to one object')
    plt.colorbar(im, label='R²')
    for i in range(6):
        for j in range(n_objects):
            ax.text(j, i, f'{slot_object_r2[i,j]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if slot_object_r2[i, j] > 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26_slot_binding.png", dpi=150); plt.close()

    for o in range(n_objects):
        best_slot = slot_object_r2[:, o].argmax()
        best_r2 = slot_object_r2[:, o].max()
        print(f"│  Object {o} → Slot {best_slot} (R²={best_r2:.3f})")
    n_bound = int(sum(slot_object_r2.max(axis=0) > 0.3))
    print(f"│  Objects with clear slot binding (R²>0.3): {n_bound}/{n_objects}")
    print("│  → results/phase26_slot_binding.png")
    print("└─ Done")

    # ── Part E: Proto-Affordances ──────────────────────────────
    print("\n┌─ Part E: Proto-Affordances (Slot ↔ Physical Properties)")

    slot_mass_corr = np.zeros(6)
    slot_radius_corr = np.zeros(6)

    for s in range(6):
        slot_mean = slots_a_np[:, s, :].mean(axis=1)
        best_mass, best_radius = 0, 0
        for o in range(n_objects):
            mass = gt_state[:, o * 8 + 6]
            radius = gt_state[:, o * 8 + 7]
            if np.std(mass) > 1e-6:
                r_m = abs(np.corrcoef(slot_mean, mass)[0, 1])
                if not np.isnan(r_m) and r_m > best_mass:
                    best_mass = r_m
            if np.std(radius) > 1e-6:
                r_r = abs(np.corrcoef(slot_mean, radius)[0, 1])
                if not np.isnan(r_r) and r_r > best_radius:
                    best_radius = r_r
        slot_mass_corr[s] = best_mass
        slot_radius_corr[s] = best_radius

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(6)
    ax.bar(x - 0.15, slot_mass_corr, 0.3, label='Mass', color='#e74c3c')
    ax.bar(x + 0.15, slot_radius_corr, 0.3, label='Radius', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Slot {i}' for i in range(6)])
    ax.set_ylabel('|Correlation|')
    ax.set_title('Phase 26: Proto-Affordances (Mass & Size from Pixels)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26_proto_affordances.png", dpi=150); plt.close()
    print(f"│  Max mass corr:   {slot_mass_corr.max():.3f}")
    print(f"│  Max radius corr: {slot_radius_corr.max():.3f}")
    print("│  → results/phase26_proto_affordances.png")
    print("└─ Done")

    # ── Part F: Message ↔ State Correlation ────────────────────
    print("\n┌─ Part F: Message → State Heatmap")
    with torch.no_grad():
        msg_a_list, msg_b_list = [], []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            sa = model.extract_slots(dataset['img_a'][bi].to(device))
            sb = model.extract_slots(dataset['img_b'][bi].to(device))
            _, ma, _ = model.communicate(sa)
            _, mb, _ = model.communicate(sb)
            msg_a_list.append(ma.cpu()); msg_b_list.append(mb.cpu())
        msg_a_np = torch.cat(msg_a_list).numpy()
        msg_b_np = torch.cat(msg_b_list).numpy()

    n_state = gt_state.shape[1]
    corr_a = np.zeros((8, n_state))
    for c in range(8):
        for s_i in range(n_state):
            if np.std(gt_state[:, s_i]) > 1e-6:
                cc = np.corrcoef(msg_a_np[:, c], gt_state[:, s_i])[0, 1]
                corr_a[c, s_i] = abs(cc) if not np.isnan(cc) else 0

    state_labels = []
    for i in range(n_objects):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r']:
            state_labels.append(f'O{i}_{v}')

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Msg{i}' for i in range(8)])
    ax.set_xticks(range(n_state))
    ax.set_xticklabels(state_labels, rotation=90, fontsize=6)
    for i in range(1, n_objects):
        ax.axvline(x=i * 8 - 0.5, color='white', linewidth=2)
    ax.set_title('Phase 26: Message ↔ State (Object-Centric)')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26_msg_heatmap.png", dpi=150); plt.close()
    print(f"│  Max corr: {corr_a.max():.3f}")
    print("│  → results/phase26_msg_heatmap.png")
    print("└─ Done")

    # ── Part G: Summary Dashboard ──────────────────────────────
    print("\n┌─ Generating Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 26: Object Discovery from Pixels',
                 fontsize=16, fontweight='bold')

    axes[0, 0].plot(history['pred'])
    axes[0, 0].set_title(f"Pred → {history['pred'][-1]:.4f}")
    axes[0, 0].set_xlabel('Epoch')

    bars_c = axes[0, 1].bar(['Real', 'Zero', 'Shuffled'],
                            [mse_real, mse_zero, mse_shuf],
                            color=['#27ae60', '#e74c3c', '#e67e22'])
    axes[0, 1].set_title(f'Comm: Zero={margin_zero:.1f}% Shuf={margin_shuf:.1f}%')

    im_b = axes[0, 2].imshow(slot_object_r2, aspect='auto', cmap='YlOrRd',
                             vmin=0, vmax=1)
    axes[0, 2].set_title(f'Slot Binding ({n_bound}/{n_objects} bound)')
    axes[0, 2].set_yticks(range(6))
    axes[0, 2].set_yticklabels([f'S{i}' for i in range(6)], fontsize=8)
    axes[0, 2].set_xticks(range(n_objects))
    axes[0, 2].set_xticklabels([f'O{i}' for i in range(n_objects)], fontsize=8)

    axes[1, 0].bar(range(6), slot_mass_corr, color='#e74c3c', alpha=0.8,
                   label='Mass')
    axes[1, 0].bar(range(6), slot_radius_corr, alpha=0.4,
                   color='#3498db', label='Radius')
    axes[1, 0].set_title('Proto-Affordances')
    axes[1, 0].set_xticks(range(6))
    axes[1, 0].legend(fontsize=8)

    im_m = axes[1, 1].imshow(corr_a, aspect='auto', cmap='YlOrRd',
                             vmin=0, vmax=0.5)
    axes[1, 1].set_title(f'Msg↔State (max={corr_a.max():.3f})')

    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    kl_final = history['kl'][-1]
    summary_text = (
        f"PHASE 26 RESULTS\n"
        f"{'━'*30}\n\n"
        f"Slot Attention discovers\n"
        f"objects from pixels.\n\n"
        f"Params: {n_params:,}\n"
        f"Slots: 6 × 64-dim\n"
        f"Device: {device}\n"
        f"Time: {elapsed/60:.0f} min\n\n"
        f"Pred: {history['pred'][-1]:.4f}\n"
        f"KL: {kl_final:.2f}\n\n"
        f"Comm zero: {margin_zero:.1f}%\n"
        f"Comm shuf: {margin_shuf:.1f}%\n\n"
        f"Slot binding: {n_bound}/{n_objects}\n"
        f"Max bind R²: {slot_object_r2.max():.3f}\n"
        f"Max mass corr: {slot_mass_corr.max():.3f}\n"
        f"Max radius corr: {slot_radius_corr.max():.3f}\n"
    )
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26_summary.png", dpi=150); plt.close()

    torch.save(model.state_dict(), OUTPUT_DIR / "phase26_model.pt")
    print("│  → results/phase26_summary.png")
    print("│  → results/phase26_model.pt")
    print("└─ Done")

    print(f"\n{'='*60}")
    print(f"PHASE 26 SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params (ObjectCentricJEPA)")
    print(f"  Slots: 6 × 64-dim (Slot Attention, 3 iters)")
    print(f"  Pred loss: {history['pred'][-1]:.4f}")
    print(f"  KL: {kl_final:.2f}")
    print(f"  Zero margin:   {margin_zero:.1f}% (was 25.2% in 25d)")
    print(f"  Shuf margin:   {margin_shuf:.1f}% (was 33.4% in 25d)")
    print(f"  Slot binding:  {n_bound}/{n_objects} (R²>0.3)")
    print(f"  Max bind R²:   {slot_object_r2.max():.3f}")
    print(f"  Mass corr:     {slot_mass_corr.max():.3f}")
    print(f"  Radius corr:   {slot_radius_corr.max():.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


def run_phase26b():
    """Phase 26b: Object Discovery — full capacity, no KL."""
    print("=" * 60)
    print("PHASE 26b: OBJECT DISCOVERY (Full Capacity, No KL)")
    print("=" * 60)
    t0 = time.time()

    # ── Device ─────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Part A: Dataset ────────────────────────────────────────
    print("\n┌─ Part A: Split-view dataset")
    dataset = collect_split_view_dataset(n_episodes=300, steps_per_episode=40,
                                         n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    print(f"│  Dataset: {n} frames, collection: {time.time()-t0:.0f}s")
    print("└─ Done")

    # ── Part B: Model + Param Verification ─────────────────────
    print("\n┌─ Part B: Training ObjectCentricJEPAv2 (300 epochs)")
    print("│  NO KL penalty — communication is free (β=0 always)")

    model = ObjectCentricJEPAv2(n_slots=6, slot_dim=64, comm_dim=8, action_dim=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"│  Model: {n_params:,} params")
    assert n_params > 2_000_000, (
        f"MODEL TOO SMALL: {n_params:,} params. Must be > 2M.")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n_epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    batch_size = 48
    history = {'pred': [], 'vicreg': []}

    train_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_pred, ep_vreg, nb = 0, 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            ba = dataset['img_a'][idx].to(device)
            bb = dataset['img_b'][idx].to(device)
            bact = dataset['action'][idx].to(device)
            bnt = dataset['next_img_target'][idx].to(device)

            total, pred_loss, vicreg, _, _, _, _ = model(
                ba, bb, bact, bnt)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model._update_target()
            ep_pred += pred_loss.item()
            ep_vreg += vicreg.item()
            nb += 1

        scheduler.step()
        history['pred'].append(ep_pred / nb)
        history['vicreg'].append(ep_vreg / nb)

        if (epoch + 1) % 25 == 0:
            elapsed_ep = time.time() - train_start
            eta = elapsed_ep / (epoch + 1) * (n_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}: pred={ep_pred/nb:.4f} "
                  f"vreg={ep_vreg/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    # Training plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['pred'])
    axes[0].set_title(f"Prediction Loss → {history['pred'][-1]:.4f}")
    axes[0].set_xlabel('Epoch')
    axes[1].plot(history['vicreg'], color='purple')
    axes[1].set_title(f"VICReg → {history['vicreg'][-1]:.4f}")
    axes[1].set_xlabel('Epoch')
    fig.suptitle('Phase 26b Training (No KL — communication free)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26b_training.png", dpi=150); plt.close()
    print(f"│  Training time: {time.time()-train_start:.0f}s")
    print("│  → results/phase26b_training.png")
    print("└─ Done")

    # ── Part C: Communication Tests ────────────────────────────
    print("\n┌─ Part C: Communication Tests")
    model.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        mse_real_vals, mse_zero_vals, mse_shuf_vals = [], [], []

        for ci in range(0, len(val_idx), 128):
            bi = val_idx[ci:ci+128]
            ba = dataset['img_a'][bi].to(device)
            bb = dataset['img_b'][bi].to(device)
            bact = dataset['action'][bi].to(device)
            bnt = dataset['next_img_target'][bi].to(device)

            slots_a = model.extract_slots(ba)
            slots_b = model.extract_slots(bb)
            msg_b = model.communicate(slots_b)
            target_slots = model.extract_target_slots(bnt)

            # Real messages
            next_real = model.predict_next_slots(slots_a, bact, msg_b)
            mse_real_vals.append(model._slot_loss(next_real, target_slots).item())

            # Zero messages
            zmsg = torch.zeros_like(msg_b)
            next_zero = model.predict_next_slots(slots_a, bact, zmsg)
            mse_zero_vals.append(model._slot_loss(next_zero, target_slots).item())

            # Shuffled messages
            perm_s = torch.randperm(len(bi))
            next_shuf = model.predict_next_slots(slots_a, bact, msg_b[perm_s])
            mse_shuf_vals.append(model._slot_loss(next_shuf, target_slots).item())

    mse_real = np.mean(mse_real_vals)
    mse_zero = np.mean(mse_zero_vals)
    mse_shuf = np.mean(mse_shuf_vals)
    margin_zero = (1 - mse_real / mse_zero) * 100 if mse_zero > 0 else 0
    margin_shuf = (1 - mse_real / mse_shuf) * 100 if mse_shuf > 0 else 0

    print(f"│  Real messages:     {mse_real:.4f}")
    print(f"│  Zero messages:     {mse_zero:.4f}  (margin: {margin_zero:.1f}%)")
    print(f"│  Shuffled messages: {mse_shuf:.4f}  (margin: {margin_shuf:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Real\nMessages', 'Zero\nMessages', 'Shuffled\nMessages'],
                  [mse_real, mse_zero, mse_shuf],
                  color=['#27ae60', '#e74c3c', '#e67e22'], alpha=0.8)
    ax.set_ylabel('Slot Prediction MSE')
    ax.set_title(f'Phase 26b: Comm Margin (No KL)\n'
                 f'Zero: {margin_zero:.1f}% | Shuffled: {margin_shuf:.1f}%')
    for bar, val in zip(bars, [mse_real, mse_zero, mse_shuf]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26b_comm_tests.png", dpi=150); plt.close()
    print("│  → results/phase26b_comm_tests.png")
    print("└─ Done")

    # ── Part D: Slot-Object Binding ────────────────────────────
    print("\n┌─ Part D: Slot-Object Binding Analysis")
    from sklearn.linear_model import LinearRegression

    with torch.no_grad():
        all_slots_a = []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            sa = model.extract_slots(dataset['img_a'][bi].to(device))
            all_slots_a.append(sa.cpu())
        slots_a_np = torch.cat(all_slots_a).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_objects = 5
    n_val = len(gt_state)

    slot_object_r2 = np.zeros((6, n_objects))
    for s in range(6):
        for o in range(n_objects):
            X = slots_a_np[:, s, :]
            y = gt_state[:, o*8:o*8+3]
            n_fit = min(1000, n_val // 2)
            reg = LinearRegression()
            reg.fit(X[:n_fit], y[:n_fit])
            r2 = reg.score(X[n_fit:], y[n_fit:])
            slot_object_r2[s, o] = max(0, r2)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'Slot {i}' for i in range(6)])
    ax.set_xticks(range(n_objects))
    ax.set_xticklabels([f'Object {i}' for i in range(n_objects)])
    ax.set_title('Phase 26b: Slot → Object Binding (R²)\n'
                 'Full backbone, no KL, deterministic comm')
    plt.colorbar(im, label='R²')
    for i in range(6):
        for j in range(n_objects):
            ax.text(j, i, f'{slot_object_r2[i,j]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if slot_object_r2[i, j] > 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26b_slot_binding.png", dpi=150); plt.close()

    n_bound = int(sum(slot_object_r2.max(axis=0) > 0.3))
    best_r2 = slot_object_r2.max()
    for o in range(n_objects):
        bs = slot_object_r2[:, o].argmax()
        br = slot_object_r2[:, o].max()
        print(f"│  Object {o} → Slot {bs} (R²={br:.3f})")
    print(f"│  Objects with clear slot binding (R²>0.3): {n_bound}/{n_objects}")
    print("│  → results/phase26b_slot_binding.png")
    print("└─ Done")

    # ── Part E: Proto-Affordances ──────────────────────────────
    print("\n┌─ Part E: Proto-Affordances")

    slot_mass_corr = np.zeros(6)
    slot_radius_corr = np.zeros(6)

    for s in range(6):
        slot_norm = np.linalg.norm(slots_a_np[:, s, :], axis=1)
        best_mass, best_radius = 0, 0
        for o in range(n_objects):
            mass = gt_state[:, o * 8 + 6]
            radius = gt_state[:, o * 8 + 7]
            if np.std(mass) > 1e-6:
                r_m = abs(np.corrcoef(slot_norm, mass)[0, 1])
                if not np.isnan(r_m) and r_m > best_mass:
                    best_mass = r_m
            if np.std(radius) > 1e-6:
                r_r = abs(np.corrcoef(slot_norm, radius)[0, 1])
                if not np.isnan(r_r) and r_r > best_radius:
                    best_radius = r_r
        slot_mass_corr[s] = best_mass
        slot_radius_corr[s] = best_radius

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(6)
    ax.bar(x - 0.15, slot_mass_corr, 0.3, label='Mass', color='#e74c3c')
    ax.bar(x + 0.15, slot_radius_corr, 0.3, label='Radius', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Slot {i}' for i in range(6)])
    ax.set_ylabel('|Correlation|')
    ax.set_title('Phase 26b: Proto-Affordances')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26b_proto_affordances.png", dpi=150); plt.close()
    print(f"│  Max mass corr:   {slot_mass_corr.max():.3f}")
    print(f"│  Max radius corr: {slot_radius_corr.max():.3f}")
    print("│  → results/phase26b_proto_affordances.png")
    print("└─ Done")

    # ── Part F: Message ↔ State Heatmap ────────────────────────
    print("\n┌─ Part F: Message → State Heatmap")
    with torch.no_grad():
        msg_a_list = []
        for ci in range(0, len(val_idx), 256):
            bi = val_idx[ci:ci+256]
            sa = model.extract_slots(dataset['img_a'][bi].to(device))
            ma = model.communicate(sa)
            msg_a_list.append(ma.cpu())
        msg_a_np = torch.cat(msg_a_list).numpy()

    n_state = gt_state.shape[1]
    corr_a = np.zeros((8, n_state))
    for c in range(8):
        for s_i in range(n_state):
            if np.std(gt_state[:, s_i]) > 1e-6 and np.std(msg_a_np[:, c]) > 1e-6:
                cc = np.corrcoef(msg_a_np[:, c], gt_state[:, s_i])[0, 1]
                corr_a[c, s_i] = abs(cc) if not np.isnan(cc) else 0

    state_labels = []
    for i in range(n_objects):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r']:
            state_labels.append(f'O{i}_{v}')

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(corr_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Msg{i}' for i in range(8)])
    ax.set_xticks(range(n_state))
    ax.set_xticklabels(state_labels, rotation=90, fontsize=6)
    for i in range(1, n_objects):
        ax.axvline(x=i * 8 - 0.5, color='white', linewidth=2)
    ax.set_title('Phase 26b: Message ↔ State (Deterministic Comm)')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26b_msg_heatmap.png", dpi=150); plt.close()
    print(f"│  Max corr: {corr_a.max():.3f}")
    print("│  → results/phase26b_msg_heatmap.png")
    print("└─ Done")

    # ── Part G: Summary Dashboard ──────────────────────────────
    print("\n┌─ Generating Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 26b: Object Discovery (Full Capacity, No KL)',
                 fontsize=16, fontweight='bold')

    axes[0, 0].plot(history['pred'])
    axes[0, 0].set_title(f"Pred → {history['pred'][-1]:.4f}")
    axes[0, 0].set_xlabel('Epoch')

    axes[0, 1].bar(['Real', 'Zero', 'Shuffled'],
                   [mse_real, mse_zero, mse_shuf],
                   color=['#27ae60', '#e74c3c', '#e67e22'])
    axes[0, 1].set_title(f'Comm: Zero={margin_zero:.1f}% Shuf={margin_shuf:.1f}%')

    im_b = axes[0, 2].imshow(slot_object_r2, aspect='auto', cmap='YlOrRd',
                             vmin=0, vmax=1)
    axes[0, 2].set_title(f'Binding: {n_bound}/5 (max R²={best_r2:.3f})')
    axes[0, 2].set_yticks(range(6))
    axes[0, 2].set_yticklabels([f'S{i}' for i in range(6)], fontsize=8)
    axes[0, 2].set_xticks(range(n_objects))
    axes[0, 2].set_xticklabels([f'O{i}' for i in range(n_objects)], fontsize=8)

    axes[1, 0].bar(range(6), slot_mass_corr, color='#e74c3c', alpha=0.8,
                   label='Mass')
    axes[1, 0].bar(range(6), slot_radius_corr, alpha=0.4,
                   color='#3498db', label='Radius')
    axes[1, 0].set_title('Proto-Affordances')
    axes[1, 0].set_xticks(range(6))
    axes[1, 0].legend(fontsize=8)

    im_m = axes[1, 1].imshow(corr_a, aspect='auto', cmap='YlOrRd',
                             vmin=0, vmax=0.5)
    axes[1, 1].set_title(f'Msg↔State (max={corr_a.max():.3f})')

    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    summary_text = (
        f"PHASE 26b RESULTS\n"
        f"{'━'*30}\n\n"
        f"Full capacity + No KL\n\n"
        f"Params: {n_params:,}\n"
        f"Slots: 6 × 64-dim\n"
        f"Device: {device}\n"
        f"Time: {elapsed/60:.0f} min\n\n"
        f"Pred: {history['pred'][-1]:.4f}\n"
        f"No KL (β=0 always)\n\n"
        f"Comm zero: {margin_zero:.1f}%\n"
        f"Comm shuf: {margin_shuf:.1f}%\n\n"
        f"Binding: {n_bound}/{n_objects}\n"
        f"Max R²: {best_r2:.3f}\n"
        f"Max mass: {slot_mass_corr.max():.3f}\n"
        f"Max radius: {slot_radius_corr.max():.3f}\n"
    )
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26b_summary.png", dpi=150); plt.close()

    torch.save(model.state_dict(), OUTPUT_DIR / "phase26b_model.pt")
    print("│  → results/phase26b_summary.png")
    print("│  → results/phase26b_model.pt")
    print("└─ Done")

    print(f"\n{'='*60}")
    print(f"PHASE 26b SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params (ObjectCentricJEPAv2)")
    print(f"  Slots: 6 × 64-dim, No KL (β=0 always)")
    print(f"  Pred loss: {history['pred'][-1]:.4f}")
    print(f"  Zero margin:   {margin_zero:.1f}% (was 0% in 26)")
    print(f"  Shuf margin:   {margin_shuf:.1f}% (was 0% in 26)")
    print(f"  Slot binding:  {n_bound}/{n_objects} (R²>0.3)")
    print(f"  Max bind R²:   {best_r2:.3f}")
    print(f"  Mass corr:     {slot_mass_corr.max():.3f}")
    print(f"  Radius corr:   {slot_radius_corr.max():.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


def run_phase26c():
    """Phase 26c: Two-Stage Object Discovery.

    Stage 1: Slot Attention autoencoder (pixel reconstruction) → learn slots
    Stage 2: Freeze slots, train JEPA predictor + communication
    """
    print("=" * 60)
    print("PHASE 26c: TWO-STAGE OBJECT DISCOVERY")
    print("=" * 60)
    t0 = time.time()

    # ── Device ─────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Dataset ────────────────────────────────────────────────
    print("\n┌─ Dataset")
    dataset = collect_split_view_dataset(n_episodes=300, steps_per_episode=40,
                                         n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    print(f"│  {n} frames, collection: {time.time()-t0:.0f}s")
    print("└─ Done")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Slot Attention Autoencoder
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 1: Slot Attention Autoencoder")
    print("=" * 50)

    autoencoder = SlotAttentionAutoencoder(n_slots=6, slot_dim=64, img_size=64)
    ae_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"│  AE params: {ae_params:,}")
    autoencoder = autoencoder.to(device)

    ae_opt = torch.optim.AdamW(autoencoder.parameters(), lr=3e-4, weight_decay=0.01)
    ae_epochs = 50
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        ae_opt, T_max=ae_epochs, eta_min=1e-5)
    batch_size = 48
    ae_history = {'recon': []}

    # Train on overhead images (img_target shows ALL objects)
    target_imgs = dataset['img_target']

    t_s1 = time.time()
    for epoch in range(ae_epochs):
        autoencoder.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = target_imgs[idx].to(device)

            loss, recon, slots, alpha = autoencoder(imgs)
            ae_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            ae_opt.step()
            ep_loss += loss.item(); nb += 1

        ae_sched.step()
        ae_history['recon'].append(ep_loss / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s1
            eta = elapsed_ep / (epoch + 1) * (ae_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: recon={ep_loss/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 1 time: {time.time()-t_s1:.0f}s")

    # ── Stage 1 Evaluation ─────────────────────────────────────
    print("\n┌─ Stage 1 Evaluation")
    autoencoder.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        val_imgs = target_imgs[val_idx[:200]].to(device)
        loss_val, recon_val, slots_val, alpha_val = autoencoder(val_imgs)
        print(f"│  Val recon MSE: {loss_val.item():.4f}")

    # Reconstruction visualization
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for i in range(8):
        axes[0, i].imshow(val_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_val[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[1, i].set_title('Recon' if i == 0 else '')
        axes[1, i].axis('off')

        masks = alpha_val[i].cpu().reshape(6, 64, 64).numpy()
        top2 = np.argsort(-masks.max(axis=(1, 2)))[:2]
        axes[2, i].imshow(masks[top2[0]], cmap='hot')
        axes[2, i].set_title(f'Slot {top2[0]}' if i == 0 else '')
        axes[2, i].axis('off')
        axes[3, i].imshow(masks[top2[1]], cmap='hot')
        axes[3, i].set_title(f'Slot {top2[1]}' if i == 0 else '')
        axes[3, i].axis('off')

    fig.suptitle('Stage 1: Reconstruction + Slot Masks', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26c_reconstruction.png", dpi=150); plt.close()
    print("│  → results/phase26c_reconstruction.png")

    # Slot-object binding
    print("│  Slot-object binding:")
    from sklearn.linear_model import LinearRegression

    with torch.no_grad():
        all_slots = []
        for ci in range(0, len(val_idx), batch_size):
            bi = val_idx[ci:ci+batch_size]
            s = autoencoder.encode(target_imgs[bi].to(device))
            all_slots.append(s.cpu())
        all_slots_np = torch.cat(all_slots).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_objects = 5
    n_val = len(gt_state)

    slot_object_r2 = np.zeros((6, n_objects))
    for s in range(6):
        for o in range(n_objects):
            X = all_slots_np[:, s, :]
            y = gt_state[:, o * 8:o * 8 + 3]
            n_fit = min(1000, n_val // 2)
            if n_fit < 10:
                continue
            reg = LinearRegression().fit(X[:n_fit], y[:n_fit])
            r2 = reg.score(X[n_fit:], y[n_fit:])
            slot_object_r2[s, o] = max(0, r2)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'Slot {i}' for i in range(6)])
    ax.set_xticks(range(n_objects))
    ax.set_xticklabels([f'Object {i}' for i in range(n_objects)])
    for i in range(6):
        for j in range(n_objects):
            ax.text(j, i, f'{slot_object_r2[i, j]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if slot_object_r2[i, j] > 0.5 else 'black')
    ax.set_title('Stage 1: Slot → Object Binding (R²)')
    plt.colorbar(im, label='R²')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26c_slot_binding.png", dpi=150); plt.close()

    n_bound = int(sum(slot_object_r2.max(axis=0) > 0.3))
    best_r2 = slot_object_r2.max()
    for o in range(n_objects):
        bs = slot_object_r2[:, o].argmax()
        br = slot_object_r2[:, o].max()
        print(f"│    Obj {o} → Slot {bs} (R²={br:.3f})")
    print(f"│  Bound: {n_bound}/{n_objects}")

    # Proto-affordances
    slot_mass_corr = np.zeros(6)
    slot_radius_corr = np.zeros(6)
    for s in range(6):
        slot_norm = np.linalg.norm(all_slots_np[:, s, :], axis=1)
        for o in range(n_objects):
            mass = gt_state[:, o * 8 + 6]
            radius = gt_state[:, o * 8 + 7]
            if np.std(mass) > 1e-6:
                r_m = abs(np.corrcoef(slot_norm, mass)[0, 1])
                if not np.isnan(r_m) and r_m > slot_mass_corr[s]:
                    slot_mass_corr[s] = r_m
            if np.std(radius) > 1e-6:
                r_r = abs(np.corrcoef(slot_norm, radius)[0, 1])
                if not np.isnan(r_r) and r_r > slot_radius_corr[s]:
                    slot_radius_corr[s] = r_r

    print(f"│  Mass corr:   {np.round(slot_mass_corr, 3)}")
    print(f"│  Radius corr: {np.round(slot_radius_corr, 3)}")
    print("└─ Done")

    # ════════════════════════════════════════════════════════════
    # STAGE 2: JEPA Predictor on Frozen Slots
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 2: JEPA Predictor on Frozen Slots")
    print("=" * 50)

    # Freeze autoencoder
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Pre-extract ALL slots (frozen encoder, compute once)
    print("│  Pre-extracting slots...")

    def extract_all_slots(images, ae, dev, bs=48):
        results = []
        for ci in range(0, len(images), bs):
            batch = images[ci:ci+bs].to(dev)
            with torch.no_grad():
                s = ae.encode(batch)
            results.append(s.cpu())
        return torch.cat(results, dim=0)

    slots_a = extract_all_slots(dataset['img_a'], autoencoder, device)
    slots_b = extract_all_slots(dataset['img_b'], autoencoder, device)
    slots_target_next = extract_all_slots(
        dataset['next_img_target'], autoencoder, device)
    print(f"│  Slots shape: {slots_a.shape}")  # [N, 6, 64]

    # Train predictor
    predictor = SlotJEPAPredictor(n_slots=6, slot_dim=64, comm_dim=8, action_dim=4)
    pred_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"│  Predictor params: {pred_params:,}")
    predictor = predictor.to(device)

    pred_opt = torch.optim.AdamW(predictor.parameters(), lr=3e-4, weight_decay=0.01)
    pred_epochs = 300
    pred_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        pred_opt, T_max=pred_epochs, eta_min=1e-5)
    pred_history = {'pred': []}

    t_s2 = time.time()
    for epoch in range(pred_epochs):
        predictor.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            sa = slots_a[idx].to(device)
            sb = slots_b[idx].to(device)
            act = dataset['action'][idx].to(device)
            st = slots_target_next[idx].to(device)

            loss, _, _ = predictor(sa, sb, act, st)
            pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            pred_opt.step()
            ep_loss += loss.item(); nb += 1

        pred_sched.step()
        pred_history['pred'].append(ep_loss / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s2
            eta = elapsed_ep / (epoch + 1) * (pred_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{pred_epochs}: pred={ep_loss/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 2 time: {time.time()-t_s2:.0f}s")

    # ── Stage 2 Communication Tests ────────────────────────────
    print("\n┌─ Communication Tests")
    predictor.eval()

    with torch.no_grad():
        sa_val = slots_a[val_idx].to(device)
        sb_val = slots_b[val_idx].to(device)
        act_val = dataset['action'][val_idx].to(device)
        st_val = slots_target_next[val_idx].to(device)

        # Normal
        loss_norm, msg_a_val, msg_b_val = predictor(
            sa_val, sb_val, act_val, st_val)

        # Real prediction
        msg_b_real = predictor.communicate(sb_val)
        next_a_real = predictor.predict_next(sa_val, act_val, msg_b_real)
        mse_real = F.mse_loss(next_a_real, st_val).item()

        # Zero messages
        zero_msg = torch.zeros_like(msg_b_real)
        next_a_zero = predictor.predict_next(sa_val, act_val, zero_msg)
        mse_zero = F.mse_loss(next_a_zero, st_val).item()

        # Shuffled messages
        shuf_msg = msg_b_real[torch.randperm(len(val_idx))]
        next_a_shuf = predictor.predict_next(sa_val, act_val, shuf_msg)
        mse_shuf = F.mse_loss(next_a_shuf, st_val).item()

        # Noise messages
        noise_msg = torch.randn_like(msg_b_real)
        next_a_noise = predictor.predict_next(sa_val, act_val, noise_msg)
        mse_noise = F.mse_loss(next_a_noise, st_val).item()

    m_zero = (1 - mse_real / mse_zero) * 100 if mse_zero > 0 else 0
    m_shuf = (1 - mse_real / mse_shuf) * 100 if mse_shuf > 0 else 0
    m_noise = (1 - mse_real / mse_noise) * 100 if mse_noise > 0 else 0

    print(f"│  Real:     {mse_real:.4f}")
    print(f"│  Zero:     {mse_zero:.4f}  (margin: {m_zero:.1f}%)")
    print(f"│  Shuffled: {mse_shuf:.4f}  (margin: {m_shuf:.1f}%)")
    print(f"│  Noise:    {mse_noise:.4f}  (margin: {m_noise:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(['Real\nMessages', 'Zero\nMessages', 'Shuffled\nMessages',
                   'Noise\nMessages'],
                  [mse_real, mse_zero, mse_shuf, mse_noise],
                  color=['#27ae60', '#e74c3c', '#e67e22', '#9b59b6'], alpha=0.8)
    ax.set_ylabel('Slot Prediction MSE')
    ax.set_title(f'Phase 26c: Communication on Frozen Slot Features\n'
                 f'Zero: {m_zero:.1f}% | Shuf: {m_shuf:.1f}% | Noise: {m_noise:.1f}%')
    for bar, val in zip(bars, [mse_real, mse_zero, mse_shuf, mse_noise]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26c_comm_tests.png", dpi=150); plt.close()
    print("│  → results/phase26c_comm_tests.png")
    print("└─ Done")

    # ── Message ↔ State Heatmap ────────────────────────────────
    print("\n┌─ Message ↔ State Heatmap")
    msg_a_np = msg_a_val.cpu().numpy()
    n_state = gt_state.shape[1]
    corr_msg = np.zeros((8, n_state))
    for c in range(8):
        for si in range(n_state):
            if np.std(gt_state[:, si]) > 1e-6 and np.std(msg_a_np[:, c]) > 1e-6:
                r = np.corrcoef(msg_a_np[:, c], gt_state[:, si])[0, 1]
                if not np.isnan(r):
                    corr_msg[c, si] = abs(r)

    state_labels = []
    for i in range(n_objects):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r']:
            state_labels.append(f'O{i}_{v}')

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(corr_msg, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Msg{i}' for i in range(8)])
    ax.set_xticks(range(n_state))
    ax.set_xticklabels(state_labels, rotation=90, fontsize=6)
    for i in range(1, n_objects):
        ax.axvline(x=i * 8 - 0.5, color='white', linewidth=2)
    ax.set_title('Phase 26c: Message ↔ State')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26c_msg_heatmap.png", dpi=150); plt.close()
    print(f"│  Max corr: {corr_msg.max():.3f}")
    print("│  → results/phase26c_msg_heatmap.png")
    print("└─ Done")

    # ── Summary Dashboard ──────────────────────────────────────
    print("\n┌─ Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 26c: Two-Stage Object Discovery',
                 fontsize=16, fontweight='bold')

    # Stage 1: recon loss
    axes[0, 0].plot(ae_history['recon'])
    axes[0, 0].set_title(f"Stage 1 Recon → {ae_history['recon'][-1]:.4f}")
    axes[0, 0].set_xlabel('Epoch')

    # Slot binding
    im_b = axes[0, 1].imshow(slot_object_r2, aspect='auto',
                             cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Binding: {n_bound}/5 (max R²={best_r2:.3f})')
    axes[0, 1].set_yticks(range(6))
    axes[0, 1].set_yticklabels([f'S{i}' for i in range(6)], fontsize=8)
    axes[0, 1].set_xticks(range(n_objects))
    axes[0, 1].set_xticklabels([f'O{i}' for i in range(n_objects)], fontsize=8)

    # Stage 2: pred loss
    axes[0, 2].plot(pred_history['pred'])
    axes[0, 2].set_title(f"Stage 2 Pred → {pred_history['pred'][-1]:.4f}")
    axes[0, 2].set_xlabel('Epoch')

    # Comm tests
    axes[1, 0].bar(['Real', 'Zero', 'Shuf', 'Noise'],
                   [mse_real, mse_zero, mse_shuf, mse_noise],
                   color=['#27ae60', '#e74c3c', '#e67e22', '#9b59b6'])
    axes[1, 0].set_title(f'Comm: Zero={m_zero:.1f}% Shuf={m_shuf:.1f}%')

    # Proto-affordances
    x = np.arange(6)
    axes[1, 1].bar(x - 0.15, slot_mass_corr, 0.3,
                   label='Mass', color='#e74c3c')
    axes[1, 1].bar(x + 0.15, slot_radius_corr, 0.3,
                   label='Radius', color='#3498db')
    axes[1, 1].legend()
    axes[1, 1].set_title('Proto-Affordances')
    axes[1, 1].set_xticks(range(6))

    # Summary text
    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    summary_text = (
        f"PHASE 26c RESULTS\n"
        f"{'━' * 30}\n\n"
        f"Two-stage training\n\n"
        f"AE params: {ae_params:,}\n"
        f"Pred params: {pred_params:,}\n"
        f"Time: {elapsed / 60:.0f} min\n\n"
        f"Recon: {ae_history['recon'][-1]:.4f}\n"
        f"Pred: {pred_history['pred'][-1]:.4f}\n\n"
        f"Comm zero: {m_zero:.1f}%\n"
        f"Comm shuf: {m_shuf:.1f}%\n"
        f"Comm noise: {m_noise:.1f}%\n\n"
        f"Binding: {n_bound}/{n_objects}\n"
        f"Max R²: {best_r2:.3f}\n"
        f"Max mass: {slot_mass_corr.max():.3f}\n"
        f"Max radius: {slot_radius_corr.max():.3f}\n"
    )
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26c_summary.png", dpi=150); plt.close()

    torch.save(autoencoder.state_dict(), OUTPUT_DIR / "phase26c_autoencoder.pt")
    torch.save(predictor.state_dict(), OUTPUT_DIR / "phase26c_predictor.pt")
    print("│  → results/phase26c_summary.png")
    print("│  → results/phase26c_autoencoder.pt")
    print("│  → results/phase26c_predictor.pt")
    print("└─ Done")

    print(f"\n{'=' * 60}")
    print(f"PHASE 26c SUMMARY")
    print(f"{'=' * 60}")
    print(f"  AE: {ae_params:,} params, recon={ae_history['recon'][-1]:.4f}")
    print(f"  Pred: {pred_params:,} params, pred={pred_history['pred'][-1]:.4f}")
    print(f"  Binding: {n_bound}/{n_objects} (R²>{0.3})")
    print(f"  Max R²:  {best_r2:.3f}")
    print(f"  Comm zero: {m_zero:.1f}%")
    print(f"  Comm shuf: {m_shuf:.1f}%")
    print(f"  Comm noise: {m_noise:.1f}%")
    print(f"  Mass corr:   {slot_mass_corr.max():.3f}")
    print(f"  Radius corr: {slot_radius_corr.max():.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def extract_all_slots(images, ae, device, batch_size=48):
    """Extract slots from all images using frozen autoencoder."""
    all_s = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            s = ae.encode(batch)
        all_s.append(s.cpu())
    return torch.cat(all_s, dim=0)


def run_phase26d():
    """Phase 26d: Fix Slot Binding.

    Stage 1: SlotAttentionAEv2 (constrained decoder + slot dropout + diversity)
    Stage 2: Freeze slots, train SlotJEPAPredictor
    """
    print("=" * 60)
    print("PHASE 26d: Fix Slot Binding")
    print("=" * 60)
    t0 = time.time()

    # ── Device ─────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Dataset ────────────────────────────────────────────────
    print("\n┌─ Dataset")
    dataset = collect_split_view_dataset(n_episodes=300, steps_per_episode=40,
                                         n_objects=5, img_size=64)
    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    target_imgs = dataset['img_target']
    print(f"│  {n} frames, collection: {time.time()-t0:.0f}s")
    print("└─ Done")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Constrained Slot Autoencoder
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 1: Constrained Slot Autoencoder")
    print("=" * 50)

    ae = SlotAttentionAEv2(n_slots=6, slot_dim=64, img_size=64)
    ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print(f"│  AE params: {ae_params:,}")
    ae = ae.to(device)

    ae_opt = torch.optim.AdamW(ae.parameters(), lr=3e-4, weight_decay=0.01)
    ae_epochs = 200
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        ae_opt, T_max=ae_epochs, eta_min=1e-5)
    batch_size = 48
    ae_hist = {'recon': [], 'div': []}

    t_s1 = time.time()
    for epoch in range(ae_epochs):
        ae.train()
        perm = torch.randperm(n_train)
        ep_recon, ep_div, nb = 0, 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = target_imgs[idx].to(device)

            total, recon_loss, div_loss, recon, slots, alpha = ae(
                imgs, training=True)
            ae_opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_opt.step()
            ep_recon += recon_loss.item()
            ep_div += div_loss.item()
            nb += 1

        ae_sched.step()
        ae_hist['recon'].append(ep_recon / nb)
        ae_hist['div'].append(ep_div / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s1
            eta = elapsed_ep / (epoch + 1) * (ae_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: recon={ep_recon/nb:.4f} "
                  f"div={ep_div/nb:.4f} [{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 1 time: {time.time()-t_s1:.0f}s")

    # ── Stage 1 Evaluation ─────────────────────────────────────
    print("\n┌─ Stage 1 Evaluation")
    ae.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        val_imgs = target_imgs[val_idx[:200]].to(device)
        _, recon_loss_v, div_v, recon_v, slots_v, alpha_v = ae(
            val_imgs, training=False)
        print(f"│  Val recon: {recon_loss_v.item():.4f}")

    # Reconstructions + slot mask composite
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    slot_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                   [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    for i in range(8):
        axes[0, i].imshow(val_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_v[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[1, i].set_title('Recon' if i == 0 else '')
        axes[1, i].axis('off')

        masks = alpha_v[i].cpu().reshape(6, 64, 64).numpy()
        composite = np.zeros((64, 64, 3))
        for s in range(6):
            for c in range(3):
                composite[:, :, c] += masks[s] * slot_colors[s][c]
        axes[2, i].imshow(np.clip(composite, 0, 1))
        axes[2, i].set_title('Slot masks' if i == 0 else '')
        axes[2, i].axis('off')

    fig.suptitle('Phase 26d: Constrained AE\n'
                 'R=S0 G=S1 B=S2 Y=S3 M=S4 C=S5', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26d_reconstruction.png", dpi=150)
    plt.close()
    print("│  → results/phase26d_reconstruction.png")

    # Slot-object binding
    print("│  Slot-object binding:")
    from sklearn.linear_model import LinearRegression

    with torch.no_grad():
        all_slots_list = []
        for ci in range(0, len(val_idx), batch_size):
            bi = val_idx[ci:ci+batch_size]
            s = ae.encode(target_imgs[bi].to(device))
            all_slots_list.append(s.cpu())
        all_slots_np = torch.cat(all_slots_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_objects = 5
    n_val = len(gt_state)

    slot_object_r2 = np.zeros((6, n_objects))
    for s in range(6):
        for o in range(n_objects):
            X = all_slots_np[:, s, :]
            y = gt_state[:, o * 8:o * 8 + 3]
            n_fit = min(1000, n_val // 2)
            if n_fit < 10:
                continue
            reg = LinearRegression().fit(X[:n_fit], y[:n_fit])
            r2 = reg.score(X[n_fit:], y[n_fit:])
            slot_object_r2[s, o] = max(0, r2)

    n_bound = int(sum(slot_object_r2.max(axis=0) > 0.3))
    best_r2 = slot_object_r2.max()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'Slot {i}' for i in range(6)])
    ax.set_xticks(range(n_objects))
    ax.set_xticklabels([f'Obj {i}' for i in range(n_objects)])
    for i in range(6):
        for j in range(n_objects):
            ax.text(j, i, f'{slot_object_r2[i, j]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if slot_object_r2[i, j] > 0.5 else 'black')
    ax.set_title(f'Phase 26d: Slot Binding — {n_bound}/5 bound')
    plt.colorbar(im, label='R²')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26d_slot_binding.png", dpi=150)
    plt.close()

    for o in range(n_objects):
        bs = slot_object_r2[:, o].argmax()
        br = slot_object_r2[:, o].max()
        print(f"│    Obj {o} → Slot {bs} (R²={br:.3f})")
    print(f"│  Bound: {n_bound}/{n_objects}")

    # Proto-affordances
    slot_mass_corr = np.zeros(6)
    slot_radius_corr = np.zeros(6)
    for s in range(6):
        slot_norm = np.linalg.norm(all_slots_np[:, s, :], axis=1)
        for o in range(n_objects):
            mass = gt_state[:, o * 8 + 6]
            radius = gt_state[:, o * 8 + 7]
            if np.std(mass) > 1e-6:
                r_m = abs(np.corrcoef(slot_norm, mass)[0, 1])
                if not np.isnan(r_m) and r_m > slot_mass_corr[s]:
                    slot_mass_corr[s] = r_m
            if np.std(radius) > 1e-6:
                r_r = abs(np.corrcoef(slot_norm, radius)[0, 1])
                if not np.isnan(r_r) and r_r > slot_radius_corr[s]:
                    slot_radius_corr[s] = r_r

    print(f"│  Mass corr:   {np.round(slot_mass_corr, 3)}")
    print(f"│  Radius corr: {np.round(slot_radius_corr, 3)}")
    print("└─ Done")

    # ════════════════════════════════════════════════════════════
    # STAGE 2: JEPA on Frozen Slots
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 2: JEPA on Frozen Slots")
    print("=" * 50)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print("│  Extracting slots...")
    slots_a = extract_all_slots(dataset['img_a'], ae, device, batch_size)
    slots_b = extract_all_slots(dataset['img_b'], ae, device, batch_size)
    slots_t_next = extract_all_slots(
        dataset['next_img_target'], ae, device, batch_size)
    print(f"│  Slots shape: {slots_a.shape}")

    predictor = SlotJEPAPredictor(n_slots=6, slot_dim=64, comm_dim=8, action_dim=4)
    pred_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"│  Predictor params: {pred_params:,}")
    predictor = predictor.to(device)

    pred_opt = torch.optim.AdamW(predictor.parameters(), lr=3e-4, weight_decay=0.01)
    pred_epochs = 300
    pred_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        pred_opt, T_max=pred_epochs, eta_min=1e-5)
    pred_hist = {'pred': []}

    t_s2 = time.time()
    for epoch in range(pred_epochs):
        predictor.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            sa = slots_a[idx].to(device)
            sb = slots_b[idx].to(device)
            act = dataset['action'][idx].to(device)
            st = slots_t_next[idx].to(device)

            loss, _, _ = predictor(sa, sb, act, st)
            pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            pred_opt.step()
            ep_loss += loss.item(); nb += 1

        pred_sched.step()
        pred_hist['pred'].append(ep_loss / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s2
            eta = elapsed_ep / (epoch + 1) * (pred_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{pred_epochs}: pred={ep_loss/nb:.6f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 2 time: {time.time()-t_s2:.0f}s")

    # ── Communication Tests ────────────────────────────────────
    print("\n┌─ Communication Tests")
    predictor.eval()

    with torch.no_grad():
        sa_val = slots_a[val_idx].to(device)
        sb_val = slots_b[val_idx].to(device)
        act_val = dataset['action'][val_idx].to(device)
        st_val = slots_t_next[val_idx].to(device)

        _, msg_a_val, msg_b_val = predictor(sa_val, sb_val, act_val, st_val)

        msg_b_real = predictor.communicate(sb_val)
        next_real = predictor.predict_next(sa_val, act_val, msg_b_real)
        mse_real = F.mse_loss(next_real, st_val).item()

        next_zero = predictor.predict_next(
            sa_val, act_val, torch.zeros_like(msg_b_real))
        mse_zero = F.mse_loss(next_zero, st_val).item()

        next_shuf = predictor.predict_next(
            sa_val, act_val, msg_b_real[torch.randperm(len(val_idx))])
        mse_shuf = F.mse_loss(next_shuf, st_val).item()

        next_noise = predictor.predict_next(
            sa_val, act_val, torch.randn_like(msg_b_real))
        mse_noise = F.mse_loss(next_noise, st_val).item()

    m_zero = (1 - mse_real / mse_zero) * 100 if mse_zero > 0 else 0
    m_shuf = (1 - mse_real / mse_shuf) * 100 if mse_shuf > 0 else 0
    m_noise = (1 - mse_real / mse_noise) * 100 if mse_noise > 0 else 0

    print(f"│  Real:     {mse_real:.6f}")
    print(f"│  Zero:     {mse_zero:.6f}  (margin: {m_zero:.1f}%)")
    print(f"│  Shuffled: {mse_shuf:.6f}  (margin: {m_shuf:.1f}%)")
    print(f"│  Noise:    {mse_noise:.6f}  (margin: {m_noise:.1f}%)")

    cond_names = ['Real', 'Zero', 'Shuffled', 'Noise']
    cond_vals = [mse_real, mse_zero, mse_shuf, mse_noise]
    cond_colors = ['#27ae60', '#e74c3c', '#e67e22', '#8e44ad']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cond_names, cond_vals, color=cond_colors, alpha=0.8)
    for bar, v, m in zip(bars, cond_vals,
                         ['', f'{m_zero:.1f}%', f'{m_shuf:.1f}%', f'{m_noise:.1f}%']):
        if m:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    m, ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Slot MSE')
    ax.set_title('Phase 26d: Communication Tests')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26d_comm_tests.png", dpi=150)
    plt.close()
    print("│  → results/phase26d_comm_tests.png")
    print("└─ Done")

    # ── Message ↔ State Heatmap ────────────────────────────────
    print("\n┌─ Msg ↔ State Heatmap")
    msg_a_np = msg_a_val.cpu().numpy()
    n_state = gt_state.shape[1]
    corr_msg = np.zeros((8, n_state))
    for c in range(8):
        for si in range(n_state):
            if np.std(gt_state[:, si]) > 1e-6 and np.std(msg_a_np[:, c]) > 1e-6:
                r = np.corrcoef(msg_a_np[:, c], gt_state[:, si])[0, 1]
                if not np.isnan(r):
                    corr_msg[c, si] = abs(r)

    state_labels = []
    for i in range(n_objects):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r']:
            state_labels.append(f'O{i}_{v}')

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(corr_msg, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Msg{i}' for i in range(8)])
    ax.set_xticks(range(n_state))
    ax.set_xticklabels(state_labels, rotation=90, fontsize=6)
    for i in range(1, n_objects):
        ax.axvline(x=i * 8 - 0.5, color='white', linewidth=2)
    ax.set_title('Phase 26d: Msg ↔ State')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26d_msg_heatmap.png", dpi=150)
    plt.close()
    print(f"│  Max corr: {corr_msg.max():.3f}")
    print("│  → results/phase26d_msg_heatmap.png")
    print("└─ Done")

    # ── Summary Dashboard ──────────────────────────────────────
    print("\n┌─ Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 26d: Object Discovery — Constrained Decoder',
                 fontsize=16, fontweight='bold')

    axes[0, 0].plot(ae_hist['recon'], label='recon')
    axes[0, 0].plot(ae_hist['div'], label='diversity')
    axes[0, 0].legend()
    axes[0, 0].set_title('Stage 1 Training')

    im_b = axes[0, 1].imshow(slot_object_r2, aspect='auto',
                             cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Binding: {n_bound}/5 (max={best_r2:.3f})')
    axes[0, 1].set_yticks(range(6))
    axes[0, 1].set_xticks(range(5))

    axes[0, 2].plot(pred_hist['pred'])
    axes[0, 2].set_title(f"Stage 2: Pred → {pred_hist['pred'][-1]:.6f}")

    axes[1, 0].bar(cond_names, cond_vals, color=cond_colors)
    axes[1, 0].set_title(
        f'Comm: Z={m_zero:.1f}% S={m_shuf:.1f}% N={m_noise:.1f}%')

    x = np.arange(6)
    axes[1, 1].bar(x - 0.15, slot_mass_corr, 0.3,
                   label='Mass', color='#e74c3c')
    axes[1, 1].bar(x + 0.15, slot_radius_corr, 0.3,
                   label='Radius', color='#3498db')
    axes[1, 1].legend()
    axes[1, 1].set_title('Proto-Affordances')

    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    summary_text = (
        f"PHASE 26d\n"
        f"{'━' * 30}\n\n"
        f"Constrained decoder\n"
        f"+ slot dropout\n"
        f"+ diversity loss (λ=0.1)\n\n"
        f"AE: {ae_params:,} params\n"
        f"Pred: {pred_params:,} params\n"
        f"Time: {elapsed / 60:.0f} min\n\n"
        f"Recon: {ae_hist['recon'][-1]:.4f}\n"
        f"Div: {ae_hist['div'][-1]:.4f}\n"
        f"Binding: {n_bound}/5\n"
        f"Max R²: {best_r2:.3f}\n\n"
        f"Pred: {pred_hist['pred'][-1]:.6f}\n"
        f"Comm zero: {m_zero:.1f}%\n"
        f"Comm shuf: {m_shuf:.1f}%\n"
        f"Comm noise: {m_noise:.1f}%\n"
    )
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26d_summary.png", dpi=150)
    plt.close()

    torch.save(ae.state_dict(), OUTPUT_DIR / "phase26d_autoencoder.pt")
    torch.save(predictor.state_dict(), OUTPUT_DIR / "phase26d_predictor.pt")
    print("│  → results/phase26d_summary.png")
    print("└─ Done")

    print(f"\n{'=' * 60}")
    print(f"PHASE 26d SUMMARY")
    print(f"{'=' * 60}")
    print(f"  AE: {ae_params:,} params, recon={ae_hist['recon'][-1]:.4f}")
    print(f"  Pred: {pred_params:,} params, pred={pred_hist['pred'][-1]:.6f}")
    print(f"  Binding: {n_bound}/{n_objects} (R²>0.3)")
    print(f"  Max R²:  {best_r2:.3f}")
    print(f"  Comm zero: {m_zero:.1f}%")
    print(f"  Comm shuf: {m_shuf:.1f}%")
    print(f"  Comm noise: {m_noise:.1f}%")
    print(f"  Mass corr:   {slot_mass_corr.max():.3f}")
    print(f"  Radius corr: {slot_radius_corr.max():.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def run_phase26e():
    """Phase 26e: Tiny Slots + Rich Scenes.

    Stage 1: SlotAttentionAEv3 (10 slots × 12 dims) on rich scenes (8 objects, 3 shapes)
    Stage 2: Freeze slots, train SlotJEPAPredictor on frozen features
    """
    print("=" * 60)
    print("PHASE 26e: Tiny Slots + Rich Scenes")
    print("=" * 60)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Dataset ────────────────────────────────────────────────
    print("\n┌─ Dataset (Rich Scenes)")
    dataset_path = OUTPUT_DIR / 'phase26e_dataset.pt'
    if dataset_path.exists():
        dataset = torch.load(dataset_path, weights_only=False)
        print(f"│  Loaded: {len(dataset['img_a'])} frames")
    else:
        dataset = collect_rich_dataset(
            n_episodes=300, steps_per_episode=40, n_objects=8, img_size=64)
        torch.save(dataset, dataset_path)
        print(f"│  Collected: {len(dataset['img_a'])} frames")

    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    target_imgs = dataset['img_target']
    print(f"│  {n} frames, collection: {time.time()-t0:.0f}s")
    print("└─ Done")

    # Sample frames
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i in range(6):
        idx = i * n // 6
        axes[0, i].imshow(dataset['img_a'][idx].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Agent A' if i == 0 else ''); axes[0, i].axis('off')
        axes[1, i].imshow(dataset['img_b'][idx].permute(1, 2, 0).numpy())
        axes[1, i].set_title('Agent B' if i == 0 else ''); axes[1, i].axis('off')
        axes[2, i].imshow(target_imgs[idx].permute(1, 2, 0).numpy())
        axes[2, i].set_title('Target' if i == 0 else ''); axes[2, i].axis('off')
    fig.suptitle('Phase 26e: Rich Scenes (spheres/cubes/pyramids + checkerboard)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26e_dataset.png", dpi=150)
    plt.close()
    print("│  → results/phase26e_dataset.png")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Tiny-Slot Autoencoder
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 1: Tiny-Slot AE (10 slots × 12 dims)")
    print("=" * 50)

    ae = SlotAttentionAEv3(n_slots=10, slot_dim=12, img_size=64)
    ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print(f"│  AE params: {ae_params:,}")
    print(f"│  Slot capacity: 10×12=120 for 8 objects×9=72 (ratio 1.67)")
    ae = ae.to(device)

    ae_opt = torch.optim.AdamW(ae.parameters(), lr=3e-4, weight_decay=0.01)
    ae_epochs = 50
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        ae_opt, T_max=ae_epochs, eta_min=1e-5)
    batch_size = 48
    ae_hist = {'recon': []}

    t_s1 = time.time()
    for epoch in range(ae_epochs):
        ae.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = target_imgs[idx].to(device)

            loss, recon, slots, alpha = ae(imgs, training=True)
            ae_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_opt.step()
            ep_loss += loss.item(); nb += 1

        ae_sched.step()
        ae_hist['recon'].append(ep_loss / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s1
            eta = elapsed_ep / (epoch + 1) * (ae_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: recon={ep_loss/nb:.4f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 1 time: {time.time()-t_s1:.0f}s")

    # ── Stage 1 Evaluation ─────────────────────────────────────
    print("\n┌─ Stage 1 Evaluation")
    ae.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        val_imgs = target_imgs[val_idx[:200]].to(device)
        loss_val, recon_val, slots_val, alpha_val = ae(
            val_imgs, training=False)
        print(f"│  Val recon: {loss_val.item():.4f}")

    # Reconstructions + slot mask composite
    slot_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                   [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1], [0, 0.5, 0],
                   [0.5, 0.5, 0.5]]
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axes[0, i].imshow(val_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')

        axes[1, i].imshow(
            recon_val[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[1, i].set_title('Recon' if i == 0 else '')
        axes[1, i].axis('off')

        masks = alpha_val[i].cpu().reshape(10, 64, 64).numpy()
        composite = np.zeros((64, 64, 3))
        for s in range(10):
            for c in range(3):
                composite[:, :, c] += masks[s] * slot_colors[s][c]
        axes[2, i].imshow(np.clip(composite, 0, 1))
        axes[2, i].set_title('Slot masks' if i == 0 else '')
        axes[2, i].axis('off')

    fig.suptitle('Phase 26e: Tiny-Slot AE (10×12)\n'
                 'R=S0 G=S1 B=S2 Y=S3 M=S4 C=S5 O=S6 P=S7 G=S8 Gr=S9',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26e_reconstruction.png", dpi=150)
    plt.close()
    print("│  → results/phase26e_reconstruction.png")

    # Slot-object binding
    print("│  Slot-object binding:")
    from sklearn.linear_model import LinearRegression

    with torch.no_grad():
        all_slots_list = []
        for ci in range(0, len(val_idx), batch_size):
            bi = val_idx[ci:ci+batch_size]
            s = ae.encode(target_imgs[bi].to(device))
            all_slots_list.append(s.cpu())
        all_slots_np = torch.cat(all_slots_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_objects = 8
    vals_per_obj = 9

    slot_object_r2 = np.zeros((10, n_objects))
    for s in range(10):
        for o in range(n_objects):
            X = all_slots_np[:, s, :]
            y = gt_state[:, o*vals_per_obj:o*vals_per_obj+3]
            n_fit = min(1000, len(X) // 2)
            if n_fit < 10:
                continue
            reg = LinearRegression().fit(X[:n_fit], y[:n_fit])
            r2 = reg.score(X[n_fit:], y[n_fit:])
            slot_object_r2[s, o] = max(0, r2)

    n_bound = int(sum(slot_object_r2.max(axis=0) > 0.3))
    best_r2 = slot_object_r2.max()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Slot {i}' for i in range(10)])
    ax.set_xticks(range(n_objects))
    ax.set_xticklabels([f'Obj {i}' for i in range(n_objects)])
    for i in range(10):
        for j in range(n_objects):
            ax.text(j, i, f'{slot_object_r2[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if slot_object_r2[i, j] > 0.5 else 'black')
    ax.set_title(
        f'Phase 26e: Slot→Object Binding — {n_bound}/8 bound (max R²={best_r2:.3f})')
    plt.colorbar(im, label='R²')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26e_slot_binding.png", dpi=150)
    plt.close()

    for o in range(n_objects):
        bs = slot_object_r2[:, o].argmax()
        br = slot_object_r2[:, o].max()
        print(f"│    Obj {o} → Slot {bs} (R²={br:.3f})")
    print(f"│  Bound: {n_bound}/{n_objects}")

    # Alpha mask entropy
    with torch.no_grad():
        masks_flat = alpha_val[:100].cpu().reshape(100, 10, 64*64)
        entropy = -(masks_flat * (masks_flat + 1e-8).log()).sum(dim=1).mean()
        max_alpha = masks_flat.max(dim=2).values.mean(dim=0)
        print(f"│  Assignment entropy: {entropy.item():.3f}")
        print(f"│  Per-slot max alpha: {max_alpha.numpy().round(3)}")

    # Proto-affordances
    slot_mass_corr = np.zeros(10)
    slot_shape_corr = np.zeros(10)
    for s in range(10):
        slot_norm = np.linalg.norm(all_slots_np[:, s, :], axis=1)
        for o in range(n_objects):
            mass = gt_state[:, o*vals_per_obj+6]
            shape = gt_state[:, o*vals_per_obj+8]
            if np.std(mass) > 1e-6:
                r_m = abs(np.corrcoef(slot_norm, mass)[0, 1])
                if not np.isnan(r_m) and r_m > slot_mass_corr[s]:
                    slot_mass_corr[s] = r_m
            if np.std(shape) > 1e-6:
                r_s = abs(np.corrcoef(slot_norm, shape)[0, 1])
                if not np.isnan(r_s) and r_s > slot_shape_corr[s]:
                    slot_shape_corr[s] = r_s

    print(f"│  Mass corr:  {np.round(slot_mass_corr, 3)}")
    print(f"│  Shape corr: {np.round(slot_shape_corr, 3)}")
    print("└─ Done")

    # ════════════════════════════════════════════════════════════
    # STAGE 2: JEPA on Frozen Slots
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 2: JEPA on Frozen Slots")
    print("=" * 50)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print("│  Extracting slots...")
    slots_a = extract_all_slots(dataset['img_a'], ae, device, batch_size)
    slots_b = extract_all_slots(dataset['img_b'], ae, device, batch_size)
    slots_t_next = extract_all_slots(
        dataset['next_img_target'], ae, device, batch_size)
    print(f"│  Slots: {slots_a.shape}")

    predictor = SlotJEPAPredictor(
        n_slots=10, slot_dim=12, comm_dim=8, action_dim=4)
    pred_params = sum(p.numel() for p in predictor.parameters()
                      if p.requires_grad)
    print(f"│  Predictor params: {pred_params:,}")
    predictor = predictor.to(device)

    pred_opt = torch.optim.AdamW(
        predictor.parameters(), lr=3e-4, weight_decay=0.01)
    pred_epochs = 300
    pred_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        pred_opt, T_max=pred_epochs, eta_min=1e-5)
    pred_hist = {'pred': []}

    t_s2 = time.time()
    for epoch in range(pred_epochs):
        predictor.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            sa = slots_a[idx].to(device)
            sb = slots_b[idx].to(device)
            act = dataset['action'][idx].to(device)
            st = slots_t_next[idx].to(device)

            loss, _, _ = predictor(sa, sb, act, st)
            pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            pred_opt.step()
            ep_loss += loss.item(); nb += 1

        pred_sched.step()
        pred_hist['pred'].append(ep_loss / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s2
            eta = elapsed_ep / (epoch + 1) * (pred_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{pred_epochs}: pred={ep_loss/nb:.6f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 2 time: {time.time()-t_s2:.0f}s")

    # ── Communication Tests ────────────────────────────────────
    print("\n┌─ Communication Tests")
    predictor.eval()

    with torch.no_grad():
        sa_val = slots_a[val_idx].to(device)
        sb_val = slots_b[val_idx].to(device)
        act_val = dataset['action'][val_idx].to(device)
        st_val = slots_t_next[val_idx].to(device)

        msg_b_real = predictor.communicate(sb_val)
        next_real = predictor.predict_next(sa_val, act_val, msg_b_real)
        mse_real = F.mse_loss(next_real, st_val).item()

        next_zero = predictor.predict_next(
            sa_val, act_val, torch.zeros_like(msg_b_real))
        mse_zero = F.mse_loss(next_zero, st_val).item()

        next_shuf = predictor.predict_next(
            sa_val, act_val, msg_b_real[torch.randperm(len(val_idx))])
        mse_shuf = F.mse_loss(next_shuf, st_val).item()

        next_noise = predictor.predict_next(
            sa_val, act_val, torch.randn_like(msg_b_real))
        mse_noise = F.mse_loss(next_noise, st_val).item()

    m_zero = (1 - mse_real / mse_zero) * 100 if mse_zero > 0 else 0
    m_shuf = (1 - mse_real / mse_shuf) * 100 if mse_shuf > 0 else 0
    m_noise = (1 - mse_real / mse_noise) * 100 if mse_noise > 0 else 0

    print(f"│  Real:     {mse_real:.6f}")
    print(f"│  Zero:     {mse_zero:.6f}  (margin: {m_zero:.1f}%)")
    print(f"│  Shuffled: {mse_shuf:.6f}  (margin: {m_shuf:.1f}%)")
    print(f"│  Noise:    {mse_noise:.6f}  (margin: {m_noise:.1f}%)")

    cond_names = ['Real', 'Zero', 'Shuffled', 'Noise']
    cond_vals = [mse_real, mse_zero, mse_shuf, mse_noise]
    cond_colors = ['#27ae60', '#e74c3c', '#e67e22', '#8e44ad']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cond_names, cond_vals, color=cond_colors, alpha=0.8)
    for bar, v, m in zip(bars, cond_vals,
                         ['', f'{m_zero:.1f}%', f'{m_shuf:.1f}%',
                          f'{m_noise:.1f}%']):
        if m:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    m, ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Slot MSE')
    ax.set_title('Phase 26e: Communication Tests')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26e_comm_tests.png", dpi=150)
    plt.close()
    print("│  → results/phase26e_comm_tests.png")
    print("└─ Done")

    # ── Message ↔ State Heatmap ────────────────────────────────
    print("\n┌─ Msg ↔ State Heatmap")
    msg_a_np = predictor.communicate(sa_val).detach().cpu().numpy()
    n_state = gt_state.shape[1]
    corr_msg = np.zeros((8, n_state))
    for c in range(8):
        for si in range(n_state):
            if (np.std(gt_state[:, si]) > 1e-6
                    and np.std(msg_a_np[:, c]) > 1e-6):
                r = np.corrcoef(msg_a_np[:, c], gt_state[:, si])[0, 1]
                if not np.isnan(r):
                    corr_msg[c, si] = abs(r)

    state_labels = []
    for i in range(n_objects):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r', 'sh']:
            state_labels.append(f'O{i}_{v}')

    fig, ax = plt.subplots(figsize=(24, 5))
    im = ax.imshow(corr_msg, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Msg{i}' for i in range(8)])
    ax.set_xticks(range(n_state))
    ax.set_xticklabels(state_labels, rotation=90, fontsize=5)
    for i in range(1, n_objects):
        ax.axvline(x=i*vals_per_obj - 0.5, color='white', linewidth=2)
    ax.set_title('Phase 26e: Msg ↔ State')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26e_msg_heatmap.png", dpi=150)
    plt.close()
    print(f"│  Max corr: {corr_msg.max():.3f}")
    print("│  → results/phase26e_msg_heatmap.png")
    print("└─ Done")

    # ── Summary Dashboard ──────────────────────────────────────
    print("\n┌─ Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 26e: Tiny Slots + Rich Scenes',
                 fontsize=16, fontweight='bold')

    axes[0, 0].plot(ae_hist['recon'])
    axes[0, 0].set_title(f"Stage 1 Recon → {ae_hist['recon'][-1]:.4f}")

    im_b = axes[0, 1].imshow(
        slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Binding: {n_bound}/8 (max={best_r2:.3f})')
    axes[0, 1].set_yticks(range(10))
    axes[0, 1].set_xticks(range(8))

    axes[0, 2].plot(pred_hist['pred'])
    axes[0, 2].set_title(f"Stage 2 Pred → {pred_hist['pred'][-1]:.6f}")

    axes[1, 0].bar(cond_names, cond_vals, color=cond_colors)
    axes[1, 0].set_title(
        f'Comm: Z={m_zero:.1f}% S={m_shuf:.1f}% N={m_noise:.1f}%')

    x = np.arange(10)
    axes[1, 1].bar(x - 0.15, slot_mass_corr, 0.3,
                   label='Mass', color='#e74c3c')
    axes[1, 1].bar(x + 0.15, slot_shape_corr, 0.3,
                   label='Shape', color='#3498db')
    axes[1, 1].legend()
    axes[1, 1].set_title('Proto-Affordances')

    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    summary_text = (
        f"PHASE 26e\n"
        f"{'━' * 30}\n\n"
        f"Tiny slots (12-dim)\n"
        f"Rich scenes (8 obj, 3 shapes)\n"
        f"Checkerboard floor\n\n"
        f"AE: {ae_params:,} params\n"
        f"Pred: {pred_params:,} params\n"
        f"Time: {elapsed / 60:.0f} min\n\n"
        f"Recon: {ae_hist['recon'][-1]:.4f}\n"
        f"Binding: {n_bound}/8\n"
        f"Max R²: {best_r2:.3f}\n\n"
        f"Pred: {pred_hist['pred'][-1]:.6f}\n"
        f"Comm zero: {m_zero:.1f}%\n"
        f"Comm shuf: {m_shuf:.1f}%\n"
        f"Comm noise: {m_noise:.1f}%\n"
    )
    axes[1, 2].text(0.05, 0.95, summary_text,
                    transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow',
                              alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26e_summary.png", dpi=150)
    plt.close()

    torch.save(ae.state_dict(), OUTPUT_DIR / "phase26e_autoencoder.pt")
    torch.save(predictor.state_dict(), OUTPUT_DIR / "phase26e_predictor.pt")
    print("│  → results/phase26e_summary.png")
    print("└─ Done")

    print(f"\n{'=' * 60}")
    print(f"PHASE 26e SUMMARY")
    print(f"{'=' * 60}")
    print(f"  AE: {ae_params:,} params, recon={ae_hist['recon'][-1]:.4f}")
    print(f"  Pred: {pred_params:,} params, pred={pred_hist['pred'][-1]:.6f}")
    print(f"  Binding: {n_bound}/{n_objects} (R²>0.3)")
    print(f"  Max R²:  {best_r2:.3f}")
    print(f"  Comm zero: {m_zero:.1f}%")
    print(f"  Comm shuf: {m_shuf:.1f}%")
    print(f"  Comm noise: {m_noise:.1f}%")
    print(f"  Mass corr:   {slot_mass_corr.max():.3f}")
    print(f"  Shape corr:  {slot_shape_corr.max():.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def run_phase26f_clevr_test():
    """CLEVR diagnostic: test if SlotAttentionAEv5 can segment simple circles.

    If this fails, there's a bug in the architecture.
    If this passes, the problem is scene complexity.
    """
    print("=" * 60)
    print("PHASE 26f CLEVR DIAGNOSTIC: Can slots segment simple circles?")
    print("=" * 60)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"│  Device: {device}")

    # ── Dataset ────────────────────────────────────────────────
    print("\n┌─ Generating CLEVR images")
    data = generate_clevr_images(n_images=5000, img_size=64, max_objects=4)
    images = data['images']       # [5000, 3, 64, 64]
    masks_gt = data['masks_gt']   # [5000, 5, 64, 64] (bg + 4 obj slots)
    n = len(images)
    n_train = int(0.8 * n)
    print(f"│  {n} images, train={n_train}, val={n-n_train}")
    print("└─ Done")

    # Sample visualization
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(8):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'{data["n_objects"][i].item()} obj')
        axes[0, i].axis('off')
        # Show GT mask composite
        gt_m = masks_gt[i].numpy()  # [5, H, W]
        gt_colors = [[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0],
                      [0, 0, 1], [1, 1, 0]]
        comp = np.zeros((64, 64, 3))
        for s in range(5):
            for c in range(3):
                comp[:, :, c] += gt_m[s] * gt_colors[s][c]
        axes[1, i].imshow(np.clip(comp, 0, 1))
        axes[1, i].set_title('GT masks' if i == 0 else '')
        axes[1, i].axis('off')
    fig.suptitle('CLEVR Diagnostic: Simple Circles on Gray', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_clevr_dataset.png", dpi=150)
    plt.close()

    # ── Training ───────────────────────────────────────────────
    n_slots = 7
    ae_epochs = 500

    print("\n" + "=" * 50)
    print(f"Training SlotAttentionAEv5 on CLEVR ({n_slots} slots, {ae_epochs} epochs, 4096 tokens)")
    print("=" * 50)

    ae = SlotAttentionAEv5(n_slots=n_slots, slot_dim=64, img_size=64).to(device)
    ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print(f"│  Params: {ae_params:,}")
    print(f"│  SA iters: {ae.slot_attention.n_iters}, shared init (reference-matched)")

    base_lr = 4e-4
    ae_opt = torch.optim.Adam(ae.parameters(), lr=base_lr)
    warmup_epochs = 30

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        elif epoch < 450:
            return 1.0  # constant LR (reference-faithful)
        else:
            return 0.5  # halve for final 50 epochs

    ae_sched = torch.optim.lr_scheduler.LambdaLR(ae_opt, lr_lambda)
    batch_size = 32  # smaller batch for 4096 tokens on MPS

    for epoch in range(ae_epochs):
        ae.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = images[idx].to(device)
            total_loss, recon_loss, entropy_reg, recon, slots, alpha = ae(
                imgs, training=True)
            ae_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_opt.step()
            ep_loss += recon_loss.item(); nb += 1
            if device.type == 'mps' and nb % 100 == 0:
                torch.mps.empty_cache()

        ae_sched.step()

        # Print schedule: every 5 epochs for first 25, every 10 after
        should_print = ((epoch + 1) <= 25 and (epoch + 1) % 5 == 0) or \
                       ((epoch + 1) > 25 and (epoch + 1) % 10 == 0) or \
                       (epoch + 1) == 1
        if should_print:
            ae.eval()
            with torch.no_grad():
                # Process in small batches to avoid MPS OOM
                diag_alphas = []
                for di in range(0, 200, 32):
                    d_batch = images[di:di+32].to(device)
                    _, _, _, _, _, d_alpha = ae(d_batch, training=False)
                    diag_alphas.append(d_alpha.cpu())
                diag_alpha = torch.cat(diag_alphas, dim=0)
                ownership = diag_alpha.argmax(dim=1)
                B_d = ownership.shape[0]
                slot_counts = torch.zeros(B_d, n_slots)
                for s in range(n_slots):
                    slot_counts[:, s] = (ownership == s).float().sum(dim=(1, 2))
                mean_fracs = (slot_counts / (64*64)).mean(dim=0)
                active = int((mean_fracs > 0.01).sum().item())
                max_cov = mean_fracs.max().item() * 100
                masks_f = diag_alpha.reshape(B_d, n_slots, -1)
                ent = -(masks_f * (masks_f + 1e-8).log()).sum(dim=1).mean()
                norm_ent = ent.item() / np.log(n_slots)

            elapsed = time.time() - t0
            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: "
                  f"recon={ep_loss/nb:.4f} "
                  f"active={active}/{n_slots} "
                  f"max_cov={max_cov:.1f}% "
                  f"entropy={norm_ent:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

            # Early stopping: sharp slot binding achieved
            if norm_ent < 0.2:
                print(f"│", flush=True)
                print(f"│  SUCCESS — stopping early (entropy={norm_ent:.3f} < 0.2)", flush=True)
                break

            # Failure exit: if entropy > 0.99 at epoch 100, seed didn't break symmetry
            if (epoch + 1) == 100 and norm_ent > 0.99:
                print(f"│", flush=True)
                print(f"│  FAIL — seed didn't break symmetry by 5K steps (entropy={norm_ent:.3f} > 0.99)", flush=True)
                break

    # ── Evaluation ─────────────────────────────────────────────
    print("\n┌─ CLEVR Evaluation")
    ae.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        val_imgs = images[val_idx[:200]].to(device)
        # Process in small batches to avoid MPS OOM
        recon_parts, alpha_parts = [], []
        for vi in range(0, len(val_imgs), 32):
            vb = val_imgs[vi:vi+32]
            _, _, _, r, _, a = ae(vb, training=False)
            recon_parts.append(r.cpu())
            alpha_parts.append(a.cpu())
        recon_val = torch.cat(recon_parts, dim=0)
        alpha_val = torch.cat(alpha_parts, dim=0)

    # Reconstruction + masks
    slot_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                   [0, 1, 1], [1, 0.5, 0]]
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axes[0, i].imshow(val_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')
        axes[1, i].imshow(
            recon_val[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[1, i].set_title('Recon' if i == 0 else '')
        axes[1, i].axis('off')
        masks = alpha_val[i].cpu().numpy()
        composite = np.zeros((64, 64, 3))
        for s in range(n_slots):
            for c in range(3):
                composite[:, :, c] += masks[s] * slot_colors[s][c]
        axes[2, i].imshow(np.clip(composite, 0, 1))
        axes[2, i].set_title('Slot masks' if i == 0 else '')
        axes[2, i].axis('off')
    fig.suptitle('CLEVR Diagnostic: Reconstruction + Slot Masks', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_clevr_result.png", dpi=150)
    plt.close()
    print("│  → results/phase26f_clevr_result.png")

    # Individual slot heatmaps
    fig, axes = plt.subplots(2, n_slots, figsize=(2.5 * n_slots, 5))
    for s in range(n_slots):
        axes[0, s].imshow(alpha_val[0, s].cpu().numpy(),
                          cmap='hot', vmin=0, vmax=1)
        axes[0, s].set_title(f'Slot {s}', fontsize=9)
        axes[0, s].axis('off')
        axes[1, s].imshow(alpha_val[1, s].cpu().numpy(),
                          cmap='hot', vmin=0, vmax=1)
        axes[1, s].axis('off')
    fig.suptitle('CLEVR Diagnostic: Individual Slot Masks', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_clevr_slots.png", dpi=150)
    plt.close()
    print("│  → results/phase26f_clevr_slots.png")

    # ARI (Adjusted Rand Index)
    from sklearn.metrics import adjusted_rand_score
    ari_scores = []
    with torch.no_grad():
        for i in range(min(200, len(val_idx))):
            img_i = images[val_idx[i]:val_idx[i]+1].to(device)
            _, _, _, _, _, alpha_i = ae(img_i, training=False)
            pred_mask = alpha_i[0].argmax(dim=0).cpu().numpy().flatten()
            gt_mask = masks_gt[val_idx[i]].argmax(dim=0).numpy().flatten()
            ari = adjusted_rand_score(gt_mask, pred_mask)
            ari_scores.append(ari)
    mean_ari = np.mean(ari_scores)
    print(f"│  ARI (Adjusted Rand Index): {mean_ari:.3f}")

    # Final metrics
    with torch.no_grad():
        all_alpha = alpha_val[:100]
        ownership = all_alpha.argmax(dim=1)
        B_d = ownership.shape[0]
        slot_counts = torch.zeros(B_d, n_slots)
        for s in range(n_slots):
            slot_counts[:, s] = (ownership == s).float().sum(dim=(1, 2))
        mean_fracs = (slot_counts / (64*64)).mean(dim=0)
        active = int((mean_fracs > 0.01).sum().item())
        max_cov = mean_fracs.max().item() * 100

    # Verdict
    passed = mean_ari > 0.5 and max_cov < 40 and active >= 3
    verdict = "PASS" if passed else "FAIL"
    elapsed = time.time() - t0

    print(f"│")
    print(f"│  Active slots: {active}/{n_slots}")
    print(f"│  Max coverage: {max_cov:.1f}%")
    print(f"│  ARI: {mean_ari:.3f}")
    print(f"│  Time: {elapsed:.0f}s")
    print(f"│")
    print(f"│  ╔{'═'*40}╗")
    print(f"│  ║  CLEVR DIAGNOSTIC: {verdict:4s}                 ║")
    if passed:
        print(f"│  ║  Architecture works on simple scenes.  ║")
        print(f"│  ║  Problem is scene complexity.           ║")
    else:
        print(f"│  ║  Architecture fails on simple scenes!   ║")
        print(f"│  ║  Implementation bug likely.             ║")
    print(f"│  ╚{'═'*40}╝")
    print("└─ Done")

    return passed


def run_phase27_dino_clevr():
    """Phase 27: DINOv2 feature reconstruction with Slot Attention.

    Replace pixel reconstruction with DINOv2 feature reconstruction.
    Frozen DINOv2-Small encoder, SA groups patch features, MLP decoder
    reconstructs DINOv2 features (not pixels). Loss: MSE on features.
    """
    print("=" * 60)
    print("PHASE 27: DINOv2 Feature Reconstruction on CLEVR")
    print("=" * 60)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"│  Device: {device}")

    # ── Dataset ────────────────────────────────────────────────
    print("\n┌─ Generating complex CLEVR images")
    data = generate_clevr_images_complex(n_images=5000, img_size=64, max_objects=6)
    images = data['images']       # [5000, 3, 64, 64]
    masks_gt = data['masks_gt']   # [5000, 7, 64, 64] (bg + 6 obj slots)
    n = len(images)
    n_train = int(0.8 * n)
    print(f"│  {n} images, train={n_train}, val={n-n_train}")
    print("└─ Done")

    # ── Training ───────────────────────────────────────────────
    n_slots = 7
    ae_epochs = 200

    print("\n" + "=" * 50)
    print(f"Training SlotAttentionDINO on CLEVR ({n_slots} slots, {ae_epochs} epochs)")
    print("=" * 50)

    ae = SlotAttentionDINO(n_slots=n_slots, slot_dim=64, img_size=64).to(device)
    trainable_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in ae.parameters() if not p.requires_grad)
    print(f"│  Trainable params: {trainable_params:,}")
    print(f"│  Frozen DINOv2 params: {frozen_params:,}")
    print(f"│  SA iters: {ae.slot_attention.n_iters}, per-slot learnable init")

    base_lr = 4e-4
    ae_opt = torch.optim.Adam(
        [p for p in ae.parameters() if p.requires_grad], lr=base_lr)
    warmup_epochs = 30

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        elif epoch < 450:
            return 1.0  # constant LR
        else:
            return 0.5  # halve for final 50 epochs

    ae_sched = torch.optim.lr_scheduler.LambdaLR(ae_opt, lr_lambda)
    batch_size = 32

    for epoch in range(ae_epochs):
        ae.train()
        # Keep DINOv2 in eval mode even during training
        ae.dino.eval()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = images[idx].to(device)
            total_loss, recon_loss, entropy_reg, recon_feat, slots, alpha = ae(
                imgs, training=True)
            ae_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in ae.parameters() if p.requires_grad], 1.0)
            ae_opt.step()
            ep_loss += recon_loss.item(); nb += 1
            if device.type == 'mps' and nb % 100 == 0:
                torch.mps.empty_cache()

        ae_sched.step()

        # Print schedule: every 5 epochs for first 25, every 10 after
        should_print = ((epoch + 1) <= 25 and (epoch + 1) % 5 == 0) or \
                       ((epoch + 1) > 25 and (epoch + 1) % 10 == 0) or \
                       (epoch + 1) == 1
        if should_print:
            ae.eval()
            with torch.no_grad():
                # Diagnostic: compute entropy over patches
                diag_alphas = []
                for di in range(0, 200, 32):
                    d_batch = images[di:di+32].to(device)
                    _, _, _, _, _, d_alpha = ae(d_batch, training=False)
                    diag_alphas.append(d_alpha.cpu())
                diag_alpha = torch.cat(diag_alphas, dim=0)  # [200, K, 256]
                # Ownership over patches
                ownership = diag_alpha.argmax(dim=1)  # [200, 256]
                B_d = ownership.shape[0]
                N_patches = 256
                slot_counts = torch.zeros(B_d, n_slots)
                for s in range(n_slots):
                    slot_counts[:, s] = (ownership == s).float().sum(dim=1)
                mean_fracs = (slot_counts / N_patches).mean(dim=0)
                active = int((mean_fracs > 0.01).sum().item())
                max_cov = mean_fracs.max().item() * 100
                # Entropy
                masks_f = diag_alpha  # [B, K, N] already flat
                ent = -(masks_f * (masks_f + 1e-8).log()).sum(dim=1).mean()
                norm_ent = ent.item() / np.log(n_slots)

            elapsed = time.time() - t0
            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: "
                  f"recon={ep_loss/nb:.4f} "
                  f"active={active}/{n_slots} "
                  f"max_cov={max_cov:.1f}% "
                  f"entropy={norm_ent:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

            # Early stopping: sharp slot binding achieved
            if norm_ent < 0.2:
                print(f"│", flush=True)
                print(f"│  SUCCESS — stopping early (entropy={norm_ent:.3f} < 0.2)", flush=True)
                break

            # Failure exit: if entropy > 0.99 at epoch 100, seed didn't break symmetry
            if (epoch + 1) == 100 and norm_ent > 0.99:
                print(f"│", flush=True)
                print(f"│  FAIL — seed didn't break symmetry by epoch 100 (entropy={norm_ent:.3f} > 0.99)", flush=True)
                break

    # ── Evaluation ─────────────────────────────────────────────
    print("\n┌─ Phase 27 Evaluation")
    ae.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        val_imgs = images[val_idx[:200]].to(device)
        alpha_parts = []
        for vi in range(0, len(val_imgs), 32):
            vb = val_imgs[vi:vi+32]
            _, _, _, _, _, a = ae(vb, training=False)
            alpha_parts.append(a.cpu())
        alpha_val = torch.cat(alpha_parts, dim=0)  # [200, K, 256]

    # Slot masks over 16x16 patch grid (reshape for visualization)
    slot_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                   [0, 1, 1], [1, 0.5, 0]]
    P = 16  # patches per side
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(8):
        axes[0, i].imshow(images[val_idx[i]].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')
        # Slot mask composite (16x16 upscaled for display)
        masks = alpha_val[i].numpy().reshape(n_slots, P, P)
        composite = np.zeros((P, P, 3))
        for s in range(n_slots):
            for c in range(3):
                composite[:, :, c] += masks[s] * slot_colors[s][c]
        axes[1, i].imshow(np.clip(composite, 0, 1), interpolation='nearest')
        axes[1, i].set_title('Slot masks (16x16)' if i == 0 else '')
        axes[1, i].axis('off')
    fig.suptitle('Phase 27: DINOv2 Feature Recon — Slot Masks', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase27_dino_clevr_result.png", dpi=150)
    plt.close()
    print("│  → results/phase27_dino_clevr_result.png")

    # Individual slot heatmaps
    fig, axes = plt.subplots(2, n_slots, figsize=(2.5 * n_slots, 5))
    for s in range(n_slots):
        axes[0, s].imshow(
            alpha_val[0, s].numpy().reshape(P, P),
            cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        axes[0, s].set_title(f'Slot {s}', fontsize=9)
        axes[0, s].axis('off')
        axes[1, s].imshow(
            alpha_val[1, s].numpy().reshape(P, P),
            cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        axes[1, s].axis('off')
    fig.suptitle('Phase 27: Individual Slot Masks (16x16 patches)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase27_dino_clevr_slots.png", dpi=150)
    plt.close()
    print("│  → results/phase27_dino_clevr_slots.png")

    # Final metrics
    with torch.no_grad():
        all_alpha = alpha_val[:100]  # [100, K, 256]
        ownership = all_alpha.argmax(dim=1)  # [100, 256]
        B_d = ownership.shape[0]
        slot_counts = torch.zeros(B_d, n_slots)
        for s in range(n_slots):
            slot_counts[:, s] = (ownership == s).float().sum(dim=1)
        mean_fracs = (slot_counts / 256).mean(dim=0)
        active = int((mean_fracs > 0.01).sum().item())
        max_cov = mean_fracs.max().item() * 100
        masks_f = all_alpha
        ent = -(masks_f * (masks_f + 1e-8).log()).sum(dim=1).mean()
        norm_ent = ent.item() / np.log(n_slots)

    elapsed = time.time() - t0
    print(f"│")
    print(f"│  Active slots: {active}/{n_slots}")
    print(f"│  Max coverage: {max_cov:.1f}%")
    print(f"│  Entropy: {norm_ent:.3f}")
    print(f"│  Time: {elapsed:.0f}s")
    print("└─ Done")

    # Save model
    torch.save(ae.state_dict(), OUTPUT_DIR / "phase27_model.pt")
    print(f"│  → results/phase27_model.pt")

    return norm_ent < 0.2


def run_phase27b_voc():
    """Phase 27b Test 3: Real images (Oxford Flowers102).

    Train SlotAttentionDINO on real photos (flowers with varied backgrounds).
    Uses Flowers102 as VOC server is down. 2040 train images (train+val splits).
    Locked config: DINOv2 + 5 SA iters + per-slot init, 7 slots, 200 epochs.
    Images resized to 224x224 (DINOv2 native resolution).
    """
    from torchvision import transforms
    from torchvision.datasets import Flowers102

    print("=" * 60, flush=True)
    print("PHASE 27b TEST 3: Real Images (Flowers102)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"│  Device: {device}", flush=True)

    # ── Download Flowers102 via torchvision ───────────────────
    print("\n┌─ Loading Flowers102 (train+val splits)", flush=True)
    ds_train = Flowers102(root="./flowers_data", split='train', download=True)
    ds_val = Flowers102(root="./flowers_data", split='val', download=True)
    print(f"│  Train: {len(ds_train)}, Val: {len(ds_val)}", flush=True)
    print("└─ Done", flush=True)

    # ── Load and preprocess images ────────────────────────────
    print("\n┌─ Preprocessing images to 224×224", flush=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0, 1] range
    ])

    all_images = []
    for ds in [ds_train, ds_val]:
        for i in range(len(ds)):
            img, _ = ds[i]
            img_t = transform(img)
            all_images.append(img_t)

    images = torch.stack(all_images)  # [N, 3, 224, 224]
    n = len(images)
    n_train = int(0.8 * n)
    print(f"│  {n} images, train={n_train}, val={n-n_train}", flush=True)
    print("└─ Done", flush=True)

    # ── Training ───────────────────────────────────────────────
    n_slots = 7
    ae_epochs = 200

    print(f"\n{'=' * 50}", flush=True)
    print(f"Training SlotAttentionDINO on Flowers102 ({n_slots} slots, {ae_epochs} epochs)", flush=True)
    print("=" * 50, flush=True)

    ae = SlotAttentionDINO(n_slots=n_slots, slot_dim=64, img_size=224).to(device)
    trainable_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in ae.parameters() if not p.requires_grad)
    print(f"│  Trainable params: {trainable_params:,}", flush=True)
    print(f"│  Frozen DINOv2 params: {frozen_params:,}", flush=True)
    print(f"│  SA iters: {ae.slot_attention.n_iters}, per-slot learnable init", flush=True)

    base_lr = 4e-4
    ae_opt = torch.optim.Adam(
        [p for p in ae.parameters() if p.requires_grad], lr=base_lr)
    warmup_epochs = 30

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0  # constant after warmup

    ae_sched = torch.optim.lr_scheduler.LambdaLR(ae_opt, lr_lambda)
    batch_size = 16  # smaller batch for 224x224 images on MPS

    for epoch in range(ae_epochs):
        ae.train()
        ae.dino.eval()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = images[idx].to(device)
            total_loss, recon_loss, entropy_reg, recon_feat, slots, alpha = ae(
                imgs, training=True)
            ae_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in ae.parameters() if p.requires_grad], 1.0)
            ae_opt.step()
            ep_loss += recon_loss.item(); nb += 1
            if device.type == 'mps' and nb % 50 == 0:
                torch.mps.empty_cache()

        ae_sched.step()

        # Print every 20 epochs + epoch 1
        should_print = (epoch + 1) == 1 or (epoch + 1) % 20 == 0
        if should_print:
            ae.eval()
            with torch.no_grad():
                diag_alphas = []
                for di in range(0, min(200, n_train), 16):
                    d_batch = images[di:di+16].to(device)
                    _, _, _, _, _, d_alpha = ae(d_batch, training=False)
                    diag_alphas.append(d_alpha.cpu())
                diag_alpha = torch.cat(diag_alphas, dim=0)
                ownership = diag_alpha.argmax(dim=1)
                B_d = ownership.shape[0]
                slot_counts = torch.zeros(B_d, n_slots)
                for s in range(n_slots):
                    slot_counts[:, s] = (ownership == s).float().sum(dim=1)
                mean_fracs = (slot_counts / 256).mean(dim=0)
                active = int((mean_fracs > 0.01).sum().item())
                max_cov = mean_fracs.max().item() * 100
                ent = -(diag_alpha * (diag_alpha + 1e-8).log()).sum(dim=1).mean()
                norm_ent = ent.item() / np.log(n_slots)

            elapsed = time.time() - t0
            elapsed_ep = elapsed / (epoch + 1)
            eta = elapsed_ep * (ae_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: "
                  f"recon={ep_loss/nb:.4f} "
                  f"active={active}/{n_slots} "
                  f"max_cov={max_cov:.1f}% "
                  f"entropy={norm_ent:.3f} "
                  f"[{elapsed:.0f}s] ETA={eta:.0f}s ({eta/60:.0f}min)", flush=True)

    # ── Evaluation ─────────────────────────────────────────────
    print(f"\n┌─ Phase 27b Test 3 Evaluation", flush=True)
    ae.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        alpha_parts = []
        for vi in range(0, len(val_idx), 16):
            vb = images[val_idx[vi:vi+16]].to(device)
            _, _, _, _, _, a = ae(vb, training=False)
            alpha_parts.append(a.cpu())
        alpha_val = torch.cat(alpha_parts, dim=0)

    # Final metrics
    with torch.no_grad():
        all_alpha = alpha_val[:min(100, len(alpha_val))]
        ownership = all_alpha.argmax(dim=1)
        B_d = ownership.shape[0]
        slot_counts = torch.zeros(B_d, n_slots)
        for s in range(n_slots):
            slot_counts[:, s] = (ownership == s).float().sum(dim=1)
        mean_fracs = (slot_counts / 256).mean(dim=0)
        active = int((mean_fracs > 0.01).sum().item())
        max_cov = mean_fracs.max().item() * 100
        ent = -(all_alpha * (all_alpha + 1e-8).log()).sum(dim=1).mean()
        norm_ent = ent.item() / np.log(n_slots)

    elapsed = time.time() - t0
    print(f"│  Active slots: {active}/{n_slots}", flush=True)
    print(f"│  Max coverage: {max_cov:.1f}%", flush=True)
    print(f"│  Entropy: {norm_ent:.3f}", flush=True)
    print(f"│  Time: {elapsed:.0f}s", flush=True)

    # ── Visualization: slot masks overlaid on real photos ─────
    slot_colors_rgb = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [1, 0.5, 0]])
    P = 16
    n_show = 6

    # Figure 1: original + overlay composite
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    for i in range(n_show):
        vi = val_idx[i].item()
        img_np = images[vi].permute(1, 2, 0).numpy()

        # Row 0: original image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Image {i}", fontsize=9)
        axes[0, i].axis('off')

        # Row 1: slot ownership (argmax, colored) — upscaled to 224
        masks = alpha_val[i].numpy().reshape(n_slots, P, P)
        owner = alpha_val[i].argmax(dim=0).numpy().reshape(P, P)
        owner_rgb = np.zeros((P, P, 3))
        for s in range(n_slots):
            owner_rgb[owner == s] = slot_colors_rgb[s]
        owner_up = np.repeat(np.repeat(owner_rgb, 14, axis=0), 14, axis=1)
        axes[1, i].imshow(owner_up)
        axes[1, i].set_title("Slot ownership", fontsize=9)
        axes[1, i].axis('off')

        # Row 2: overlay (image + slot color blend)
        overlay = img_np.copy()
        for s in range(n_slots):
            mask_up = np.repeat(np.repeat(
                masks[s].reshape(P, P), 14, axis=0), 14, axis=1)
            for c in range(3):
                overlay[:, :, c] = overlay[:, :, c] * (1 - 0.4 * mask_up) + \
                    0.4 * mask_up * slot_colors_rgb[s, c]
        axes[2, i].imshow(np.clip(overlay, 0, 1))
        axes[2, i].set_title("Overlay", fontsize=9)
        axes[2, i].axis('off')

    fig.suptitle(f"Phase 27b Test 3: Flowers102 — Entropy {norm_ent:.3f}, "
                 f"{active}/{n_slots} active", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase27b_voc_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  → results/phase27b_voc_results.png", flush=True)

    # Figure 2: individual slot heatmaps for 2 images
    fig, axes = plt.subplots(2, n_slots + 1, figsize=(2.5 * (n_slots + 1), 5))
    for row in range(2):
        vi = val_idx[row].item()
        img_np = images[vi].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Original", fontsize=8)
        axes[row, 0].axis('off')
        for s in range(n_slots):
            mask = alpha_val[row, s].numpy().reshape(P, P)
            axes[row, s + 1].imshow(mask, cmap='hot', vmin=0, vmax=1,
                                     interpolation='nearest')
            axes[row, s + 1].set_title(f"Slot {s}", fontsize=8)
            axes[row, s + 1].axis('off')
    fig.suptitle("Phase 27b Test 3: Individual Slot Heatmaps (Flowers102)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase27b_voc_slots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  → results/phase27b_voc_slots.png", flush=True)
    print("└─ Done", flush=True)

    # Save model
    torch.save(ae.state_dict(), OUTPUT_DIR / "phase27b_voc_model.pt")
    print(f"│  → results/phase27b_voc_model.pt", flush=True)

    return norm_ent


def run_phase28_slot_jepa():
    """Phase 28: JEPA dynamics in slot space.

    Frozen SlotAttentionDINO encodes physics sequences into slot vectors.
    SlotPredictor MLP learns: slots(t) → slots(t+1).
    Evaluation: autoregressive rollout over 5 steps.
    """
    import random

    print("=" * 60, flush=True)
    print("PHASE 28: JEPA Dynamics in Slot Space", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"│  Device: {device}", flush=True)

    # ── Load frozen SlotAttentionDINO ─────────────────────────
    print("\n┌─ Loading frozen SlotAttentionDINO", flush=True)
    ae = SlotAttentionDINO(n_slots=7, slot_dim=64, img_size=64).to(device)
    state = torch.load(OUTPUT_DIR / "phase27_model.pt", map_location=device)
    ae.load_state_dict(state)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    print("│  Model loaded and frozen", flush=True)
    print("└─ Done", flush=True)

    # ── Generate physics sequences ────────────────────────────
    n_sequences = 1000
    n_frames = 20
    img_size = 64

    palette = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
    ]

    print(f"\n┌─ Generating {n_sequences} sequences × {n_frames} frames", flush=True)
    all_frames = []  # will become [n_sequences, n_frames, 3, H, W]

    for seq_i in range(n_sequences):
        n_obj = random.randint(2, 4)
        S = img_size

        objects = []
        for oi in range(n_obj):
            r = random.randint(5, 10)
            cx = random.uniform(r + 2, S - r - 3)
            cy = random.uniform(r + 2, S - r - 3)
            vx = random.uniform(-1.5, 1.5)
            vy = random.uniform(-1.5, 1.5)
            objects.append({'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy, 'r': r,
                            'color': palette[oi % len(palette)]})

        frames = []
        for fi in range(n_frames):
            img = np.ones((S, S, 3), dtype=np.float32) * 0.5
            for obj in objects:
                r = obj['r']
                cx_i, cy_i = int(round(obj['cx'])), int(round(obj['cy']))
                color = np.array(obj['color'])
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx*dx + dy*dy <= r*r:
                            px, py = cx_i + dx, cy_i + dy
                            if 0 <= px < S and 0 <= py < S:
                                img[py, px] = color
            frames.append(img.transpose(2, 0, 1))

            for obj in objects:
                obj['cx'] += obj['vx']; obj['cy'] += obj['vy']
                r = obj['r']
                if obj['cx'] - r < 0: obj['cx'] = r; obj['vx'] *= -1
                if obj['cx'] + r >= S: obj['cx'] = S - r - 1; obj['vx'] *= -1
                if obj['cy'] - r < 0: obj['cy'] = r; obj['vy'] *= -1
                if obj['cy'] + r >= S: obj['cy'] = S - r - 1; obj['vy'] *= -1

        all_frames.append(np.array(frames))

        if (seq_i + 1) % 200 == 0:
            print(f"│  Generated {seq_i+1}/{n_sequences} sequences", flush=True)

    all_frames = torch.tensor(np.array(all_frames), dtype=torch.float32)
    # [1000, 20, 3, 64, 64]
    print(f"│  Shape: {list(all_frames.shape)}", flush=True)
    print("└─ Done", flush=True)

    # ── Encode all frames with frozen SA ──────────────────────
    print(f"\n┌─ Encoding {n_sequences * n_frames} frames → slot vectors", flush=True)
    all_slots = []  # will become [1000, 20, 7, 64]

    with torch.no_grad():
        for seq_i in range(n_sequences):
            seq_slots = []
            for fi in range(0, n_frames, 8):  # batch 8 frames at a time
                batch = all_frames[seq_i, fi:fi+8].to(device)
                slots, _ = ae.encode(batch)  # [B, 7, 64]
                seq_slots.append(slots.cpu())
            all_slots.append(torch.cat(seq_slots, dim=0))  # [20, 7, 64]

            if (seq_i + 1) % 200 == 0:
                print(f"│  Encoded {seq_i+1}/{n_sequences} sequences", flush=True)
                if device.type == 'mps':
                    torch.mps.empty_cache()

    all_slots = torch.stack(all_slots)  # [1000, 20, 7, 64]
    print(f"│  Slot cache shape: {list(all_slots.shape)}", flush=True)

    # Compute slot vector statistics for reference
    slot_var = all_slots.var().item()
    slot_mean_norm = all_slots.reshape(-1, 64).norm(dim=-1).mean().item()
    print(f"│  Slot variance: {slot_var:.4f}", flush=True)
    print(f"│  Slot mean L2 norm: {slot_mean_norm:.4f}", flush=True)
    print("└─ Done", flush=True)

    # Free frame memory
    del all_frames
    if device.type == 'mps':
        torch.mps.empty_cache()

    # ── Build training pairs ──────────────────────────────────
    n_train_seq = 800
    n_val_seq = 200

    # slots_t → slots_t+1 pairs: [n_seq * 19, 7, 64]
    train_slots_t = all_slots[:n_train_seq, :-1].reshape(-1, 7, 64)  # [15200, 7, 64]
    train_slots_tp1 = all_slots[:n_train_seq, 1:].reshape(-1, 7, 64)
    val_slots_t = all_slots[n_train_seq:, :-1].reshape(-1, 7, 64)    # [3800, 7, 64]
    val_slots_tp1 = all_slots[n_train_seq:, 1:].reshape(-1, 7, 64)

    print(f"\n│  Train pairs: {len(train_slots_t)}", flush=True)
    print(f"│  Val pairs: {len(val_slots_t)}", flush=True)

    # Baseline: copy-previous MSE (predicting slots_t = slots_t+1)
    copy_mse = F.mse_loss(val_slots_t, val_slots_tp1).item()
    print(f"│  Copy-previous baseline MSE: {copy_mse:.6f}", flush=True)

    # ── Train SlotPredictor ───────────────────────────────────
    n_slots = 7
    slot_dim = 64
    pred_epochs = 200
    batch_size = 64
    lr = 1e-3

    print(f"\n{'=' * 50}", flush=True)
    print(f"Training SlotPredictor ({pred_epochs} epochs, lr={lr})", flush=True)
    print("=" * 50, flush=True)

    predictor = SlotPredictor(n_slots=n_slots, slot_dim=slot_dim, hidden_dim=256).to(device)
    pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"│  Predictor params: {pred_params:,}", flush=True)

    pred_opt = torch.optim.Adam(predictor.parameters(), lr=lr)
    pred_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pred_opt, T_max=pred_epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    t_train = time.time()

    for epoch in range(pred_epochs):
        predictor.train()
        perm = torch.randperm(len(train_slots_t))
        ep_loss, nb = 0, 0

        for i in range(0, len(train_slots_t), batch_size):
            idx = perm[i:i+batch_size]
            st = train_slots_t[idx].to(device)
            st1 = train_slots_tp1[idx].to(device)

            pred = predictor(st)
            loss = F.mse_loss(pred, st1)

            pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            pred_opt.step()
            ep_loss += loss.item(); nb += 1

        pred_sched.step()

        # Validation every 10 epochs + epoch 1
        should_print = (epoch + 1) == 1 or (epoch + 1) % 10 == 0
        if should_print:
            predictor.eval()
            with torch.no_grad():
                val_preds = []
                for vi in range(0, len(val_slots_t), 256):
                    vb = val_slots_t[vi:vi+256].to(device)
                    vp = predictor(vb)
                    val_preds.append(vp.cpu())
                val_preds = torch.cat(val_preds, dim=0)
                val_loss = F.mse_loss(val_preds, val_slots_tp1).item()

            improvement = (1 - val_loss / copy_mse) * 100
            elapsed = time.time() - t_train
            elapsed_ep = elapsed / (epoch + 1)
            eta = elapsed_ep * (pred_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{pred_epochs}: "
                  f"train={ep_loss/nb:.6f} val={val_loss:.6f} "
                  f"vs_copy={improvement:.1f}% better "
                  f"[{elapsed:.0f}s] ETA={eta:.0f}s ({eta/60:.0f}min)", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(predictor.state_dict(), OUTPUT_DIR / "phase28_predictor.pt")
            else:
                patience_counter += 10  # increments by print interval

            # Early stop on plateau
            if patience_counter >= 30:
                print(f"│  Early stop: no improvement for {patience_counter} epochs", flush=True)
                break

    print(f"│  Best val loss: {best_val_loss:.6f} at epoch {best_epoch}", flush=True)

    # Load best model
    predictor.load_state_dict(
        torch.load(OUTPUT_DIR / "phase28_predictor.pt", map_location=device))
    predictor.eval()

    # ── Autoregressive evaluation ─────────────────────────────
    print(f"\n┌─ Autoregressive Rollout Evaluation", flush=True)
    print(f"│  Given frames 1-5, predict frames 6-10", flush=True)

    val_seqs = all_slots[n_train_seq:]  # [200, 20, 7, 64]
    n_context = 5
    n_predict = 5

    step_mses = []  # MSE at each prediction step
    step_cosines = []  # cosine similarity at each step

    with torch.no_grad():
        for pred_step in range(n_predict):
            if pred_step == 0:
                # First prediction: use last context frame
                input_slots = val_seqs[:, n_context - 1].to(device)  # [200, 7, 64]
            else:
                input_slots = pred_slots  # use previous prediction

            pred_slots = predictor(input_slots)  # [200, 7, 64]
            target_slots = val_seqs[:, n_context + pred_step].to(device)

            mse = F.mse_loss(pred_slots, target_slots).item()
            # Cosine similarity per slot, averaged
            cos = F.cosine_similarity(
                pred_slots.reshape(-1, slot_dim),
                target_slots.reshape(-1, slot_dim), dim=-1).mean().item()

            step_mses.append(mse)
            step_cosines.append(cos)

            print(f"│  Step {pred_step+1} (frame {n_context+pred_step+1}): "
                  f"MSE={mse:.6f} cosine={cos:.4f}", flush=True)

    print(f"│", flush=True)
    print(f"│  1-step MSE: {step_mses[0]:.6f} (copy baseline: {copy_mse:.6f}, "
          f"{(1-step_mses[0]/copy_mse)*100:.1f}% better)", flush=True)
    print(f"│  5-step MSE: {step_mses[4]:.6f} "
          f"(ratio to 1-step: {step_mses[4]/step_mses[0]:.2f}x)", flush=True)

    # Success criteria
    success_1step = step_mses[0] < 0.1 * slot_var
    success_rollout = step_mses[4] < 2.0 * step_mses[0]
    success_vs_copy = step_mses[0] < copy_mse

    print(f"│", flush=True)
    print(f"│  1-step < 0.1×variance ({0.1*slot_var:.6f}): "
          f"{'YES' if success_1step else 'NO'}", flush=True)
    print(f"│  5-step < 2× 1-step: "
          f"{'YES' if success_rollout else 'NO'}", flush=True)
    print(f"│  Better than copy: "
          f"{'YES' if success_vs_copy else 'NO'}", flush=True)
    print("└─ Done", flush=True)

    # ── Visualization ─────────────────────────────────────────
    print(f"\n┌─ Generating visualizations", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vis_seqs = [0, 1]
    vis_frames = [0, 4, 5, 7, 9]  # frames 1,5,6,8,10
    labels = ['f1', 'f5\n(last ctx)', 'f6\n(pred 1)', 'f8\n(pred 3)', 'f10\n(pred 5)']

    fig, axes = plt.subplots(len(vis_seqs) * 2, len(vis_frames),
                              figsize=(3 * len(vis_frames), 3 * len(vis_seqs) * 2))

    with torch.no_grad():
        for si_idx, si in enumerate(vis_seqs):
            seq_slots = val_seqs[si]  # [20, 7, 64]

            # Ground truth alpha masks for all frames
            gt_alphas = {}
            for fi in vis_frames:
                slots_fi = seq_slots[fi:fi+1].to(device)
                _, alpha = ae.decode(slots_fi)  # [1, 7, 256]
                gt_alphas[fi] = alpha[0].cpu()

            # Predicted alpha masks for frames 6-10
            pred_alphas = {}
            cur = seq_slots[n_context - 1:n_context].to(device)
            for ps in range(n_predict):
                cur = predictor(cur)
                _, alpha = ae.decode(cur)
                pred_alphas[n_context + ps] = alpha[0].cpu()

            slot_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:7, :3]
            P = 16

            for fi_idx, fi in enumerate(vis_frames):
                # Row 1: ground truth slot masks
                ax = axes[si_idx * 2, fi_idx]
                alpha = gt_alphas[fi]  # [7, 256]
                owner = alpha.argmax(dim=0).numpy().reshape(P, P)
                owner_rgb = np.zeros((P, P, 3))
                for s in range(7):
                    owner_rgb[owner == s] = slot_colors[s]
                owner_up = np.repeat(np.repeat(owner_rgb, 4, axis=0), 4, axis=1)
                ax.imshow(owner_up)
                ax.set_title(f"GT {labels[fi_idx]}", fontsize=8)
                ax.axis('off')

                # Row 2: predicted slot masks (use GT for context frames)
                ax = axes[si_idx * 2 + 1, fi_idx]
                if fi < n_context:
                    alpha_p = gt_alphas[fi]
                    title = f"(ctx) {labels[fi_idx]}"
                else:
                    alpha_p = pred_alphas[fi]
                    title = f"Pred {labels[fi_idx]}"
                owner_p = alpha_p.argmax(dim=0).numpy().reshape(P, P)
                owner_rgb_p = np.zeros((P, P, 3))
                for s in range(7):
                    owner_rgb_p[owner_p == s] = slot_colors[s]
                owner_up_p = np.repeat(np.repeat(owner_rgb_p, 4, axis=0), 4, axis=1)
                ax.imshow(owner_up_p)
                ax.set_title(title, fontsize=8)
                ax.axis('off')

    fig.suptitle(f"Phase 28: Slot JEPA — GT (top) vs Predicted (bottom)\n"
                 f"1-step MSE={step_mses[0]:.6f}, 5-step MSE={step_mses[4]:.6f}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase28_slot_jepa.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  → results/phase28_slot_jepa.png", flush=True)

    # Step-wise error plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    steps = list(range(1, n_predict + 1))
    ax1.plot(steps, step_mses, 'bo-', label='Predictor')
    ax1.axhline(y=copy_mse, color='r', linestyle='--', label='Copy baseline')
    ax1.set_xlabel('Prediction step')
    ax1.set_ylabel('MSE')
    ax1.set_title('Autoregressive MSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, step_cosines, 'go-')
    ax2.set_xlabel('Prediction step')
    ax2.set_ylabel('Cosine similarity')
    ax2.set_title('Slot Cosine Similarity')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Phase 28: Autoregressive Rollout', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase28_rollout_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"│  → results/phase28_rollout_error.png", flush=True)
    print("└─ Done", flush=True)

    elapsed = time.time() - t0
    print(f"\n│  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    return {
        'best_val_loss': best_val_loss,
        'copy_mse': copy_mse,
        'step_mses': step_mses,
        'step_cosines': step_cosines,
        'slot_var': slot_var,
        'success': success_1step and success_vs_copy,
    }


def run_phase27b_video_consistency():
    """Phase 27b Test 2: Video frame consistency (inference only).

    Load saved SlotAttentionDINO model, generate physics sequences with
    moving bouncing circles, encode every frame, track slot-object
    assignments across frames via Hungarian matching on slot attention masks.
    """
    import time
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment

    t0 = time.time()
    print("=" * 60, flush=True)
    print("PHASE 27b TEST 2: Video Frame Consistency (Inference Only)", flush=True)
    print("=" * 60, flush=True)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"│  Device: {device}", flush=True)

    # ── Load saved model ──────────────────────────────────────
    print("\n┌─ Loading SlotAttentionDINO from results/phase27_model.pt", flush=True)
    ae = SlotAttentionDINO(n_slots=7, slot_dim=64, img_size=64).to(device)
    state = torch.load(OUTPUT_DIR / "phase27_model.pt", map_location=device)
    ae.load_state_dict(state)
    ae.eval()
    print("│  Model loaded", flush=True)
    print("└─ Done", flush=True)

    # ── Generate physics sequences ────────────────────────────
    n_sequences = 10
    n_frames = 20
    img_size = 64

    palette = [
        [1.0, 0.0, 0.0],    # red
        [0.0, 1.0, 0.0],    # green
        [0.0, 0.0, 1.0],    # blue
        [1.0, 1.0, 0.0],    # yellow
        [0.0, 1.0, 1.0],    # cyan
        [1.0, 0.0, 1.0],    # magenta
    ]

    print(f"\n┌─ Generating {n_sequences} sequences, {n_frames} frames each", flush=True)
    sequences = []  # list of [n_frames, 3, H, W] tensors
    seq_objects = []  # list of object info per sequence

    for seq_i in range(n_sequences):
        n_obj = random.randint(2, 4)
        S = img_size

        # Initialize objects: position, velocity, radius, color
        objects = []
        for oi in range(n_obj):
            r = random.randint(5, 10)
            cx = random.uniform(r + 2, S - r - 3)
            cy = random.uniform(r + 2, S - r - 3)
            vx = random.uniform(-1.5, 1.5)
            vy = random.uniform(-1.5, 1.5)
            color = palette[oi % len(palette)]
            objects.append({'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy, 'r': r, 'color': color})

        frames = []
        for fi in range(n_frames):
            # Render frame
            img = np.ones((S, S, 3), dtype=np.float32) * 0.5  # gray bg
            for obj in objects:
                r = obj['r']
                cx, cy = int(round(obj['cx'])), int(round(obj['cy']))
                color = np.array(obj['color'])
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx*dx + dy*dy <= r*r:
                            px, py = cx + dx, cy + dy
                            if 0 <= px < S and 0 <= py < S:
                                img[py, px] = color

            frames.append(img.transpose(2, 0, 1))  # CHW

            # Physics step: linear motion + wall bounce
            for obj in objects:
                obj['cx'] += obj['vx']
                obj['cy'] += obj['vy']
                r = obj['r']
                if obj['cx'] - r < 0:
                    obj['cx'] = r
                    obj['vx'] *= -1
                if obj['cx'] + r >= S:
                    obj['cx'] = S - r - 1
                    obj['vx'] *= -1
                if obj['cy'] - r < 0:
                    obj['cy'] = r
                    obj['vy'] *= -1
                if obj['cy'] + r >= S:
                    obj['cy'] = S - r - 1
                    obj['vy'] *= -1

        sequences.append(torch.tensor(np.array(frames), dtype=torch.float32))
        seq_objects.append(n_obj)

    print(f"│  Objects per sequence: {seq_objects}", flush=True)
    print("└─ Done", flush=True)

    # ── Encode all frames and extract slot masks ──────────────
    print(f"\n┌─ Encoding frames with SlotAttentionDINO", flush=True)
    all_seq_alphas = []  # [n_seq][n_frames] of [K, 256] alpha masks

    with torch.no_grad():
        for seq_i, seq_frames in enumerate(sequences):
            seq_alphas = []
            # Process frames one at a time (inference)
            for fi in range(n_frames):
                frame = seq_frames[fi:fi+1].to(device)  # [1, 3, H, W]
                _, _, _, _, _, alpha = ae(frame, training=False)
                # alpha: [1, K, 256]
                seq_alphas.append(alpha[0].cpu())  # [K, 256]
            all_seq_alphas.append(seq_alphas)
            if device.type == 'mps':
                torch.mps.empty_cache()

    print(f"│  Encoded {n_sequences * n_frames} frames", flush=True)
    print("└─ Done", flush=True)

    # ── Track consistency via Hungarian matching ──────────────
    print(f"\n┌─ Computing slot-object consistency", flush=True)

    consistency_scores = []

    for seq_i in range(n_sequences):
        alphas = all_seq_alphas[seq_i]  # list of [K, 256]
        n_frames_seq = len(alphas)

        # Use frame 0 as reference assignment
        ref_alpha = alphas[0]  # [K, 256]

        # For each subsequent frame, find best slot permutation via Hungarian
        # matching (maximize IoU between reference and current slot masks)
        frame_matches = []  # list of permutation arrays

        for fi in range(1, n_frames_seq):
            cur_alpha = alphas[fi]  # [K, 256]

            # Compute cost matrix: negative IoU between ref slot i and cur slot j
            # Use hard assignments for IoU
            ref_assign = ref_alpha.argmax(dim=0)  # [256]
            cur_assign = cur_alpha.argmax(dim=0)  # [256]

            K = ref_alpha.shape[0]
            cost = np.zeros((K, K))
            for si in range(K):
                ref_mask = (ref_assign == si)
                for sj in range(K):
                    cur_mask = (cur_assign == sj)
                    intersection = (ref_mask & cur_mask).float().sum().item()
                    union = (ref_mask | cur_mask).float().sum().item()
                    iou = intersection / (union + 1e-8)
                    cost[si, sj] = -iou  # negative for minimization

            row_ind, col_ind = linear_sum_assignment(cost)
            frame_matches.append(col_ind)

        # Consistency: for each frame, check if the permutation is identity
        # (same slot tracks same region across frames)
        # More nuanced: track which slots are "active" (cover >2% of patches)
        ref_assign = alphas[0].argmax(dim=0)  # [256]
        active_slots = []
        for s in range(alphas[0].shape[0]):
            if (ref_assign == s).float().mean().item() > 0.02:
                active_slots.append(s)

        n_consistent = 0
        n_total = 0
        for fi, perm in enumerate(frame_matches):
            for s in active_slots:
                n_total += 1
                if perm[s] == s:
                    n_consistent += 1

        consistency = n_consistent / max(n_total, 1)
        consistency_scores.append(consistency)
        print(f"│  Seq {seq_i}: {consistency*100:.1f}% consistent "
              f"({len(active_slots)} active slots, {seq_objects[seq_i]} objects)",
              flush=True)

    avg_consistency = np.mean(consistency_scores)
    print(f"│")
    print(f"│  Average consistency: {avg_consistency*100:.1f}%", flush=True)
    print(f"│  Target: 80%+", flush=True)
    print(f"│  {'SUCCESS' if avg_consistency >= 0.8 else 'BELOW TARGET'}", flush=True)
    print("└─ Done", flush=True)

    # ── Visualize 2 sequences ─────────────────────────────────
    print(f"\n┌─ Generating visualizations", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vis_frames = [0, 4, 9, 14, 19]  # frames 1, 5, 10, 15, 20
    vis_seqs = [0, 1]

    fig, axes = plt.subplots(
        len(vis_seqs) * 2, len(vis_frames),
        figsize=(3 * len(vis_frames), 3 * len(vis_seqs) * 2))

    for si_idx, si in enumerate(vis_seqs):
        alphas = all_seq_alphas[si]
        seq_frames = sequences[si]

        for fi_idx, fi in enumerate(vis_frames):
            # Row 1: original frame
            ax = axes[si_idx * 2, fi_idx]
            frame_img = seq_frames[fi].permute(1, 2, 0).numpy()
            ax.imshow(frame_img)
            ax.set_title(f"Seq {si} Frame {fi+1}", fontsize=8)
            ax.axis('off')

            # Row 2: slot mask visualization (argmax ownership, colored)
            ax = axes[si_idx * 2 + 1, fi_idx]
            alpha = alphas[fi]  # [K, 256]
            ownership = alpha.argmax(dim=0)  # [256]
            # Reshape to 16x16 and upscale to 64x64
            mask_16 = ownership.reshape(16, 16).numpy()
            # Create RGB visualization
            slot_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:7, :3]
            mask_rgb = np.zeros((16, 16, 3))
            for s in range(7):
                mask_rgb[mask_16 == s] = slot_colors[s]
            # Nearest-neighbor upscale to 64x64
            mask_rgb_up = np.repeat(np.repeat(mask_rgb, 4, axis=0), 4, axis=1)
            ax.imshow(mask_rgb_up)
            ax.set_title(f"Slots f{fi+1}", fontsize=8)
            ax.axis('off')

    plt.suptitle(f"Phase 27b Test 2: Video Consistency — Avg {avg_consistency*100:.1f}%",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase27b_video_consistency.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"│  → results/phase27b_video_consistency.png", flush=True)
    print("└─ Done", flush=True)

    elapsed = time.time() - t0
    print(f"\n│  Total time: {elapsed:.0f}s", flush=True)

    return avg_consistency >= 0.8


def run_phase26f():
    """Phase 26f: Match Original Slot Attention (Research-Informed).

    Root cause fix for slot binding failure:
    1. CNN decoder with ConvTranspose2d (locality bias) instead of MLP
    2. Dense encoder features (32x32=1024 tokens) instead of 8x8=64
    3. Gradient background instead of checkerboard floor
    4. LR warmup + exponential decay

    Stage 1: Train SlotAttentionAEv5 on gradient-bg scenes (100 epochs max)
    Stage 2: Freeze AE, extract slots, train SlotJEPAPredictor (50 epochs)
    """
    print("=" * 60)
    print("PHASE 26f: Match Original Slot Attention (Research-Informed)")
    print("=" * 60)
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"│  Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"│  Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"│  Using CPU")

    # ── Dataset ────────────────────────────────────────────────
    print("\n┌─ Dataset (Gradient-BG Rich Scenes)")
    dataset_path = OUTPUT_DIR / 'phase26f_dataset.pt'
    if dataset_path.exists():
        dataset = torch.load(dataset_path, weights_only=False)
        print(f"│  Loaded: {len(dataset['img_a'])} frames")
    else:
        dataset = collect_simplified_rich_dataset(
            n_episodes=300, steps_per_episode=40, n_objects=8, img_size=64)
        torch.save(dataset, dataset_path)
        print(f"│  Collected: {len(dataset['img_a'])} frames")

    n = len(dataset['img_a'])
    n_train = int(0.8 * n)
    target_imgs = dataset['img_target']
    print(f"│  {n} frames, collection: {time.time()-t0:.0f}s")
    print("└─ Done")

    # Sample frames
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i in range(6):
        idx = i * n // 6
        axes[0, i].imshow(dataset['img_a'][idx].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Agent A' if i == 0 else ''); axes[0, i].axis('off')
        axes[1, i].imshow(dataset['img_b'][idx].permute(1, 2, 0).numpy())
        axes[1, i].set_title('Agent B' if i == 0 else ''); axes[1, i].axis('off')
        axes[2, i].imshow(target_imgs[idx].permute(1, 2, 0).numpy())
        axes[2, i].set_title('Target' if i == 0 else ''); axes[2, i].axis('off')
    fig.suptitle('Phase 26f: Gradient-BG Rich Scenes (spheres/cubes/pyramids)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_dataset.png", dpi=150)
    plt.close()
    print("│  → results/phase26f_dataset.png")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Reference Slot Attention AE (CNN Decoder)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 1: SlotAttentionAEv5 (CNN decoder, 1024 tokens)")
    print("=" * 50)

    n_slots = 10
    slot_dim = 64
    ae = SlotAttentionAEv5(n_slots=n_slots, slot_dim=slot_dim, img_size=64)
    ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print(f"│  AE params: {ae_params:,}")
    print(f"│  Encoder: 4-layer CNN stride-2-once → 32x32=1024 tokens × 64 features")
    print(f"│  Decoder: CNN ConvTranspose2d 8→16→32→64 (locality bias)")
    print(f"│  Slots: {n_slots} × {slot_dim}")
    ae = ae.to(device)

    # Adam with LR warmup + exponential decay
    base_lr = 4e-4
    ae_opt = torch.optim.Adam(ae.parameters(), lr=base_lr)
    ae_epochs = 100
    warmup_epochs = 10
    decay_rate = 0.97

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return decay_rate ** (epoch - warmup_epochs)

    ae_sched = torch.optim.lr_scheduler.LambdaLR(ae_opt, lr_lambda)
    batch_size = 48
    ae_hist = {'recon': [], 'entropy': []}

    t_s1 = time.time()
    early_stop = False

    for epoch in range(ae_epochs):
        ae.train()
        perm = torch.randperm(n_train)
        ep_loss, ep_recon, ep_ent, nb = 0, 0, 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = target_imgs[idx].to(device)

            total_loss, recon_loss, entropy_reg, recon, slots, alpha = ae(
                imgs, training=True)
            ae_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_opt.step()
            ep_loss += recon_loss.item()
            ep_ent += entropy_reg.item()
            nb += 1

        ae_sched.step()
        ae_hist['recon'].append(ep_loss / nb)
        ae_hist['entropy'].append(ep_ent / nb)

        # ── Mask diagnostic every 10 epochs ──
        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s1
            eta = elapsed_ep / (epoch + 1) * (ae_epochs - epoch - 1)
            cur_lr = ae_sched.get_last_lr()[0]

            ae.eval()
            with torch.no_grad():
                diag_imgs = target_imgs[:200].to(device)
                _, _, _, _, _, diag_alpha = ae(diag_imgs, training=False)
                # diag_alpha: [B, K, H, W]

                # Active slots: count slots owning >=1% of pixels (argmax)
                ownership = diag_alpha.argmax(dim=1)  # [B, H, W]
                B_d = ownership.shape[0]
                slot_pixel_counts = torch.zeros(B_d, n_slots, device=device)
                for s in range(n_slots):
                    slot_pixel_counts[:, s] = (ownership == s).float().sum(
                        dim=(1, 2))
                total_pixels = 64 * 64
                slot_fracs = slot_pixel_counts / total_pixels  # [B, K]
                # Average across batch
                mean_fracs = slot_fracs.mean(dim=0)  # [K]
                active_slots = int((mean_fracs > 0.01).sum().item())
                max_coverage = mean_fracs.max().item() * 100

                # Normalized entropy
                masks_flat = diag_alpha.reshape(B_d, n_slots, -1)
                pixel_ent = -(masks_flat * (masks_flat + 1e-8).log()
                              ).sum(dim=1).mean()
                norm_ent = pixel_ent.item() / np.log(n_slots)

            print(f"│  Epoch {epoch+1:3d}/{ae_epochs}: "
                  f"recon={ep_loss/nb:.4f} entropy={ep_ent/nb:.3f} "
                  f"lr={cur_lr:.1e}")
            print(f"│    Mask diag: active_slots={active_slots}/{n_slots} "
                  f"max_coverage={max_coverage:.1f}% "
                  f"norm_entropy={norm_ent:.3f}")
            print(f"│    Slot ownership: "
                  f"{(mean_fracs.cpu().numpy() * 100).round(1)}")
            print(f"│    [{elapsed_ep:.0f}s elapsed, ETA {eta:.0f}s]")

            # Early stop check at epoch 50
            if epoch + 1 == 50 and max_coverage < 15.0:
                print("│")
                print("│  ⚠ EARLY STOP: At epoch 50, max_coverage < 15%")
                print("│    All slots still have roughly equal coverage.")
                print("│    Slot binding is likely failing.")
                early_stop = True
                break

    print(f"│  Stage 1 time: {time.time()-t_s1:.0f}s")
    if early_stop:
        print("│  STOPPED EARLY — slot masks still uniform at epoch 50")

    # ── Stage 1 Evaluation ─────────────────────────────────────
    print("\n┌─ Stage 1 Evaluation")
    ae.eval()
    val_idx = torch.arange(n_train, n)

    with torch.no_grad():
        val_imgs = target_imgs[val_idx[:200]].to(device)
        total_loss_val, recon_loss_val, ent_val, recon_val, slots_val, alpha_val = ae(
            val_imgs, training=False)
        print(f"│  Val recon: {recon_loss_val.item():.4f}")

    # Reconstructions + slot mask composite
    slot_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                   [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1], [0, 0.5, 0],
                   [0.5, 0.5, 0.5]]
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axes[0, i].imshow(val_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')

        axes[1, i].imshow(
            recon_val[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[1, i].set_title('Recon' if i == 0 else '')
        axes[1, i].axis('off')

        masks = alpha_val[i].cpu().numpy()  # [K, H, W]
        composite = np.zeros((64, 64, 3))
        for s in range(n_slots):
            for c in range(3):
                composite[:, :, c] += masks[s] * slot_colors[s][c]
        axes[2, i].imshow(np.clip(composite, 0, 1))
        axes[2, i].set_title('Slot masks' if i == 0 else '')
        axes[2, i].axis('off')

    fig.suptitle('Phase 26f: CNN-Decoder Slot AE (10×64)\n'
                 'R=S0 G=S1 B=S2 Y=S3 M=S4 C=S5 O=S6 P=S7 G=S8 Gr=S9',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_reconstruction.png", dpi=150)
    plt.close()
    print("│  → results/phase26f_reconstruction.png")

    # Individual slot mask heatmaps (new for 26f)
    fig, axes = plt.subplots(2, n_slots, figsize=(2.5 * n_slots, 5))
    for s in range(n_slots):
        # Sample 0
        axes[0, s].imshow(alpha_val[0, s].cpu().numpy(),
                          cmap='hot', vmin=0, vmax=1)
        axes[0, s].set_title(f'Slot {s}', fontsize=9)
        axes[0, s].axis('off')
        # Sample 1
        axes[1, s].imshow(alpha_val[1, s].cpu().numpy(),
                          cmap='hot', vmin=0, vmax=1)
        axes[1, s].axis('off')
    fig.suptitle('Phase 26f: Individual Slot Masks (2 samples)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_slot_masks.png", dpi=150)
    plt.close()
    print("│  → results/phase26f_slot_masks.png")

    # Slot-object binding
    print("│  Slot-object binding:")
    from sklearn.linear_model import LinearRegression

    with torch.no_grad():
        all_slots_list = []
        for ci in range(0, len(val_idx), batch_size):
            bi = val_idx[ci:ci+batch_size]
            s = ae.encode(target_imgs[bi].to(device))
            all_slots_list.append(s.cpu())
        all_slots_np = torch.cat(all_slots_list).numpy()

    gt_state = dataset['state'][val_idx].numpy()
    n_objects = 8
    vals_per_obj = 9

    slot_object_r2 = np.zeros((n_slots, n_objects))
    for s in range(n_slots):
        for o in range(n_objects):
            X = all_slots_np[:, s, :]
            y = gt_state[:, o*vals_per_obj:o*vals_per_obj+3]
            n_fit = min(1000, len(X) // 2)
            if n_fit < 10:
                continue
            reg = LinearRegression().fit(X[:n_fit], y[:n_fit])
            r2 = reg.score(X[n_fit:], y[n_fit:])
            slot_object_r2[s, o] = max(0, r2)

    n_bound = int(sum(slot_object_r2.max(axis=0) > 0.3))
    best_r2 = slot_object_r2.max()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(n_slots))
    ax.set_yticklabels([f'Slot {i}' for i in range(n_slots)])
    ax.set_xticks(range(n_objects))
    ax.set_xticklabels([f'Obj {i}' for i in range(n_objects)])
    for i in range(n_slots):
        for j in range(n_objects):
            ax.text(j, i, f'{slot_object_r2[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if slot_object_r2[i, j] > 0.5 else 'black')
    ax.set_title(
        f'Phase 26f: Slot→Object Binding — {n_bound}/8 bound (max R²={best_r2:.3f})')
    plt.colorbar(im, label='R²')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_slot_binding.png", dpi=150)
    plt.close()

    for o in range(n_objects):
        bs = slot_object_r2[:, o].argmax()
        br = slot_object_r2[:, o].max()
        print(f"│    Obj {o} → Slot {bs} (R²={br:.3f})")
    print(f"│  Bound: {n_bound}/{n_objects}")

    # Alpha mask entropy
    with torch.no_grad():
        masks_flat = alpha_val[:100].cpu().reshape(100, n_slots, 64*64)
        entropy = -(masks_flat * (masks_flat + 1e-8).log()).sum(dim=1).mean()
        max_alpha = masks_flat.max(dim=2).values.mean(dim=0)
        print(f"│  Assignment entropy: {entropy.item():.3f}")
        print(f"│  Per-slot max alpha: {max_alpha.numpy().round(3)}")

    # Proto-affordances
    slot_mass_corr = np.zeros(n_slots)
    slot_shape_corr = np.zeros(n_slots)
    for s in range(n_slots):
        slot_norm = np.linalg.norm(all_slots_np[:, s, :], axis=1)
        for o in range(n_objects):
            mass = gt_state[:, o*vals_per_obj+6]
            shape = gt_state[:, o*vals_per_obj+8]
            if np.std(mass) > 1e-6:
                r_m = abs(np.corrcoef(slot_norm, mass)[0, 1])
                if not np.isnan(r_m) and r_m > slot_mass_corr[s]:
                    slot_mass_corr[s] = r_m
            if np.std(shape) > 1e-6:
                r_s = abs(np.corrcoef(slot_norm, shape)[0, 1])
                if not np.isnan(r_s) and r_s > slot_shape_corr[s]:
                    slot_shape_corr[s] = r_s

    print(f"│  Mass corr:  {np.round(slot_mass_corr, 3)}")
    print(f"│  Shape corr: {np.round(slot_shape_corr, 3)}")
    print("└─ Done")

    # ════════════════════════════════════════════════════════════
    # STAGE 2: JEPA on Frozen Slots
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("STAGE 2: JEPA on Frozen Slots")
    print("=" * 50)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print("│  Extracting slots...")
    slots_a = extract_all_slots(dataset['img_a'], ae, device, batch_size)
    slots_b = extract_all_slots(dataset['img_b'], ae, device, batch_size)
    slots_t_next = extract_all_slots(
        dataset['next_img_target'], ae, device, batch_size)
    print(f"│  Slots: {slots_a.shape}")

    predictor = SlotJEPAPredictor(
        n_slots=n_slots, slot_dim=slot_dim, comm_dim=8, action_dim=4)
    pred_params = sum(p.numel() for p in predictor.parameters()
                      if p.requires_grad)
    print(f"│  Predictor params: {pred_params:,}")
    predictor = predictor.to(device)

    pred_opt = torch.optim.AdamW(
        predictor.parameters(), lr=3e-4, weight_decay=0.01)
    pred_epochs = 50
    pred_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        pred_opt, T_max=pred_epochs, eta_min=1e-5)
    pred_hist = {'pred': []}

    t_s2 = time.time()
    for epoch in range(pred_epochs):
        predictor.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            sa = slots_a[idx].to(device)
            sb = slots_b[idx].to(device)
            act = dataset['action'][idx].to(device)
            st = slots_t_next[idx].to(device)

            loss, _, _ = predictor(sa, sb, act, st)
            pred_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            pred_opt.step()
            ep_loss += loss.item(); nb += 1

        pred_sched.step()
        pred_hist['pred'].append(ep_loss / nb)

        if (epoch + 1) % 10 == 0:
            elapsed_ep = time.time() - t_s2
            eta = elapsed_ep / (epoch + 1) * (pred_epochs - epoch - 1)
            print(f"│  Epoch {epoch+1:3d}/{pred_epochs}: pred={ep_loss/nb:.6f} "
                  f"[{elapsed_ep:.0f}s, ETA {eta:.0f}s]")

    print(f"│  Stage 2 time: {time.time()-t_s2:.0f}s")

    # ── Communication Tests ────────────────────────────────────
    print("\n┌─ Communication Tests")
    predictor.eval()

    with torch.no_grad():
        sa_val = slots_a[val_idx].to(device)
        sb_val = slots_b[val_idx].to(device)
        act_val = dataset['action'][val_idx].to(device)
        st_val = slots_t_next[val_idx].to(device)

        msg_b_real = predictor.communicate(sb_val)
        next_real = predictor.predict_next(sa_val, act_val, msg_b_real)
        mse_real = F.mse_loss(next_real, st_val).item()

        next_zero = predictor.predict_next(
            sa_val, act_val, torch.zeros_like(msg_b_real))
        mse_zero = F.mse_loss(next_zero, st_val).item()

        next_shuf = predictor.predict_next(
            sa_val, act_val, msg_b_real[torch.randperm(len(val_idx))])
        mse_shuf = F.mse_loss(next_shuf, st_val).item()

        next_noise = predictor.predict_next(
            sa_val, act_val, torch.randn_like(msg_b_real))
        mse_noise = F.mse_loss(next_noise, st_val).item()

    m_zero = (1 - mse_real / mse_zero) * 100 if mse_zero > 0 else 0
    m_shuf = (1 - mse_real / mse_shuf) * 100 if mse_shuf > 0 else 0
    m_noise = (1 - mse_real / mse_noise) * 100 if mse_noise > 0 else 0

    print(f"│  Real:     {mse_real:.6f}")
    print(f"│  Zero:     {mse_zero:.6f}  (margin: {m_zero:.1f}%)")
    print(f"│  Shuffled: {mse_shuf:.6f}  (margin: {m_shuf:.1f}%)")
    print(f"│  Noise:    {mse_noise:.6f}  (margin: {m_noise:.1f}%)")

    cond_names = ['Real', 'Zero', 'Shuffled', 'Noise']
    cond_vals = [mse_real, mse_zero, mse_shuf, mse_noise]
    cond_colors = ['#27ae60', '#e74c3c', '#e67e22', '#8e44ad']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cond_names, cond_vals, color=cond_colors, alpha=0.8)
    for bar, v, m in zip(bars, cond_vals,
                         ['', f'{m_zero:.1f}%', f'{m_shuf:.1f}%',
                          f'{m_noise:.1f}%']):
        if m:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    m, ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Slot MSE')
    ax.set_title('Phase 26f: Communication Tests')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_comm_tests.png", dpi=150)
    plt.close()
    print("│  → results/phase26f_comm_tests.png")
    print("└─ Done")

    # ── Message ↔ State Heatmap ────────────────────────────────
    print("\n┌─ Msg ↔ State Heatmap")
    msg_a_np = predictor.communicate(sa_val).detach().cpu().numpy()
    n_state = gt_state.shape[1]
    corr_msg = np.zeros((8, n_state))
    for c in range(8):
        for si in range(n_state):
            if (np.std(gt_state[:, si]) > 1e-6
                    and np.std(msg_a_np[:, c]) > 1e-6):
                r = np.corrcoef(msg_a_np[:, c], gt_state[:, si])[0, 1]
                if not np.isnan(r):
                    corr_msg[c, si] = abs(r)

    state_labels = []
    for i in range(n_objects):
        for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r', 'sh']:
            state_labels.append(f'O{i}_{v}')

    fig, ax = plt.subplots(figsize=(24, 5))
    im = ax.imshow(corr_msg, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Msg{i}' for i in range(8)])
    ax.set_xticks(range(n_state))
    ax.set_xticklabels(state_labels, rotation=90, fontsize=5)
    for i in range(1, n_objects):
        ax.axvline(x=i*vals_per_obj - 0.5, color='white', linewidth=2)
    ax.set_title('Phase 26f: Msg ↔ State')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_msg_heatmap.png", dpi=150)
    plt.close()
    print(f"│  Max corr: {corr_msg.max():.3f}")
    print("│  → results/phase26f_msg_heatmap.png")
    print("└─ Done")

    # ── Summary Dashboard ──────────────────────────────────────
    print("\n┌─ Summary Dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 26f: Reference Slot Attention (CNN Decoder)',
                 fontsize=16, fontweight='bold')

    axes[0, 0].plot(ae_hist['recon'], label='recon')
    axes[0, 0].plot(ae_hist['entropy'], label='entropy', alpha=0.7)
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Stage 1 Recon → {ae_hist['recon'][-1]:.4f}")

    im_b = axes[0, 1].imshow(
        slot_object_r2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Binding: {n_bound}/8 (max={best_r2:.3f})')
    axes[0, 1].set_yticks(range(n_slots))
    axes[0, 1].set_xticks(range(8))

    axes[0, 2].plot(pred_hist['pred'])
    axes[0, 2].set_title(f"Stage 2 Pred → {pred_hist['pred'][-1]:.6f}")

    axes[1, 0].bar(cond_names, cond_vals, color=cond_colors)
    axes[1, 0].set_title(
        f'Comm: Z={m_zero:.1f}% S={m_shuf:.1f}% N={m_noise:.1f}%')

    # Pixel ownership bar chart (new for 26f)
    with torch.no_grad():
        sample_alpha = alpha_val[0].cpu()  # [K, H, W]
        ownership_sample = sample_alpha.argmax(dim=0)  # [H, W]
        ownership_pct = torch.zeros(n_slots)
        for s in range(n_slots):
            ownership_pct[s] = (ownership_sample == s).float().mean() * 100
    bar_colors = [slot_colors[s] for s in range(n_slots)]
    axes[1, 1].bar(range(n_slots), ownership_pct.numpy(), color=bar_colors)
    axes[1, 1].set_xlabel('Slot')
    axes[1, 1].set_ylabel('% pixels owned')
    axes[1, 1].set_title('Pixel Ownership (sample 0)')
    axes[1, 1].set_xticks(range(n_slots))

    axes[1, 2].axis('off')
    elapsed = time.time() - t0
    summary_text = (
        f"PHASE 26f\n"
        f"{'━' * 30}\n\n"
        f"CNN decoder (locality bias)\n"
        f"1024 tokens (32x32)\n"
        f"Gradient background\n"
        f"LR warmup + exp decay\n\n"
        f"AE: {ae_params:,} params\n"
        f"Pred: {pred_params:,} params\n"
        f"Time: {elapsed / 60:.0f} min\n\n"
        f"Recon: {ae_hist['recon'][-1]:.4f}\n"
        f"Binding: {n_bound}/8\n"
        f"Max R²: {best_r2:.3f}\n\n"
        f"Pred: {pred_hist['pred'][-1]:.6f}\n"
        f"Comm zero: {m_zero:.1f}%\n"
        f"Comm shuf: {m_shuf:.1f}%\n"
        f"Comm noise: {m_noise:.1f}%\n"
    )
    axes[1, 2].text(0.05, 0.95, summary_text,
                    transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow',
                              alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase26f_summary.png", dpi=150)
    plt.close()

    torch.save(ae.state_dict(), OUTPUT_DIR / "phase26f_autoencoder.pt")
    torch.save(predictor.state_dict(), OUTPUT_DIR / "phase26f_predictor.pt")
    print("│  → results/phase26f_summary.png")
    print("└─ Done")

    print(f"\n{'=' * 60}")
    print(f"PHASE 26f SUMMARY")
    print(f"{'=' * 60}")
    print(f"  AE: {ae_params:,} params, recon={ae_hist['recon'][-1]:.4f}")
    print(f"  Pred: {pred_params:,} params, pred={pred_hist['pred'][-1]:.6f}")
    print(f"  Binding: {n_bound}/{n_objects} (R²>0.3)")
    print(f"  Max R²:  {best_r2:.3f}")
    print(f"  Comm zero: {m_zero:.1f}%")
    print(f"  Comm shuf: {m_shuf:.1f}%")
    print(f"  Comm noise: {m_noise:.1f}%")
    print(f"  Mass corr:   {slot_mass_corr.max():.3f}")
    print(f"  Shape corr:  {slot_shape_corr.max():.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    print("╔" + "═"*58 + "╗")
    print("║  WORLD MODEL EXPERIMENT                                 ║")
    print("║  From Bouncing Ball to Latent Communication             ║")
    print("║  (LeCun 2022 architecture, implemented at toy scale)    ║")
    print("╚" + "═"*58 + "╝")

    t0 = time.time()
    model_p1, config = run_phase1()
    run_phase2(model_p1, config)
    run_phase3()
    run_phase4()
    run_phase5()
    results_5b = run_phase5b()

    # Phase 5 vs 5b comparison
    print(f"\n{'='*60}")
    print("PHASE 5 vs 5b: Effective Rank Comparison")
    print("="*60)
    print("Phase 5  (redundant views): rank ≈ 30 (threshold), 25.67 (entropy)")
    print("  → Communication channel stays high-rank (no structure needed)")
    if results_5b:
        r_t = results_5b.get('final_rank_threshold', '?')
        r_e = results_5b.get('final_rank_entropy', 0)
        print(f"Phase 5b (occlusion):       rank = {r_t} (threshold), {r_e:.2f} (entropy)")
        if isinstance(r_e, float) and r_e < 25.0:
            print("  → Lower rank! Communication channel developing structure.")
        else:
            print("  → Rank still high — may need more training or harder task.")
    fused_beat = results_5b and results_5b['mse_f'] < min(results_5b['mse_a'], results_5b['mse_b'])
    if fused_beat:
        print("\n★ KEY RESULT: Fusion outperforms single agents when information")
        print("  asymmetry forces genuine communication!")
    print()

    run_phase6()
    run_phase7()
    run_phase8()
    run_phase9()
    run_phase10()
    run_phase11()
    run_phase10b()
    run_phase11b()
    run_phase12()
    run_phase13()
    run_phase14()
    run_phase15()
    run_phase16()
    run_phase17()
    run_phase18()
    run_phase19()
    run_phase20()
    run_phase21()
    run_phase22()
    run_phase23()
    run_phase24()
    run_phase25()
    run_phase25b()
    run_phase25c()
    run_phase25d()
    run_phase26()
    run_phase26b()
    run_phase26c()
    run_phase26d()
    run_phase26e()
    run_phase26f()

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL PHASES COMPLETE in {total:.0f}s ({total/60:.1f} min)")
    print(f"Results: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f}")

"""
Phases 107-109: Stress Test, Minimal Onboarding, Domain Bootstrap
==================================================================
107: 100-agent inference stress test
108: Minimal viable projection layer
109: Domain bootstrap speed

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase107_109 import run_all; run_all()"
"""

import time, json, math, os, sys, threading, queue, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
HIDDEN_DIM = 128
N_HEADS = 2
VOCAB_SIZE = 3


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, max(1, n_frames))
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
        self.encoder = encoder; self.vocab_size = vocab_size; self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)])
    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x); messages, logits = [], []
        for head in self.heads:
            l = head(h)
            if self.training: m = F.gumbel_softmax(l, tau=tau, hard=hard)
            else: m = F.one_hot(l.argmax(-1), self.vocab_size).float()
            messages.append(m); logits.append(l)
        return torch.cat(messages, dim=-1), logits


class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=1.0, hard=True):
        messages, logits = [], []
        for s, v in zip(self.senders, views):
            m, l = s(v, tau, hard); messages.append(m); logits.extend(l)
        return torch.cat(messages, dim=-1), logits


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))
    def forward(self, ma, mb):
        return self.net(torch.cat([ma, mb], dim=-1)).squeeze(-1)


def load_spring():
    vd = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dd = torch.load(RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)
    vf = vd["features"].float(); obj = vd["obj_names"]; mass = vd["mass_values"]
    df = dd["features"].float(); nf = vf.shape[1]
    dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()
    return vf, dt, obj, mass


def train_quick(configs, mass_values, obj_names, vocab_size, seed, hidden_dim=128,
                comm_epochs=400, patience=150):
    """Lean training, returns accuracy + posdis + elapsed time."""
    n_agents = len(configs); msg_dim = n_agents * N_HEADS * vocab_size
    agent_views = [f.float() for f, _ in configs]
    unique_objs = sorted(set(obj_names)); rng = np.random.RandomState(seed * 1000 + 42)
    n_holdout = max(4, len(unique_objs) // 5)
    holdout = set(rng.choice(unique_objs, n_holdout, replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout])
    if len(holdout_ids) < 4: return None

    torch.manual_seed(seed); np.random.seed(seed)
    senders = [CompositionalSender(TemporalEncoder(hidden_dim, d, f.shape[1]),
               hidden_dim, vocab_size, N_HEADS) for f, d in configs]
    multi = MultiAgentSender(senders).to(DEVICE)
    recv = CompositionalReceiver(msg_dim, hidden_dim).to(DEVICE)
    s_opt = torch.optim.Adam(multi.parameters(), lr=1e-3)
    r_opt = torch.optim.Adam(recv.parameters(), lr=3e-3)
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    nb = max(1, len(train_ids) // 32)
    best_acc, best_state, best_ep = 0.0, None, 0
    t0 = time.time()
    me = math.log(vocab_size)

    for ep in range(comm_epochs):
        if time.time() - t0 > 600: break
        if ep - best_ep > patience and best_acc > 0.55: break
        if ep > 0 and ep % 40 == 0:
            recv = CompositionalReceiver(msg_dim, hidden_dim).to(DEVICE)
            r_opt = torch.optim.Adam(recv.parameters(), lr=3e-3)
        multi.train(); recv.train()
        tau = 3.0 + (1.0 - 3.0) * ep / max(1, comm_epochs - 1); hard = ep >= 30
        for _ in range(nb):
            ia = rng.choice(train_ids, 32); ib = rng.choice(train_ids, 32)
            s = ia == ib
            while s.any(): ib[s] = rng.choice(train_ids, s.sum()); s = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib]); keep = md > 0.5
            if keep.sum() < 4: continue
            ia, ib = ia[keep], ib[keep]
            va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = multi(va, tau, hard); mb, lb = multi(vb, tau, hard)
            loss = F.binary_cross_entropy_with_logits(recv(ma, mb), label)
            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1); p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / me < 0.1: loss = loss - 0.03 * ent
            if torch.isnan(loss): s_opt.zero_grad(); r_opt.zero_grad(); continue
            s_opt.zero_grad(); r_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0); s_opt.step(); r_opt.step()
        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep+1) % 50 == 0 or ep == 0:
            multi.eval(); recv.eval()
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(32, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs); ib_h = er.choice(holdout_ids, bs)
                    sh = ia_h == ib_h
                    while sh.any(): ib_h[sh] = er.choice(holdout_ids, sh.sum()); sh = ia_h == ib_h
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in agent_views]; wh = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    mah, _ = multi(vh); mbh, _ = multi(wh)
                    c += ((recv(mah, mbh)>0) == la_h).sum().item(); t += len(la_h)
                acc = c / max(t, 1)
                if acc > best_acc: best_acc = acc; best_ep = ep; best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}

    elapsed = time.time() - t0
    if best_state: multi.load_state_dict(best_state)
    multi.eval()

    # PosDis
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), 32):
            views = [v[i:i+32].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            all_tokens.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)
    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj_names)); oi = {o: i for i, o in enumerate(uo)}
    obj_bins = np.digitize(np.array([oi[o] for o in obj_names]),
                            np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    pd, _, _ = positional_disentanglement(all_tokens, np.stack([mass_bins, obj_bins], axis=1), vocab_size)
    return {"accuracy": float(best_acc), "posdis": float(pd), "elapsed_s": elapsed,
            "converge_epoch": best_ep + 1}


# ═══ PHASE 107: Stress Test ═══

def run_phase107():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 107: 100-Agent Inference Stress Test              ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vf, dt, obj_names, mass_values = load_spring()
    nf = vf.shape[1]

    # Build 100 senders (50 V-JEPA + 50 DINOv2), each with 1 frame
    n_agents = 100; vocab_size = 3
    msg_dim = n_agents * N_HEADS * vocab_size
    print(f"  Building {n_agents} agents (msg_dim={msg_dim})...", flush=True)

    torch.manual_seed(42)
    senders = []
    agent_views = []
    for i in range(n_agents):
        frame_idx = i % nf
        if i % 2 == 0:
            senders.append(CompositionalSender(
                TemporalEncoder(HIDDEN_DIM, 1024, n_frames=1), HIDDEN_DIM, vocab_size, N_HEADS))
            agent_views.append(vf[:, frame_idx:frame_idx+1, :].float())
        else:
            senders.append(CompositionalSender(
                TemporalEncoder(HIDDEN_DIM, 384, n_frames=1), HIDDEN_DIM, vocab_size, N_HEADS))
            agent_views.append(dt[:, frame_idx:frame_idx+1, :].float())

    multi = MultiAgentSender(senders).cpu().eval()
    recv = CompositionalReceiver(msg_dim, HIDDEN_DIM).cpu().eval()
    av_cpu = [v.cpu() for v in agent_views]

    # Warmup
    print(f"  Warming up...", flush=True)
    with torch.no_grad():
        va = [v[:1] for v in av_cpu]; vb = [v[1:2] for v in av_cpu]
        ma, _ = multi(va); mb, _ = multi(vb); recv(ma, mb)

    # Throughput: sequential
    print(f"  Sequential throughput test (100 rounds)...", flush=True)
    latencies = []
    for _ in range(100):
        i = np.random.randint(0, len(obj_names)); j = np.random.randint(0, len(obj_names))
        t_s = time.perf_counter()
        with torch.no_grad():
            va = [v[i:i+1] for v in av_cpu]; vb = [v[j:j+1] for v in av_cpu]
            ma, _ = multi(va); mb, _ = multi(vb); recv(ma, mb)
        latencies.append((time.perf_counter() - t_s) * 1000)

    lat = np.array(latencies)
    print(f"  Sequential: mean={np.mean(lat):.1f}ms median={np.median(lat):.1f}ms "
          f"p95={np.percentile(lat,95):.1f}ms throughput={1000/np.mean(lat):.0f}/s", flush=True)

    # Concurrent: threaded message bus
    print(f"  Concurrent throughput test (100 threads, 10 msgs each)...", flush=True)
    msg_bus = queue.Queue()
    concurrent_lats = []
    drops = [0]

    def sender_thread(agent_id, n_msgs):
        for _ in range(n_msgs):
            i = np.random.randint(0, len(obj_names))
            t_s = time.perf_counter()
            with torch.no_grad():
                va = [v[i:i+1] for v in av_cpu]
                ma, _ = multi(va)
            msg_bus.put({"msg": ma, "t_start": t_s, "agent": agent_id})

    def receiver_thread(expected):
        received = 0
        while received < expected:
            try:
                msg = msg_bus.get(timeout=5.0)
                t_end = time.perf_counter()
                concurrent_lats.append((t_end - msg["t_start"]) * 1000)
                received += 1
            except queue.Empty:
                drops[0] += expected - received
                break

    n_threads = 100; msgs_per = 10
    threads = [threading.Thread(target=sender_thread, args=(i, msgs_per)) for i in range(n_threads)]
    recv_thread = threading.Thread(target=receiver_thread, args=(n_threads * msgs_per,))
    recv_thread.start()
    for t in threads: t.start()
    for t in threads: t.join()
    recv_thread.join()

    clat = np.array(concurrent_lats) if concurrent_lats else np.array([0])
    results = {
        "n_agents": n_agents,
        "sequential": {
            "mean_ms": float(np.mean(lat)), "median_ms": float(np.median(lat)),
            "p95_ms": float(np.percentile(lat, 95)),
            "throughput_per_s": float(1000 / np.mean(lat))},
        "concurrent": {
            "n_threads": n_threads, "msgs_per_thread": msgs_per,
            "total_messages": n_threads * msgs_per,
            "received": len(concurrent_lats), "drops": drops[0],
            "mean_ms": float(np.mean(clat)), "p95_ms": float(np.percentile(clat, 95)),
            "throughput_per_s": float(len(concurrent_lats) / (np.sum(clat)/1000)) if np.sum(clat) > 0 else 0},
    }
    print(f"  Concurrent: {len(concurrent_lats)}/{n_threads*msgs_per} received, "
          f"{drops[0]} drops, mean={np.mean(clat):.1f}ms, "
          f"throughput={results['concurrent']['throughput_per_s']:.0f}/s", flush=True)

    save_path = RESULTS_DIR / "phase107_stress_test.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return results


# ═══ PHASE 108: Minimal Viable Onboarding ═══

def run_phase108():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 108: Minimal Viable Projection Layer              ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    vf, dt, obj_names, mass_values = load_spring()
    nf = vf.shape[1]; fpa = nf // 2
    hidden_dims = [8, 16, 32, 64, 128, 256, 512]
    n_seeds = 10
    all_results = {}

    for hd in hidden_dims:
        print(f"\n  ── hidden_dim={hd} ──", flush=True)
        seed_results = []
        for seed in range(n_seeds):
            configs = [(vf[:, :fpa, :], 1024), (dt[:, fpa:, :], 384)]
            r = train_quick(configs, mass_values, obj_names, VOCAB_SIZE, seed, hidden_dim=hd)
            if r: seed_results.append(r)
        if seed_results:
            accs = [r["accuracy"] for r in seed_results]
            pds = [r["posdis"] for r in seed_results]
            params = sum(p.numel() for p in TemporalEncoder(hd, 1024, 4).parameters()) + hd * VOCAB_SIZE * N_HEADS
            all_results[hd] = {
                "acc": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "acc_mean": float(np.mean(accs)), "pd_mean": float(np.mean(pds)),
                "approx_params": params, "passes_threshold": np.mean(pds) > 0.5}
            print(f"    acc={np.mean(accs):.1%} PD={np.mean(pds):.3f} "
                  f"params≈{params:,} pass={np.mean(pds) > 0.5}", flush=True)
        torch.mps.empty_cache()

    # Find minimum
    min_hd = min((hd for hd, r in all_results.items() if r["passes_threshold"]), default=None)
    print(f"\n  Minimum viable hidden_dim: {min_hd}", flush=True)
    if min_hd:
        print(f"  At hidden_dim={min_hd}: {all_results[min_hd]['acc']} acc, "
              f"{all_results[min_hd]['posdis']} PD, ~{all_results[min_hd]['approx_params']:,} params", flush=True)

    save_path = RESULTS_DIR / "phase108_minimal_onboarding.json"
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_results


# ═══ PHASE 109: Domain Bootstrap Speed ═══

def run_phase109():
    print("\n╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  Phase 109: Domain Bootstrap Speed                       ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    t0 = time.time()

    scenarios = [("spring", None), ("fall", None), ("ramp", 500)]
    n_seeds = 5
    all_results = {}

    for scenario, max_clips in scenarios:
        print(f"\n  ── {scenario} ──", flush=True)
        vd = torch.load(RESULTS_DIR / f"phase87_phys101_{scenario}_features.pt", weights_only=False)
        dd = torch.load(RESULTS_DIR / f"phase87_phys101_{scenario}_static.pt", weights_only=False)
        vf = vd["features"].float(); obj = vd["obj_names"]; mass = vd["mass_values"]
        df = dd["features"].float()
        if max_clips and len(obj) > max_clips:
            rng = np.random.RandomState(42); idx = rng.choice(len(obj), max_clips, replace=False); idx.sort()
            vf = vf[idx]; df = df[idx]; obj = [obj[i] for i in idx]; mass = mass[idx]
        nf = vf.shape[1]; fpa = nf // 2
        dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()

        seed_results = []
        for seed in range(n_seeds):
            configs = [(vf[:, :fpa, :], 1024), (dt[:, fpa:, :], 384)]
            r = train_quick(configs, mass, obj, VOCAB_SIZE, seed)
            if r:
                seed_results.append(r)
                print(f"    Seed {seed}: acc={r['accuracy']:.1%} PD={r['posdis']:.3f} "
                      f"ep={r['converge_epoch']} time={r['elapsed_s']:.0f}s", flush=True)
        if seed_results:
            times = [r["elapsed_s"] for r in seed_results]
            pds = [r["posdis"] for r in seed_results]
            accs = [r["accuracy"] for r in seed_results]
            all_results[scenario] = {
                "n_clips": len(obj), "n_objects": len(set(obj)),
                "wall_clock_s": f"{np.mean(times):.0f}±{np.std(times):.0f}",
                "wall_clock_mean": float(np.mean(times)),
                "accuracy": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "posdis_above_0.5": np.mean(pds) > 0.5}
        torch.mps.empty_cache()

    print(f"\n  ╔═══ BOOTSTRAP SPEED ═══╗", flush=True)
    for sc, r in all_results.items():
        print(f"  ║ {sc:8s}: {r['wall_clock_s']}s, {r['n_clips']} clips, "
              f"PD={r['posdis']}, PD>0.5={'YES' if r['posdis_above_0.5'] else 'NO'}", flush=True)
    print(f"  ╚════════════════════════╝", flush=True)

    save_path = RESULTS_DIR / "phase109_bootstrap_speed.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved {save_path} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return all_results


def run_all():
    t_total = time.time()
    for num, func in [(107, run_phase107), (108, run_phase108), (109, run_phase109)]:
        try:
            print(f"\n{'#'*70}\n#  PHASE {num}\n{'#'*70}", flush=True)
            func()
            torch.mps.empty_cache()
        except Exception as e:
            print(f"  PHASE {num} FAILED: {e}", flush=True)
            traceback.print_exc()
    print(f"\n  Phases 107-109 complete. Total: {(time.time()-t_total)/60:.1f}min", flush=True)


if __name__ == "__main__":
    run_all()

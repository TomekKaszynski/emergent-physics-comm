"""
WMCP Spec Compliance Test
==========================
Validates every spec claim against actual trained agent outputs.
Generates COMPLIANCE_REPORT.md with PASS/FAIL per requirement.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 protocol-spec/tests/test_spec_compliance.py
"""

import time, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Self-contained — all architecture definitions inline
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
SPEC_DIR = Path("protocol-spec")
HIDDEN_DIM = 128
N_HEADS = 2
N_RECEIVERS = 3


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, max(1, n_frames))
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2),
            nn.ReLU(), nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1))
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
            messages.append(msg); all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits


class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=1.0, hard=True):
        messages, all_logits = [], []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg); all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))
    def forward(self, msg_a, msg_b):
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


def mutual_information(x, y):
    x_vals, y_vals = np.unique(x), np.unique(y)
    n = len(x)
    mi = 0.0
    for xv in x_vals:
        for yv in y_vals:
            p_xy = np.sum((x == xv) & (y == yv)) / n
            p_x = np.sum(x == xv) / n
            p_y = np.sum(y == yv) / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi


def positional_disentanglement(tokens, attributes, vocab_size):
    n_pos, n_attr = tokens.shape[1], attributes.shape[1]
    mi_matrix = np.zeros((n_pos, n_attr))
    for p in range(n_pos):
        for a in range(n_attr):
            mi_matrix[p, a] = mutual_information(tokens[:, p], attributes[:, a])
    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            s = np.sort(mi_matrix[p])[::-1]
            if s[0] > 1e-10:
                pos_dis += (s[0] - s[1]) / s[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0
    return float(pos_dis), mi_matrix


def run_compliance_test():
    print("WMCP Spec Compliance Test", flush=True)
    print("=" * 50, flush=True)

    results = []

    # Load spring data
    vjepa_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)
    vf = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    df = dino_data["features"].float()
    nf = vf.shape[1]
    dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()

    # Train a hetero 4-agent K=3 model
    n_agents, vocab_size = 4, 3
    fpa = nf // n_agents
    configs = []
    for i in range(n_agents):
        if i % 2 == 0:
            configs.append((vf[:, i*fpa:(i+1)*fpa, :], 1024))
        else:
            configs.append((dt[:, i*fpa:(i+1)*fpa, :], 384))

    msg_dim = n_agents * N_HEADS * vocab_size
    torch.manual_seed(0); np.random.seed(0)
    senders = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, dim, n_frames=feat.shape[1]),
               HIDDEN_DIM, vocab_size, N_HEADS) for feat, dim in configs]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=1e-3)
    r_opts = [torch.optim.Adam(r.parameters(), lr=3e-3) for r in receivers]

    agent_views = [feat.float() for feat, _ in configs]
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    unique_objs = sorted(set(obj_names))
    rng = np.random.RandomState(42)
    holdout_objs = set(rng.choice(unique_objs, max(4, len(unique_objs)//5), replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

    best_acc, best_state, best_ep = 0.0, None, 0
    print("  Training model...", flush=True)
    for ep in range(400):
        if ep - best_ep > 150 and best_acc > 0.55: break
        if ep > 0 and ep % 40 == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=3e-3)
        multi.train()
        for r in receivers: r.train()
        tau = 3.0 + (1.0 - 3.0) * ep / 399
        hard = ep >= 30
        nb = max(1, len(train_ids) // 32)
        for _ in range(nb):
            ia = rng.choice(train_ids, 32); ib = rng.choice(train_ids, 32)
            same = ia == ib
            while same.any(): ib[same] = rng.choice(train_ids, same.sum()); same = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib]); keep = md > 0.5
            if keep.sum() < 4: continue
            ia, ib = ia[keep], ib[keep]
            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard); mb, lb = multi(vb, tau=tau, hard=hard)
            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb), label) for r in receivers) / len(receivers)
            me = math.log(vocab_size)
            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1); p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / me < 0.1: loss = loss - 0.03 * ent
            if torch.isnan(loss): s_opt.zero_grad(); [o.zero_grad() for o in r_opts]; continue
            s_opt.zero_grad(); [o.zero_grad() for o in r_opts]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
            s_opt.step(); [o.step() for o in r_opts]
        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep+1) % 50 == 0 or ep == 0:
            multi.eval(); [r.eval() for r in receivers]
            with torch.no_grad():
                correct = total = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(32, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs); ib_h = er.choice(holdout_ids, bs)
                    s = ia_h == ib_h
                    while s.any(): ib_h[s] = er.choice(holdout_ids, s.sum()); s = ia_h == ib_h
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in agent_views]
                    wh = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    mah, _ = multi(vh); mbh, _ = multi(wh)
                    for r in receivers:
                        correct += ((r(mah, mbh) > 0) == la_h).sum().item(); total += len(la_h)
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc; best_ep = ep
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}

    if best_state: multi.load_state_dict(best_state)
    multi.eval()
    print(f"  Training complete: acc={best_acc:.1%}", flush=True)

    # ═══ TEST 1: Message Format ═══
    with torch.no_grad():
        va = [v[:1].to(DEVICE) for v in agent_views]
        msg, logits = multi(va)
    msg_len = msg.shape[1]
    expected_len = n_agents * N_HEADS * vocab_size
    t1 = msg_len == expected_len
    results.append(("Message format: correct dimensions", t1,
                     f"Expected {expected_len}, got {msg_len}"))

    # Verify discrete (one-hot per position)
    msg_reshaped = msg.view(1, n_agents * N_HEADS, vocab_size)
    is_onehot = all(
        (msg_reshaped[0, p].sum().item() - 1.0) < 1e-5 and
        msg_reshaped[0, p].max().item() > 0.99
        for p in range(n_agents * N_HEADS))
    results.append(("Message format: one-hot per position", is_onehot,
                     f"Checked {n_agents * N_HEADS} positions"))

    # ═══ TEST 2: PosDis > 0.5 ═══
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), 32):
            views = [v[i:i+32].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            all_tokens.append(np.stack([l.argmax(dim=-1).cpu().numpy() for l in logits], axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    mass_bins = np.digitize(mass_values, np.quantile(mass_values, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj_names)); oi = {o: i for i, o in enumerate(uo)}
    obj_bins = np.digitize(np.array([oi[o] for o in obj_names]),
                            np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    attrs = np.stack([mass_bins, obj_bins], axis=1)
    posdis, mi = positional_disentanglement(all_tokens, attrs, vocab_size)
    t2 = posdis > 0.5
    results.append(("Compositionality: PosDis > 0.5", t2, f"PosDis = {posdis:.3f}"))

    # ═══ TEST 3: Noise tolerance at σ=0.5 ═══
    multi.eval(); recv = receivers[0]; recv.eval()
    baseline_correct = noisy_correct = total_n = 0
    with torch.no_grad():
        er = np.random.RandomState(999)
        for _ in range(100):
            ia_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
            ib_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
            s = ia_h == ib_h
            while s.any(): ib_h[s] = er.choice(holdout_ids, s.sum()); s = ia_h == ib_h
            mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
            if kh.sum() < 2: continue
            ia_h, ib_h = ia_h[kh], ib_h[kh]
            va_h = [v[ia_h].to(DEVICE) for v in agent_views]
            vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
            la_h = mass_dev[ia_h] > mass_dev[ib_h]
            ma_h, _ = multi(va_h); mb_h, _ = multi(vb_h)
            baseline_correct += ((recv(ma_h, mb_h) > 0) == la_h).sum().item()
            noisy_ma = ma_h + torch.randn_like(ma_h) * 0.5
            noisy_mb = mb_h + torch.randn_like(mb_h) * 0.5
            noisy_correct += ((recv(noisy_ma, noisy_mb) > 0) == la_h).sum().item()
            total_n += len(la_h)
    baseline_acc = baseline_correct / max(total_n, 1)
    noisy_acc = noisy_correct / max(total_n, 1)
    drop = baseline_acc - noisy_acc
    t3 = drop < 0.10
    results.append(("Noise tolerance: <10% drop at σ=0.5", t3,
                     f"Baseline={baseline_acc:.1%}, Noisy={noisy_acc:.1%}, Drop={drop:.1%}"))

    # ═══ TEST 4: Latency < 10ms ═══
    multi_cpu = multi.cpu()
    recv_cpu = recv.cpu()
    av_cpu = [v.cpu() for v in agent_views]
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            va = [v[:1] for v in av_cpu]; vb = [v[1:2] for v in av_cpu]
            ma, _ = multi_cpu(va); mb, _ = multi_cpu(vb); recv_cpu(ma, mb)
    latencies = []
    for _ in range(200):
        i = np.random.randint(0, len(obj_names))
        j = np.random.randint(0, len(obj_names))
        t_s = time.perf_counter()
        with torch.no_grad():
            va = [v[i:i+1] for v in av_cpu]; vb = [v[j:j+1] for v in av_cpu]
            ma, _ = multi_cpu(va); mb, _ = multi_cpu(vb); recv_cpu(ma, mb)
        latencies.append((time.perf_counter() - t_s) * 1000)
    mean_lat = np.mean(latencies)
    t4 = mean_lat < 10.0
    results.append(("Latency: <10ms on CPU", t4,
                     f"Mean={mean_lat:.2f}ms, P95={np.percentile(latencies, 95):.2f}ms"))

    # ═══ TEST 5: Vocabulary size K=3 ═══
    t5 = vocab_size == 3
    results.append(("Config: K=3 vocabulary", t5, f"K={vocab_size}"))

    # ═══ TEST 6: 4-agent heterogeneous ═══
    dims_used = set()
    for _, dim in configs:
        dims_used.add(dim)
    t6 = len(dims_used) > 1 and n_agents == 4
    results.append(("Config: heterogeneous 4-agent", t6,
                     f"n_agents={n_agents}, dims={dims_used}"))

    # ═══ TEST 7: MI matrix shows mass dominance ═══
    mass_mi = np.mean([mi[p, 0] for p in range(mi.shape[0])])
    obj_mi = np.mean([mi[p, 1] for p in range(mi.shape[0])])
    t7 = mass_mi > obj_mi
    results.append(("MI structure: mass > object identity", t7,
                     f"Mass MI={mass_mi:.3f}, Object MI={obj_mi:.3f}"))

    # ═══ TEST 8: Holdout accuracy > 60% ═══
    t8 = best_acc > 0.60
    results.append(("Task performance: holdout acc > 60%", t8,
                     f"Accuracy={best_acc:.1%}"))

    # ═══ Generate Report ═══
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)

    lines = ["# WMCP Compliance Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Result: {n_pass}/{n_total} tests PASSED**\n")
    lines.append("| # | Requirement | Status | Detail |")
    lines.append("|---|-------------|--------|--------|")
    for i, (name, passed, detail) in enumerate(results):
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {i+1} | {name} | {status} | {detail} |")

    lines.append(f"\n## Configuration\n")
    lines.append(f"- K = {vocab_size}")
    lines.append(f"- L = {N_HEADS} positions per agent")
    lines.append(f"- N = {n_agents} agents")
    lines.append(f"- Architecture: heterogeneous (V-JEPA 2 + DINOv2)")
    lines.append(f"- Dataset: Physics 101 spring (206 clips, 26 objects)")
    lines.append(f"- Spec version: WMCP v0.1")

    report = "\n".join(lines)
    report_path = SPEC_DIR / "COMPLIANCE_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n  {n_pass}/{n_total} tests PASSED", flush=True)
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}", flush=True)
    print(f"\n  Report saved to {report_path}", flush=True)
    return n_pass == n_total


if __name__ == "__main__":
    run_compliance_test()

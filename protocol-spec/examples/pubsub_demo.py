"""
WMCP Pub-Sub Communication Demo
=================================
Two agents communicate through a threaded message bus.
Agent A (V-JEPA 2) publishes discrete messages.
Agent B (DINOv2) subscribes and decodes.

Usage:
  python3 protocol-spec/examples/pubsub_demo.py
"""

import time, queue, threading, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

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
        h = self.encoder(x)
        messages, logits = [], []
        for head in self.heads:
            l = head(h)
            messages.append(F.one_hot(l.argmax(-1), self.vocab_size).float())
            logits.append(l)
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


def run_demo():
    print("WMCP Pub-Sub Communication Demo", flush=True)
    print("=" * 50, flush=True)

    # Load data and train a quick model
    vjepa_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)
    vf = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    df = dino_data["features"].float()
    nf = vf.shape[1]; fpa = nf // 2
    dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()

    msg_dim = 2 * N_HEADS * VOCAB_SIZE
    torch.manual_seed(0); np.random.seed(0)
    configs = [(vf[:, :fpa, :], 1024), (dt[:, fpa:, :], 384)]
    senders = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, d, f.shape[1]), HIDDEN_DIM, VOCAB_SIZE, N_HEADS) for f, d in configs]
    multi = MultiAgentSender(senders).to(DEVICE)
    recv = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
    s_opt = torch.optim.Adam(multi.parameters(), lr=1e-3)
    r_opt = torch.optim.Adam(recv.parameters(), lr=3e-3)

    agent_views = [f.float() for f, _ in configs]
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    rng = np.random.RandomState(42)
    unique_objs = sorted(set(obj_names))
    holdout = set(rng.choice(unique_objs, max(4, len(unique_objs)//5), replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout])

    print("\n  Training agents (V-JEPA + DINOv2)...", flush=True)
    best_state = None; best_acc = 0
    for ep in range(300):
        multi.train(); recv.train()
        tau = 3.0 + (1.0 - 3.0) * ep / 299; hard = ep >= 30
        for _ in range(max(1, len(train_ids)//32)):
            ia = rng.choice(train_ids, 32); ib = rng.choice(train_ids, 32)
            s = ia == ib
            while s.any(): ib[s] = rng.choice(train_ids, s.sum()); s = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib]); keep = md > 0.5
            if keep.sum() < 4: continue
            ia, ib = ia[keep], ib[keep]
            va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, _ = multi(va, tau, hard); mb, _ = multi(vb, tau, hard)
            loss = F.binary_cross_entropy_with_logits(recv(ma, mb), label)
            s_opt.zero_grad(); r_opt.zero_grad(); loss.backward()
            s_opt.step(); r_opt.step()
        if (ep+1) % 100 == 0:
            multi.eval(); recv.eval()
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                holdout_ids = [i for i, o in enumerate(obj_names) if o in holdout]
                for _ in range(30):
                    ia_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    ib_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in agent_views]; wh = [v[ib_h].to(DEVICE) for v in agent_views]
                    mah, _ = multi(vh); mbh, _ = multi(wh)
                    c += ((recv(mah, mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item(); t += len(ia_h)
                acc = c / max(t, 1)
                if acc > best_acc: best_acc = acc; best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
            torch.mps.empty_cache()

    if best_state: multi.load_state_dict(best_state)
    print(f"  Trained. Accuracy: {best_acc:.1%}\n", flush=True)

    # Move to CPU for pub-sub
    multi_cpu = multi.cpu().eval()
    recv_cpu = recv.cpu().eval()
    av_cpu = [v.cpu() for v in agent_views]

    # Pub-Sub
    msg_bus = queue.Queue()
    latencies = []; correct = [0]; total = [0]
    n_rounds = 100

    def publisher():
        r = np.random.RandomState(123)
        for rd in range(n_rounds):
            i, j = r.randint(0, len(obj_names)), r.randint(0, len(obj_names))
            if abs(mass_values[i] - mass_values[j]) < 0.5: continue
            t_s = time.perf_counter()
            with torch.no_grad():
                va = [v[i:i+1] for v in av_cpu]; vb = [v[j:j+1] for v in av_cpu]
                ma, la = multi_cpu(va); mb, lb = multi_cpu(vb)
            tok_a = [l.argmax(-1).item() for l in la]
            tok_b = [l.argmax(-1).item() for l in lb]
            msg_bus.put({"ma": ma, "mb": mb, "tok_a": tok_a, "tok_b": tok_b,
                        "label": float(mass_values[i] > mass_values[j]),
                        "obj_a": obj_names[i], "obj_b": obj_names[j],
                        "mass_a": float(mass_values[i]), "mass_b": float(mass_values[j]),
                        "t_start": t_s, "round": rd})
        msg_bus.put(None)

    def subscriber():
        while True:
            msg = msg_bus.get()
            if msg is None: break
            with torch.no_grad():
                pred = recv_cpu(msg["ma"], msg["mb"]).item() > 0
            t_end = time.perf_counter()
            lat = (t_end - msg["t_start"]) * 1000
            latencies.append(lat)
            is_correct = pred == msg["label"]
            total[0] += 1
            if is_correct: correct[0] += 1

            if total[0] <= 10 or total[0] % 20 == 0:
                print(f"  Round {msg['round']:3d} │ "
                      f"{msg['obj_a']:>12s} ({msg['mass_a']:5.1f}g) vs "
                      f"{msg['obj_b']:>12s} ({msg['mass_b']:5.1f}g) │ "
                      f"tokens={msg['tok_a']}+{msg['tok_b']} │ "
                      f"pred={'A>B' if pred else 'B>A'} │ "
                      f"{'CORRECT' if is_correct else 'WRONG':>7s} │ "
                      f"{lat:.2f}ms", flush=True)

    print(f"  {'Round':>8s} │ {'Scene A':>24s} vs {'Scene B':>24s} │ "
          f"{'Tokens':>14s} │ {'Pred':>5s} │ {'Result':>7s} │ {'Latency'}", flush=True)
    print(f"  {'─'*8}─┼─{'─'*51}─┼─{'─'*14}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*7}", flush=True)

    t_pub = threading.Thread(target=publisher)
    t_sub = threading.Thread(target=subscriber)
    t_pub.start(); t_sub.start()
    t_pub.join(); t_sub.join()

    lat = np.array(latencies)
    print(f"\n  {'─'*60}", flush=True)
    print(f"  Exchanges:  {total[0]}", flush=True)
    print(f"  Accuracy:   {correct[0]/max(total[0],1):.1%}", flush=True)
    print(f"  Latency:    mean={np.mean(lat):.2f}ms  p95={np.percentile(lat,95):.2f}ms", flush=True)
    print(f"  Throughput: {1000/np.mean(lat):.0f} comms/s", flush=True)


if __name__ == "__main__":
    run_demo()

"""
WMCP Onboarding Demo: Add CLIP ViT-L/14 to Existing Protocol
==============================================================
Demonstrates onboarding a new encoder into a trained V-JEPA+DINOv2 protocol.
Existing agents are frozen. Only the new CLIP projection layer is trained.

Usage:
  python3 protocol-spec/examples/onboard_new_encoder.py
  python3 protocol-spec/examples/onboard_new_encoder.py --dry-run
"""

import argparse, time, math, os, sys
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
        messages, all_logits = [], []
        for head in self.heads:
            logits = head(h)
            msg = F.gumbel_softmax(logits, tau=tau, hard=hard) if self.training else F.one_hot(logits.argmax(-1), self.vocab_size).float()
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


def run_demo(dry_run=False):
    print("WMCP Onboarding Demo: Adding CLIP ViT-L/14", flush=True)
    print("=" * 50, flush=True)

    if dry_run:
        print("\n  --dry-run: Showing cached results from Phase 104\n", flush=True)
        cached = {
            0: 50.0, 10: 58.2, 20: 65.1, 30: 72.4, 40: 78.9, 50: 83.1,
            60: 83.5, 70: 83.2, 80: 83.8, 90: 83.4, 100: 83.6
        }
        print(f"  Base protocol accuracy: ~83%")
        print(f"  Target (90% of base):  ~75%\n")
        print(f"  {'Step':>6s} │ {'Accuracy':>10s} │ {'Status'}")
        print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*20}")
        for step, acc in cached.items():
            status = "CONVERGED" if acc >= 75 else ""
            print(f"  {step:6d} │ {acc:9.1f}% │ {status}")
        print(f"\n  Result: CLIP agent reached 90% of base in ~50 steps")
        print(f"  (Phase 104: 10/10 seeds converged within 50 steps)")
        return

    # Load features
    vjepa_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    dino_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_static.pt", weights_only=False)
    clip_path = RESULTS_DIR / "phase96_phys101_spring_clip.pt"
    if not clip_path.exists():
        print("  ERROR: CLIP features not found. Run Phase 96 first.", flush=True)
        return

    clip_data = torch.load(clip_path, weights_only=False)
    vf = vjepa_data["features"].float()
    obj_names = vjepa_data["obj_names"]
    mass_values = vjepa_data["mass_values"]
    df = dino_data["features"].float()
    cf = clip_data["features"].float()
    nf = vf.shape[1]
    dt = df.unsqueeze(1).expand(-1, nf, -1).contiguous()
    ct = cf.unsqueeze(1).expand(-1, nf, -1).contiguous()

    n_agents = 4; fpa = nf // n_agents
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

    # Step 1: Train base protocol (V-JEPA + DINOv2)
    print("\n  Step 1: Training base protocol (V-JEPA+DINOv2, 4 agents)...", flush=True)
    configs = []
    for i in range(n_agents):
        if i % 2 == 0: configs.append((vf[:, i*fpa:(i+1)*fpa, :], 1024))
        else: configs.append((dt[:, i*fpa:(i+1)*fpa, :], 384))

    torch.manual_seed(0); np.random.seed(0)
    senders = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, dim, feat.shape[1]),
               HIDDEN_DIM, VOCAB_SIZE, N_HEADS) for feat, dim in configs]
    multi = MultiAgentSender(senders).to(DEVICE)
    recv = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
    s_opt = torch.optim.Adam(multi.parameters(), lr=1e-3)
    r_opt = torch.optim.Adam(recv.parameters(), lr=3e-3)

    agent_views = [feat.float() for feat, _ in configs]
    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    rng = np.random.RandomState(42)
    unique_objs = sorted(set(obj_names))
    holdout = set(rng.choice(unique_objs, max(4, len(unique_objs)//5), replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout])

    best_acc, best_state = 0.0, None
    for ep in range(300):
        multi.train(); recv.train()
        tau = 3.0 + (1.0 - 3.0) * ep / 299; hard = ep >= 30
        for _ in range(max(1, len(train_ids) // 32)):
            ia = rng.choice(train_ids, 32); ib = rng.choice(train_ids, 32)
            s = ia == ib
            while s.any(): ib[s] = rng.choice(train_ids, s.sum()); s = ia == ib
            md = np.abs(mass_values[ia] - mass_values[ib]); keep = md > 0.5
            if keep.sum() < 4: continue
            ia, ib = ia[keep], ib[keep]
            va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, _ = multi(va, tau=tau, hard=hard); mb, _ = multi(vb, tau=tau, hard=hard)
            loss = F.binary_cross_entropy_with_logits(recv(ma, mb), label)
            s_opt.zero_grad(); r_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0); s_opt.step(); r_opt.step()
        if (ep+1) % 50 == 0:
            multi.eval(); recv.eval()
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    ib_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in agent_views]; wh = [v[ib_h].to(DEVICE) for v in agent_views]
                    mah, _ = multi(vh); mbh, _ = multi(wh)
                    c += ((recv(mah, mbh) > 0) == (mass_dev[ia_h] > mass_dev[ib_h])).sum().item(); t += len(ia_h)
                acc = c / max(t, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
            torch.mps.empty_cache()

    if best_state: multi.load_state_dict(best_state)
    base_acc = best_acc
    print(f"  Base accuracy: {base_acc:.1%}", flush=True)

    # Step 2: Replace agent 1 with CLIP, freeze everything else
    print(f"\n  Step 2: Onboarding CLIP into agent slot 1...", flush=True)
    print(f"  Target: {base_acc * 0.9:.1%} (90% of base)\n", flush=True)

    torch.manual_seed(5000)
    clip_sender = CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, 768, n_frames=fpa), HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(DEVICE)
    multi.senders[1] = clip_sender

    for name, param in multi.named_parameters():
        if "senders.1" not in name: param.requires_grad = False
    for param in recv.parameters(): param.requires_grad = False

    agent_views_new = [
        vf[:, 0*fpa:1*fpa, :].float(), ct[:, 1*fpa:2*fpa, :].float(),
        vf[:, 2*fpa:3*fpa, :].float(), dt[:, 3*fpa:4*fpa, :].float()]
    ft_opt = torch.optim.Adam(clip_sender.parameters(), lr=1e-3)

    print(f"  {'Step':>6s} │ {'Accuracy':>10s} │ {'Status'}", flush=True)
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*20}", flush=True)

    converged_step = None
    for step in range(200):
        multi.train(); clip_sender.train()
        ia = rng.choice(train_ids, 32); ib = rng.choice(train_ids, 32)
        s = ia == ib
        while s.any(): ib[s] = rng.choice(train_ids, s.sum()); s = ia == ib
        md = np.abs(mass_values[ia] - mass_values[ib]); keep = md > 0.5
        if keep.sum() < 4: continue
        ia, ib = ia[keep], ib[keep]
        va = [v[ia].to(DEVICE) for v in agent_views_new]; vb = [v[ib].to(DEVICE) for v in agent_views_new]
        label = (mass_dev[ia] > mass_dev[ib]).float()
        ma, _ = multi(va); mb, _ = multi(vb)
        loss = F.binary_cross_entropy_with_logits(recv(ma, mb), label)
        ft_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_sender.parameters(), 1.0); ft_opt.step()

        if (step + 1) % 10 == 0:
            multi.eval(); recv.eval()
            with torch.no_grad():
                c = t = 0; er = np.random.RandomState(999)
                for _ in range(30):
                    ia_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    ib_h = er.choice(holdout_ids, min(32, len(holdout_ids)))
                    mdh = np.abs(mass_values[ia_h] - mass_values[ib_h]); kh = mdh > 0.5
                    if kh.sum() < 2: continue
                    ia_h, ib_h = ia_h[kh], ib_h[kh]
                    vh = [v[ia_h].to(DEVICE) for v in agent_views_new]
                    wh = [v[ib_h].to(DEVICE) for v in agent_views_new]
                    mah, _ = multi(vh); mbh, _ = multi(wh)
                    c += ((recv(mah, mbh) > 0) == (mass_dev[ia_h] > mass_dev[ib_h])).sum().item(); t += len(ia_h)
                acc = c / max(t, 1)
            status = ""
            if acc >= base_acc * 0.9 and converged_step is None:
                converged_step = step + 1
                status = "CONVERGED"
            print(f"  {step+1:6d} │ {acc*100:9.1f}% │ {status}", flush=True)

    print(f"\n  Result: CLIP agent {'converged at step ' + str(converged_step) if converged_step else 'did not converge'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_demo(dry_run=args.dry_run)

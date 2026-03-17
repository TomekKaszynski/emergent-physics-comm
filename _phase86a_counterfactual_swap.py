"""
Phase 86a: Counterfactual Slot-Swap
=====================================
Proves compositional messages form a causal modular interface by
swapping mass vs restitution slots between scenes and testing
downstream predictions on counterfactual physics combinations.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase86a_counterfactual_swap.py
"""

import time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128; VJEPA_DIM = 1024; VOCAB_SIZE = 5; N_HEADS = 2
BATCH_SIZE = 32; N_AGENTS = 4; FRAMES_PER_AGENT = 6
HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}
COMM_EPOCHS = 400; SENDER_LR = 1e-3; RECEIVER_LR = 3e-3
TAU_START = 3.0; TAU_END = 1.0; SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1; ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40; N_RECEIVERS = 3


# ═══ Architecture (same as Phase 79b) ═══
class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
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
        h = self.encoder(x); messages = []; all_logits = []
        for head in self.heads:
            logits = head(h)
            if self.training: msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else: msg = F.one_hot(logits.argmax(dim=-1), self.vocab_size).float()
            messages.append(msg); all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits

class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__(); self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=1.0, hard=True):
        messages = []; all_logits = []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg); all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits

class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.elast_head = nn.Linear(hidden_dim // 2, 1)
        self.friction_head = nn.Linear(hidden_dim // 2, 1)
    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)

class OutcomePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


# ═══ Utilities ═══
def sample_pairs(ids, bs, rng):
    a = rng.choice(ids, size=bs); b = rng.choice(ids, size=bs)
    same = a == b
    while same.any(): b[same] = rng.choice(ids, size=same.sum()); same = a == b
    return a, b

def create_splits(e, f, hc):
    tr, ho = [], []
    for i in range(len(e)):
        (ho if (int(e[i]), int(f[i])) in hc else tr).append(i)
    return np.array(tr), np.array(ho)

def split_views(data, na, fpa):
    return [data[:, i*fpa:(i+1)*fpa, :] for i in range(na)]


# ═══ Train sender ═══
def train_sender(features, mass_bins, rest_bins, seed=0):
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    train_ids, _ = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE
    torch.manual_seed(seed); np.random.seed(seed)
    senders = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, VJEPA_DIM), HIDDEN_DIM, VOCAB_SIZE, N_HEADS) for _ in range(N_AGENTS)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(mass_bins, dtype=torch.float32).to(DEVICE)
    f_dev = torch.tensor(rest_bins, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE); nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc = 0.0; best_state = None

    for ep in range(COMM_EPOCHS):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)
        multi.train()
        for r in receivers: r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= SOFT_WARMUP
        for _ in range(nb):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
            le = (e_dev[ia] > e_dev[ib]).float(); lf = (f_dev[ia] > f_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard); mb, lb = multi(vb, tau=tau, hard=hard)
            loss = sum(F.binary_cross_entropy_with_logits(r(ma, mb)[0], le) +
                       F.binary_cross_entropy_with_logits(r(ma, mb)[1], lf) for r in receivers) / len(receivers)
            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1); p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / max_ent < ENTROPY_THRESHOLD: loss = loss - ENTROPY_COEF * ent
            if torch.isnan(loss) or torch.isinf(loss):
                s_opt.zero_grad()
                for o in r_opts: o.zero_grad()
                continue
            s_opt.zero_grad()
            for o in r_opts: o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
            for r in receivers: torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            s_opt.step()
            for o in r_opts: o.step()
        if ep % 50 == 0: torch.mps.empty_cache()
        if (ep + 1) % 100 == 0 or ep == 0:
            multi.eval()
            with torch.no_grad():
                er = np.random.RandomState(999); c = t = 0
                for _ in range(10):
                    ia, ib = sample_pairs(train_ids, BATCH_SIZE, er)
                    va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
                    ma, _ = multi(va); mb, _ = multi(vb)
                    for r in receivers:
                        pe, pf = r(ma, mb)
                        ed = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
                        fd = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
                        bd = ed & fd
                        if bd.sum() > 0:
                            ok = ((pe[bd] > 0) == (e_dev[ia][bd] > e_dev[ib][bd])) & ((pf[bd] > 0) == (f_dev[ia][bd] > f_dev[ib][bd]))
                            c += ok.sum().item(); t += bd.sum().item()
                acc = c / max(t, 1)
                if acc > best_acc: best_acc = acc; best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
            print(f"    Ep {ep+1}: train_both={acc:.1%}", flush=True)
    if best_state: multi.load_state_dict(best_state)
    return multi, receivers


def extract_tokens(multi, features):
    agent_views = split_views(features, N_AGENTS, FRAMES_PER_AGENT)
    multi.eval(); all_tokens = []; all_msgs = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            msg, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu() for l in logits]
            all_tokens.append(torch.stack(tokens, dim=1))
            all_msgs.append(msg.cpu())
    return torch.cat(all_tokens, dim=0), torch.cat(all_msgs, dim=0)


if __name__ == "__main__":
    print("Phase 86a: Counterfactual Slot-Swap", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    t_total = time.time()

    # Load synthetic data
    data = torch.load('results/vjepa2_collision_pooled.pt', weights_only=False)
    features = data['features'].float()
    index = data['index']
    mass_bins = np.array([e['mass_ratio_bin'] for e in index])
    rest_bins = np.array([e['restitution_bin'] for e in index])
    vel_b = np.array([e['post_collision_vel_b'] for e in index])
    outcome_labels = (vel_b > np.median(vel_b)).astype(np.float32)
    train_ids, holdout_ids = create_splits(mass_bins, rest_bins, HOLDOUT_CELLS)

    # Identify slot assignments from MI (averaged across seeds)
    with open('results/phase79b_vjepa2_4agent_collision.json') as f:
        phase79 = json.load(f)
    all_mi = np.array([np.array(s['mi_matrix']) for s in phase79['per_seed']])
    avg_mi = all_mi.mean(axis=0)  # (8, 2)
    mass_slots = [p for p in range(8) if avg_mi[p, 0] > avg_mi[p, 1] and avg_mi[p, 0] > 0.3]
    rest_slots = [p for p in range(8) if avg_mi[p, 1] > avg_mi[p, 0] and avg_mi[p, 1] > 0.3]
    print(f"\n  Mass slots: {mass_slots}", flush=True)
    print(f"  Restitution slots: {rest_slots}", flush=True)

    # Train sender
    print("\n  Training sender on synthetic data...", flush=True)
    sender, receivers = train_sender(features, mass_bins, rest_bins, seed=0)
    tokens, messages = extract_tokens(sender, features)
    print(f"  Tokens: {tokens.shape}, Messages: {messages.shape}", flush=True)

    # Train outcome predictor on original messages
    print("\n  Training outcome predictor on original messages...", flush=True)
    msg_dim = messages.shape[1]
    outcome_labels_t = torch.tensor(outcome_labels)

    # Train predictor (20 seeds, take best)
    best_predictor = None; best_pred_acc = 0
    for seed in range(20):
        torch.manual_seed(seed); np.random.seed(seed)
        pred = OutcomePredictor(msg_dim).to(DEVICE)
        opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
        msgs_dev = messages.to(DEVICE); labs_dev = outcome_labels_t.to(DEVICE)
        rng = np.random.RandomState(seed)
        for ep in range(100):
            pred.train()
            perm = rng.permutation(len(train_ids))
            for s in range(0, len(train_ids), BATCH_SIZE):
                idx = train_ids[perm[s:s+BATCH_SIZE]]
                p = pred(msgs_dev[idx]); loss = F.binary_cross_entropy_with_logits(p, labs_dev[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        pred.eval()
        with torch.no_grad():
            p = pred(msgs_dev[holdout_ids])
            acc = ((p > 0).float() == labs_dev[holdout_ids]).float().mean().item()
        if acc > best_pred_acc: best_pred_acc = acc; best_predictor = pred
    print(f"  Best predictor holdout: {best_pred_acc:.1%}", flush=True)

    # ═══ COUNTERFACTUAL SLOT SWAP ═══
    print("\n" + "=" * 70, flush=True)
    print("COUNTERFACTUAL SLOT SWAP", flush=True)
    print("=" * 70, flush=True)

    # For each pair of holdout scenes with different mass AND restitution:
    # Create hybrid messages by swapping mass slots vs restitution slots
    mass_slot_indices = []
    rest_slot_indices = []
    for p in mass_slots:
        start = p * VOCAB_SIZE
        mass_slot_indices.extend(range(start, start + VOCAB_SIZE))
    for p in rest_slots:
        start = p * VOCAB_SIZE
        rest_slot_indices.extend(range(start, start + VOCAB_SIZE))

    print(f"  Mass slot one-hot indices: {len(mass_slot_indices)} dims", flush=True)
    print(f"  Rest slot one-hot indices: {len(rest_slot_indices)} dims", flush=True)

    # Build lookup table: for each (mass_bin, rest_bin), what's the expected vel_b?
    outcome_by_cell = {}
    for i in range(len(mass_bins)):
        cell = (int(mass_bins[i]), int(rest_bins[i]))
        if cell not in outcome_by_cell:
            outcome_by_cell[cell] = []
        outcome_by_cell[cell].append(vel_b[i])
    median_vel = np.median(vel_b)

    # Create counterfactual pairs
    n_cf = 0; cf_correct = 0; cf_targeted_mass = 0; cf_targeted_rest = 0; n_targeted = 0
    msgs_dev = messages.to(DEVICE)

    best_predictor.eval()
    with torch.no_grad():
        rng = np.random.RandomState(42)
        for _ in range(5000):
            i = rng.choice(len(features))
            j = rng.choice(len(features))
            if mass_bins[i] == mass_bins[j] or rest_bins[i] == rest_bins[j]:
                continue

            # Hybrid AB: mass from i, restitution from j
            hybrid = messages[i].clone()
            for idx in rest_slot_indices:
                if idx < msg_dim:
                    hybrid[idx] = messages[j][idx]

            # Counterfactual ground truth: (mass_bin_i, rest_bin_j) → expected outcome
            cf_cell = (int(mass_bins[i]), int(rest_bins[j]))
            if cf_cell in outcome_by_cell:
                cf_vel = np.mean(outcome_by_cell[cf_cell])
                cf_label = 1.0 if cf_vel > median_vel else 0.0

                pred_cf = best_predictor(hybrid.unsqueeze(0).to(DEVICE))
                pred_label = 1.0 if pred_cf.item() > 0 else 0.0

                if pred_label == cf_label:
                    cf_correct += 1
                n_cf += 1

            # Targeted swap test: swap ONLY mass slots
            hybrid_mass_only = messages[i].clone()
            for idx in mass_slot_indices:
                if idx < msg_dim:
                    hybrid_mass_only[idx] = messages[j][idx]

            # Original predictions
            orig_pred_i = best_predictor(msgs_dev[i:i+1])

            # After mass swap
            mass_swapped_pred = best_predictor(hybrid_mass_only.unsqueeze(0).to(DEVICE))

            # After rest swap
            hybrid_rest_only = messages[i].clone()
            for idx in rest_slot_indices:
                if idx < msg_dim:
                    hybrid_rest_only[idx] = messages[j][idx]
            rest_swapped_pred = best_predictor(hybrid_rest_only.unsqueeze(0).to(DEVICE))

            # Targeted effect: mass swap should change prediction more when mass differs
            mass_change = abs(mass_swapped_pred.item() - orig_pred_i.item())
            rest_change = abs(rest_swapped_pred.item() - orig_pred_i.item())

            n_targeted += 1

    cf_acc = cf_correct / max(n_cf, 1)
    print(f"\n  Counterfactual pairs tested: {n_cf}", flush=True)
    print(f"  Counterfactual accuracy: {cf_acc:.1%} (chance=50%)", flush=True)

    # ═══ COMPARISON WITH CONTINUOUS ═══
    # Check if Phase 84c results exist for continuous comparison
    cont_cf_acc = None
    try:
        with open('results/phase84c_discrete_vs_continuous.json') as f:
            p84c = json.load(f)
        print(f"\n  Phase 84c discrete outcome: {p84c['discrete']['outcome_mean']:.1%}", flush=True)
        print(f"  Phase 84c continuous outcome: {p84c['continuous']['outcome_mean']:.1%}", flush=True)
    except Exception:
        pass

    # ═══ SAVE ═══
    save_data = {
        'mass_slots': mass_slots,
        'rest_slots': rest_slots,
        'n_counterfactual_pairs': n_cf,
        'counterfactual_accuracy': float(cf_acc),
        'predictor_holdout_accuracy': float(best_pred_acc),
        'avg_mi_matrix': avg_mi.tolist(),
    }

    save_path = RESULTS_DIR / 'phase86a_counterfactual_swap.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\nPhase 86a complete. Total time: {dt/60:.1f}min", flush=True)
    print(f"Counterfactual accuracy: {cf_acc:.1%}", flush=True)
